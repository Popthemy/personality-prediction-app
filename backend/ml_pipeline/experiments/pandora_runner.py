"""
Django-free factorial experiment engine for the PANDORA-big5 pipeline.
=====================================================================

This module reproduces the train/eval glue that normally lives in the
Django-coupled ``pipeline_orchestrator`` (which reads/writes the ORM), but drives
the *exact same* Django-free service classes directly on in-memory PANDORA data.
It is what the Google Colab notebook imports and calls; nothing here imports
Django.

PANDORA maps cleanly onto the training problem:
    one proxy-user  = one "volunteer"
    their comments  = their "posts"
    their (O,C,E,A,N) percentile scores = the ground-truth labels

Factorial design (2 x 2 x 2 = 8 conditions)
-------------------------------------------
Both models are trained under **every** combination of the two pipeline factors,
so the effect of each factor is isolated and the two models are directly
comparable on a shared metric:

    factor 1  selection : baseline-select   vs  Q-learning-select
    factor 2  GAN        : no augmentation   vs  GAN-augmented training fold
    factor 3  model      : Lasso (continuous OCEAN) vs LSTM (continuous OCEAN)

    condition id          selection   gan    model
    --------------------  ----------  -----  -----
    lasso_baseline        baseline    off    Lasso
    lasso_qlearn          qlearning   off    Lasso
    lasso_baseline_gan    baseline    on     Lasso
    lasso_qlearn_gan      qlearning   on     Lasso
    lstm_baseline         baseline    off    LSTM
    lstm_qlearn           qlearning   off    LSTM
    lstm_baseline_gan     baseline    on     LSTM
    lstm_qlearn_gan       qlearning   on     LSTM

Why this design (so the claims are defensible)
----------------------------------------------
  * **Both Lasso and LSTM are continuous 5-output OCEAN regressors** and run in
    all four (selection x GAN) cells. They are compared at matched conditions,
    never confounded with selection or augmentation. Binary High/Low
    thresholding is a downstream decision analysis on those continuous scores.
  * **Isolated factor effects.** ``factor_effects()`` reports the Q-learning
    effect (qlearning - baseline, holding model+GAN fixed) and the GAN effect
    (gan - no-gan, holding model+selection fixed) as matched-pair deltas.
  * **One shared metric family.** Both models emit ``(N, 5)`` continuous
    predictions on the same held-out test participants. ``metrics_engine``
    computes MAE/MSE/RMSE/R²/Pearson, the High/Low threshold sweep, ROC-AUC,
    and PR-AUC. The runner does not re-derive any metric formula.
  * **One participant-level train/val/test split shared by all 8 conditions**,
    and an **identical Q-learning policy** across the four Q-learning cells.
    Baseline and Q-learning use the same selection budget ``top_k``.
  * **GAN augmentation** uses ``services/gan_augmenter.py``, a joint
    (embedding, OCEAN) GAN. It is ``fit`` on training participants only
    (never val/test) and ``generate`` returns paired ``(X_syn, y_syn)``.
    Those pairs stay row-aligned; synthetic embeddings are never given
    arbitrary real-user labels.
  * **Threshold selection is not done on the test set.** A High/Low cutoff
    and a decision threshold are chosen on validation, frozen, then applied
    once to test. ``evaluate()`` may still emit the full test sweep for
    plots; that sweep is analysis-only.
  * BERT encoding -- the dominant cost -- is cached on disk by text hash, so
    re-runs (and the comments shared between the baseline / Q-learning
    selections) are near-instant.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Django-free service classes (reused unchanged). Importing these triggers
# backend/ml_pipeline/services/__init__.py, which is Django-free (Orchestrator
# is exposed lazily there via PEP 562). torch/transformers/sklearn load here.
#
# Interface binding (pandora branch) -- these are the ACTUAL service seams the
# Colab runner drives, so the experiment numbers match what the Django pipeline
# will produce once the ML side is finalized:
#   qlearning_agent.QLearningAgent / run_training_loop  -> comment selection
#   bert_encoder.BERTEncoder                            -> 768-d embeddings
#   gan_augmenter.GANAugmenter                          -> joint (embedding, OCEAN) GAN
#   lasso_regressor.LassoTrainer                        -> per-trait ElasticNet
#   lstm_classifier.LSTMTrainer                         -> continuous 5-output OCEAN LSTM
#   metrics_engine.evaluate (+ component fns)           -> canonical metrics
from backend.ml_pipeline.services.data.pandora import PreparedUserComments
from backend.ml_pipeline.services.qlearning_agent import QLearningAgent, run_training_loop
from backend.ml_pipeline.services.bert_encoder import BERTEncoder
from backend.ml_pipeline.services.gan_augmenter import GANAugmenter
from backend.ml_pipeline.services.lasso_regressor import LassoTrainer
from backend.ml_pipeline.services.lstm_classifier import LSTMTrainer, set_seed
from backend.ml_pipeline.services import metrics_engine as me

logger = logging.getLogger("ml_pipeline")

# ---------------------------------------------------------------------------
# Trait bookkeeping. The dataset stores columns O,C,E,A,N in that order; these
# two tuples stay index-aligned so labels_unit[:, i] <-> OCEAN_TRAITS[i].
# ---------------------------------------------------------------------------
TRAIT_KEYS: Tuple[str, ...] = ("O", "C", "E", "A", "N")
OCEAN_TRAITS: List[str] = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]

# Full 2 x 2 x 2 factorial: {selection} x {gan} x {model}. Both models appear in
# every (selection, gan) cell so Lasso-vs-LSTM is a matched comparison and the
# Q-learning / GAN effects are isolable per model. Insertion order is grouped by
# model then selection then gan for stable, readable tables.
EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "lasso_baseline":     {"selection": "baseline",  "gan": False, "model": "lasso",
                           "label": "Lasso | baseline-select"},
    "lasso_qlearn":       {"selection": "qlearning", "gan": False, "model": "lasso",
                           "label": "Lasso | Q-learning-select"},
    "lasso_baseline_gan": {"selection": "baseline",  "gan": True,  "model": "lasso",
                           "label": "Lasso | baseline-select + GAN"},
    "lasso_qlearn_gan":   {"selection": "qlearning", "gan": True,  "model": "lasso",
                           "label": "Lasso | Q-learning-select + GAN"},
    "lstm_baseline":      {"selection": "baseline",  "gan": False, "model": "lstm",
                           "label": "LSTM | baseline-select"},
    "lstm_qlearn":        {"selection": "qlearning", "gan": False, "model": "lstm",
                           "label": "LSTM | Q-learning-select"},
    "lstm_baseline_gan":  {"selection": "baseline",  "gan": True,  "model": "lstm",
                           "label": "LSTM | baseline-select + GAN"},
    "lstm_qlearn_gan":    {"selection": "qlearning", "gan": True,  "model": "lstm",
                           "label": "LSTM | Q-learning-select + GAN"},
}

# The four matched (selection, gan) cells in which Lasso and LSTM are compared.
_CELLS: Tuple[Tuple[str, bool], ...] = (
    ("baseline", False),
    ("qlearning", False),
    ("baseline", True),
    ("qlearning", True),
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """All knobs for one factorial sweep. Defaults target a fast report-figure run."""

    # Sampling ----------------------------------------------------------------
    sample_n_users: int = 40          # small fixed sample for fast iteration
    min_comments_per_user: int = 5    # proxy-users with fewer are skipped
    seed: int = 42

    # Selection ---------------------------------------------------------------
    top_k: int = 10                   # comments kept per user (selection budget)
    qlearning_train_epochs: int = 3   # 0 => cold (approx random) policy; >0 trains the agent

    # Split -------------------------------------------------------------------
    val_ratio: float = 0.2            # participant-level validation fraction
    test_ratio: float = 0.2           # participant-level final-evaluation fraction

    # GAN (joint embedding+OCEAN GAN, services/gan_augmenter.py) --------------
    synthetic_weight: float = 0.35    # down-weight synthetic rows vs real train rows
    gan_latent_dim: int = 64
    gan_hidden_dim: int = 128
    gan_epochs: int = 150             # adversarial training epochs (small data -> fast)
    gan_batch_size: int = 16
    gan_learning_rate: float = 2e-4

    # BERT --------------------------------------------------------------------
    bert_max_length: int = 256        # comments are short; 256 is faster than 512

    # Lasso -------------------------------------------------------------------
    lasso_alpha: float = 0.001
    lasso_l1_ratio: float = 0.5
    lasso_max_iter: int = 10000
    lasso_regularization: str = "elasticnet"

    # LSTM --------------------------------------------------------------------
    lstm_epochs: int = 35
    lstm_batch_size: int = 4
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_learning_rate: float = 1e-3

    # I/O (typically Google Drive paths) --------------------------------------
    output_dir: Optional[str] = None           # artifacts (metrics, model states)
    embedding_cache_dir: Optional[str] = None  # BERT embedding cache (.npy)


# ---------------------------------------------------------------------------
# Sample & feature containers
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """A fixed, deterministic sample of proxy-users ready for experimentation."""

    user_ids: List[str]
    texts: List[List[str]]        # cleaned comment strings per user
    labels_raw: np.ndarray        # (n, 5) original O,C,E,A,N
    labels_unit: np.ndarray       # (n, 5) scaled to [0,1]
    scale: str                    # human-readable detected scale (for the report)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)


@dataclass
class Features:
    """Per-user features under one selection policy."""

    mode: str                       # 'baseline' | 'qlearning'
    pooled: np.ndarray              # (n, 768) mean-pooled selected-comment vectors
    sequences: List[np.ndarray]     # per user: (k_i, 768) ordered selected vectors
    n_selected: List[int]           # how many comments each user actually contributed


# ---------------------------------------------------------------------------
# Label scaling
# ---------------------------------------------------------------------------

def scale_labels_to_unit(labels_raw: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Auto-detect the O/C/E/A/N value range and map it to [0, 1].

    PANDORA's exact scale isn't documented in the parquet export, so this
    detects the common cases from the observed min/max across the whole sampled
    label matrix (kept consistent across all five traits):

        [0, 1]         -> already unit                   (identity)
        [1, 5]         -> Likert                          (v - 1) / 4
        [0, 100]       -> percentile                      v / 100
        anything else  -> min-max normalize (e.g. z-scores) + warn

    The notebook's EDA cell shows the histogram so this choice is auditable.
    """
    finite = labels_raw[np.isfinite(labels_raw)]
    if finite.size == 0:
        raise ValueError("Label matrix has no finite values to scale.")
    lo = float(np.min(finite))
    hi = float(np.max(finite))

    if lo >= 0.0 and hi <= 1.0:
        unit, scale = labels_raw.copy(), "unit[0,1] (identity)"
    elif lo >= 1.0 and hi <= 5.0:
        unit, scale = (labels_raw - 1.0) / 4.0, "likert[1,5] -> (v-1)/4"
    elif 5.0 < hi <= 100.0 and lo >= 0.0:
        unit, scale = labels_raw / 100.0, "percentile[0,100] -> v/100"
    else:
        unit = (labels_raw - lo) / (hi - lo + 1e-12)
        scale = f"min-max[{lo:.3f},{hi:.3f}] (non-standard range)"
        logger.warning(
            "O/C/E/A/N range [%.3f, %.3f] didn't match a known scale; "
            "falling back to min-max normalization.", lo, hi,
        )

    unit = np.clip(unit, 0.0, 1.0)
    logger.info("Label scale detected: %s", scale)
    return unit, scale


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_users(prepared: List[PreparedUserComments], cfg: ExperimentConfig) -> Sample:
    """
    Deterministically pick a small fixed sample of proxy-users that have traits
    and at least ``min_comments_per_user`` cleaned comments.

    Determinism: candidates are sorted by user_id, then sampled with a local
    ``random.Random(seed)`` so the sample is independent of any other RNG state.
    """
    eligible = [
        u for u in prepared
        if u.traits is not None and len(u.comments) >= cfg.min_comments_per_user
    ]
    eligible.sort(key=lambda u: u.user_id)

    if not eligible:
        raise ValueError(
            "No proxy-users satisfy the sampling filter "
            f"(need traits + >= {cfg.min_comments_per_user} comments)."
        )

    rng = random.Random(cfg.seed)
    if len(eligible) > cfg.sample_n_users:
        chosen = rng.sample(eligible, cfg.sample_n_users)
    else:
        logger.warning(
            "Only %d eligible users (< requested %d); using all of them.",
            len(eligible), cfg.sample_n_users,
        )
        chosen = eligible
    chosen.sort(key=lambda u: u.user_id)  # stable order for splitting/reporting

    user_ids: List[str] = []
    texts: List[List[str]] = []
    labels_raw: List[List[float]] = []
    for u in chosen:
        user_texts = [c.cleaned_text for c in u.comments if c.cleaned_text]
        if not user_texts:
            continue
        user_ids.append(u.user_id)
        texts.append(user_texts)
        t = u.traits
        labels_raw.append([float(t.O), float(t.C), float(t.E), float(t.A), float(t.N)])

    labels_arr = np.asarray(labels_raw, dtype=float)
    labels_unit, scale = scale_labels_to_unit(labels_arr)

    logger.info(
        "Sampled %d users | comments/user: min=%d max=%d | label scale: %s",
        len(user_ids),
        min(len(t) for t in texts),
        max(len(t) for t in texts),
        scale,
    )
    return Sample(
        user_ids=user_ids,
        texts=texts,
        labels_raw=labels_arr,
        labels_unit=labels_unit,
        scale=scale,
    )


def prepare_sample(prepared: List[PreparedUserComments], cfg: ExperimentConfig) -> Sample:
    """Public alias for :func:`sample_users` (kept as the plan's named entry point)."""
    return sample_users(prepared, cfg)


# ---------------------------------------------------------------------------
# BERT embedding with a disk cache
# ---------------------------------------------------------------------------

def get_encoder() -> BERTEncoder:
    """Instantiate the (globally-cached) BERT encoder. GPU-aware."""
    return BERTEncoder()


def _cache_path(cache_dir: Path, text: str, max_length: int) -> Path:
    key = hashlib.sha1(f"{max_length}␟{text}".encode("utf-8", "replace")).hexdigest()
    return cache_dir / key[:2] / f"{key}.npy"


def _embed_one(encoder: Any, text: str, cfg: ExperimentConfig) -> np.ndarray:
    """
    Encode one comment to a 768-vec, using the Drive-backed cache when configured.

    ``encoder`` only needs an ``encode_text(text, max_length) -> {'embedding': [...]}``
    method, so a fake encoder can be injected for local testing without BERT.
    """
    cache_dir = Path(cfg.embedding_cache_dir) if cfg.embedding_cache_dir else None
    if cache_dir is not None:
        path = _cache_path(cache_dir, text, cfg.bert_max_length)
        if path.exists():
            try:
                return np.load(path)
            except Exception as exc:  # corrupt cache entry — re-encode
                logger.debug("Cache read failed for %s (%s); re-encoding.", path, exc)

    vec = np.asarray(
        encoder.encode_text(text, max_length=cfg.bert_max_length)["embedding"],
        dtype=np.float32,
    )

    if cache_dir is not None:
        path = _cache_path(cache_dir, text, cfg.bert_max_length)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.save(path, vec)
        except Exception as exc:  # caching is best-effort
            logger.debug("Cache write failed for %s (%s).", path, exc)

    return vec


# ---------------------------------------------------------------------------
# Selection + feature building
# ---------------------------------------------------------------------------

def _baseline_select(texts: List[str], top_k: int) -> List[str]:
    """Control policy: the first ``top_k`` comments (matches BaselineSelector)."""
    return texts[:top_k]


def _qlearning_select(agent: QLearningAgent, texts: List[str], top_k: int) -> List[str]:
    """Greedy Q-policy selection; falls back to first comment if it selects none."""
    chosen = agent.select_comments(texts, top_k=top_k, training=False)
    selected = [c["text"] for c in chosen]
    if not selected:  # cold/degenerate policy chose nothing — keep the user usable
        logger.debug("Q-policy selected 0 comments; falling back to first comment.")
        selected = texts[:1]
    return selected


def train_qlearning_agent(sample: Sample, cfg: ExperimentConfig) -> QLearningAgent:
    """
    Train ONE Q-learning agent on the whole sample's comments, reused (greedy)
    for selection in every Q-learning condition so those cells share an
    identical policy.
    """
    agent = QLearningAgent(alpha=0.1, gamma=0.99, epsilon=0.1)
    if cfg.qlearning_train_epochs <= 0:
        logger.warning("qlearning_train_epochs=0 -> cold policy (approx random selection).")
        return agent

    set_seed(cfg.seed)  # np.random drives epsilon-greedy exploration
    run_training_loop(
        agent,
        comment_batches=sample.texts,
        n_epochs=cfg.qlearning_train_epochs,
        max_selected=cfg.top_k,
    )
    logger.info("Q-learning agent trained: Q-table has %d states.", len(agent.q_table))
    return agent


def build_features(
    sample: Sample,
    mode: str,
    cfg: ExperimentConfig,
    encoder: Any,
    agent: Optional[QLearningAgent] = None,
) -> Features:
    """
    Select comments per user under ``mode`` ('baseline'|'qlearning'), BERT-encode
    the selected comments (cached), and produce both the mean-pooled vector
    (Lasso) and the ordered sequence (LSTM) per user.
    """
    if mode == "qlearning" and agent is None:
        raise ValueError("qlearning feature mode requires a trained agent.")

    pooled: List[np.ndarray] = []
    sequences: List[np.ndarray] = []
    n_selected: List[int] = []

    for idx, user_texts in enumerate(sample.texts):
        if mode == "baseline":
            selected = _baseline_select(user_texts, cfg.top_k)
        elif mode == "qlearning":
            selected = _qlearning_select(agent, user_texts, cfg.top_k)
        else:
            raise ValueError(f"Unknown selection mode: {mode!r}")

        vecs = [_embed_one(encoder, t, cfg) for t in selected]
        seq = np.vstack(vecs).astype(np.float32)  # (k, 768)
        sequences.append(seq)
        pooled.append(seq.mean(axis=0))
        n_selected.append(len(selected))

        if (idx + 1) % 10 == 0:
            logger.info("  [%s] encoded %d/%d users", mode, idx + 1, sample.n_users)

    logger.info(
        "Features built (%s): pooled=%s, mean comments/user=%.1f",
        mode, (sample.n_users, 768), float(np.mean(n_selected)) if n_selected else 0.0,
    )
    return Features(
        mode=mode,
        pooled=np.vstack(pooled).astype(np.float32),
        sequences=sequences,
        n_selected=n_selected,
    )


# ---------------------------------------------------------------------------
# Train/val split (participant-level, shared across all conditions)
# ---------------------------------------------------------------------------

def make_split(
    n_users: int,
    cfg: ExperimentConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic participant-level train/val/test index split, shared by every
    condition so they differ only by their selection/GAN/model factors, not the
    split. No participant appears in more than one fold.
    """
    if n_users < 3:
        raise ValueError("Need at least 3 users to form a train/val/test split.")
    rng = np.random.RandomState(cfg.seed)
    perm = rng.permutation(n_users)
    test_count = max(1, round(n_users * cfg.test_ratio))
    val_count = max(1, round(n_users * cfg.val_ratio))
    if test_count + val_count >= n_users:
        test_count = max(1, n_users - 2)
        val_count = 1
    test_idx = np.sort(perm[:test_count])
    val_idx = np.sort(perm[test_count:test_count + val_count])
    train_idx = np.sort(perm[test_count + val_count:])
    logger.info(
        "Split: train=%d, val=%d, test=%d (of %d users).",
        len(train_idx), len(val_idx), len(test_idx), n_users,
    )
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# GAN augmentation (joint embedding+OCEAN GAN, training participants only)
# ---------------------------------------------------------------------------

def _make_gan(embedding_dim: int, cfg: ExperimentConfig) -> GANAugmenter:
    """
    Build ``GANAugmenter`` from ``services/gan_augmenter.py``.

    OCEAN domain is [0, 1] because the runner trains and evaluates on
    ``labels_unit``. The GAN never sees validation or test participants;
    the caller must pass training pairs only.
    """
    return GANAugmenter(
        embedding_dim=embedding_dim,
        latent_dim=cfg.gan_latent_dim,
        hidden_dim=cfg.gan_hidden_dim,
        epochs=cfg.gan_epochs,
        batch_size=cfg.gan_batch_size,
        learning_rate=cfg.gan_learning_rate,
        seed=cfg.seed,
        ocean_domain_min=0.0,
        ocean_domain_max=1.0,
    )


def _paired_gan_samples(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    cfg: ExperimentConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit the GAN on training (embedding, OCEAN) pairs only and generate the
    same number of paired synthetic samples. ``X_syn[i]`` and ``y_syn[i]``
    come from the same generator forward pass.
    """
    X_tr = np.asarray(X_tr, dtype=np.float32)
    y_tr = np.asarray(y_tr, dtype=np.float32)
    empty_x = np.empty((0, X_tr.shape[1]), dtype=np.float32)
    empty_y = np.empty((0, 5), dtype=np.float32)
    if y_tr.ndim != 2 or y_tr.shape[1] != 5:
        raise ValueError(f"GAN ocean_scores must have shape (N, 5); got {y_tr.shape}")
    if len(X_tr) != len(y_tr):
        raise ValueError(
            f"GAN training pairs must be row-aligned (got {len(X_tr)} embeddings, {len(y_tr)} labels)."
        )
    if len(X_tr) < 2:
        logger.warning("Too few train users (%d) to fit the GAN; skipping augmentation.", len(X_tr))
        return empty_x, empty_y

    gan = _make_gan(X_tr.shape[1], cfg)
    gan.fit(X_tr, ocean_scores=y_tr)
    X_syn, y_syn, _metadata = gan.generate(n_samples=len(X_tr))
    return np.asarray(X_syn, dtype=np.float32), np.asarray(y_syn, dtype=np.float32)


def _select_thresholds_on_validation(
    y_val: np.ndarray,
    pred_val: np.ndarray,
    cutoff: Optional[float] = None,
) -> Dict[str, Any]:
    """Choose one High/Low decision threshold per trait on the validation fold."""
    if cutoff is None:
        cutoff = me.DEFAULT_GROUND_TRUTH_CUTOFF
    per_trait: Dict[str, Any] = {}
    for i, key in enumerate(TRAIT_KEYS):
        y_bin, used = me.derive_binary_ground_truth(y_val[:, i], cutoff)
        sweep = me.sweep_thresholds_on_scores(y_bin, pred_val[:, i])
        per_trait[key] = {
            "split": "validation",
            "ground_truth_cutoff": used,
            "threshold": sweep["best_threshold"],
            "accuracy": sweep["accuracy"],
            "precision": sweep["precision"],
            "recall": sweep["recall"],
            "f1": sweep["f1"],
        }
    return {
        "split": "validation",
        "ground_truth_cutoff": float(cutoff),
        "per_trait": per_trait,
        "aggregate": {
            "f1": _mean([per_trait[k]["f1"] for k in TRAIT_KEYS]),
            "accuracy": _mean([per_trait[k]["accuracy"] for k in TRAIT_KEYS]),
        },
    }


def _score_test_predictions(
    y_test: np.ndarray,
    pred_test: np.ndarray,
    threshold_selection: Dict[str, Any],
    cutoff: Optional[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Score continuous test predictions via ``metrics_engine``.

    The full test threshold sweep / ROC / PR are analysis outputs.
    Official High/Low metrics use the *validation-selected* threshold,
    never a threshold chosen on these test labels.
    """
    if cutoff is None:
        cutoff = me.DEFAULT_GROUND_TRUTH_CUTOFF
    per_trait: Dict[str, Any] = {}
    for i, key in enumerate(TRAIT_KEYS):
        reg = me.compute_regression_metrics(y_test[:, i], pred_test[:, i])
        y_bin, used = me.derive_binary_ground_truth(y_test[:, i], cutoff)
        sweep = me.sweep_thresholds_on_scores(y_bin, pred_test[:, i])
        roc = me.compute_roc_curve_metrics(y_bin, pred_test[:, i])
        pr = me.compute_precision_recall_curve_metrics(y_bin, pred_test[:, i])
        tau = threshold_selection["per_trait"][key]["threshold"]
        official = me.compute_classification_metrics_at_threshold(y_bin, pred_test[:, i], tau)
        per_trait[key] = {
            "mae": _f(reg["mae"]),
            "mse": _f(reg["mse"]),
            "rmse": _f(reg["rmse"]),
            "r2": _f(reg["r2"]),
            "correlation": _f(reg["correlation"]),
            "threshold_sweep": sweep,
            "best_threshold": _f(tau),
            "analysis_best_threshold": sweep["best_threshold"],
            "roc_auc": roc["auc"],
            "pr_auc": pr["average_precision"],
            "official_binary": {
                "threshold": _f(tau),
                "source": "validation",
                "accuracy": official["accuracy"],
                "precision": official["precision"],
                "recall": official["recall"],
                "f1": official["f1_score"],
            },
            "ground_truth_cutoff": used,
        }
    overall = {
        "mae": _mean([per_trait[k]["mae"] for k in TRAIT_KEYS]),
        "mse": _mean([per_trait[k]["mse"] for k in TRAIT_KEYS]),
        "rmse": _mean([per_trait[k]["rmse"] for k in TRAIT_KEYS]),
        "r2": _mean([per_trait[k]["r2"] for k in TRAIT_KEYS]),
        "correlation": _mean([per_trait[k]["correlation"] for k in TRAIT_KEYS]),
        "official_f1": _mean([per_trait[k]["official_binary"]["f1"] for k in TRAIT_KEYS]),
        "official_accuracy": _mean([per_trait[k]["official_binary"]["accuracy"] for k in TRAIT_KEYS]),
        "roc_auc": _mean([per_trait[k]["roc_auc"] for k in TRAIT_KEYS]),
        "pr_auc": _mean([per_trait[k]["pr_auc"] for k in TRAIT_KEYS]),
    }
    return per_trait, overall


# ---------------------------------------------------------------------------
# Modeling: Lasso and LSTM (each runnable with/without GAN)
# ---------------------------------------------------------------------------

def _run_lasso(
    features: Features,
    sample: Sample,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: ExperimentConfig,
    use_gan: bool,
) -> Tuple[Dict[str, Any], LassoTrainer, Dict[str, np.ndarray]]:
    """
    Per-trait ElasticNet on mean-pooled selected-comment embeddings.

    GAN (when enabled) is fit on training (X, y) pairs only and produces
    paired synthetic (X_syn, y_syn). The feature scaler is fit on real
    training embeddings only. Metrics come from ``metrics_engine``.
    """
    X = features.pooled
    y = np.asarray(sample.labels_unit, dtype=float)
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    trainer = LassoTrainer(
        alpha=cfg.lasso_alpha,
        max_iter=cfg.lasso_max_iter,
        regularization=cfg.lasso_regularization,
        l1_ratio=cfg.lasso_l1_ratio,
    )
    # Scaler on REAL training embeddings only. Dummy Likert labels satisfy
    # prepare_training_data's (v-1)/4 contract; trait targets stay unit-scale.
    X_tr_scaled, _ = trainer.prepare_training_data(X_tr, 1.0 + 4.0 * y_tr[:, 0])
    X_val_scaled = trainer.transform_features(X_val)
    X_te_scaled = trainer.transform_features(X_te)

    X_fit, y_fit = X_tr_scaled, y_tr
    sample_weight = None
    if use_gan:
        X_syn, y_syn = _paired_gan_samples(X_tr, y_tr, cfg)
        if len(X_syn) > 0:
            X_syn_scaled = trainer.transform_features(X_syn)
            X_fit = np.vstack([X_tr_scaled, X_syn_scaled])
            y_fit = np.concatenate([y_tr, y_syn], axis=0)
            sample_weight = np.concatenate([
                np.ones(len(X_tr_scaled), dtype=float),
                np.full(len(X_syn_scaled), cfg.synthetic_weight, dtype=float),
            ])

    val_pred = np.zeros_like(y_val)
    test_pred = np.zeros_like(y_te)
    train_mae: Dict[str, Optional[float]] = {}
    for ti, trait in enumerate(OCEAN_TRAITS):
        train_metrics = trainer.train_trait_model(
            X_fit, y_fit[:, ti], trait,
            validate_X=X_val_scaled, validate_y=y_val[:, ti],
            sample_weight=sample_weight,
        )
        val_pred[:, ti] = trainer.predict_trait(trait, X_val_scaled)
        test_pred[:, ti] = trainer.predict_trait(trait, X_te_scaled)
        train_mae[TRAIT_KEYS[ti]] = _f(train_metrics.get("train_mae"))

    cutoff = me.DEFAULT_GROUND_TRUTH_CUTOFF
    threshold_selection = _select_thresholds_on_validation(y_val, val_pred, cutoff)
    per_trait, overall = _score_test_predictions(y_te, test_pred, threshold_selection, cutoff)
    for i, key in enumerate(TRAIT_KEYS):
        val_reg = me.compute_regression_metrics(y_val[:, i], val_pred[:, i])
        per_trait[key]["train_mae"] = train_mae[key]
        per_trait[key]["val_mae"] = _f(val_reg["mae"])
    overall["val_mae"] = _mean([per_trait[k]["val_mae"] for k in TRAIT_KEYS])

    raw = {
        "y_test": y_te,
        "predictions": test_pred,
        "y_val": y_val,
        "val_predictions": val_pred,
    }
    result = {
        "per_trait": per_trait,
        "overall": overall,
        "threshold_selection": threshold_selection,
        "final_test_evaluation": {
            "split": "test",
            "ground_truth_cutoff": cutoff,
            "note": (
                "Official High/Low metrics use the validation-selected threshold. "
                "The test threshold sweep is analysis-only and does not choose the threshold."
            ),
        },
    }
    return result, trainer, raw


def _run_lstm(
    features: Features,
    sample: Sample,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: ExperimentConfig,
    use_gan: bool,
) -> Tuple[Dict[str, Any], LSTMTrainer, Dict[str, np.ndarray]]:
    """
    Continuous 5-output OCEAN LSTM on ordered selected-comment sequences.

    Real training data stay sequences. The current GAN emits paired vectors,
    not sequences, so synthetic samples (when enabled) are length-1 sequences
    of the generated embedding with the matching generated OCEAN target.
    GAN is fit on participant-level pooled training embeddings + (N, 5)
    labels only — never val/test, and never with borrowed real-user labels.
    """
    seqs = features.sequences
    y = np.asarray(sample.labels_unit, dtype=float)
    tr_seqs = [seqs[i] for i in train_idx]
    val_seqs = [seqs[i] for i in val_idx]
    te_seqs = [seqs[i] for i in test_idx]
    y_tr, y_val, y_te = y[train_idx], y[val_idx], y[test_idx]

    fit_seqs: List[np.ndarray] = list(tr_seqs)
    fit_y = y_tr
    sample_weights = None
    if use_gan:
        # Participant-level pairs only: one pooled embedding + one OCEAN vector.
        X_tr_pooled = features.pooled[train_idx]
        X_syn, y_syn = _paired_gan_samples(X_tr_pooled, y_tr, cfg)
        if len(X_syn) > 0:
            synth_seqs = [row.reshape(1, -1).astype(np.float32) for row in X_syn]
            fit_seqs = tr_seqs + synth_seqs
            fit_y = np.concatenate([y_tr, y_syn], axis=0)
            sample_weights = np.concatenate([
                np.ones(len(tr_seqs), dtype=float),
                np.full(len(synth_seqs), cfg.synthetic_weight, dtype=float),
            ])

    trainer = LSTMTrainer(
        hidden_dim=cfg.lstm_hidden_dim,
        num_layers=cfg.lstm_num_layers,
        dropout=cfg.lstm_dropout,
        learning_rate=cfg.lstm_learning_rate,
    )
    set_seed(cfg.seed)
    trainer.train(
        sequences=fit_seqs,
        targets=fit_y,
        val_sequences=val_seqs,
        val_targets=y_val,
        epochs=cfg.lstm_epochs,
        batch_size=cfg.lstm_batch_size,
        sample_weights=sample_weights,
        seed=cfg.seed,
    )
    val_pred = trainer.predict(val_seqs)
    test_pred = trainer.predict(te_seqs)

    cutoff = me.DEFAULT_GROUND_TRUTH_CUTOFF
    threshold_selection = _select_thresholds_on_validation(y_val, val_pred, cutoff)
    per_trait, overall = _score_test_predictions(y_te, test_pred, threshold_selection, cutoff)
    for i, key in enumerate(TRAIT_KEYS):
        val_reg = me.compute_regression_metrics(y_val[:, i], val_pred[:, i])
        per_trait[key]["val_mae"] = _f(val_reg["mae"])
    overall["val_mae"] = _mean([per_trait[k]["val_mae"] for k in TRAIT_KEYS])

    raw = {
        "y_test": y_te,
        "predictions": test_pred,
        "y_val": y_val,
        "val_predictions": val_pred,
    }
    result = {
        "per_trait": per_trait,
        "overall": overall,
        "threshold_selection": threshold_selection,
        "final_test_evaluation": {
            "split": "test",
            "ground_truth_cutoff": cutoff,
            "note": (
                "Official High/Low metrics use the validation-selected threshold. "
                "The test threshold sweep is analysis-only and does not choose the threshold."
            ),
        },
    }
    return result, trainer, raw


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _run_condition(
    sample: Sample,
    exp_id: str,
    cfg: ExperimentConfig,
    features: Features,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[Dict[str, Any], Any, Dict[str, np.ndarray]]:
    """
    Run a single condition and return ``(out_dict, fitted_model, raw_arrays)``.

    ``raw_arrays`` holds held-out test (and val) continuous arrays used by
    ``hybrid_cell_evaluations`` to pair Lasso + LSTM through
    ``metrics_engine.evaluate``. It never enters the JSON bundle.
    """
    spec = EXPERIMENTS[exp_id]
    logger.info("=== %s: %s ===", exp_id, spec["label"])

    if spec["model"] == "lasso":
        result, model, raw = _run_lasso(
            features, sample, train_idx, val_idx, test_idx, cfg, use_gan=spec["gan"],
        )
    elif spec["model"] == "lstm":
        result, model, raw = _run_lstm(
            features, sample, train_idx, val_idx, test_idx, cfg, use_gan=spec["gan"],
        )
    else:
        raise ValueError(f"Unknown model for {exp_id}: {spec['model']}")

    out = {
        "experiment": exp_id,
        "label": spec["label"],
        "selection": spec["selection"],
        "gan": spec["gan"],
        "model": spec["model"],
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "mean_comments_selected": float(np.mean(features.n_selected)),
        "per_trait": result["per_trait"],
        "overall": result["overall"],
        "threshold_selection": result["threshold_selection"],
        "final_test_evaluation": result["final_test_evaluation"],
    }
    logger.info(
        "%s done | test MAE=%s R2=%s official F1=%s",
        exp_id,
        f"{out['overall']['mae']:.4f}" if out["overall"]["mae"] is not None else "n/a",
        f"{out['overall']['r2']:.4f}" if out["overall"]["r2"] is not None else "n/a",
        f"{out['overall']['official_f1']:.3f}" if out["overall"]["official_f1"] is not None else "n/a",
    )
    return out, model, raw


def run_experiment(
    sample: Sample,
    exp_id: str,
    cfg: ExperimentConfig,
    *,
    features: Features,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    return_model: bool = False,
):
    """
    Run a single condition given already-built features and a shared split, and
    return its metrics dict. (``run_all`` builds features/agent/split once and
    runs each of the 8 conditions.)
    """
    out, model, _raw = _run_condition(
        sample, exp_id, cfg, features, train_idx, val_idx, test_idx,
    )
    return (out, model) if return_model else out


def run_all(
    prepared: List[PreparedUserComments],
    cfg: ExperimentConfig,
    *,
    encoder: Any = None,
) -> Dict[str, Any]:
    """
    End-to-end factorial sweep: sample -> train agent -> build baseline+Q-learning
    features -> shared split -> run all 8 conditions -> compare / analyze factors
    -> (optionally) save artifacts.

    Returns a bundle dict with per-condition metrics, the sample metadata, the
    ``comparison`` table, the Lasso-vs-LSTM ``model_comparison``, the
    ``factor_effects`` (Q-learning & GAN), the ``hybrid_cell_evaluations`` (the
    canonical ``metrics_engine.evaluate`` pairing of Lasso + LSTM per matched
    cell), and JSON-safe ``findings``. ``encoder`` may be injected for testing;
    otherwise a real BERT encoder is created lazily.
    """
    set_seed(cfg.seed)
    sample = sample_users(prepared, cfg)
    if encoder is None:
        encoder = get_encoder()

    agent = train_qlearning_agent(sample, cfg)

    # Build features once per selection policy; both models and both GAN settings
    # reuse them. The BERT cache also dedupes the comments the two selections
    # share.
    feats = {
        "baseline": build_features(sample, "baseline", cfg, encoder),
        "qlearning": build_features(sample, "qlearning", cfg, encoder, agent=agent),
    }
    train_idx, val_idx, test_idx = make_split(sample.n_users, cfg)

    results: Dict[str, Any] = {}
    models: Dict[str, Any] = {}
    raws: Dict[str, Dict[str, np.ndarray]] = {}
    for exp_id, spec in EXPERIMENTS.items():
        out, model, raw = _run_condition(
            sample, exp_id, cfg,
            feats[spec["selection"]],
            train_idx, val_idx, test_idx,
        )
        results[exp_id] = out
        models[exp_id] = model
        raws[exp_id] = raw

    comparison = comparison_table(results)
    model_cmp = model_comparison(results)
    effects = factor_effects(results)
    hybrid = hybrid_cell_evaluations(results, raws, cfg)
    findings = summarize_findings(results)

    logger.info("Comparison:\n%s", comparison.to_string(index=False))
    for note in findings["notes"]:
        logger.info("FINDING: %s", note)

    bundle = {
        "config": asdict(cfg),
        "sample": {
            "n_users": sample.n_users,
            "user_ids": sample.user_ids,
            "label_scale": sample.scale,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
        },
        "results": results,
        "comparison": comparison,
        "model_comparison": model_cmp,
        "factor_effects": effects,
        "hybrid_cell_evaluations": hybrid,
        "findings": findings,
    }

    if cfg.output_dir:
        save_artifacts(bundle, models, agent, cfg)

    return bundle


# ---------------------------------------------------------------------------
# Lightweight, Django-free orchestration object (the notebook's handle)
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Lightweight, Django-free stand-in for the pipeline's orchestration layer.

    The Django ``PipelineOrchestrator`` couples training to the ORM -- it reads
    ``VOLUNTEER`` rows and writes results back to the database. For the Colab
    experiments we want none of that: this runner drives the *same* Django-free
    service classes (BERT / Q-learning / GAN / Lasso / LSTM / metrics_engine)
    over in-memory PANDORA ``PreparedUserComments`` and returns plain dicts and
    pandas DataFrames. It is the single object the notebook holds onto.

    Usage
    -----
        runner = ExperimentRunner(prepared, cfg)
        bundle = runner.run()              # sample -> 8 conditions -> analyze -> save
        runner.comparison                  # the 8-condition headline table
        runner.model_comparison            # Lasso vs LSTM at each matched cell
        runner.factor_effects              # {'qlearning_effect', 'gan_effect'}
        runner.hybrid_cell_evaluations     # metrics_engine.evaluate per cell
        runner.findings["notes"]           # ready-to-cite claim sentences

    ``run()`` is thin on purpose: it delegates to :func:`run_all`, so anything
    that function does (Drive caching of embeddings, artifact persistence when
    ``cfg.output_dir`` is set, one shared split + one Q-learning policy across
    all 8 conditions) applies unchanged. Re-running ``run()`` recomputes the
    bundle; the BERT cache keeps that cheap.
    """

    def __init__(
        self,
        prepared: List[PreparedUserComments],
        cfg: ExperimentConfig,
        *,
        encoder: Any = None,
    ) -> None:
        self.prepared = prepared
        self.cfg = cfg
        self.encoder = encoder
        self.bundle: Optional[Dict[str, Any]] = None

    def run(self) -> Dict[str, Any]:
        """Run the full 8-condition factorial sweep and cache the bundle."""
        self.bundle = run_all(self.prepared, self.cfg, encoder=self.encoder)
        return self.bundle

    def _require_run(self) -> Dict[str, Any]:
        if self.bundle is None:
            raise RuntimeError("ExperimentRunner: call .run() before accessing results.")
        return self.bundle

    @property
    def results(self) -> Dict[str, Any]:
        return self._require_run()["results"]

    @property
    def comparison(self):
        return self._require_run()["comparison"]

    @property
    def model_comparison(self):
        return self._require_run()["model_comparison"]

    @property
    def factor_effects(self) -> Dict[str, Any]:
        return self._require_run()["factor_effects"]

    @property
    def hybrid_cell_evaluations(self) -> Dict[str, Any]:
        return self._require_run()["hybrid_cell_evaluations"]

    @property
    def findings(self) -> Dict[str, Any]:
        return self._require_run()["findings"]

    @property
    def sample(self) -> Dict[str, Any]:
        return self._require_run()["sample"]


# ---------------------------------------------------------------------------
# Reporting: comparison table, model comparison, factor effects, findings
# ---------------------------------------------------------------------------

def _find(results: Dict[str, Any], model: str, selection: str, gan: bool) -> Optional[Dict[str, Any]]:
    """Return the result whose (model, selection, gan) matches, or None."""
    for r in results.values():
        if r["model"] == model and r["selection"] == selection and bool(r["gan"]) == bool(gan):
            return r
    return None


def _winner(a: Optional[float], b: Optional[float], name_a: str, name_b: str,
            eps: float = 1e-4, higher_is_better: bool = True) -> str:
    if a is None or b is None:
        return "n/a"
    if abs(a - b) < eps:
        return "tie"
    if higher_is_better:
        return name_a if a > b else name_b
    return name_a if a < b else name_b


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a - b)


def comparison_table(results: Dict[str, Any]):
    """All 8 conditions x headline metrics as a pandas DataFrame (report table)."""
    import pandas as pd

    rows = []
    for exp_id in EXPERIMENTS:
        if exp_id not in results:
            continue
        r = results[exp_id]
        o = r["overall"]
        rows.append({
            "condition": exp_id,
            "description": r["label"],
            "model": r["model"],
            "selection": r["selection"],
            "gan": r["gan"],
            "test_mae": o.get("mae"),
            "test_r2": o.get("r2"),
            "test_pearson": o.get("correlation"),
            "official_f1": o.get("official_f1"),
            "roc_auc": o.get("roc_auc"),
            "pr_auc": o.get("pr_auc"),
            "val_mae": o.get("val_mae"),
        })
    return pd.DataFrame(rows)


def model_comparison(results: Dict[str, Any]):
    """
    Head-to-head **Lasso vs LSTM** at each matched (selection, gan) cell, on
    continuous test MAE (lower is better) and R² (higher is better).
    """
    import pandas as pd

    rows = []
    for sel, gan in _CELLS:
        la = _find(results, "lasso", sel, gan)
        ls = _find(results, "lstm", sel, gan)
        if not la or not ls:
            continue
        la_mae, ls_mae = la["overall"].get("mae"), ls["overall"].get("mae")
        la_r2, ls_r2 = la["overall"].get("r2"), ls["overall"].get("r2")
        rows.append({
            "selection": sel,
            "gan": gan,
            "lasso_mae": la_mae,
            "lstm_mae": ls_mae,
            "mae_winner": _winner(la_mae, ls_mae, "Lasso", "LSTM", higher_is_better=False),
            "lasso_r2": la_r2,
            "lstm_r2": ls_r2,
            "r2_winner": _winner(la_r2, ls_r2, "Lasso", "LSTM"),
        })
    return pd.DataFrame(rows)


def factor_effects(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Isolated main effects as matched-pair deltas (positive => the factor helped):

      * ``qlearning_effect``: Q-learning-select minus baseline-select, holding
        (model, gan) fixed  -> "does the Q-learning comment selection help?"
      * ``gan_effect``: GAN minus no-GAN, holding (model, selection) fixed
        -> "does GAN augmentation help?"

    Returns pandas DataFrames keyed by effect name (one row per matched pair).
    """
    import pandas as pd

    q_rows = []
    for model in ("lasso", "lstm"):
        for gan in (False, True):
            base = _find(results, model, "baseline", gan)
            ql = _find(results, model, "qlearning", gan)
            if not base or not ql:
                continue
            q_rows.append({
                "model": model,
                "gan": gan,
                "mae_baseline": base["overall"].get("mae"),
                "mae_qlearning": ql["overall"].get("mae"),
                "delta_mae": _delta(ql["overall"].get("mae"), base["overall"].get("mae")),
                "delta_r2": _delta(ql["overall"].get("r2"), base["overall"].get("r2")),
                "delta_official_f1": _delta(
                    ql["overall"].get("official_f1"), base["overall"].get("official_f1"),
                ),
            })

    g_rows = []
    for model in ("lasso", "lstm"):
        for sel in ("baseline", "qlearning"):
            nog = _find(results, model, sel, False)
            gon = _find(results, model, sel, True)
            if not nog or not gon:
                continue
            g_rows.append({
                "model": model,
                "selection": sel,
                "mae_no_gan": nog["overall"].get("mae"),
                "mae_gan": gon["overall"].get("mae"),
                "delta_mae": _delta(gon["overall"].get("mae"), nog["overall"].get("mae")),
                "delta_r2": _delta(gon["overall"].get("r2"), nog["overall"].get("r2")),
                "delta_official_f1": _delta(
                    gon["overall"].get("official_f1"), nog["overall"].get("official_f1"),
                ),
            })

    return {
        "qlearning_effect": pd.DataFrame(q_rows),
        "gan_effect": pd.DataFrame(g_rows),
    }


def hybrid_cell_evaluations(
    results: Dict[str, Any],
    raws: Dict[str, Dict[str, np.ndarray]],
    cfg: ExperimentConfig,
) -> Dict[str, Any]:
    """
    Run ``metrics_engine.evaluate`` once per matched (selection, GAN) cell,
    pairing that cell's Lasso and LSTM **continuous** ``(N, 5)`` test
    predictions on the same held-out test participants.

    The evaluate() test sweep is analysis/plotting only. Official High/Low
    thresholds live on each condition's ``threshold_selection`` (validation).
    """
    out: Dict[str, Any] = {}
    cutoff = me.DEFAULT_GROUND_TRUTH_CUTOFF
    for sel, gan in _CELLS:
        la = _find(results, "lasso", sel, gan)
        ls = _find(results, "lstm", sel, gan)
        if not la or not ls:
            continue
        lasso_id, lstm_id = la["experiment"], ls["experiment"]
        if lasso_id not in raws or lstm_id not in raws:
            continue
        lr, sr = raws[lasso_id], raws[lstm_id]
        if lr["y_test"].shape != sr["y_test"].shape:
            raise ValueError(
                f"Test labels misaligned for cell {sel}/{gan}: "
                f"{lr['y_test'].shape} vs {sr['y_test'].shape}"
            )
        cell_label = sel + (" + GAN" if gan else "")
        out[cell_label] = me.evaluate(
            y_true=lr["y_test"],
            lasso_predictions=lr["predictions"],
            lstm_predictions=sr["predictions"],
            trait_names=list(TRAIT_KEYS),
            ground_truth_cutoff=cutoff,
        )
    return out


def summarize_findings(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    JSON-safe headline claims: best condition by test MAE, mean Lasso vs LSTM
    regression metrics, and mean Q-learning / GAN effects.
    """
    def _model_mean(model: str, metric: str) -> Optional[float]:
        vals = [r["overall"].get(metric) for r in results.values()
                if r["model"] == model and r["overall"].get(metric) is not None]
        return float(np.mean(vals)) if vals else None

    def _mean_delta(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        vals = [row[key] for row in rows if row.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    scored = {k: r for k, r in results.items() if r["overall"].get("mae") is not None}
    best_id = min(scored, key=lambda k: scored[k]["overall"]["mae"]) if scored else None
    best = None
    if best_id is not None:
        b = results[best_id]
        best = {
            "condition": best_id,
            "label": b["label"],
            "mae": b["overall"].get("mae"),
            "r2": b["overall"].get("r2"),
            "official_f1": b["overall"].get("official_f1"),
        }

    lasso_mae, lstm_mae = _model_mean("lasso", "mae"), _model_mean("lstm", "mae")
    lasso_r2, lstm_r2 = _model_mean("lasso", "r2"), _model_mean("lstm", "r2")

    effects = factor_effects(results)
    q_df, g_df = effects["qlearning_effect"], effects["gan_effect"]
    q_rows = q_df.to_dict("records") if hasattr(q_df, "to_dict") else []
    g_rows = g_df.to_dict("records") if hasattr(g_df, "to_dict") else []
    q_mae, q_r2 = _mean_delta(q_rows, "delta_mae"), _mean_delta(q_rows, "delta_r2")
    g_mae, g_r2 = _mean_delta(g_rows, "delta_mae"), _mean_delta(g_rows, "delta_r2")

    better_by_mae = _winner(lasso_mae, lstm_mae, "Lasso", "LSTM", higher_is_better=False)
    better_by_r2 = _winner(lasso_r2, lstm_r2, "Lasso", "LSTM")

    notes: List[str] = []
    if best is not None:
        notes.append(
            f"Best condition: {best['condition']} ({best['label']}) - "
            f"test MAE {best['mae']:.3f}, R2 {best['r2']:.3f}."
        )
    if lasso_mae is not None and lstm_mae is not None:
        notes.append(
            f"Model comparison (mean over the 4 matched cells): "
            f"Lasso MAE {lasso_mae:.3f} vs LSTM {lstm_mae:.3f} -> {better_by_mae} wins on MAE; "
            f"Lasso R2 {lasso_r2:.3f} vs LSTM {lstm_r2:.3f} -> {better_by_r2} wins on R2."
        )
    if q_mae is not None:
        verdict = "helps" if q_mae < 0 else ("hurts" if q_mae > 0 else "is neutral")
        notes.append(
            f"Q-learning selection {verdict} on average: mean delta MAE {q_mae:+.3f}, "
            f"mean delta R2 {q_r2:+.3f} (vs baseline-select, over model x GAN)."
        )
    if g_mae is not None:
        verdict = "helps" if g_mae < 0 else ("hurts" if g_mae > 0 else "is neutral")
        notes.append(
            f"GAN augmentation {verdict} on average: mean delta MAE {g_mae:+.3f}, "
            f"mean delta R2 {g_r2:+.3f} (vs no-GAN, over model x selection)."
        )

    return {
        "best_condition": best,
        "model_means": {
            "lasso": {"mae": lasso_mae, "r2": lasso_r2},
            "lstm": {"mae": lstm_mae, "r2": lstm_r2},
        },
        "better_model": {"by_mae": better_by_mae, "by_r2": better_by_r2},
        "qlearning_effect_mean": {"delta_mae": q_mae, "delta_r2": q_r2},
        "gan_effect_mean": {"delta_mae": g_mae, "delta_r2": g_r2},
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

# Bundle keys that hold pandas objects (or dicts of them) — excluded from the
# JSON run summary and written as their own CSVs instead.
_NON_JSON_BUNDLE_KEYS = ("comparison", "model_comparison", "factor_effects")


def save_artifacts(
    bundle: Dict[str, Any],
    models: Dict[str, Any],
    agent: QLearningAgent,
    cfg: ExperimentConfig,
) -> None:
    """
    Persist metrics + analysis tables + model states to ``cfg.output_dir``
    (typically on Drive):

        <out>/comparison.csv            all 8 conditions x headline metrics
        <out>/model_comparison.csv      Lasso vs LSTM at each matched cell
        <out>/qlearning_effect.csv      Q-learning matched-pair deltas
        <out>/gan_effect.csv            GAN matched-pair deltas
        <out>/findings.json             JSON-safe headline claims
        <out>/hybrid_evaluation.json    metrics_engine.evaluate per matched cell
        <out>/run_summary.json          config + sample + per-condition results + findings
        <out>/q_table.json              trained Q-learning policy
        <out>/<condition>/metrics.json  full per-trait metrics
        <out>/<condition>/lasso_state.json   (Lasso conditions)
        <out>/<condition>/lstm_state.pt      (LSTM conditions)
    """
    import torch

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Analysis tables (CSV).
    bundle["comparison"].to_csv(out / "comparison.csv", index=False)
    bundle["model_comparison"].to_csv(out / "model_comparison.csv", index=False)
    effects = bundle.get("factor_effects", {})
    if "qlearning_effect" in effects:
        effects["qlearning_effect"].to_csv(out / "qlearning_effect.csv", index=False)
    if "gan_effect" in effects:
        effects["gan_effect"].to_csv(out / "gan_effect.csv", index=False)

    with (out / "findings.json").open("w", encoding="utf-8") as fh:
        json.dump(bundle.get("findings", {}), fh, ensure_ascii=False, indent=2)

    with (out / "hybrid_evaluation.json").open("w", encoding="utf-8") as fh:
        json.dump(bundle.get("hybrid_cell_evaluations", {}), fh, ensure_ascii=False, indent=2)

    with (out / "q_table.json").open("w", encoding="utf-8") as fh:
        json.dump(agent.save_state(), fh)

    # JSON run summary (drop the pandas-bearing keys; findings stays, it's JSON-safe).
    summary = {k: v for k, v in bundle.items() if k not in _NON_JSON_BUNDLE_KEYS}
    with (out / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # Per-condition metrics + trained model state.
    for exp_id, result in bundle["results"].items():
        exp_dir = out / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        with (exp_dir / "metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)

        model = models[exp_id]
        if EXPERIMENTS[exp_id]["model"] == "lasso":
            with (exp_dir / "lasso_state.json").open("w", encoding="utf-8") as fh:
                json.dump(model.save_state(), fh)
        else:  # lstm — save state_dicts + the config needed to rebuild the nets
            torch.save(
                {
                    "state_dicts": {t: m.state_dict() for t, m in model.models.items()},
                    "config": {
                        "hidden_dim": cfg.lstm_hidden_dim,
                        "num_layers": cfg.lstm_num_layers,
                        "dropout": cfg.lstm_dropout,
                        "input_dim": 768,
                    },
                },
                exp_dir / "lstm_state.pt",
            )

    logger.info("Artifacts saved under %s", out)


# ---------------------------------------------------------------------------
# Small numeric helpers (keep metrics JSON-clean: plain floats / None)
# ---------------------------------------------------------------------------

def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return float(np.mean(vals)) if vals else None
