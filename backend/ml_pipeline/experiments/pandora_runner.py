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
    factor 3  model      : Lasso (regressor) vs  LSTM (3-class sequence model)

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
  * **Both Lasso and LSTM run in all four (selection x GAN) cells.** That is what
    lets us support a claim about *which model is better* -- they are compared at
    matched conditions, never confounded with selection or augmentation.
  * **Isolated factor effects.** ``factor_effects()`` reports the Q-learning
    effect (qlearning - baseline, holding model+GAN fixed) and the GAN effect
    (gan - no-gan, holding model+selection fixed) as matched-pair deltas, so
    "does Q-learning help?" / "does GAN help?" are answered per model.
  * **One shared metric across both model families.** Lasso is a regressor and
    LSTM a 3-class classifier, so every condition is also scored on a common
    *tertile* Low/Med/High view: Lasso's continuous predictions are binned with
    the same train-derived cut points the LSTM classifies into. Accuracy and
    macro-F1 are therefore comparable Lasso-vs-LSTM. (Lasso additionally reports
    native regression MAE/RMSE/R2/Pearson.)
  * **One participant-level train/val split shared by all 8 conditions**, and an
    **identical Q-learning policy** across the four Q-learning cells, so results
    differ only by the factor under study -- not by split luck or policy drift.
  * **GAN augmentation uses the real adversarial GAN** in
    ``services/augmentation/gan.py`` (an MLP Generator/Discriminator trained
    with a BCE adversarial loss), not the simplified noise augmenter. For each
    GAN condition the GAN is ``fit`` on *that model's own training fold only*
    (never val) and asked to ``generate`` synthetic training examples, which are
    appended to the training fold down-weighted by ``synthetic_weight``. For
    Lasso the GAN is fit on the pooled training vectors and generates one
    synthetic pooled vector per real training user; for LSTM it is fit on the
    real training timestep vectors and generates one same-length synthetic
    sequence per real training user. Fitting per feature space keeps the GAN's
    train-only data boundary intact for both models.
  * **All metrics come from ``metrics_engine``** -- the same canonical module the
    Django ``PipelineOrchestrator`` evaluates through. Regression metrics
    (MAE/MSE/RMSE/R2/Pearson) via ``compute_regression_metrics``; 3-class
    Low/Med/High metrics via ``compute_multiclass_metrics``; the Lasso decision
    threshold sweep via ``sweep_thresholds_on_scores``; and, per matched cell,
    the full plug-and-play ``evaluate()`` entry point pairing Lasso + LSTM
    predictions. The runner does not re-derive any metric formula.
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
#   augmentation.gan.GANAugmenter                       -> REAL adversarial GAN
#   lasso_regressor.LassoTrainer                        -> per-trait ElasticNet
#   lstm_classifier.LSTMTrainer                         -> per-trait 3-class LSTM
#   metrics_engine.evaluate (+ component fns)           -> canonical metrics
#
# GAN note: there are two GANAugmenter classes in the tree. The Django
# orchestrator today calls the simplified MVP one (services/gan_augmenter.py:
# Gaussian noise + z-norm). The experiments instead bind to the *real*
# adversarial GAN in services/augmentation/gan.py (Generator/Discriminator,
# fit(X_train) -> generate(n)), because the point of the sweep is to measure
# the effect of a genuine GAN before wiring it into Django. The augmentation
# package __init__ is empty, so we import the submodule directly.
from backend.ml_pipeline.services.data.pandora import PreparedUserComments
from backend.ml_pipeline.services.qlearning_agent import QLearningAgent, run_training_loop
from backend.ml_pipeline.services.bert_encoder import BERTEncoder
from backend.ml_pipeline.services.augmentation.gan import GANAugmenter
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
    val_ratio: float = 0.2            # participant-level held-out fraction

    # GAN (real adversarial GAN, services/augmentation/gan.py) ----------------
    synthetic_weight: float = 0.35    # mirrors orchestrator SYNTHETIC_SAMPLE_WEIGHT
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


def _tertile_cuts(train_units: np.ndarray) -> Tuple[float, float]:
    """Low/High cut points at the 33rd/67th percentiles of the TRAIN labels only."""
    lo, hi = np.quantile(np.asarray(train_units, dtype=float), [1.0 / 3.0, 2.0 / 3.0])
    lo, hi = float(lo), float(hi)
    if lo >= hi:  # degenerate (near-constant labels) — fall back to fixed unit thirds
        lo, hi = 1.0 / 3.0, 2.0 / 3.0
    return lo, hi


def to_tertile_classes(values: np.ndarray, low_cut: float, high_cut: float) -> np.ndarray:
    """Bin unit-scale values into {0:Low, 1:Medium, 2:High} using given cut points."""
    v = np.asarray(values, dtype=float)
    cls = np.ones(len(v), dtype=np.int64)  # default Medium
    cls[v < low_cut] = 0
    cls[v >= high_cut] = 2
    return cls


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

def make_split(n_users: int, cfg: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic participant-level train/val index split, shared by every
    condition so they differ only by their selection/GAN/model factors, not the
    split.
    """
    if n_users < 2:
        raise ValueError("Need at least 2 users to form a train/val split.")
    rng = np.random.RandomState(cfg.seed)
    perm = rng.permutation(n_users)
    val_count = max(1, round(n_users * cfg.val_ratio))
    val_count = min(val_count, n_users - 1)  # always leave >=1 for training
    val_idx = np.sort(perm[:val_count])
    train_idx = np.sort(perm[val_count:])
    logger.info("Split: train=%d, val=%d (of %d users).", len(train_idx), len(val_idx), n_users)
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# GAN augmentation (real adversarial GAN, fit on each model's own train fold)
# ---------------------------------------------------------------------------

def _make_gan(embedding_dim: int, cfg: ExperimentConfig) -> GANAugmenter:
    """
    Build the real adversarial ``GANAugmenter`` (services/augmentation/gan.py)
    configured from ``cfg``. It seeds itself from ``seed`` and resolves the
    device to CUDA when available, so it trains on the Colab GPU.
    """
    return GANAugmenter(
        embedding_dim=embedding_dim,
        latent_dim=cfg.gan_latent_dim,
        hidden_dim=cfg.gan_hidden_dim,
        epochs=cfg.gan_epochs,
        batch_size=cfg.gan_batch_size,
        learning_rate=cfg.gan_learning_rate,
        seed=cfg.seed,
    )


def _augment_pooled_gan(X_tr: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    """
    Fit the adversarial GAN on the pooled TRAIN vectors only, then generate one
    synthetic pooled vector per real training user. Used by the Lasso path.
    Deterministic given ``cfg.seed`` (the GAN seeds training and generation).
    """
    X_tr = np.asarray(X_tr, dtype=np.float32)
    if len(X_tr) < 2:  # GAN needs >= 2 real samples to fit
        logger.warning("Too few train users (%d) to fit the GAN; skipping augmentation.", len(X_tr))
        return np.empty((0, X_tr.shape[1]), dtype=np.float32)
    gan = _make_gan(X_tr.shape[1], cfg).fit(X_tr)
    synth, _, _ = gan.generate(len(X_tr))
    return np.asarray(synth, dtype=np.float32)


def _augment_sequences_gan(train_seqs: List[np.ndarray], cfg: ExperimentConfig) -> List[np.ndarray]:
    """
    LSTM path: fit the adversarial GAN on *all* real TRAIN timestep vectors
    (never val), then generate one same-length synthetic sequence per real
    training user. Generating the full synthetic pool in a single ``generate``
    call keeps the synthetic timesteps distinct across sequences.
    Deterministic given ``cfg.seed``.
    """
    all_vecs = np.vstack(train_seqs).astype(np.float32) if train_seqs else np.empty((0, 768), np.float32)
    if len(all_vecs) < 2:
        logger.warning("Too few train timesteps (%d) to fit the GAN; reusing real sequences.", len(all_vecs))
        return [np.asarray(s, dtype=np.float32) for s in train_seqs]
    gan = _make_gan(all_vecs.shape[1], cfg).fit(all_vecs)
    total = int(sum(len(s) for s in train_seqs))
    synth_all, _, _ = gan.generate(total)
    synth_all = np.asarray(synth_all, dtype=np.float32)
    out: List[np.ndarray] = []
    pos = 0
    for seq in train_seqs:
        k = len(seq)
        out.append(synth_all[pos:pos + k])
        pos += k
    return out


# ---------------------------------------------------------------------------
# Modeling: Lasso and LSTM (each runnable with/without GAN)
# ---------------------------------------------------------------------------

def _run_lasso(
    features: Features,
    sample: Sample,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: ExperimentConfig,
    use_gan: bool,
) -> Tuple[Dict[str, Any], LassoTrainer, Dict[str, np.ndarray]]:
    """
    Per-trait ElasticNet on mean-pooled features, mirroring the orchestrator's
    ``_fit_trait_variant`` (elasticnet, alpha=0.001, l1_ratio=0.5). One trainer
    holds all five trait models and the shared feature scaler.

    Every reported number comes from ``metrics_engine`` (``me``), the same
    canonical module the Django orchestrator evaluates through -- regression
    MAE/RMSE/R2/Pearson in the normalized [0,1] domain via
    ``compute_regression_metrics``; the shared tertile Low/Med/High accuracy &
    macro-P/R/F1 via ``compute_multiclass_metrics`` (cut points from the TRAIN
    labels only); and the 5-candidate decision-threshold sweep on Lasso's own
    continuous predictions via ``sweep_thresholds_on_scores``. The runner does
    not re-derive any metric formula here.

    Also returns the held-out val-fold arrays (continuous truth, Lasso
    predictions, tertile-truth classes) so ``run_all`` can pair them with the
    matched LSTM cell through ``metrics_engine.evaluate``.
    """
    X = features.pooled
    X_tr, X_val = X[train_idx], X[val_idx]

    trainer = LassoTrainer(
        alpha=cfg.lasso_alpha,
        max_iter=cfg.lasso_max_iter,
        regularization=cfg.lasso_regularization,
        l1_ratio=cfg.lasso_l1_ratio,
    )
    # Fit the feature scaler once on the REAL training fold (leakage-safe). The
    # returned normalized labels for trait 0 are unused; per-trait unit labels
    # are taken directly below.
    X_tr_scaled, _ = trainer.prepare_training_data(X_tr, 1.0 + 4.0 * sample.labels_unit[train_idx, 0])
    X_val_scaled = trainer.transform_features(X_val)

    # Synthetic augmentation: trait-agnostic, so generate once and reuse each
    # trait's labels for the synthetic rows. Synthetic rows pass through the same
    # (real-fit) scaler and are down-weighted by synthetic_weight. If the GAN
    # can't fit (too few train users) it returns no rows and we train unaugmented.
    synth_scaled = None
    sample_weight = None
    if use_gan:
        synth = _augment_pooled_gan(X_tr, cfg)
        if len(synth) > 0:
            synth_scaled = trainer.transform_features(synth)
            sample_weight = np.concatenate([
                np.ones(len(X_tr_scaled), dtype=float),
                np.full(len(synth_scaled), cfg.synthetic_weight, dtype=float),
            ])

    n_val = len(val_idx)
    n_traits = len(OCEAN_TRAITS)
    true_unit = np.zeros((n_val, n_traits), dtype=float)
    lasso_pred_mat = np.zeros((n_val, n_traits), dtype=float)
    true_classes = np.zeros((n_val, n_traits), dtype=int)

    per_trait: Dict[str, Any] = {}
    for ti, trait in enumerate(OCEAN_TRAITS):
        unit = sample.labels_unit[:, ti]
        y_tr_unit = unit[train_idx]
        y_val_unit = unit[val_idx]

        if synth_scaled is not None:
            X_fit = np.vstack([X_tr_scaled, synth_scaled])
            y_fit = np.concatenate([y_tr_unit, y_tr_unit])
        else:
            X_fit, y_fit = X_tr_scaled, y_tr_unit

        train_metrics = trainer.train_trait_model(
            X_fit, y_fit, trait,
            validate_X=X_val_scaled, validate_y=y_val_unit,
            sample_weight=sample_weight,
        )
        val_pred = trainer.predict_trait(trait, X_val_scaled)

        # Common tertile-classification view (cut points from real train labels),
        # so Lasso is scored on the SAME Low/Med/High target the LSTM classifies.
        low_cut, high_cut = _tertile_cuts(y_tr_unit)
        y_val_cls = to_tertile_classes(y_val_unit, low_cut, high_cut)
        pred_cls = to_tertile_classes(val_pred, low_cut, high_cut)

        # All metric formulas from metrics_engine (single source of truth).
        reg = me.compute_regression_metrics(y_val_unit, val_pred)
        cls = me.compute_multiclass_metrics(y_val_cls, pred_cls, labels=[0, 1, 2])
        y_bin, gt_cut = me.derive_binary_ground_truth(y_val_unit)
        sweep = me.sweep_thresholds_on_scores(y_bin, val_pred)

        true_unit[:, ti] = y_val_unit
        lasso_pred_mat[:, ti] = val_pred
        true_classes[:, ti] = y_val_cls

        per_trait[trait] = {
            "val_mae": _f(reg["mae"]),
            "val_rmse": _f(reg["rmse"]),
            "val_r2": _f(reg["r2"]),
            "val_pearson": _f(reg["correlation"]),
            "train_mae": _f(train_metrics.get("train_mae")),
            "accuracy": _f(cls["accuracy"]),
            "macro_f1": _f(cls["f1"]),
            "macro_precision": _f(cls["precision"]),
            "macro_recall": _f(cls["recall"]),
            "confusion_matrix": cls["confusion_matrix"],
            "threshold_sweep": {
                "best_threshold": sweep["best_threshold"],
                "best_f1": sweep["best_f1"],
                "ground_truth_cutoff": _f(gt_cut),
            },
            "tertile_cuts": [low_cut, high_cut],
            "sparse_features": train_metrics.get("sparse_features"),
        }

    overall = {
        "val_mae": _mean([per_trait[t]["val_mae"] for t in OCEAN_TRAITS]),
        "val_rmse": _mean([per_trait[t]["val_rmse"] for t in OCEAN_TRAITS]),
        "val_r2": _mean([per_trait[t]["val_r2"] for t in OCEAN_TRAITS]),
        "val_pearson": _mean([per_trait[t]["val_pearson"] for t in OCEAN_TRAITS]),
        "accuracy": _mean([per_trait[t]["accuracy"] for t in OCEAN_TRAITS]),
        "macro_f1": _mean([per_trait[t]["macro_f1"] for t in OCEAN_TRAITS]),
    }
    raw = {"true_unit": true_unit, "lasso_pred": lasso_pred_mat, "true_classes": true_classes}
    return {"per_trait": per_trait, "overall": overall}, trainer, raw


def _run_lstm(
    features: Features,
    sample: Sample,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: ExperimentConfig,
    use_gan: bool,
) -> Tuple[Dict[str, Any], LSTMTrainer, Dict[str, np.ndarray]]:
    """
    Per-trait 3-class LSTM on the ordered selected-comment sequences. Tertile
    labels are derived from TRAIN labels only and applied identically to val
    (fixed class boundaries decided before evaluation).

    Predictions come from the trainer's own ``predict_trait`` (raw argmax
    classes on the val fold); every reported number is then computed by
    ``metrics_engine.compute_multiclass_metrics`` (accuracy, macro P/R/F1,
    confusion matrix over labels [0,1,2]) -- the same canonical module the
    Django orchestrator evaluates through, so the runner re-derives no metric.

    With ``use_gan`` the training fold is augmented with one synthetic sequence
    per real training participant (same real adversarial GAN as the Lasso path),
    appended to the real sequences and down-weighted by ``synthetic_weight`` via
    the LSTM's ``sample_weights``. Augmentation is trait-agnostic so it's
    generated once.

    Also returns the held-out val-fold arrays (continuous truth, predicted
    classes, tertile-truth classes) so ``run_all`` can pair them with the
    matched Lasso cell through ``metrics_engine.evaluate``.
    """
    seqs = features.sequences
    tr_seqs = [seqs[i] for i in train_idx]
    val_seqs = [seqs[i] for i in val_idx]

    # Trait-agnostic synthetic sequences (generated once, reused per trait).
    synth_seqs: Optional[List[np.ndarray]] = None
    sample_weights = None
    if use_gan:
        synth_seqs = _augment_sequences_gan(tr_seqs, cfg)
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

    n_val = len(val_idx)
    n_traits = len(OCEAN_TRAITS)
    true_unit = np.zeros((n_val, n_traits), dtype=float)
    pred_classes_mat = np.zeros((n_val, n_traits), dtype=int)
    true_classes = np.zeros((n_val, n_traits), dtype=int)

    per_trait: Dict[str, Any] = {}
    for ti, trait in enumerate(OCEAN_TRAITS):
        unit = sample.labels_unit[:, ti]
        low_cut, high_cut = _tertile_cuts(unit[train_idx])
        tr_labels = to_tertile_classes(unit[train_idx], low_cut, high_cut)
        val_labels = to_tertile_classes(unit[val_idx], low_cut, high_cut)

        if use_gan:
            fit_seqs = tr_seqs + synth_seqs
            fit_labels = np.concatenate([tr_labels, tr_labels])
        else:
            fit_seqs = tr_seqs
            fit_labels = tr_labels

        set_seed(cfg.seed)
        trainer.train_trait_model(
            trait, fit_seqs, None,  # targets ignored — precomputed_labels drives it
            val_sequences=val_seqs,
            precomputed_labels=fit_labels,
            val_precomputed_labels=val_labels,
            epochs=cfg.lstm_epochs,
            batch_size=cfg.lstm_batch_size,
            sample_weights=sample_weights,
            seed=cfg.seed,
        )
        # Raw predictions from the trainer; metrics from metrics_engine.
        _probs, pred_cls, _labels = trainer.predict_trait(trait, val_seqs)
        cls = me.compute_multiclass_metrics(val_labels, pred_cls, labels=[0, 1, 2])

        true_unit[:, ti] = unit[val_idx]
        pred_classes_mat[:, ti] = pred_cls
        true_classes[:, ti] = val_labels

        per_trait[trait] = {
            "val_mae": None,  # classification model — no regression MAE
            "accuracy": _f(cls["accuracy"]),
            "macro_f1": _f(cls["f1"]),
            "macro_precision": _f(cls["precision"]),
            "macro_recall": _f(cls["recall"]),
            "confusion_matrix": cls["confusion_matrix"],
            "tertile_cuts": [low_cut, high_cut],
        }

    overall = {
        "val_mae": None,
        "accuracy": _mean([per_trait[t]["accuracy"] for t in OCEAN_TRAITS]),
        "macro_f1": _mean([per_trait[t]["macro_f1"] for t in OCEAN_TRAITS]),
    }
    raw = {"true_unit": true_unit, "lstm_pred_classes": pred_classes_mat, "true_classes": true_classes}
    return {"per_trait": per_trait, "overall": overall}, trainer, raw


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
) -> Tuple[Dict[str, Any], Any, Dict[str, np.ndarray]]:
    """
    Run a single condition and return ``(out_dict, fitted_model, raw_arrays)``.

    ``raw_arrays`` holds the held-out val-fold numpy arrays (continuous truth,
    model predictions, tertile-truth classes) used by ``hybrid_cell_evaluations``
    to pair Lasso + LSTM through ``metrics_engine.evaluate``. It never enters the
    JSON bundle. This is the shared core behind the public ``run_experiment``.
    """
    spec = EXPERIMENTS[exp_id]
    logger.info("=== %s: %s ===", exp_id, spec["label"])

    if spec["model"] == "lasso":
        result, model, raw = _run_lasso(features, sample, train_idx, val_idx, cfg, use_gan=spec["gan"])
    elif spec["model"] == "lstm":
        result, model, raw = _run_lstm(features, sample, train_idx, val_idx, cfg, use_gan=spec["gan"])
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
        "mean_comments_selected": float(np.mean(features.n_selected)),
        "per_trait": result["per_trait"],
        "overall": result["overall"],
    }
    logger.info(
        "%s done | overall: MAE=%s acc=%.3f macroF1=%.3f",
        exp_id,
        f"{out['overall']['val_mae']:.4f}" if out["overall"]["val_mae"] is not None else "n/a",
        out["overall"]["accuracy"], out["overall"]["macro_f1"],
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
    return_model: bool = False,
):
    """
    Run a single condition given already-built features and a shared split, and
    return its metrics dict. (``run_all`` builds features/agent/split once and
    runs each of the 8 conditions.)

    With ``return_model=True`` the return is ``(metrics_dict, fitted_model)`` so
    a caller can persist the trained model; the default is just the dict.
    """
    out, model, _raw = _run_condition(sample, exp_id, cfg, features, train_idx, val_idx)
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
    train_idx, val_idx = make_split(sample.n_users, cfg)

    results: Dict[str, Any] = {}
    models: Dict[str, Any] = {}
    raws: Dict[str, Dict[str, np.ndarray]] = {}
    for exp_id, spec in EXPERIMENTS.items():
        out, model, raw = _run_condition(
            sample, exp_id, cfg,
            feats[spec["selection"]],
            train_idx, val_idx,
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
            eps: float = 1e-4) -> str:
    if a is None or b is None:
        return "n/a"
    if abs(a - b) < eps:
        return "tie"
    return name_a if a > b else name_b


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
            "val_mae": o["val_mae"],
            "accuracy": o["accuracy"],
            "macro_f1": o["macro_f1"],
        })
    return pd.DataFrame(rows)


def model_comparison(results: Dict[str, Any]):
    """
    Head-to-head **Lasso vs LSTM** at each matched (selection, gan) cell, on the
    shared tertile accuracy & macro-F1. This is the table that supports a
    'which model produces the best result' claim, because the two models are
    compared under identical selection + augmentation.
    """
    import pandas as pd

    rows = []
    for sel, gan in _CELLS:
        la = _find(results, "lasso", sel, gan)
        ls = _find(results, "lstm", sel, gan)
        if not la or not ls:
            continue
        la_acc, ls_acc = la["overall"]["accuracy"], ls["overall"]["accuracy"]
        la_f1, ls_f1 = la["overall"]["macro_f1"], ls["overall"]["macro_f1"]
        rows.append({
            "selection": sel,
            "gan": gan,
            "lasso_accuracy": la_acc,
            "lstm_accuracy": ls_acc,
            "acc_winner": _winner(la_acc, ls_acc, "Lasso", "LSTM"),
            "lasso_macro_f1": la_f1,
            "lstm_macro_f1": ls_f1,
            "f1_winner": _winner(la_f1, ls_f1, "Lasso", "LSTM"),
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
                "acc_baseline": base["overall"]["accuracy"],
                "acc_qlearning": ql["overall"]["accuracy"],
                "delta_accuracy": _delta(ql["overall"]["accuracy"], base["overall"]["accuracy"]),
                "delta_macro_f1": _delta(ql["overall"]["macro_f1"], base["overall"]["macro_f1"]),
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
                "acc_no_gan": nog["overall"]["accuracy"],
                "acc_gan": gon["overall"]["accuracy"],
                "delta_accuracy": _delta(gon["overall"]["accuracy"], nog["overall"]["accuracy"]),
                "delta_macro_f1": _delta(gon["overall"]["macro_f1"], nog["overall"]["macro_f1"]),
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
    Run the canonical ``metrics_engine.evaluate`` once per matched (selection,
    GAN) cell, pairing that cell's **Lasso** continuous predictions with its
    **LSTM** 3-class predictions on the *same* held-out val users.

    ``evaluate`` is the single plug-and-play entry point the Django
    ``PipelineOrchestrator`` also evaluates through, so these numbers are
    directly comparable to what production will report. For each cell it returns
    ``{'lasso': {...}, 'lstm': {...}, 'threshold': {...}}`` (each with per-trait +
    aggregate blocks); the Lasso side carries regression MAE/MSE/RMSE/R2/Pearson,
    the LSTM side the 3-class accuracy/precision/recall/F1/specificity, and the
    threshold block the 5-candidate decision sweep on Lasso's continuous scores.

    The tertile ground-truth classes are passed explicitly (``lstm_true_classes``)
    from the train-derived cut points already used in the sweep, so ``evaluate``
    does not re-derive them. Output is JSON-safe (metrics_engine rounds to plain
    floats / lists), keyed by a human ``"<selection>[ + GAN]"`` cell label.
    """
    out: Dict[str, Any] = {}
    for sel, gan in _CELLS:
        la = _find(results, "lasso", sel, gan)
        ls = _find(results, "lstm", sel, gan)
        if not la or not ls:
            continue
        lasso_id, lstm_id = la["experiment"], ls["experiment"]
        if lasso_id not in raws or lstm_id not in raws:
            continue
        lr, sr = raws[lasso_id], raws[lstm_id]
        cell_label = sel + (" + GAN" if gan else "")
        out[cell_label] = me.evaluate(
            y_true=lr["true_unit"],
            lasso_predictions=lr["lasso_pred"],
            lstm_predictions=sr["lstm_pred_classes"],
            lstm_true_classes=lr["true_classes"],
            trait_names=list(OCEAN_TRAITS),
        )
    return out


def summarize_findings(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    JSON-safe headline claims the report can cite directly: the best condition,
    each model's mean tertile accuracy/macro-F1, the better model overall, and
    the mean Q-learning / GAN effects. Also emits human-readable ``notes``.
    """
    def _model_mean(model: str, metric: str) -> Optional[float]:
        vals = [r["overall"][metric] for r in results.values()
                if r["model"] == model and r["overall"][metric] is not None]
        return float(np.mean(vals)) if vals else None

    def _mean_delta(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        vals = [row[key] for row in rows if row.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    # Best condition by shared tertile accuracy.
    scored = {k: r for k, r in results.items() if r["overall"]["accuracy"] is not None}
    best_id = max(scored, key=lambda k: scored[k]["overall"]["accuracy"]) if scored else None
    best = None
    if best_id is not None:
        b = results[best_id]
        best = {
            "condition": best_id,
            "label": b["label"],
            "accuracy": b["overall"]["accuracy"],
            "macro_f1": b["overall"]["macro_f1"],
        }

    lasso_acc, lstm_acc = _model_mean("lasso", "accuracy"), _model_mean("lstm", "accuracy")
    lasso_f1, lstm_f1 = _model_mean("lasso", "macro_f1"), _model_mean("lstm", "macro_f1")

    effects = factor_effects(results)
    q_df, g_df = effects["qlearning_effect"], effects["gan_effect"]
    q_rows = q_df.to_dict("records") if hasattr(q_df, "to_dict") else []
    g_rows = g_df.to_dict("records") if hasattr(g_df, "to_dict") else []
    q_acc, q_f1 = _mean_delta(q_rows, "delta_accuracy"), _mean_delta(q_rows, "delta_macro_f1")
    g_acc, g_f1 = _mean_delta(g_rows, "delta_accuracy"), _mean_delta(g_rows, "delta_macro_f1")

    better_by_acc = _winner(lasso_acc, lstm_acc, "Lasso", "LSTM")
    better_by_f1 = _winner(lasso_f1, lstm_f1, "Lasso", "LSTM")

    # Human-readable claim strings (guarded against None).
    notes: List[str] = []
    if best is not None:
        notes.append(
            f"Best condition: {best['condition']} ({best['label']}) - "
            f"tertile accuracy {best['accuracy']:.3f}, macro-F1 {best['macro_f1']:.3f}."
        )
    if lasso_acc is not None and lstm_acc is not None:
        notes.append(
            f"Model comparison (mean over the 4 matched cells): "
            f"Lasso accuracy {lasso_acc:.3f} vs LSTM {lstm_acc:.3f} -> {better_by_acc} wins on accuracy; "
            f"Lasso macro-F1 {lasso_f1:.3f} vs LSTM {lstm_f1:.3f} -> {better_by_f1} wins on macro-F1."
        )
    if q_acc is not None:
        verdict = "helps" if q_acc > 0 else ("hurts" if q_acc < 0 else "is neutral")
        notes.append(
            f"Q-learning selection {verdict} on average: mean delta accuracy {q_acc:+.3f}, "
            f"mean delta macro-F1 {q_f1:+.3f} (vs baseline-select, over model x GAN)."
        )
    if g_acc is not None:
        verdict = "helps" if g_acc > 0 else ("hurts" if g_acc < 0 else "is neutral")
        notes.append(
            f"GAN augmentation {verdict} on average: mean delta accuracy {g_acc:+.3f}, "
            f"mean delta macro-F1 {g_f1:+.3f} (vs no-GAN, over model x selection)."
        )

    return {
        "best_condition": best,
        "model_means": {
            "lasso": {"accuracy": lasso_acc, "macro_f1": lasso_f1},
            "lstm": {"accuracy": lstm_acc, "macro_f1": lstm_f1},
        },
        "better_model": {"by_accuracy": better_by_acc, "by_macro_f1": better_by_f1},
        "qlearning_effect_mean": {"delta_accuracy": q_acc, "delta_macro_f1": q_f1},
        "gan_effect_mean": {"delta_accuracy": g_acc, "delta_macro_f1": g_f1},
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
