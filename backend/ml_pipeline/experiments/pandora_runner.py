"""
Django-free PANDORA experiment runner for the final 2x2x2 pipeline.

The current project direction is:

    PANDORA comments -> baseline or Q-learning comment selection
                     -> BERT embeddings
                     -> optional paired GAN augmentation on the training fold
                     -> Lasso final model or LSTM final model
                     -> validation threshold sweep
                     -> held-out test metrics

This runner trains all eight combinations of:

    selection: baseline or Q-learning
    GAN:       off or on
    model:     Lasso or LSTM

Lasso emits five normalized continuous OCEAN scores. LSTM emits five
probabilities, one P(High) for each OCEAN trait. Low/High decisions are made
with the supervisor-facing candidate thresholds:

    0.30, 0.40, 0.50, 0.60, 0.70
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
from backend.ml_pipeline.services.lstm_classifier import (
    LSTMTrainer,
    OCEAN_TRAITS,
    TRAIT_KEYS,
    set_seed,
)
from backend.ml_pipeline.services import metrics_engine as me

logger = logging.getLogger("ml_pipeline")

PRESENTATION_METRICS: Tuple[str, ...] = (
    "accuracy",
    "f1",
    "specificity",
    "precision",
    "recall",
    "roc_auc",
    "pr_auc",
)

THRESHOLD_PLOT_METRICS: Tuple[str, ...] = (
    "accuracy",
    "f1_score",
    "specificity",
    "precision",
    "recall",
)

_CELLS: Tuple[Tuple[str, bool], ...] = (
    ("baseline", False),
    ("qlearning", False),
    ("baseline", True),
    ("qlearning", True),
)


EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "lasso_baseline": {
        "selection": "baseline",
        "gan": False,
        "model": "lasso",
        "label": "Lasso | baseline-select",
    },
    "lstm_baseline": {
        "selection": "baseline",
        "gan": False,
        "model": "lstm",
        "label": "LSTM | baseline-select",
    },
    "lasso_baseline_gan": {
        "selection": "baseline",
        "gan": True,
        "model": "lasso",
        "label": "Lasso | baseline-select + GAN",
    },
    "lstm_baseline_gan": {
        "selection": "baseline",
        "gan": True,
        "model": "lstm",
        "label": "LSTM | baseline-select + GAN",
    },
    "lasso_qlearn": {
        "selection": "qlearning",
        "gan": False,
        "model": "lasso",
        "label": "Lasso | Q-learning-select",
    },
    "lstm_qlearn": {
        "selection": "qlearning",
        "gan": False,
        "model": "lstm",
        "label": "LSTM | Q-learning-select",
    },
    "lasso_qlearn_gan": {
        "selection": "qlearning",
        "gan": True,
        "model": "lasso",
        "label": "Lasso | Q-learning-select + GAN",
    },
    "lstm_qlearn_gan": {
        "selection": "qlearning",
        "gan": True,
        "model": "lstm",
        "label": "LSTM | Q-learning-select + GAN",
    },
}


@dataclass
class ExperimentConfig:
    """Knobs for one local PANDORA LSTM sweep."""

    sample_n_users: int = 40
    min_comments_per_user: int = 5
    seed: int = 42

    top_k: int = 10
    qlearning_train_epochs: int = 3

    # Split -------------------------------------------------------------------
    val_ratio: float = 0.2            # participant-level held-out fraction
    test_ratio: float = 0.2           # reserved; current make_split is train/val

    # GAN (real adversarial GAN, services/augmentation/gan.py) ----------------
    synthetic_weight: float = 0.35    # mirrors orchestrator SYNTHETIC_SAMPLE_WEIGHT
    gan_latent_dim: int = 64
    gan_hidden_dim: int = 128
    gan_epochs: int = 150
    gan_batch_size: int = 16
    gan_learning_rate: float = 2e-4

    bert_max_length: int = 256

    lasso_alpha: float = 0.001
    lasso_l1_ratio: float = 0.5
    lasso_max_iter: int = 10000
    lasso_regularization: str = "elasticnet"

    lstm_epochs: int = 35
    lstm_batch_size: int = 4
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_learning_rate: float = 1e-3

    ground_truth_cutoff: float = me.DEFAULT_GROUND_TRUTH_CUTOFF
    candidate_thresholds: Tuple[float, ...] = tuple(me.CANDIDATE_THRESHOLDS)

    output_dir: Optional[str] = None
    embedding_cache_dir: Optional[str] = None


@dataclass
class Sample:
    """A deterministic participant-level sample from prepared PANDORA data."""

    user_ids: List[str]
    texts: List[List[str]]
    labels_raw: np.ndarray
    labels_unit: np.ndarray
    scale: str

    @property
    def n_users(self) -> int:
        return len(self.user_ids)


@dataclass
class Features:
    """Selected comment embeddings under one selection policy."""

    mode: str
    pooled: np.ndarray
    sequences: List[np.ndarray]
    n_selected: List[int]


def scale_labels_to_unit(labels_raw: np.ndarray) -> Tuple[np.ndarray, str]:
    """Detect common OCEAN scales and convert labels to [0, 1]."""
    labels_raw = np.asarray(labels_raw, dtype=float)
    finite = labels_raw[np.isfinite(labels_raw)]
    if finite.size == 0:
        raise ValueError("Label matrix has no finite values to scale.")
    lo = float(np.min(finite))
    hi = float(np.max(finite))

    if lo >= 0.0 and hi <= 1.0:
        unit, scale = labels_raw.copy(), "unit[0,1] (identity)"
    elif lo >= 1.0 and hi <= 5.0:
        unit, scale = (labels_raw - 1.0) / 4.0, "likert[1,5] -> (v-1)/4"
    elif lo >= 0.0 and 5.0 < hi <= 100.0:
        unit, scale = labels_raw / 100.0, "percentile[0,100] -> v/100"
    else:
        unit = (labels_raw - lo) / (hi - lo + 1e-12)
        scale = f"min-max[{lo:.3f},{hi:.3f}]"
        logger.warning("Unknown label range [%.3f, %.3f]; using min-max scaling.", lo, hi)

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
    """Choose a stable subset of PANDORA users with enough cleaned comments."""
    eligible = [
        u
        for u in prepared
        if u.traits is not None and len(u.comments) >= cfg.min_comments_per_user
    ]
    eligible.sort(key=lambda u: u.user_id)
    if not eligible:
        raise ValueError(
            "No PANDORA users have traits and enough comments. "
            f"Need at least {cfg.min_comments_per_user} comments per user."
        )

    rng = random.Random(cfg.seed)
    chosen = rng.sample(eligible, cfg.sample_n_users) if len(eligible) > cfg.sample_n_users else eligible
    chosen.sort(key=lambda u: u.user_id)

    user_ids: List[str] = []
    texts: List[List[str]] = []
    labels: List[List[float]] = []
    for user in chosen:
        cleaned = [c.cleaned_text for c in user.comments if c.cleaned_text]
        if not cleaned:
            continue
        assert user.traits is not None
        user_ids.append(user.user_id)
        texts.append(cleaned)
        labels.append([
            float(user.traits.O),
            float(user.traits.C),
            float(user.traits.E),
            float(user.traits.A),
            float(user.traits.N),
        ])

    labels_raw = np.asarray(labels, dtype=float)
    labels_unit, scale = scale_labels_to_unit(labels_raw)
    logger.info(
        "Sampled %d users; comments/user min=%d max=%d.",
        len(user_ids),
        min(len(t) for t in texts),
        max(len(t) for t in texts),
    )
    return Sample(user_ids, texts, labels_raw, labels_unit, scale)


def prepare_sample(prepared: List[PreparedUserComments], cfg: ExperimentConfig) -> Sample:
    """Public alias used by notebooks."""
    return sample_users(prepared, cfg)


def get_encoder() -> BERTEncoder:
    """Create the BERT encoder lazily."""
    return BERTEncoder()


def _cache_path(cache_dir: Path, text: str, max_length: int) -> Path:
    key = hashlib.sha1(f"{max_length}\u241f{text}".encode("utf-8", "replace")).hexdigest()
    return cache_dir / key[:2] / f"{key}.npy"


def _embed_one(encoder: Any, text: str, cfg: ExperimentConfig) -> np.ndarray:
    """Encode one comment with optional on-disk cache."""
    cache_dir = Path(cfg.embedding_cache_dir) if cfg.embedding_cache_dir else None
    if cache_dir is not None:
        path = _cache_path(cache_dir, text, cfg.bert_max_length)
        if path.exists():
            try:
                return np.load(path).astype(np.float32)
            except Exception as exc:
                logger.debug("Ignoring corrupt embedding cache %s (%s).", path, exc)

    vec = np.asarray(
        encoder.encode_text(text, max_length=cfg.bert_max_length)["embedding"],
        dtype=np.float32,
    )
    if cache_dir is not None:
        path = _cache_path(cache_dir, text, cfg.bert_max_length)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vec)
    return vec


def _baseline_select(texts: List[str], top_k: int) -> List[str]:
    return texts[:top_k]


def _qlearning_select(agent: QLearningAgent, texts: List[str], top_k: int) -> List[str]:
    selected = [c["text"] for c in agent.select_comments(texts, top_k=top_k, training=False)]
    return selected or texts[:1]


def train_qlearning_agent(sample: Sample, cfg: ExperimentConfig) -> QLearningAgent:
    """Train a single Q-learning policy reused by all Q-learning conditions."""
    agent = QLearningAgent(alpha=0.1, gamma=0.99, epsilon=0.1)
    if cfg.qlearning_train_epochs <= 0:
        logger.warning("qlearning_train_epochs=0; using an untrained greedy policy.")
        return agent

    set_seed(cfg.seed)
run_training_loop(
        agent,
        comment_batches=sample.texts,
        n_epochs=cfg.qlearning_train_epochs,
        max_selected=cfg.top_k,
    )
    logger.info("Q-learning trained with %d Q-table states.", len(agent.q_table))
    return agent


def build_features(
    sample: Sample,
    mode: str,
    cfg: ExperimentConfig,
    encoder: Any,
    agent: Optional[QLearningAgent] = None,
) -> Features:
    """Select comments and build both pooled and ordered embedding features."""
    if mode == "qlearning" and agent is None:
        raise ValueError("qlearning mode requires an agent.")

    pooled: List[np.ndarray] = []
    sequences: List[np.ndarray] = []
    n_selected: List[int] = []
    for idx, user_texts in enumerate(sample.texts):
        if mode == "baseline":
            selected = _baseline_select(user_texts, cfg.top_k)
        elif mode == "qlearning":
            selected = _qlearning_select(agent, user_texts, cfg.top_k)  # type: ignore[arg-type]
        else:
            raise ValueError(f"Unknown selection mode: {mode!r}")

        vectors = [_embed_one(encoder, text, cfg) for text in selected]
        seq = np.vstack(vectors).astype(np.float32)
        sequences.append(seq)
        pooled.append(seq.mean(axis=0))
        n_selected.append(len(selected))

        if (idx + 1) % 10 == 0:
            logger.info("[%s] encoded %d/%d users.", mode, idx + 1, sample.n_users)

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
    test_count = max(1, round(n_users * cfg.test_ratio))
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
        ocean_domain_min=0.0,
        ocean_domain_max=1.0,
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
    y_train: np.ndarray,
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
    y_train: np.ndarray,
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
        "model": spec["model"],
        "selection": spec["selection"],
        "gan": bool(spec["gan"]),
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
    test_idx: np.ndarray,
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


run_condition = run_experiment


def run_all(
    prepared: List[PreparedUserComments],
    cfg: ExperimentConfig,
    *,
    encoder: Any = None,
) -> Dict[str, Any]:
    """Run the full local PANDORA 2x2x2 experiment."""
    set_seed(cfg.seed)
    sample = sample_users(prepared, cfg)
    encoder = encoder or get_encoder()
    agent = train_qlearning_agent(sample, cfg)

    features = {
        "baseline": build_features(sample, "baseline", cfg, encoder),
        "qlearning": build_features(sample, "qlearning", cfg, encoder, agent=agent),
    }
    train_idx, val_idx = make_split(sample.n_users, cfg)

    results: Dict[str, Any] = {}
    models: Dict[str, Union[LassoTrainer, LSTMTrainer]] = {}
    for exp_id, spec in EXPERIMENTS.items():
        out, model, _raw = _run_condition(
            sample, exp_id, cfg,
            features[spec["selection"]],
            train_idx, val_idx,
        )
        results[exp_id] = out
        models[exp_id] = model

    comparison = comparison_table(results)
    effects = factor_effects(results)
    findings = summarize_findings(results)
    presentation_metrics = presentation_metric_table(results)
    threshold_sweeps = threshold_sweep_table(results)
    prediction_evidence = prediction_evidence_table(results)
    audit = audit_classification_metrics(results)

    logger.info("Comparison:\n%s", comparison.to_string(index=False))
    for note in findings["notes"]:
        logger.info("FINDING: %s", note)
    if audit["status"] != "PASS":
        logger.warning("Classification metric audit failed: %s", audit)

    bundle = {
        "config": asdict(cfg),
        "sample": {
            "n_users": sample.n_users,
            "user_ids": sample.user_ids,
            "label_scale": sample.scale,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(val_idx)),
        },
        "results": results,
        "comparison": comparison,
        "presentation_metrics": presentation_metrics,
        "threshold_sweeps": threshold_sweeps,
        "prediction_evidence": prediction_evidence,
        "factor_effects": effects,
        "findings": findings,
        "audit": audit,
    }
    if cfg.output_dir:
        save_artifacts(bundle, models, agent, cfg)
    return bundle


class ExperimentRunner:
    """Notebook-friendly object wrapper around ``run_all``."""

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
        self.bundle = run_all(self.prepared, self.cfg, encoder=self.encoder)
        return self.bundle

    def _require_run(self) -> Dict[str, Any]:
        if self.bundle is None:
            raise RuntimeError("Call .run() before accessing experiment outputs.")
        return self.bundle

    @property
    def results(self) -> Dict[str, Any]:
        return self._require_run()["results"]

    @property
    def comparison(self):
        return self._require_run()["comparison"]

    @property
    def factor_effects(self) -> Dict[str, Any]:
        return self._require_run()["factor_effects"]

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
    """Return headline metrics for the eight Selection/GAN/Model conditions."""
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
            "selection": r["selection"],
            "gan": r["gan"],
            "model": r["model"],
            "selection": r["selection"],
            "gan": r["gan"],
            "val_mae": o["val_mae"],
            "accuracy": o["accuracy"],
            "macro_f1": o["macro_f1"],
        })
    return pd.DataFrame(rows)


def presentation_metric_table(results: Dict[str, Any]):
    """Long-form headline metrics for presentation CSVs."""
    import pandas as pd

    rows = []
    for exp_id, row in results.items():
        overall = row.get("overall") or {}
        for metric in ("val_mae", "accuracy", "macro_f1"):
            rows.append({
                "condition": exp_id,
                "description": row.get("label"),
                "selection": row.get("selection"),
                "gan": row.get("gan"),
                "model": row.get("model"),
                "metric": metric,
                "value": overall.get(metric),
            })
    return pd.DataFrame(rows)


def threshold_sweep_table(results: Dict[str, Any]):
    """Flatten per-trait threshold sweeps when they exist."""
    import pandas as pd

    rows = []
    for exp_id, row in results.items():
        per_trait = row.get("per_trait") or {}
        for trait, block in per_trait.items():
            sweep = (block or {}).get("threshold_sweep") or {}
            if not sweep:
                continue
            rows.append({
                "condition": exp_id,
                "split": "validation",
                "trait": trait,
                "threshold": sweep.get("best_threshold"),
                "accuracy": (block or {}).get("accuracy"),
                "f1_score": sweep.get("best_f1"),
                "specificity": None,
                "precision": (block or {}).get("macro_precision"),
                "recall": (block or {}).get("macro_recall"),
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


def prediction_evidence_table(results: Dict[str, Any]):
    """Return row-level validation/test predictions that back the metrics."""
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for exp_id in EXPERIMENTS:
        if exp_id not in results:
            continue
        for split_rows in results[exp_id].get("prediction_evidence", {}).values():
            rows.extend(split_rows)
    return pd.DataFrame(rows)


def _recompute_binary_metrics_from_evidence(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(1 for r in rows if r["outcome"] == "TP")
    fp = sum(1 for r in rows if r["outcome"] == "FP")
    tn = sum(1 for r in rows if r["outcome"] == "TN")
    fn = sum(1 for r in rows if r["outcome"] == "FN")
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def audit_classification_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute saved metrics from prediction evidence and flag mismatches."""
    tolerance = 1e-4
    rows: List[Dict[str, Any]] = []
    mismatches: List[Dict[str, Any]] = []
    threshold_issues: List[Dict[str, Any]] = []
    for exp_id in EXPERIMENTS:
        if exp_id not in results:
            continue
        result = results[exp_id]
        expected_thresholds = [
            float(t)
            for t in result.get("validation", {}).get("candidate_thresholds", me.CANDIDATE_THRESHOLDS)
        ]
        evidence = result.get("prediction_evidence", {})
        for split in ("validation", "test"):
            split_rows = evidence.get(split, [])
            for trait in TRAIT_KEYS:
                trait_rows = [r for r in split_rows if r["trait"] == trait]
                recomputed = _recompute_binary_metrics_from_evidence(trait_rows)
                stored = result[split]["per_trait"][trait]
                audit_row = {
                    "condition": exp_id,
                    "split": split,
                    "trait": trait,
                    "stored_accuracy": stored.get("accuracy"),
                    "recomputed_accuracy": recomputed["accuracy"],
                    "stored_f1": stored.get("f1"),
                    "recomputed_f1": recomputed["f1"],
                    "stored_specificity": stored.get("specificity"),
                    "recomputed_specificity": recomputed["specificity"],
                    "tp": recomputed["tp"],
                    "fp": recomputed["fp"],
                    "tn": recomputed["tn"],
                    "fn": recomputed["fn"],
                }
                rows.append(audit_row)
                for metric in ("accuracy", "precision", "recall", "f1", "specificity"):
                    stored_value = stored.get(metric)
                    if stored_value is None:
                        continue
                    if abs(float(stored_value) - float(recomputed[metric])) > tolerance:
                        mismatches.append({
                            "condition": exp_id,
                            "split": split,
                            "trait": trait,
                            "metric": metric,
                            "stored": stored_value,
                            "recomputed": recomputed[metric],
                        })

                thresholds = [
                    round(float(item.get("threshold")), 2)
                    for item in stored.get("threshold_sweep", [])
                ]
                if thresholds != [round(t, 2) for t in expected_thresholds]:
                    threshold_issues.append({
                        "condition": exp_id,
                        "split": split,
                        "trait": trait,
                        "thresholds": thresholds,
                        "expected": expected_thresholds,
                    })

    return {
        "status": "PASS" if not mismatches and not threshold_issues else "FAIL",
        "n_metric_checks": len(rows) * 5,
        "n_threshold_checks": len(rows),
        "mismatches": mismatches,
        "threshold_issues": threshold_issues,
        "rows": rows,
    }


def _save_plot_file(fig: Any, target: Path, *, dpi: int = 180) -> Path:
    """Save a plot through a temporary file, then replace the previous image."""
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.stem}.{os.getpid()}.{time.time_ns()}.tmp.png")
    fig.savefig(tmp, dpi=dpi)
    try:
        tmp.replace(target)
        return target
    except OSError as exc:
        fallback = target.with_name(
            f"{target.stem}_{time.strftime('%Y%m%d_%H%M%S')}{target.suffix}"
        )
        tmp.replace(fallback)
        logger.warning(
            "Could not replace existing plot %s (%s). Saved new plot as %s.",
            target,
            exc,
            fallback,
        )
        return fallback


def save_presentation_plots(results: Dict[str, Any], out: Path) -> None:
    """Create report-ready metric and threshold figures."""
    import matplotlib.pyplot as plt

    plot_dir = (out / "plots").resolve()
    plot_dir.mkdir(parents=True, exist_ok=True)

    comparison = comparison_table(results)
    if len(comparison):
        x = np.arange(len(comparison))
        labels = comparison["condition"].tolist()
        for metric in ("test_accuracy", "test_f1", "test_specificity", "test_precision", "test_recall"):
            fig, ax = plt.subplots(figsize=(12, 5))
            colors = ["#4C78A8" if model == "lasso" else "#F58518" for model in comparison["model"]]
            ax.bar(x, comparison[metric].astype(float), color=colors)
            ax.set_title(metric.replace("_", " ").title())
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            _save_plot_file(fig, plot_dir / f"{metric}_by_condition.png")
            plt.close(fig)

    thresholds = threshold_sweep_table(results)
    if len(thresholds):
        test_thresholds = thresholds[thresholds["split"] == "test"]
        for metric in THRESHOLD_PLOT_METRICS:
            grouped = (
                test_thresholds
                .groupby(["condition", "threshold"], as_index=False)[metric]
                .mean()
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            for condition, group in grouped.groupby("condition"):
                ax.plot(
                    group["threshold"].astype(float),
                    group[metric].astype(float),
                    marker="o",
                    linewidth=1.6,
                    label=condition,
                )
            ax.set_title(f"Test Threshold Sweep - {metric.replace('_', ' ').title()}")
            ax.set_xlabel("Decision threshold")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            _save_plot_file(fig, plot_dir / f"threshold_sweep_test_{metric}.png")
            plt.close(fig)

        for condition, condition_df in test_thresholds.groupby("condition"):
            fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=True)
            axes_flat = axes.flatten()
            for axis, trait in zip(axes_flat, TRAIT_KEYS):
                trait_df = condition_df[condition_df["trait"] == trait]
                for metric in ("accuracy", "f1_score", "specificity"):
                    axis.plot(
                        trait_df["threshold"].astype(float),
                        trait_df[metric].astype(float),
                        marker="o",
                        label=metric,
                    )
                axis.set_title(trait)
                axis.grid(alpha=0.25)
            axes_flat[-1].axis("off")
            handles, legend_labels = axes_flat[0].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc="lower center", ncol=3)
            fig.suptitle(f"{condition} - Five Thresholds by Trait", y=0.98)
            fig.tight_layout(rect=(0, 0.04, 1, 0.96))
            _save_plot_file(fig, plot_dir / f"{condition}_thresholds_by_trait.png")
            plt.close(fig)


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a - b)


def factor_effects(results: Dict[str, Any]) -> Dict[str, Any]:
    """Matched deltas for Q-learning, GAN augmentation, and final model choice."""
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
        "model_comparison": model_comparison(results),
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
    notes: List[str] = []
    if best_id:
        r = results[best_id]
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
    models: Dict[str, Union[LassoTrainer, LSTMTrainer]],
    agent: QLearningAgent,
    cfg: ExperimentConfig,
) -> Path:
    """Write presentation tables, evidence, audit files, figures, and model states."""
    import torch

    archive_dir = Path(cfg.output_dir or "pandora_personality/artifacts")
    archive_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out = archive_dir / run_id
    counter = 1
    while out.exists():
        out = archive_dir / f"{run_id}_{counter:02d}"
        counter += 1
    out.mkdir(parents=True, exist_ok=True)
    bundle["artifact_dir"] = str(out.resolve())

    bundle["comparison"].to_csv(out / "comparison.csv", index=False)
    bundle["presentation_metrics"].to_csv(out / "presentation_metrics_long.csv", index=False)
    bundle["threshold_sweeps"].to_csv(out / "threshold_sweeps_long.csv", index=False)
    bundle["prediction_evidence"].to_csv(out / "prediction_evidence.csv", index=False)
    effects = bundle["factor_effects"]
    effects["qlearning_effect"].to_csv(out / "qlearning_effect.csv", index=False)
    effects["gan_effect"].to_csv(out / "gan_effect.csv", index=False)
    effects["model_comparison"].to_csv(out / "model_comparison.csv", index=False)

    with (out / "findings.json").open("w", encoding="utf-8") as fh:
        json.dump(bundle["findings"], fh, ensure_ascii=False, indent=2)
    with (out / "classification_audit.json").open("w", encoding="utf-8") as fh:
        json.dump(bundle["audit"], fh, ensure_ascii=False, indent=2)
    manifest = {
        "presentation_order": [
            "comparison.csv",
            "presentation_metrics_long.csv",
            "threshold_sweeps_long.csv",
            "prediction_evidence.csv",
            "classification_audit.json",
            "qlearning_effect.csv",
            "gan_effect.csv",
            "model_comparison.csv",
            "plots/",
        ],
        "run_folder": out.name,
        "run_created_at": datetime.now().isoformat(timespec="seconds"),
        "metric_policy": {
            "official_test_metrics": "Use validation-selected thresholds only.",
            "candidate_thresholds": list(cfg.candidate_thresholds),
            "ground_truth_cutoff": float(cfg.ground_truth_cutoff),
            "audit_requirement": "classification_audit.json status must be PASS before presenting results.",
        },
        "graph_policy": {
            "condition_metric_bars": "One PNG per headline metric across all 8 conditions.",
            "threshold_sweeps": "One PNG per threshold metric across all 8 conditions plus per-condition trait graphs.",
        },
    }
    with (out / "artifact_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    with (out / "q_table.json").open("w", encoding="utf-8") as fh:
        json.dump(agent.save_state(), fh)

    summary = {
        k: v
        for k, v in bundle.items()
        if k not in {
            "comparison",
            "presentation_metrics",
            "threshold_sweeps",
            "prediction_evidence",
            "factor_effects",
        }
    }
    with (out / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    save_presentation_plots(bundle["results"], out)

    for exp_id, result in bundle["results"].items():
        exp_dir = out / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        with (exp_dir / "metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        model = models[exp_id]
        if result["model"] == "lasso":
            with (exp_dir / "lasso_state.json").open("w", encoding="utf-8") as fh:
                json.dump(model.save_state(), fh, ensure_ascii=False, indent=2)
        else:
            torch.save(model.save_state(), exp_dir / "lstm_state.pt")

    if bundle["audit"]["status"] != "PASS":
        raise RuntimeError(
            "Classification audit failed. Inspect classification_audit.json before presenting results."
        )

    logger.info("Artifacts saved under %s", out)
    return out


def _cleaned_content_from_dict(data: Dict[str, Any]) -> CleanedContent:
    signals = data.get("signals") or {}
    extracted = signals if isinstance(signals, ExtractedSignals) else ExtractedSignals(
        hashtags=list(signals.get("hashtags") or []),
        mentions=list(signals.get("mentions") or []),
        emojis=list(signals.get("emojis") or []),
        urls=list(signals.get("urls") or []),
    )
    return CleanedContent(
        content_id=str(data.get("content_id") or ""),
        content_type=str(data.get("content_type") or "tweet"),
        original_text=str(data.get("original_text") or ""),
        cleaned_text=str(data.get("cleaned_text") or ""),
        signals=extracted,
        timestamp_utc=data.get("timestamp_utc"),
        metadata=dict(data.get("metadata") or {}),
    )


def load_prepared_cache(path: str | Path) -> List[PreparedUserComments]:
    """Load cached prepared PANDORA JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    prepared: List[PreparedUserComments] = []
    for user in payload:
        traits = UserTraits(**user["traits"]) if user.get("traits") else None
        comments = [_cleaned_content_from_dict(c) for c in user.get("comments", [])]
        prepared.append(PreparedUserComments(str(user.get("user_id") or ""), traits, comments))
    return prepared


def load_or_prepare_pandora(
    pandora_file: str | Path,
    prepared_json: str | Path,
    *,
    min_text_length: int = 3,
    group_by: str = "traits",
    refresh_prepared: bool = False,
) -> List[PreparedUserComments]:
    """Use prepared JSON if present, otherwise build it from the PANDORA parquet."""
    prepared_path = Path(prepared_json)
    if prepared_path.exists() and not refresh_prepared:
        logger.info("Loading cached prepared PANDORA data from %s.", prepared_path)
        return load_prepared_cache(prepared_path)
    return load_pandora_comments(
        pandora_file,
        output_path=prepared_path,
        min_text_length=min_text_length,
        group_by=group_by,  # type: ignore[arg-type]
    )


def _default_pandora_file() -> Optional[Path]:
    candidates = sorted(Path("PANDORA").glob("**/train-*.parquet"))
    if not candidates:
        candidates = sorted(Path("PANDORA").glob("**/*.parquet"))
    return candidates[0] if candidates else None


def _parse_thresholds(value: str) -> Tuple[float, ...]:
    thresholds = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if len(thresholds) < 1:
        raise argparse.ArgumentTypeError("Provide at least one threshold.")
    return thresholds


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local PANDORA binary LSTM experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_file = _default_pandora_file()
    parser.add_argument("--pandora-file", default=str(default_file) if default_file else None)
    parser.add_argument("--work-dir", default="pandora_personality")
    parser.add_argument("--sample-n-users", type=int, default=40)
    parser.add_argument("--min-comments-per-user", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--qlearning-train-epochs", type=int, default=3)
    parser.add_argument("--lstm-epochs", type=int, default=35)
    parser.add_argument("--gan-epochs", type=int, default=150)
    parser.add_argument("--candidate-thresholds", type=_parse_thresholds, default=tuple(me.CANDIDATE_THRESHOLDS))
    parser.add_argument("--ground-truth-cutoff", type=float, default=me.DEFAULT_GROUND_TRUTH_CUTOFF)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh-prepared", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    logging.getLogger("ml_pipeline").setLevel(getattr(logging, args.log_level))

    if not args.pandora_file:
        raise SystemExit("No PANDORA parquet file found. Pass --pandora-file path/to/file.parquet.")

    work_dir = Path(args.work_dir)
    data_dir = work_dir / "data"
    cache_dir = work_dir / "cache"
    artifact_dir = work_dir / "artifacts"
    for folder in (data_dir, cache_dir, artifact_dir):
        folder.mkdir(parents=True, exist_ok=True)

    prepared = load_or_prepare_pandora(
        args.pandora_file,
        data_dir / "pandora_prepared.json",
        refresh_prepared=args.refresh_prepared,
    )
    cfg = ExperimentConfig(
        sample_n_users=args.sample_n_users,
        min_comments_per_user=args.min_comments_per_user,
        top_k=args.top_k,
        qlearning_train_epochs=args.qlearning_train_epochs,
        lstm_epochs=args.lstm_epochs,
        gan_epochs=args.gan_epochs,
        candidate_thresholds=args.candidate_thresholds,
        ground_truth_cutoff=args.ground_truth_cutoff,
        seed=args.seed,
        embedding_cache_dir=str(cache_dir),
        output_dir=str(artifact_dir),
    )
    bundle = ExperimentRunner(prepared, cfg).run()

    print("\nComparison")
    print(bundle["comparison"].to_string(index=False))
    print("\nFindings")
    for note in bundle["findings"]["notes"]:
        print(f"- {note}")
    print(f"\nArtifacts saved to: {bundle.get('artifact_dir', str(artifact_dir.resolve()))}")
    return bundle


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


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
