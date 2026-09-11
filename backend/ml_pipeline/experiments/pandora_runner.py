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
#   gan_augmenter.GANAugmenter                          -> paired GAN augmentation
#   lasso_regressor.LassoTrainer                        -> per-trait ElasticNet
#   lstm_classifier.LSTMTrainer                         -> per-trait 3-class LSTM
#   metrics_engine.evaluate (+ component fns)           -> canonical metrics
#
from backend.ml_pipeline.cleaning.cleaner import CleanedContent, ExtractedSignals
from backend.ml_pipeline.services.data.pandora import (
    PreparedUserComments,
    UserTraits,
    get_last_ingestion_quality,
    load_ingestion_quality,
    load_pandora_comments,
    quality_sidecar_path,
)
from backend.ml_pipeline.services.data.quality import (
    build_experiment_data_quality,
    comment_volume,
    exclusion,
    thesis_rows,
)
from backend.ml_pipeline.services.qlearning_agent import QLearningAgent, run_training_loop
from backend.ml_pipeline.services.bert_encoder import BERTEncoder
from backend.ml_pipeline.services.gan_augmenter import GANAugmenter
from backend.ml_pipeline.services.lasso_regressor import LassoTrainer
from backend.ml_pipeline.services.lstm_classifier import (
    LSTMTrainer,
    OCEAN_TRAITS,
    set_seed,
)
from backend.ml_pipeline.services import metrics_engine as me

logger = logging.getLogger("ml_pipeline")

TRAIT_KEYS: Tuple[str, ...] = ("O", "C", "E", "A", "N")
TRAIT_DISPLAY_ALIASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = tuple(
    (short, (short, str(full)))
    for short, full in zip(TRAIT_KEYS, OCEAN_TRAITS)
)

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
        "label": "Lasso/ElasticNet sparse regression | baseline-select",
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
        "label": "Lasso/ElasticNet sparse regression | baseline-select + GAN",
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
        "label": "Lasso/ElasticNet sparse regression | Q-learning-select",
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
        "label": "Lasso/ElasticNet sparse regression | Q-learning-select + GAN",
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
class DatasetSplits:
    sample: Sample
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    sources: Dict[str, str]


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


def combine_split_samples(train: Sample, validation: Sample, test: Sample) -> DatasetSplits:
    """Combine file-defined train/validation/test samples with explicit indexes."""
    if train.scale != validation.scale or train.scale != test.scale:
        logger.warning(
            "PANDORA split label scales differ: train=%s validation=%s test=%s.",
            train.scale, validation.scale, test.scale,
        )
    user_ids = train.user_ids + validation.user_ids + test.user_ids
    texts = train.texts + validation.texts + test.texts
    labels_raw = np.vstack([train.labels_raw, validation.labels_raw, test.labels_raw])
    labels_unit = np.vstack([train.labels_unit, validation.labels_unit, test.labels_unit])
    sample = Sample(
        user_ids=user_ids,
        texts=texts,
        labels_raw=labels_raw,
        labels_unit=labels_unit,
        scale=train.scale,
    )
    n_train = train.n_users
    n_val = validation.n_users
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, sample.n_users)
    return DatasetSplits(sample, train_idx, val_idx, test_idx, sources={})


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


def _baseline_select(texts: List[str], top_k: Optional[int] = None) -> List[str]:
    """Full-history baseline: embed every available comment for the author."""
    return list(texts)


def _qlearning_select(agent: QLearningAgent, texts: List[str]) -> List[str]:
    selected = [c["text"] for c in agent.select_comments(texts, top_k=None, training=False)]
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
        max_selected=None,
    )
    logger.info("Q-learning trained with %d Q-table states.", len(agent.q_table))
    return agent


def subset_sample(sample: Sample, indices: Sequence[int]) -> Sample:
    """Return a participant subset while preserving the sampled label scale."""
    idx = [int(i) for i in indices]
    return Sample(
        user_ids=[sample.user_ids[i] for i in idx],
        texts=[sample.texts[i] for i in idx],
        labels_raw=sample.labels_raw[idx],
        labels_unit=sample.labels_unit[idx],
        scale=sample.scale,
    )


def trait_band_distribution(
    sample: Sample,
    indices: Sequence[int],
    trait_label_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Count Low/Medium/High and binary Low/High bands for each OCEAN trait."""
    idx = np.asarray([int(i) for i in indices], dtype=int)
    labels = sample.labels_unit[idx] if len(idx) else np.empty((0, len(TRAIT_KEYS)))
    out: Dict[str, Any] = {
        "n_users": int(len(idx)),
        "band_cutoffs": {"low_max": 1.0 / 3.0, "high_min": 2.0 / 3.0},
        "binary_cutoff_source": "train_split_median_per_trait" if trait_label_thresholds else "configured_default",
        "traits": {},
    }
    for ti, trait in enumerate(TRAIT_KEYS):
        values = labels[:, ti] if len(labels) else np.asarray([], dtype=float)
        binary_cutoff = (
            _cutoff_for_trait(trait_label_thresholds, str(trait), ti)
            if trait_label_thresholds is not None
            else float(me.DEFAULT_GROUND_TRUTH_CUTOFF)
        )
        low = int(np.sum(values < (1.0 / 3.0)))
        medium = int(np.sum((values >= (1.0 / 3.0)) & (values < (2.0 / 3.0))))
        high = int(np.sum(values >= (2.0 / 3.0)))
        bin_low = int(np.sum(values < binary_cutoff))
        bin_high = int(np.sum(values >= binary_cutoff))
        nonzero = [x for x in (low, medium, high) if x > 0]
        out["traits"][trait] = {
            "mean": _f(float(np.mean(values)) if len(values) else None),
            "low": low,
            "medium": medium,
            "high": high,
            "low_medium_high": [low, medium, high],
            "imbalance_ratio": _f(max(nonzero) / min(nonzero) if nonzero else None),
            "binary_cutoff": _f(binary_cutoff),
            "binary_low": bin_low,
            "binary_high": bin_high,
        }
    return out


def derive_trait_label_thresholds(sample: Sample, train_idx: Sequence[int]) -> Dict[str, float]:
    """
    Learn the Low/High ground-truth boundary from TRAIN labels only.

    The saved dict includes both short keys (O/C/E/A/N) and full trait names so
    older reporting paths can read the same locked cutoffs without translation.
    """
    idx = np.asarray([int(i) for i in train_idx], dtype=int)
    labels = sample.labels_unit[idx] if len(idx) else sample.labels_unit
    thresholds: Dict[str, float] = {}
    for ti, short in enumerate(TRAIT_KEYS):
        value = float(np.median(labels[:, ti])) if len(labels) else float(me.DEFAULT_GROUND_TRUTH_CUTOFF)
        thresholds[str(short)] = value
        thresholds[str(OCEAN_TRAITS[ti])] = value
    return thresholds


def _cutoff_for_trait(thresholds: Dict[str, float], trait: str, ti: int) -> float:
    return float(thresholds.get(trait, thresholds.get(str(TRAIT_KEYS[ti]), me.DEFAULT_GROUND_TRUTH_CUTOFF)))


def build_imbalance_report(
    sample: Sample,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    trait_label_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Persist run-level imbalance evidence for each file-defined split."""
    return {
        "kind": "trait_band_distribution",
        "scale": sample.scale,
        "splits": {
            "train": trait_band_distribution(sample, train_idx, trait_label_thresholds),
            "validation": trait_band_distribution(sample, val_idx, trait_label_thresholds),
            "test": trait_band_distribution(sample, test_idx, trait_label_thresholds),
        },
        "notes": [
            "Low/Medium/High uses fixed normalized cut points 0.333 and 0.667.",
            "Binary Low/High uses train-derived per-trait label cutoffs when available.",
            "Training weights are derived from train split trait bands only.",
        ],
    }


def training_sample_weights(sample: Sample, train_idx: Sequence[int]) -> np.ndarray:
    """Average inverse-frequency band weights across the five traits."""
    idx = np.asarray([int(i) for i in train_idx], dtype=int)
    if len(idx) == 0:
        return np.asarray([], dtype=float)
    labels = sample.labels_unit[idx]
    weights = np.zeros(len(idx), dtype=float)
    for ti in range(labels.shape[1]):
        bands = to_tertile_classes(labels[:, ti], 1.0 / 3.0, 2.0 / 3.0)
        counts = np.bincount(bands, minlength=3).astype(float)
        counts[counts == 0] = 1.0
        trait_weights = len(bands) / (3.0 * counts)
        weights += trait_weights[bands]
    weights = weights / labels.shape[1]
    return weights / max(float(np.mean(weights)), 1e-12)


def targeted_gan_training_positions(sample: Sample, train_idx: Sequence[int]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Pick train-row positions that sit in underrepresented Low/Medium/High bands.

    This keeps GAN augmentation focused on scarce trait regions while the final
    classifier remains binary Low/High.
    """
    idx = np.asarray([int(i) for i in train_idx], dtype=int)
    labels = sample.labels_unit[idx] if len(idx) else np.empty((0, len(TRAIT_KEYS)))
    if len(labels) == 0:
        return np.asarray([], dtype=int), {"enabled": False, "reason": "empty_train_split"}

    weights = np.zeros(len(labels), dtype=float)
    traits: Dict[str, Any] = {}
    band_names = ["low", "medium", "high"]
    for ti, trait in enumerate(TRAIT_KEYS):
        bands = to_tertile_classes(labels[:, ti], 1.0 / 3.0, 2.0 / 3.0)
        counts = np.bincount(bands, minlength=3).astype(float)
        target = float(np.max(counts)) if len(counts) else 0.0
        deficits = np.maximum(target - counts, 0.0)
        if target > 0:
            weights += deficits[bands] / target
        traits[str(trait)] = {
            "counts": {band_names[i]: int(counts[i]) for i in range(3)},
            "target_count": int(target),
            "deficits": {band_names[i]: int(deficits[i]) for i in range(3)},
        }

    selected = np.flatnonzero(weights > 0)
    if len(selected) < 2:
        return np.asarray([], dtype=int), {
            "enabled": False,
            "reason": "no_minority_band_with_enough_rows",
            "traits": traits,
            "selected_train_rows": int(len(selected)),
        }
    return selected.astype(int), {
        "enabled": True,
        "strategy": "train_gan_on_rows_from_underrepresented_fixed_trait_bands",
        "band_cutoffs": {"low_max": 1.0 / 3.0, "high_min": 2.0 / 3.0},
        "selected_train_rows": int(len(selected)),
        "traits": traits,
    }


def selection_efficiency_report(
    sample: Sample,
    features: Dict[str, Features],
    cfg: ExperimentConfig,
    qlearning_effect: Any = None,
    feature_build_seconds: Optional[Dict[str, float]] = None,
    results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record the practical cost/saving of selection before BERT/model training."""
    total_comments = np.asarray([len(texts) for texts in sample.texts], dtype=float)
    total_available = int(np.sum(total_comments))
    out: Dict[str, Any] = {
        "kind": "selection_efficiency",
        "selection_unit": "author_profile_comments",
        "expensive_step": "BERT embedding is run only on selected comments.",
        "qlearning_extra_work": (
            "Q-learning scores candidate comments with cheap lexical features before BERT; "
            "it does not call BERT while selecting."
        ),
        "baseline_policy": "use_all_available_comments",
        "qlearning_policy": "learned_select_skip_over_all_comments_without_fixed_selection_budget",
        "qlearning_selection_budget": None,
        "total_profiles": int(sample.n_users),
        "total_available_comments": total_available,
        "by_selection": {},
        "feature_build_seconds": {
            str(k): _f(v)
            for k, v in (feature_build_seconds or {}).items()
        },
        "condition_training_seconds": {},
    }
    for mode, block in features.items():
        selected = np.asarray(block.n_selected, dtype=float)
        selected_total = int(np.sum(selected))
        saved = max(0, total_available - selected_total)
        out["by_selection"][mode] = {
            "profiles": int(len(selected)),
            "selected_comments_total": selected_total,
            "mean_comments_selected": _f(float(np.mean(selected)) if len(selected) else None),
            "median_comments_selected": _f(float(np.median(selected)) if len(selected) else None),
            "max_comments_selected": int(np.max(selected)) if len(selected) else 0,
            "bert_embedding_calls_saved_vs_full_history": int(saved),
            "bert_embedding_reduction_vs_full_history": _f(saved / total_available if total_available else None),
        }

    baseline = out["by_selection"].get("baseline") or {}
    qlearn = out["by_selection"].get("qlearning") or {}
    if baseline and qlearn:
        baseline_total = max(1, int(baseline["selected_comments_total"]))
        saved_vs_baseline = int(baseline["selected_comments_total"] - qlearn["selected_comments_total"])
        out["qlearning_vs_baseline"] = {
            "delta_selected_comments_total": int(
                qlearn["selected_comments_total"] - baseline["selected_comments_total"]
            ),
            "delta_mean_comments_selected": _f(
                (qlearn["mean_comments_selected"] or 0.0) - (baseline["mean_comments_selected"] or 0.0)
            ),
            "bert_embedding_calls_saved_vs_baseline": saved_vs_baseline,
            "bert_embedding_reduction_vs_baseline": _f(saved_vs_baseline / baseline_total),
            "selection_time_delta_seconds": _f(
                (out["feature_build_seconds"].get("qlearning") or 0.0) -
                (out["feature_build_seconds"].get("baseline") or 0.0)
            ),
            "interpretation": (
                "Baseline embeds every available author comment. Q-learning sees the same candidate pool, "
                "then selects a smaller learned subset before BERT. The compute tradeoff is the saved "
                "BERT calls versus any accuracy/F1 change in quality_effect_by_matched_cell."
            ),
        }
    if qlearning_effect is not None:
        try:
            out["quality_effect_by_matched_cell"] = qlearning_effect.to_dict("records")
        except AttributeError:
            out["quality_effect_by_matched_cell"] = qlearning_effect
    if results:
        for exp_id, result in results.items():
            out["condition_training_seconds"][exp_id] = _f(result.get("training_seconds"))
    return out


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
            selected = _qlearning_select(agent, user_texts)  # type: ignore[arg-type]
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


def _augment_pooled_gan(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    cfg: ExperimentConfig,
    minority_positions: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fit the adversarial GAN on the pooled TRAIN vectors only, then generate one
    synthetic pooled vector per real training user. Used by the Lasso path.
    Deterministic given ``cfg.seed`` (the GAN seeds training and generation).
    """
    X_tr = np.asarray(X_tr, dtype=np.float32)
    y_tr = np.asarray(y_tr, dtype=np.float32)
    positions = np.asarray(minority_positions if minority_positions is not None else [], dtype=int)
    if len(positions) >= 2:
        fit_X = X_tr[positions]
        fit_y = y_tr[positions]
        n_generate = len(positions)
        targeted = True
    else:
        fit_X = X_tr
        fit_y = y_tr
        n_generate = len(X_tr)
        targeted = False
    if len(fit_X) < 2:  # GAN needs >= 2 real samples to fit
        logger.warning("Too few train users (%d) to fit the GAN; skipping augmentation.", len(fit_X))
        empty_x = np.empty((0, X_tr.shape[1]), dtype=np.float32)
        empty_y = np.empty((0, y_tr.shape[1]), dtype=np.float32)
        return empty_x, empty_y, {"generated_rows": 0, "targeted": targeted, "reason": "too_few_rows"}
    gan = _make_gan(X_tr.shape[1], cfg).fit(fit_X, ocean_scores=fit_y)
    synth_x, synth_y, _ = gan.generate(n_generate)
    return (
        np.asarray(synth_x, dtype=np.float32),
        np.asarray(synth_y, dtype=np.float32),
        {"generated_rows": int(n_generate), "fit_rows": int(len(fit_X)), "targeted": targeted},
    )


def _augment_sequences_gan(
    train_seqs: List[np.ndarray],
    y_tr: np.ndarray,
    cfg: ExperimentConfig,
    minority_positions: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, Any]]:
    """
    LSTM path: fit the adversarial GAN on *all* real TRAIN timestep vectors
    (never val), then generate one same-length synthetic sequence per real
    training user. Generating the full synthetic pool in a single ``generate``
    call keeps the synthetic timesteps distinct across sequences.
    Deterministic given ``cfg.seed``.
    """
    positions = np.asarray(minority_positions if minority_positions is not None else [], dtype=int)
    if len(positions) >= 2:
        selected_seqs = [train_seqs[int(i)] for i in positions]
        selected_y = y_tr[positions]
        targeted = True
    else:
        selected_seqs = train_seqs
        selected_y = y_tr
        targeted = False
    all_vecs = np.vstack(selected_seqs).astype(np.float32) if selected_seqs else np.empty((0, 768), np.float32)
    if len(all_vecs) < 2:
        logger.warning("Too few train timesteps (%d) to fit the GAN; reusing real sequences.", len(all_vecs))
        return [], np.empty((0, y_tr.shape[1]), dtype=np.float32), {
            "generated_sequences": 0,
            "targeted": targeted,
            "reason": "too_few_timesteps",
        }
    repeated_y = np.vstack([
        np.repeat(selected_y[i:i + 1], len(seq), axis=0)
        for i, seq in enumerate(selected_seqs)
    ]).astype(np.float32)
    gan = _make_gan(all_vecs.shape[1], cfg).fit(all_vecs, ocean_scores=repeated_y)
    total = int(sum(len(s) for s in selected_seqs))
    synth_all, synth_y_all, _ = gan.generate(total)
    synth_all = np.asarray(synth_all, dtype=np.float32)
    synth_y_all = np.asarray(synth_y_all, dtype=np.float32)
    out: List[np.ndarray] = []
    y_out: List[np.ndarray] = []
    pos = 0
    for seq in selected_seqs:
        k = len(seq)
        out.append(synth_all[pos:pos + k])
        y_out.append(np.mean(synth_y_all[pos:pos + k], axis=0))
        pos += k
    return out, np.vstack(y_out).astype(np.float32), {
        "generated_sequences": int(len(out)),
        "fit_sequences": int(len(selected_seqs)),
        "fit_timesteps": int(len(all_vecs)),
        "targeted": targeted,
    }


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
    train_sample_weights: Optional[np.ndarray] = None,
    trait_label_thresholds: Optional[Dict[str, float]] = None,
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
    real_sample_weight = (
        np.asarray(train_sample_weights, dtype=float)
        if train_sample_weights is not None
        else np.ones(len(X_tr_scaled), dtype=float)
    )
    sample_weight = real_sample_weight
    synth_y_all: Optional[np.ndarray] = None
    gan_report: Dict[str, Any] = {"used": False}
    if use_gan:
        y_tr_all = sample.labels_unit[train_idx].astype(np.float32)
        minority_positions, targeting = targeted_gan_training_positions(sample, train_idx)
        synth, synth_y_all, augment_report = _augment_pooled_gan(X_tr, y_tr_all, cfg, minority_positions)
        gan_report = {"used": len(synth) > 0, "targeting": targeting, "augmentation": augment_report}
        if len(synth) > 0:
            synth_scaled = trainer.transform_features(synth)
            sample_weight = np.concatenate([
                real_sample_weight,
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
            y_fit = np.concatenate([y_tr_unit, synth_y_all[:, ti]])
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
        gt_input = (
            _cutoff_for_trait(trait_label_thresholds, trait, ti)
            if trait_label_thresholds is not None
            else cfg.ground_truth_cutoff
        )
        y_bin, gt_cut = me.derive_binary_ground_truth(y_val_unit, gt_input)
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
    return {"per_trait": per_trait, "overall": overall, "targeted_gan": gan_report}, trainer, raw


def _run_lstm(
    features: Features,
    sample: Sample,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: ExperimentConfig,
    use_gan: bool,
    train_sample_weights: Optional[np.ndarray] = None,
    trait_label_thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, Any], LSTMTrainer, Dict[str, np.ndarray]]:
    """Train the current joint 5-output LSTM and score validation Low/High metrics."""
    seqs = features.sequences
    tr_seqs = [seqs[i] for i in train_idx]
    val_seqs = [seqs[i] for i in val_idx]
    y_tr = sample.labels_unit[train_idx].astype(np.float32)
    y_val = sample.labels_unit[val_idx].astype(np.float32)
    real_sample_weights = (
        np.asarray(train_sample_weights, dtype=float)
        if train_sample_weights is not None
        else np.ones(len(tr_seqs), dtype=float)
    )

    synth_seqs: Optional[List[np.ndarray]] = None
    sample_weights = real_sample_weights
    gan_report: Dict[str, Any] = {"used": False}
    if use_gan:
        minority_positions, targeting = targeted_gan_training_positions(sample, train_idx)
        synth_seqs, synth_y, augment_report = _augment_sequences_gan(tr_seqs, y_tr, cfg, minority_positions)
        gan_report = {"used": len(synth_seqs) > 0, "targeting": targeting, "augmentation": augment_report}
        fit_targets = np.concatenate([y_tr, synth_y], axis=0) if len(synth_seqs) else y_tr
        sample_weights = np.concatenate([
            real_sample_weights,
            np.full(len(synth_seqs), cfg.synthetic_weight, dtype=float),
        ]) if len(synth_seqs) else real_sample_weights
        fit_seqs = tr_seqs + synth_seqs if len(synth_seqs) else tr_seqs
    else:
        fit_seqs = tr_seqs
        fit_targets = y_tr

    trainer = LSTMTrainer(
        hidden_dim=cfg.lstm_hidden_dim,
        num_layers=cfg.lstm_num_layers,
        dropout=cfg.lstm_dropout,
        learning_rate=cfg.lstm_learning_rate,
    )
    trainer.train(
        "OCEAN",
        fit_seqs,
        fit_targets,
        val_sequences=val_seqs,
        val_targets=y_val,
        epochs=cfg.lstm_epochs,
        batch_size=cfg.lstm_batch_size,
        sample_weights=sample_weights,
        seed=cfg.seed,
    )
    val_pred = np.clip(trainer.predict(val_seqs), 0.0, 1.0)
    val_regression: Dict[str, Dict[str, float]] = {}
    for ti, trait in enumerate(TRAIT_KEYS):
        val_regression[trait] = me.compute_regression_metrics(y_val[:, ti], val_pred[:, ti])
    validation = me.evaluate_lstm_binary_classifier(
        y_val,
        val_pred,
        trait_names=list(TRAIT_KEYS),
        ground_truth_cutoff=trait_label_thresholds or cfg.ground_truth_cutoff,
        candidate_thresholds=None,
    )

    per_trait: Dict[str, Any] = {}
    for trait in TRAIT_KEYS:
        block = validation["per_trait"][trait]
        reg = val_regression[trait]
        per_trait[trait] = {
            "val_mae": reg["mae"],
            "val_rmse": reg["rmse"],
            "val_r2": reg["r2"],
            "val_pearson": reg["correlation"],
            "accuracy": block["accuracy"],
            "macro_f1": block["f1"],
            "macro_precision": block["precision"],
            "macro_recall": block["recall"],
            "specificity": block["specificity"],
            "roc_auc": block.get("roc_auc"),
            "pr_auc": block.get("pr_auc"),
            "best_threshold": block.get("best_threshold"),
            "threshold_sweep": block["threshold_sweep"],
        }

    overall = {
        "val_mae": _mean([per_trait[t]["val_mae"] for t in TRAIT_KEYS]),
        "val_rmse": _mean([per_trait[t]["val_rmse"] for t in TRAIT_KEYS]),
        "val_r2": _mean([per_trait[t]["val_r2"] for t in TRAIT_KEYS]),
        "val_pearson": _mean([per_trait[t]["val_pearson"] for t in TRAIT_KEYS]),
        "accuracy": validation["aggregate"]["accuracy"],
        "macro_f1": validation["aggregate"]["f1"],
        "specificity": validation["aggregate"]["specificity"],
        "roc_auc": validation["aggregate"].get("roc_auc"),
        "pr_auc": validation["aggregate"].get("pr_auc"),
    }
    raw = {"true_unit": y_val, "lstm_pred": val_pred}
    return {"per_trait": per_trait, "overall": overall, "targeted_gan": gan_report}, trainer, raw


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
    test_idx: Optional[np.ndarray] = None,
    train_sample_weights: Optional[np.ndarray] = None,
    trait_label_thresholds: Optional[Dict[str, float]] = None,
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
        result, model, raw = _run_lasso(
            features, sample, train_idx, val_idx, cfg,
            use_gan=spec["gan"],
            train_sample_weights=train_sample_weights,
            trait_label_thresholds=trait_label_thresholds,
        )
    elif spec["model"] == "lstm":
        result, model, raw = _run_lstm(
            features, sample, train_idx, val_idx, cfg,
            use_gan=spec["gan"],
            train_sample_weights=train_sample_weights,
            trait_label_thresholds=trait_label_thresholds,
        )
    else:
        raise ValueError(f"Unknown model for {exp_id}: {spec['model']}")

    validation_result = {
        "per_trait": result["per_trait"],
        "overall": _reporting_aliases(result["overall"]),
    }
    official_result = validation_result
    raw["split"] = "validation"

    if test_idx is not None and len(test_idx) > 0:
        if spec["model"] == "lstm":
            test_seqs = [features.sequences[int(i)] for i in test_idx]
            y_test = sample.labels_unit[test_idx].astype(np.float32)
            test_pred = np.clip(model.predict(test_seqs), 0.0, 1.0)
            test_regression: Dict[str, Dict[str, float]] = {}
            for ti, trait in enumerate(TRAIT_KEYS):
                test_regression[trait] = me.compute_regression_metrics(y_test[:, ti], test_pred[:, ti])
            test_eval = me.evaluate_lstm_binary_with_thresholds(
                y_test,
                test_pred,
                validation_result,
                trait_names=list(TRAIT_KEYS),
                ground_truth_cutoff=trait_label_thresholds or cfg.ground_truth_cutoff,
                candidate_thresholds=None,
            )
            test_per_trait = {}
            for trait in TRAIT_KEYS:
                block = test_eval["per_trait"][trait]
                reg = test_regression[trait]
                test_per_trait[trait] = {
                    "val_mae": reg["mae"],
                    "val_rmse": reg["rmse"],
                    "val_r2": reg["r2"],
                    "val_pearson": reg["correlation"],
                    "accuracy": block["accuracy"],
                    "macro_f1": block["f1"],
                    "f1": block["f1"],
                    "macro_precision": block["precision"],
                    "precision": block["precision"],
                    "macro_recall": block["recall"],
                    "recall": block["recall"],
                    "specificity": block["specificity"],
                    "roc_auc": block.get("roc_auc"),
                    "pr_auc": block.get("pr_auc"),
                    "selected_threshold": block.get("selected_threshold"),
                    "threshold_source": "validation",
                    "threshold_sweep": block["threshold_sweep"],
                }
            official_result = {
                "per_trait": test_per_trait,
                "overall": _reporting_aliases({
                    "val_mae": _mean([test_per_trait[t]["val_mae"] for t in TRAIT_KEYS]),
                    "val_rmse": _mean([test_per_trait[t]["val_rmse"] for t in TRAIT_KEYS]),
                    "val_r2": _mean([test_per_trait[t]["val_r2"] for t in TRAIT_KEYS]),
                    "val_pearson": _mean([test_per_trait[t]["val_pearson"] for t in TRAIT_KEYS]),
                    "accuracy": test_eval["aggregate"]["accuracy"],
                    "macro_f1": test_eval["aggregate"]["f1"],
                    "specificity": test_eval["aggregate"]["specificity"],
                    "roc_auc": test_eval["aggregate"].get("roc_auc"),
                    "pr_auc": test_eval["aggregate"].get("pr_auc"),
                }),
            }
            raw = {"true_unit": y_test, "lstm_pred": test_pred, "split": "test"}
        else:
            X_test = features.pooled[test_idx]
            X_test_scaled = model.transform_features(X_test)
            n_test = len(test_idx)
            n_traits = len(OCEAN_TRAITS)
            true_unit = np.zeros((n_test, n_traits), dtype=float)
            pred_mat = np.zeros((n_test, n_traits), dtype=float)
            true_classes = np.zeros((n_test, n_traits), dtype=int)
            test_per_trait = {}
            for ti, trait in enumerate(OCEAN_TRAITS):
                unit = sample.labels_unit[:, ti]
                y_tr_unit = unit[train_idx]
                y_test_unit = unit[test_idx]
                pred = model.predict_trait(trait, X_test_scaled)
                low_cut, high_cut = _tertile_cuts(y_tr_unit)
                y_cls = to_tertile_classes(y_test_unit, low_cut, high_cut)
                pred_cls = to_tertile_classes(pred, low_cut, high_cut)
                reg = me.compute_regression_metrics(y_test_unit, pred)
                cls = me.compute_multiclass_metrics(y_cls, pred_cls, labels=[0, 1, 2])
                selected_tau = validation_result["per_trait"][trait]["threshold_sweep"]["best_threshold"]
                gt_input = (
                    _cutoff_for_trait(trait_label_thresholds, trait, ti)
                    if trait_label_thresholds is not None
                    else cfg.ground_truth_cutoff
                )
                y_bin, gt_cut = me.derive_binary_ground_truth(y_test_unit, gt_input)
                official = me.compute_classification_metrics_at_threshold(y_bin, pred, selected_tau)
                sweep = me.sweep_thresholds_on_scores(y_bin, pred, None)
                true_unit[:, ti] = y_test_unit
                pred_mat[:, ti] = pred
                true_classes[:, ti] = y_cls
                test_per_trait[trait] = {
                    "val_mae": _f(reg["mae"]),
                    "val_rmse": _f(reg["rmse"]),
                    "val_r2": _f(reg["r2"]),
                    "val_pearson": _f(reg["correlation"]),
                    "accuracy": _f(cls["accuracy"]),
                    "macro_f1": _f(cls["f1"]),
                    "macro_precision": _f(cls["precision"]),
                    "macro_recall": _f(cls["recall"]),
                    "specificity": official["specificity"],
                    "precision": official["precision"],
                    "recall": official["recall"],
                    "f1": official["f1_score"],
                    "selected_threshold": selected_tau,
                    "threshold_source": "validation",
                    "threshold_sweep": sweep["results"],
                    "ground_truth_cutoff": _f(gt_cut),
                    "tertile_cuts": [low_cut, high_cut],
                }
            official_result = {
                "per_trait": test_per_trait,
                "overall": _reporting_aliases({
                    "val_mae": _mean([test_per_trait[t]["val_mae"] for t in OCEAN_TRAITS]),
                    "val_rmse": _mean([test_per_trait[t]["val_rmse"] for t in OCEAN_TRAITS]),
                    "val_r2": _mean([test_per_trait[t]["val_r2"] for t in OCEAN_TRAITS]),
                    "val_pearson": _mean([test_per_trait[t]["val_pearson"] for t in OCEAN_TRAITS]),
                    "accuracy": _mean([test_per_trait[t]["accuracy"] for t in OCEAN_TRAITS]),
                    "macro_f1": _mean([test_per_trait[t]["macro_f1"] for t in OCEAN_TRAITS]),
                    "specificity": _mean([test_per_trait[t]["specificity"] for t in OCEAN_TRAITS]),
                    "precision": _mean([test_per_trait[t]["precision"] for t in OCEAN_TRAITS]),
                    "recall": _mean([test_per_trait[t]["recall"] for t in OCEAN_TRAITS]),
                    "f1": _mean([test_per_trait[t]["f1"] for t in OCEAN_TRAITS]),
                }),
            }
            raw = {"true_unit": true_unit, "lasso_pred": pred_mat, "true_classes": true_classes, "split": "test"}

    out = {
        "experiment": exp_id,
        "label": spec["label"],
        "model": spec["model"],
        "selection": spec["selection"],
        "gan": bool(spec["gan"]),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)) if test_idx is not None else int(len(val_idx)),
        "mean_comments_selected": float(np.mean(features.n_selected)),
        "per_trait": official_result["per_trait"],
        "overall": official_result["overall"],
        "validation": {
            "candidate_thresholds": "validation_score_percentiles_per_trait",
            "split": "validation",
            **validation_result,
        },
        "targeted_gan": result.get("targeted_gan"),
        "test": {
            "split": "test" if test_idx is not None and len(test_idx) > 0 else "validation",
            **official_result,
            "threshold_source": "validation",
        },
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
    trait_label_thresholds = derive_trait_label_thresholds(sample, train_idx)
    out, model, _raw = _run_condition(
        sample,
        exp_id,
        cfg,
        features,
        train_idx,
        val_idx,
        test_idx,
        trait_label_thresholds=trait_label_thresholds,
    )
    return (out, model) if return_model else out


run_condition = run_experiment


def _reporting_aliases(overall: Dict[str, Any]) -> Dict[str, Any]:
    """Copy already-computed metrics onto the names interpretation/contract read."""
    out = dict(overall)
    if out.get("mae") is None:
        out["mae"] = out.get("val_mae")
    if out.get("rmse") is None:
        out["rmse"] = out.get("val_rmse")
    if out.get("r2") is None:
        out["r2"] = out.get("val_r2")
    if out.get("official_f1") is None:
        out["official_f1"] = out.get("macro_f1")
    if out.get("official_accuracy") is None:
        out["official_accuracy"] = out.get("accuracy")
    return out


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item") and not isinstance(obj, (bytes, str)):
        try:
            return obj.item()
        except (ValueError, AttributeError):
            pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    return str(obj)


def _repository_identity() -> Dict[str, Any]:
    sha = None
    dirty = None
    try:
        import subprocess

        root = Path(__file__).resolve().parents[3]
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        dirty = bool(status.strip())
    except Exception:
        sha = None
        dirty = None
    return {
        "repository_commit_sha": sha,
        "repository_dirty": dirty,
        "code_identity_reliable": bool(sha) and dirty is False,
    }


def _measure_experiment_filter(
    prepared: List[PreparedUserComments],
    sample: Sample,
    cfg: ExperimentConfig,
) -> Dict[str, Any]:
    if not prepared:
        return {
            "users_before_filter": sample.n_users,
            "users_after_filter": sample.n_users,
            "users_used": sample.n_users,
            "exclusion_reasons": {},
        }
    n_before = len(prepared)
    n_no_traits = sum(1 for user in prepared if user.traits is None)
    n_too_few = sum(
        1
        for user in prepared
        if user.traits is not None and len(user.comments) < cfg.min_comments_per_user
    )
    n_eligible = n_before - n_no_traits - n_too_few
    return {
        "users_before_filter": n_before,
        "users_after_filter": n_eligible,
        "users_used": sample.n_users,
        "exclusion_reasons": {
            "missing_traits": exclusion(
                n_no_traits,
                "Prepared user has no OCEAN traits.",
                stage="experiment_filter",
            ),
            "too_few_comments": exclusion(
                n_too_few,
                f"Prepared user has fewer than {cfg.min_comments_per_user} comments.",
                stage="experiment_filter",
            ),
        },
    }


def _assemble_data_quality(
    prepared: List[PreparedUserComments],
    sample: Sample,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: ExperimentConfig,
) -> Dict[str, Any]:
    filt = _measure_experiment_filter(prepared, sample, cfg)
    ingestion = get_last_ingestion_quality()
    if not ingestion and cfg.output_dir:
        prepared_json = Path(cfg.output_dir).parent / "data" / "pandora_prepared.json"
        ingestion = load_ingestion_quality(prepared_json)
    train_counts = [len(sample.texts[int(i)]) for i in train_idx]
    val_counts = [len(sample.texts[int(i)]) for i in val_idx]
    test_counts = [len(sample.texts[int(i)]) for i in test_idx]
    all_counts = [len(texts) for texts in sample.texts]
    volume = {
        "train": comment_volume(train_counts, split="train", population="sampled_train"),
        "val": comment_volume(val_counts, split="validation", population="sampled_held_out"),
        "test": comment_volume(test_counts, split="test", population="sampled_test"),
        "sampled_all_folds": comment_volume(all_counts, split="all", population="sampled_all_folds"),
    }
    notes = [
        "Filter counts recount the same eligibility rules used by sample_users; they do not change sampling.",
        "The current runner uses file-defined train/validation/test splits when available.",
    ]
    if not ingestion or not ingestion.get("available"):
        notes.append("Ingestion/cleaning counts were not measured in this process.")
    return build_experiment_data_quality(
        ingestion=ingestion,
        users_before_filter=filt["users_before_filter"],
        users_after_filter=filt["users_after_filter"],
        users_used=filt["users_used"],
        exclusion_reasons=filt["exclusion_reasons"],
        comment_volume=volume,
        notes=notes,
    )


def _build_experiment_data(
    sample: Sample,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: ExperimentConfig,
    split_sources: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    train_ids = [sample.user_ids[int(i)] for i in train_idx]
    val_ids = [sample.user_ids[int(i)] for i in val_idx]
    test_ids = [sample.user_ids[int(i)] for i in test_idx]
    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    overlap = {
        "train_validation": sorted(train_set & val_set),
        "train_test": sorted(train_set & test_set),
        "validation_test": sorted(val_set & test_set),
    }
    overlap_counts = {name: len(ids) for name, ids in overlap.items()}
    return {
        "kind": "experiment_data",
        "dataset_source": "PANDORA",
        "split_source": "file_defined" if split_sources else "generated",
        "split_sources": split_sources or {},
        "seed": cfg.seed,
        "n_users": sample.n_users,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "held_out_fold": "test",
        "label_scale": sample.scale,
        "user_ids": list(sample.user_ids),
        "split": {
            "train_user_ids": train_ids,
            "validation_user_ids": val_ids,
            "test_user_ids": test_ids,
            "held_out_fold": "test",
        },
        "author_overlap": overlap,
        "author_overlap_counts": overlap_counts,
        "participant_level_split_verified": all(count == 0 for count in overlap_counts.values()),
    }


def _build_reproducibility(
    sample: Sample,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: ExperimentConfig,
    experiment_data: Dict[str, Any],
) -> Dict[str, Any]:
    repo = _repository_identity()
    return {
        "random_seed": cfg.seed,
        "dataset_source": "PANDORA",
        "repository": repo,
        "repository_commit_sha": repo.get("repository_commit_sha"),
        "repository_dirty": repo.get("repository_dirty"),
        "code_identity_reliable": repo.get("code_identity_reliable"),
        "train_participant_count": int(len(train_idx)),
        "validation_participant_count": int(len(val_idx)),
        "test_participant_count": int(len(test_idx)),
        "participant_sample_size": sample.n_users,
        "split": experiment_data["split"],
        "preprocessing": {"label_scale": sample.scale, "group_by": "author"},
        "config_fingerprint": hashlib.sha1(
            json.dumps(asdict(cfg), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
    }


def _safe_interpret(fn, *args, **kwargs) -> Dict[str, Any]:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.exception("Interpretation layer failed: %s", exc)
        return {
            "kind": "interpretation_error",
            "error": str(exc),
            "uses_llm": False,
            "recalculates_metrics": False,
            "clinical_diagnosis": False,
        }


def _attach_interpretation(bundle: Dict[str, Any]) -> Dict[str, Any]:
    from backend.ml_pipeline.services.interpretation_engine import (
        interpret_condition_result,
        interpret_experiment_bundle,
    )

    data_quality = bundle.get("data_quality") or {}
    config = bundle.get("config") or {}
    for row in (bundle.get("results") or {}).values():
        if not isinstance(row, dict):
            continue
        row["interpretation"] = _safe_interpret(
            interpret_condition_result,
            row,
            data_quality=data_quality,
            config=config,
        )
    return _safe_interpret(interpret_experiment_bundle, bundle)


def run_all(
    prepared: Union[List[PreparedUserComments], DatasetSplits],
    cfg: ExperimentConfig,
    *,
    encoder: Any = None,
) -> Dict[str, Any]:
    """Run the full local PANDORA 2x2x2 experiment."""
    set_seed(cfg.seed)
    if isinstance(prepared, DatasetSplits):
        sample = prepared.sample
        train_idx = prepared.train_idx
        val_idx = prepared.val_idx
        test_idx = prepared.test_idx
        split_sources = prepared.sources
    else:
        sample = sample_users(prepared, cfg)
        train_idx, val_idx = make_split(sample.n_users, cfg)
        test_idx = val_idx
        split_sources = {}
    encoder = encoder or get_encoder()
    qlearning_train_sample = subset_sample(sample, train_idx)
    agent = train_qlearning_agent(qlearning_train_sample, cfg)
    sample_weights = training_sample_weights(sample, train_idx)
    trait_label_thresholds = derive_trait_label_thresholds(sample, train_idx)
    imbalance = build_imbalance_report(sample, train_idx, val_idx, test_idx, trait_label_thresholds)

    feature_build_seconds: Dict[str, float] = {}
    start = time.perf_counter()
    baseline_features = build_features(sample, "baseline", cfg, encoder)
    feature_build_seconds["baseline"] = time.perf_counter() - start
    start = time.perf_counter()
    qlearning_features = build_features(sample, "qlearning", cfg, encoder, agent=agent)
    feature_build_seconds["qlearning"] = time.perf_counter() - start
    features = {
        "baseline": baseline_features,
        "qlearning": qlearning_features,
    }

    results: Dict[str, Any] = {}
    models: Dict[str, Union[LassoTrainer, LSTMTrainer]] = {}
    for exp_id, spec in EXPERIMENTS.items():
        condition_start = time.perf_counter()
        out, model, _raw = _run_condition(
            sample, exp_id, cfg,
            features[spec["selection"]],
            train_idx, val_idx,
            test_idx=test_idx,
            train_sample_weights=sample_weights,
            trait_label_thresholds=trait_label_thresholds,
        )
        out["training_seconds"] = _f(time.perf_counter() - condition_start)
        results[exp_id] = out
        models[exp_id] = model

    comparison = comparison_table(results)
    effects = factor_effects(results)
    qlearning_efficiency = selection_efficiency_report(
        sample,
        features,
        cfg,
        qlearning_effect=effects.get("qlearning_effect"),
        feature_build_seconds=feature_build_seconds,
        results=results,
    )
    findings = summarize_findings(results)
    presentation_metrics = presentation_metric_table(results)
    threshold_sweeps = threshold_sweep_table(results)
    prediction_evidence = prediction_evidence_table(results)
    audit = audit_classification_metrics(results)
    targeted_gan = {
        exp_id: row.get("targeted_gan")
        for exp_id, row in results.items()
        if row.get("gan") and row.get("targeted_gan") is not None
    }

    experiment_data = _build_experiment_data(sample, train_idx, val_idx, test_idx, cfg, split_sources)
    data_quality_source = [] if isinstance(prepared, DatasetSplits) else prepared
    data_quality = _assemble_data_quality(data_quality_source, sample, train_idx, val_idx, test_idx, cfg)
    reproducibility = _build_reproducibility(sample, train_idx, val_idx, test_idx, cfg, experiment_data)

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
            "n_test": int(len(test_idx)),
            "seed": cfg.seed,
            "dataset_source": "PANDORA",
            "held_out_fold": "test",
            "split": experiment_data["split"],
            "split_sources": split_sources,
        },
        "experiment_data": experiment_data,
        "data_quality": data_quality,
        "reproducibility": reproducibility,
        "imbalance": imbalance,
        "qlearning_efficiency": qlearning_efficiency,
        "trait_label_thresholds": {
            "source": "train_split_median_per_trait",
            "thresholds": trait_label_thresholds,
            "notes": [
                "These cutoffs convert continuous OCEAN labels into binary Low/High labels.",
                "They are learned only from the train split, then reused unchanged for validation and test.",
                "Model decision thresholds are still selected on validation predictions and applied once to test.",
            ],
        },
        "targeted_gan": targeted_gan,
        "results": results,
        "comparison": comparison,
        "presentation_metrics": presentation_metrics,
        "threshold_sweeps": threshold_sweeps,
        "prediction_evidence": prediction_evidence,
        "factor_effects": effects,
        "model_comparison": effects.get("model_comparison"),
        "findings": findings,
        "audit": audit,
    }
    bundle["interpretation"] = _attach_interpretation(bundle)
    research_questions = (bundle.get("interpretation") or {}).get("research_questions")
    bundle["research_evidence"] = {
        "kind": "research_evidence",
        "quality": (bundle.get("interpretation") or {}).get("quality"),
        "effects": (bundle.get("interpretation") or {}).get("effects"),
        "evidence": (bundle.get("interpretation") or {}).get("evidence"),
        "research_questions": research_questions,
    }
    try:
        from backend.ml_pipeline.experiments.research_contract import (
            format_report,
            validate_research_contract,
        )

        contract = validate_research_contract(bundle, include_source_checks=True)
        bundle["research_contract"] = contract
        if contract.get("status") == "FAIL":
            logger.warning("Research contract FAIL (does not abort training):\n%s", format_report(contract))
        elif contract.get("status") == "WARNING":
            logger.info("Research contract WARNING:\n%s", format_report(contract))
    except Exception as exc:
        logger.exception("Research contract validation failed to run: %s", exc)
        bundle["research_contract"] = {
            "kind": "research_contract_error",
            "error": str(exc),
            "stops_execution": False,
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


def _winner_lower(a: Optional[float], b: Optional[float], name_a: str, name_b: str,
                  eps: float = 1e-4) -> str:
    if a is None or b is None:
        return "n/a"
    if abs(a - b) < eps:
        return "tie"
    return name_a if a < b else name_b


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a - b)


def _fmt_metric(value: Optional[float], *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def comparison_table(results: Dict[str, Any]):
    """Return test-first headline metrics for the eight experiment conditions."""
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
            "mean_comments_selected": r.get("mean_comments_selected"),
            "training_seconds": r.get("training_seconds"),
            "test_mae": o.get("mae", o.get("val_mae")),
            "test_rmse": o.get("rmse", o.get("val_rmse")),
            "test_r2": o.get("r2", o.get("val_r2")),
            "test_pearson": o.get("val_pearson"),
            "test_accuracy": o.get("accuracy"),
            "test_f1": o.get("f1", o.get("macro_f1")),
            "test_specificity": o.get("specificity"),
            "test_precision": o.get("precision", o.get("macro_precision")),
            "test_recall": o.get("recall", o.get("macro_recall")),
        })
    return pd.DataFrame(rows)


def presentation_metric_table(results: Dict[str, Any]):
    """Long-form headline metrics for presentation CSVs."""
    import pandas as pd

    rows = []
    for exp_id, row in results.items():
        overall = row.get("overall") or {}
        metrics = {
            "test_mae": overall.get("mae", overall.get("val_mae")),
            "test_rmse": overall.get("rmse", overall.get("val_rmse")),
            "test_r2": overall.get("r2", overall.get("val_r2")),
            "test_pearson": overall.get("val_pearson"),
            "test_accuracy": overall.get("accuracy"),
            "test_f1": overall.get("f1", overall.get("macro_f1")),
            "test_specificity": overall.get("specificity"),
            "test_precision": overall.get("precision", overall.get("macro_precision")),
            "test_recall": overall.get("recall", overall.get("macro_recall")),
        }
        for metric, value in metrics.items():
            rows.append({
                "condition": exp_id,
                "description": row.get("label"),
                "selection": row.get("selection"),
                "gan": row.get("gan"),
                "model": row.get("model"),
                "metric": metric,
                "value": value,
            })
    return pd.DataFrame(rows)


def threshold_sweep_table(results: Dict[str, Any]):
    """Flatten per-trait threshold sweeps when they exist."""
    import pandas as pd

    rows = []
    for exp_id, row in results.items():
        validation = row.get("validation") or {}
        per_trait = validation.get("per_trait") or row.get("per_trait") or {}
        for trait, block in per_trait.items():
            sweep = (block or {}).get("threshold_sweep") or {}
            if not sweep:
                continue
            if isinstance(sweep, list):
                for item in sweep:
                    rows.append({
                        "condition": exp_id,
                        "split": "validation",
                        "trait": trait,
                        "threshold": item.get("threshold"),
                        "accuracy": item.get("accuracy"),
                        "f1_score": item.get("f1_score"),
                        "specificity": item.get("specificity"),
                        "precision": item.get("precision"),
                        "recall": item.get("recall"),
                        "selection_score": item.get("selection_score"),
                        "selection_policy": item.get("selection_policy"),
                        "selected": item.get("threshold") == block.get("best_threshold"),
                    })
            else:
                rows.append({
                    "condition": exp_id,
                    "split": "validation",
                    "trait": trait,
                    "threshold": sweep.get("best_threshold"),
                    "accuracy": (block or {}).get("accuracy"),
                    "f1_score": sweep.get("best_f1"),
                    "specificity": sweep.get("specificity"),
                    "precision": (block or {}).get("macro_precision"),
                    "recall": (block or {}).get("macro_recall"),
                    "selection_score": sweep.get("selection_score"),
                    "selection_policy": sweep.get("selection_policy"),
                    "selected": True,
                })
    return pd.DataFrame(rows)


def model_comparison(results: Dict[str, Any]):
    """
    Head-to-head Lasso vs LSTM at each matched cell using test regression first.
    """
    import pandas as pd

    rows = []
    for sel, gan in _CELLS:
        la = _find(results, "lasso", sel, gan)
        ls = _find(results, "lstm", sel, gan)
        if not la or not ls:
            continue
        lo, so = la["overall"], ls["overall"]
        la_mae, ls_mae = lo.get("mae", lo.get("val_mae")), so.get("mae", so.get("val_mae"))
        la_rmse, ls_rmse = lo.get("rmse", lo.get("val_rmse")), so.get("rmse", so.get("val_rmse"))
        la_r2, ls_r2 = lo.get("r2", lo.get("val_r2")), so.get("r2", so.get("val_r2"))
        la_pearson, ls_pearson = lo.get("val_pearson"), so.get("val_pearson")
        la_acc, ls_acc = lo.get("accuracy"), so.get("accuracy")
        la_f1, ls_f1 = lo.get("f1", lo.get("macro_f1")), so.get("f1", so.get("macro_f1"))
        rows.append({
            "selection": sel,
            "gan": gan,
            "lasso_test_mae": la_mae,
            "lstm_test_mae": ls_mae,
            "mae_winner": _winner_lower(la_mae, ls_mae, "Lasso", "LSTM"),
            "lasso_test_rmse": la_rmse,
            "lstm_test_rmse": ls_rmse,
            "rmse_winner": _winner_lower(la_rmse, ls_rmse, "Lasso", "LSTM"),
            "lasso_test_r2": la_r2,
            "lstm_test_r2": ls_r2,
            "r2_winner": _winner(la_r2, ls_r2, "Lasso", "LSTM"),
            "lasso_test_pearson": la_pearson,
            "lstm_test_pearson": ls_pearson,
            "pearson_winner": _winner(la_pearson, ls_pearson, "Lasso", "LSTM"),
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
        configured_thresholds = result.get("validation", {}).get("candidate_thresholds", me.CANDIDATE_THRESHOLDS)
        expected_thresholds = (
            [float(t) for t in configured_thresholds]
            if isinstance(configured_thresholds, (list, tuple))
            else None
        )
        evidence = result.get("prediction_evidence", {})
        for split in ("validation", "test"):
            split_rows = evidence.get(split, [])
            split_result = result.get(split, result if split == "validation" else {})
            per_trait = split_result.get("per_trait") or {}
            if not per_trait:
                continue
            for trait in TRAIT_KEYS:
                if trait not in per_trait:
                    continue
                trait_rows = [r for r in split_rows if r["trait"] == trait]
                recomputed = _recompute_binary_metrics_from_evidence(trait_rows) if trait_rows else None
                stored = per_trait[trait]
                audit_row = {
                    "condition": exp_id,
                    "split": split,
                    "trait": trait,
                    "stored_accuracy": stored.get("accuracy"),
                    "recomputed_accuracy": recomputed["accuracy"] if recomputed else None,
                    "stored_f1": stored.get("f1", stored.get("macro_f1")),
                    "recomputed_f1": recomputed["f1"] if recomputed else None,
                    "stored_specificity": stored.get("specificity"),
                    "recomputed_specificity": recomputed["specificity"] if recomputed else None,
                    "tp": recomputed["tp"] if recomputed else None,
                    "fp": recomputed["fp"] if recomputed else None,
                    "tn": recomputed["tn"] if recomputed else None,
                    "fn": recomputed["fn"] if recomputed else None,
                }
                rows.append(audit_row)
                if recomputed is not None:
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

                sweep = stored.get("threshold_sweep", [])
                if isinstance(sweep, list) and expected_thresholds is not None:
                    thresholds = [round(float(item.get("threshold")), 2) for item in sweep]
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
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import pandas as pd

    plot_dir = (out / "plots").resolve()
    plot_dir.mkdir(parents=True, exist_ok=True)

    comparison = comparison_table(results)
    if len(comparison):
        x = np.arange(len(comparison))
        labels = comparison["condition"].tolist()
        plot_metrics = [
            ("test_mae" if "test_mae" in comparison else "val_mae", "test_mae"),
            ("test_rmse" if "test_rmse" in comparison else "val_rmse", "test_rmse"),
            ("test_r2" if "test_r2" in comparison else "val_r2", "test_r2"),
            ("test_pearson" if "test_pearson" in comparison else "val_pearson", "test_pearson"),
            ("test_accuracy" if "test_accuracy" in comparison else "accuracy", "accuracy"),
            ("test_f1" if "test_f1" in comparison else "macro_f1", "f1"),
            ("test_specificity" if "test_specificity" in comparison else "specificity", "specificity"),
            ("test_precision" if "test_precision" in comparison else "macro_precision", "precision"),
            ("test_recall" if "test_recall" in comparison else "macro_recall", "recall"),
        ]
        for metric, label in plot_metrics:
            if metric not in comparison:
                continue
            fig, ax = plt.subplots(figsize=(12, 5))
            colors = ["#4C78A8" if model == "lasso" else "#F58518" for model in comparison["model"]]
            ax.bar(x, comparison[metric].astype(float), color=colors)
            ax.set_title(label.replace("_", " ").title())
            ax.set_ylabel("Error" if label in {"test_mae", "test_rmse"} else "Score")
            if label not in {"test_r2", "test_pearson"}:
                ax.set_ylim(0, 1)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            _save_plot_file(fig, plot_dir / f"{label}_by_condition.png")
            plt.close(fig)

    thresholds = threshold_sweep_table(results)
    if len(thresholds):
        test_thresholds = thresholds[thresholds["split"] == "test"]
        threshold_split = "test"
        if not len(test_thresholds):
            test_thresholds = thresholds[thresholds["split"] == "validation"]
            threshold_split = "validation"
        top_threshold_rows = []
        for (condition, trait), trait_df in test_thresholds.groupby(["condition", "trait"]):
            trait_df = trait_df.sort_values("threshold").reset_index(drop=True)
            score_col = "selection_score" if "selection_score" in trait_df else "f1_score"
            top_df = (
                trait_df
                .assign(_rank_score=trait_df[score_col].astype(float))
                .sort_values(["_rank_score", "f1_score", "specificity"], ascending=False)
                .head(5)
                .sort_values("threshold")
                .reset_index(drop=True)
            )
            top_df["display_rank"] = np.arange(1, len(top_df) + 1)
            top_threshold_rows.append(top_df)
        display_thresholds = (
            pd.concat(top_threshold_rows, ignore_index=True)
            if top_threshold_rows else test_thresholds
        )
        for metric in THRESHOLD_PLOT_METRICS:
            if metric not in display_thresholds:
                continue
            grouped = (
                display_thresholds
                .groupby(["condition", "display_rank"], as_index=False)[metric]
                .mean()
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            for condition, group in grouped.groupby("condition"):
                ax.plot(
                    group["display_rank"].astype(int),
                    group[metric].astype(float),
                    marker="o",
                    linewidth=1.6,
                    label=condition,
                )
            ax.set_title(f"{threshold_split.title()} Top 5 Threshold Candidates - {metric.replace('_', ' ').title()}")
            ax.set_xlabel("Top threshold candidate")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1)
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
            fig.tight_layout(rect=(0, 0.08, 1, 1))
            _save_plot_file(fig, plot_dir / f"threshold_sweep_{threshold_split}_{metric}.png")
            plt.close(fig)

        for condition, condition_df in test_thresholds.groupby("condition"):
            fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=True)
            axes_flat = axes.flatten()
            for axis, (trait_label, trait_aliases) in zip(axes_flat, TRAIT_DISPLAY_ALIASES):
                trait_df = (
                    condition_df[condition_df["trait"].astype(str).isin(trait_aliases)]
                    .sort_values("threshold")
                    .reset_index(drop=True)
                )
                if trait_df.empty:
                    axis.set_title(trait_label)
                    axis.grid(alpha=0.25)
                    continue
                score_col = "selection_score" if "selection_score" in trait_df else "f1_score"
                top_trait_df = (
                    trait_df
                    .assign(_rank_score=trait_df[score_col].astype(float))
                    .sort_values(["_rank_score", "f1_score", "specificity"], ascending=False)
                    .head(5)
                    .sort_values("threshold")
                    .reset_index(drop=True)
                )
                trait_df = top_trait_df
                x_values = np.arange(1, len(trait_df) + 1)
                for metric in ("accuracy", "f1_score", "specificity"):
                    axis.plot(
                        x_values,
                        trait_df[metric].astype(float),
                        marker="o",
                        label=metric,
                    )
                selected_rows = trait_df[trait_df.get("selected", False).astype(bool)] if "selected" in trait_df else []
                if len(selected_rows):
                    selected_idx = int(selected_rows.index[0])
                    selected_x = selected_idx + 1
                    selected_threshold = float(trait_df.loc[selected_idx, "threshold"])
                    axis.axvline(selected_x, color="#222222", linestyle="--", linewidth=1.0, alpha=0.55)
                    y_anchor = float(trait_df.loc[selected_idx, "f1_score"]) if "f1_score" in trait_df else 0.5
                    axis.annotate(
                        f"{selected_threshold:.3f}",
                        xy=(selected_x, y_anchor),
                        xytext=(4, 6),
                        textcoords="offset points",
                        fontsize=7,
                        color="#222222",
                    )
                axis.set_title(trait_label)
                axis.set_xlabel("Top threshold candidate")
                axis.grid(alpha=0.25)
            axes_flat[-1].axis("off")
            handles, legend_labels = axes_flat[0].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc="lower center", ncol=3)
            fig.suptitle(f"{condition} - Top 5 Validation-Derived Thresholds by Trait", y=0.98)
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
            bo, qo = base["overall"], ql["overall"]
            q_rows.append({
                "model": model,
                "gan": gan,
                "baseline_test_mae": bo.get("mae", bo.get("val_mae")),
                "qlearning_test_mae": qo.get("mae", qo.get("val_mae")),
                "delta_test_mae": _delta(qo.get("mae", qo.get("val_mae")), bo.get("mae", bo.get("val_mae"))),
                "baseline_test_rmse": bo.get("rmse", bo.get("val_rmse")),
                "qlearning_test_rmse": qo.get("rmse", qo.get("val_rmse")),
                "delta_test_rmse": _delta(qo.get("rmse", qo.get("val_rmse")), bo.get("rmse", bo.get("val_rmse"))),
                "baseline_test_r2": bo.get("r2", bo.get("val_r2")),
                "qlearning_test_r2": qo.get("r2", qo.get("val_r2")),
                "delta_test_r2": _delta(qo.get("r2", qo.get("val_r2")), bo.get("r2", bo.get("val_r2"))),
                "baseline_test_pearson": bo.get("val_pearson"),
                "qlearning_test_pearson": qo.get("val_pearson"),
                "delta_test_pearson": _delta(qo.get("val_pearson"), bo.get("val_pearson")),
                "acc_baseline": bo.get("accuracy"),
                "acc_qlearning": qo.get("accuracy"),
                "delta_accuracy": _delta(qo.get("accuracy"), bo.get("accuracy")),
                "delta_macro_f1": _delta(qo.get("macro_f1"), bo.get("macro_f1")),
            })

    g_rows = []
    for model in ("lasso", "lstm"):
        for sel in ("baseline", "qlearning"):
            nog = _find(results, model, sel, False)
            gon = _find(results, model, sel, True)
            if not nog or not gon:
                continue
            no, go = nog["overall"], gon["overall"]
            g_rows.append({
                "model": model,
                "selection": sel,
                "no_gan_test_mae": no.get("mae", no.get("val_mae")),
                "gan_test_mae": go.get("mae", go.get("val_mae")),
                "delta_test_mae": _delta(go.get("mae", go.get("val_mae")), no.get("mae", no.get("val_mae"))),
                "no_gan_test_rmse": no.get("rmse", no.get("val_rmse")),
                "gan_test_rmse": go.get("rmse", go.get("val_rmse")),
                "delta_test_rmse": _delta(go.get("rmse", go.get("val_rmse")), no.get("rmse", no.get("val_rmse"))),
                "no_gan_test_r2": no.get("r2", no.get("val_r2")),
                "gan_test_r2": go.get("r2", go.get("val_r2")),
                "delta_test_r2": _delta(go.get("r2", go.get("val_r2")), no.get("r2", no.get("val_r2"))),
                "no_gan_test_pearson": no.get("val_pearson"),
                "gan_test_pearson": go.get("val_pearson"),
                "delta_test_pearson": _delta(go.get("val_pearson"), no.get("val_pearson")),
                "acc_no_gan": no.get("accuracy"),
                "acc_gan": go.get("accuracy"),
                "delta_accuracy": _delta(go.get("accuracy"), no.get("accuracy")),
                "delta_macro_f1": _delta(go.get("macro_f1"), no.get("macro_f1")),
            })

    model_rows = []
    for sel, gan in _CELLS:
        la = _find(results, "lasso", sel, gan)
        ls = _find(results, "lstm", sel, gan)
        if not la or not ls:
            continue
        lo, so = la["overall"], ls["overall"]
        model_rows.append({
            "selection": sel,
            "gan": gan,
            "delta_lstm_minus_lasso_test_mae": _delta(
                so.get("mae", so.get("val_mae")),
                lo.get("mae", lo.get("val_mae")),
            ),
            "delta_lstm_minus_lasso_test_rmse": _delta(
                so.get("rmse", so.get("val_rmse")),
                lo.get("rmse", lo.get("val_rmse")),
            ),
            "delta_lstm_minus_lasso_test_r2": _delta(
                so.get("r2", so.get("val_r2")),
                lo.get("r2", lo.get("val_r2")),
            ),
            "delta_lstm_minus_lasso_accuracy": _delta(so.get("accuracy"), lo.get("accuracy")),
            "delta_lstm_minus_lasso_macro_f1": _delta(so.get("macro_f1"), lo.get("macro_f1")),
        })

    interaction_rows = []
    for model in ("lasso", "lstm"):
        base_no = _find(results, model, "baseline", False)
        base_gan = _find(results, model, "baseline", True)
        q_no = _find(results, model, "qlearning", False)
        q_gan = _find(results, model, "qlearning", True)
        if base_no and base_gan and q_no and q_gan:
            interaction_rows.append({
                "interaction": "selection_x_gan",
                "model": model,
                "delta_test_mae": _delta(
                    _delta(q_gan["overall"].get("mae", q_gan["overall"].get("val_mae")), base_gan["overall"].get("mae", base_gan["overall"].get("val_mae"))),
                    _delta(q_no["overall"].get("mae", q_no["overall"].get("val_mae")), base_no["overall"].get("mae", base_no["overall"].get("val_mae"))),
                ),
                "delta_accuracy": _delta(
                    _delta(q_gan["overall"].get("accuracy"), base_gan["overall"].get("accuracy")),
                    _delta(q_no["overall"].get("accuracy"), base_no["overall"].get("accuracy")),
                ),
                "delta_macro_f1": _delta(
                    _delta(q_gan["overall"].get("macro_f1"), base_gan["overall"].get("macro_f1")),
                    _delta(q_no["overall"].get("macro_f1"), base_no["overall"].get("macro_f1")),
                ),
            })
    for gan in (False, True):
        base_lasso = _find(results, "lasso", "baseline", gan)
        q_lasso = _find(results, "lasso", "qlearning", gan)
        base_lstm = _find(results, "lstm", "baseline", gan)
        q_lstm = _find(results, "lstm", "qlearning", gan)
        if base_lasso and q_lasso and base_lstm and q_lstm:
            interaction_rows.append({
                "interaction": "selection_x_model",
                "gan": gan,
                "delta_test_mae": _delta(
                    _delta(q_lstm["overall"].get("mae", q_lstm["overall"].get("val_mae")), base_lstm["overall"].get("mae", base_lstm["overall"].get("val_mae"))),
                    _delta(q_lasso["overall"].get("mae", q_lasso["overall"].get("val_mae")), base_lasso["overall"].get("mae", base_lasso["overall"].get("val_mae"))),
                ),
                "delta_accuracy": _delta(
                    _delta(q_lstm["overall"].get("accuracy"), base_lstm["overall"].get("accuracy")),
                    _delta(q_lasso["overall"].get("accuracy"), base_lasso["overall"].get("accuracy")),
                ),
                "delta_macro_f1": _delta(
                    _delta(q_lstm["overall"].get("macro_f1"), base_lstm["overall"].get("macro_f1")),
                    _delta(q_lasso["overall"].get("macro_f1"), base_lasso["overall"].get("macro_f1")),
                ),
            })

    return {
        "qlearning_effect": pd.DataFrame(q_rows),
        "gan_effect": pd.DataFrame(g_rows),
        "model_effect": pd.DataFrame(model_rows),
        "interaction_effect": pd.DataFrame(interaction_rows),
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
    JSON-safe headline claims for the report.

    Continuous OCEAN regression is primary, so the best condition is chosen by
    lowest held-out test MAE. Binary High/Low classification remains secondary.
    """
    def _model_mean(model: str, metric: str) -> Optional[float]:
        vals = [
            r["overall"].get(metric)
            for r in results.values()
            if r["model"] == model and r["overall"].get(metric) is not None
        ]
        return float(np.mean(vals)) if vals else None

    def _mean_delta(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        vals = [row[key] for row in rows if row.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    scored = {
        k: r for k, r in results.items()
        if r["overall"].get("mae", r["overall"].get("val_mae")) is not None
    }
    best_id = min(
        scored,
        key=lambda k: scored[k]["overall"].get("mae", scored[k]["overall"].get("val_mae")),
    ) if scored else None
    best = None
    notes: List[str] = []
    if best_id:
        r = results[best_id]
        o = r["overall"]
        best = {
            "condition": best_id,
            "label": r["label"],
            "primary_metric": "test_mae",
            "test_mae": o.get("mae", o.get("val_mae")),
            "test_rmse": o.get("rmse", o.get("val_rmse")),
            "test_r2": o.get("r2", o.get("val_r2")),
            "test_pearson": o.get("val_pearson"),
            "test_accuracy": o.get("accuracy"),
            "test_f1": o.get("f1", o.get("macro_f1")),
        }

    lasso_mae, lstm_mae = _model_mean("lasso", "mae"), _model_mean("lstm", "mae")
    lasso_rmse, lstm_rmse = _model_mean("lasso", "rmse"), _model_mean("lstm", "rmse")
    lasso_r2, lstm_r2 = _model_mean("lasso", "r2"), _model_mean("lstm", "r2")
    lasso_pearson = _model_mean("lasso", "val_pearson")
    lstm_pearson = _model_mean("lstm", "val_pearson")
    lasso_acc, lstm_acc = _model_mean("lasso", "accuracy"), _model_mean("lstm", "accuracy")
    lasso_f1, lstm_f1 = _model_mean("lasso", "f1"), _model_mean("lstm", "f1")

    effects = factor_effects(results)
    q_df, g_df = effects["qlearning_effect"], effects["gan_effect"]
    q_rows = q_df.to_dict("records") if hasattr(q_df, "to_dict") else []
    g_rows = g_df.to_dict("records") if hasattr(g_df, "to_dict") else []
    q_mae = _mean_delta(q_rows, "delta_test_mae")
    g_mae = _mean_delta(g_rows, "delta_test_mae")
    q_acc, q_f1 = _mean_delta(q_rows, "delta_accuracy"), _mean_delta(q_rows, "delta_macro_f1")
    g_acc, g_f1 = _mean_delta(g_rows, "delta_accuracy"), _mean_delta(g_rows, "delta_macro_f1")

    better_by_mae = _winner_lower(lasso_mae, lstm_mae, "Lasso", "LSTM")
    better_by_rmse = _winner_lower(lasso_rmse, lstm_rmse, "Lasso", "LSTM")
    better_by_r2 = _winner(lasso_r2, lstm_r2, "Lasso", "LSTM")
    better_by_pearson = _winner(lasso_pearson, lstm_pearson, "Lasso", "LSTM")
    better_by_acc = _winner(lasso_acc, lstm_acc, "Lasso", "LSTM")
    better_by_f1 = _winner(lasso_f1, lstm_f1, "Lasso", "LSTM")

    # Human-readable claim strings (guarded against None).
    notes: List[str] = []
    if best is not None:
        notes.append(
            f"Best condition: {best['condition']} ({best['label']}) - "
            f"lowest test MAE {_fmt_metric(best['test_mae'])}, RMSE {_fmt_metric(best['test_rmse'])}; "
            f"secondary High/Low F1 {_fmt_metric(best['test_f1'])}."
        )
    if lasso_mae is not None and lstm_mae is not None:
        notes.append(
            f"Model comparison (mean over the 4 matched cells): "
            f"Lasso test MAE {_fmt_metric(lasso_mae)} vs LSTM {_fmt_metric(lstm_mae)} -> {better_by_mae} wins on primary MAE; "
            f"Lasso RMSE {_fmt_metric(lasso_rmse)} vs LSTM {_fmt_metric(lstm_rmse)} -> {better_by_rmse} wins on RMSE."
        )
    if q_acc is not None:
        verdict = "helps" if q_mae is not None and q_mae < 0 else ("hurts" if q_mae is not None and q_mae > 0 else "is neutral")
        notes.append(
            f"Q-learning selection {verdict} on average for primary regression: "
            f"mean delta test MAE {_fmt_metric(q_mae, signed=True)}; secondary delta accuracy {_fmt_metric(q_acc, signed=True)}, "
            f"delta macro-F1 {_fmt_metric(q_f1, signed=True)}."
        )
    if g_acc is not None:
        verdict = "helps" if g_mae is not None and g_mae < 0 else ("hurts" if g_mae is not None and g_mae > 0 else "is neutral")
        notes.append(
            f"GAN augmentation {verdict} on average for primary regression: "
            f"mean delta test MAE {_fmt_metric(g_mae, signed=True)}; secondary delta accuracy {_fmt_metric(g_acc, signed=True)}, "
            f"delta macro-F1 {_fmt_metric(g_f1, signed=True)}."
        )

    return {
        "best_condition": best,
        "model_means": {
            "lasso": {
                "test_mae": lasso_mae,
                "test_rmse": lasso_rmse,
                "test_r2": lasso_r2,
                "test_pearson": lasso_pearson,
                "test_accuracy": lasso_acc,
                "test_f1": lasso_f1,
            },
            "lstm": {
                "test_mae": lstm_mae,
                "test_rmse": lstm_rmse,
                "test_r2": lstm_r2,
                "test_pearson": lstm_pearson,
                "test_accuracy": lstm_acc,
                "test_f1": lstm_f1,
            },
        },
        "better_model": {
            "by_test_mae": better_by_mae,
            "by_test_rmse": better_by_rmse,
            "by_test_r2": better_by_r2,
            "by_test_pearson": better_by_pearson,
            "by_accuracy": better_by_acc,
            "by_macro_f1": better_by_f1,
        },
        "qlearning_effect_mean": {
            "delta_test_mae": q_mae,
            "delta_accuracy": q_acc,
            "delta_macro_f1": q_f1,
        },
        "gan_effect_mean": {
            "delta_test_mae": g_mae,
            "delta_accuracy": g_acc,
            "delta_macro_f1": g_f1,
        },
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
    effects["model_effect"].to_csv(out / "model_effect.csv", index=False)
    effects["interaction_effect"].to_csv(out / "interaction_effect.csv", index=False)
    effects["model_comparison"].to_csv(out / "model_comparison.csv", index=False)

    with (out / "findings.json").open("w", encoding="utf-8") as fh:
        json.dump(bundle["findings"], fh, ensure_ascii=False, indent=2, default=_json_default)
    with (out / "classification_audit.json").open("w", encoding="utf-8") as fh:
        json.dump(bundle["audit"], fh, ensure_ascii=False, indent=2, default=_json_default)
    for name, key in (
        ("experiment_data.json", "experiment_data"),
        ("data_quality.json", "data_quality"),
        ("reproducibility.json", "reproducibility"),
        ("interpretation.json", "interpretation"),
        ("research_evidence.json", "research_evidence"),
        ("research_contract.json", "research_contract"),
        ("imbalance_report.json", "imbalance"),
        ("qlearning_efficiency.json", "qlearning_efficiency"),
        ("trait_label_thresholds.json", "trait_label_thresholds"),
        ("targeted_gan_report.json", "targeted_gan"),
    ):
        if bundle.get(key) is not None:
            with (out / name).open("w", encoding="utf-8") as fh:
                json.dump(bundle[key], fh, ensure_ascii=False, indent=2, default=_json_default)
    questions = (bundle.get("interpretation") or {}).get("research_questions")
    if questions:
        with (out / "research_questions.json").open("w", encoding="utf-8") as fh:
            json.dump(questions, fh, ensure_ascii=False, indent=2, default=_json_default)
    thesis = (bundle.get("interpretation") or {}).get("research_summary", {}).get("thesis_rows")
    if thesis:
        import pandas as pd

        pd.DataFrame(thesis).to_csv(out / "interpretation_thesis.csv", index=False)
    dq_rows = (bundle.get("data_quality") or {}).get("thesis_rows") or thesis_rows(bundle.get("data_quality") or {})
    if dq_rows:
        import pandas as pd

        pd.DataFrame(dq_rows).to_csv(out / "data_quality_thesis.csv", index=False)
    manifest = {
        "presentation_order": [
            "comparison.csv",
            "presentation_metrics_long.csv",
            "threshold_sweeps_long.csv",
            "prediction_evidence.csv",
            "classification_audit.json",
            "qlearning_effect.csv",
            "qlearning_efficiency.json",
            "gan_effect.csv",
            "model_effect.csv",
            "interaction_effect.csv",
            "model_comparison.csv",
            "experiment_data.json",
            "data_quality.json",
            "interpretation.json",
            "research_evidence.json",
            "research_questions.json",
            "research_contract.json",
            "plots/",
        ],
        "run_folder": out.name,
        "run_created_at": datetime.now().isoformat(timespec="seconds"),
        "metric_policy": {
            "official_test_metrics": "Use validation-selected thresholds only.",
            "candidate_thresholds": "validation_score_percentiles_per_trait",
            "threshold_selection": "max_harmonic_mean_f1_specificity",
            "ground_truth_cutoff": "train_split_median_per_trait",
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
            "model_comparison",
        }
    }
    with (out / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=_json_default)

    save_presentation_plots(bundle["results"], out)

    for exp_id, result in bundle["results"].items():
        exp_dir = out / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        with (exp_dir / "metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2, default=_json_default)
        if result.get("interpretation"):
            with (exp_dir / "interpretation.json").open("w", encoding="utf-8") as fh:
                json.dump(result["interpretation"], fh, ensure_ascii=False, indent=2, default=_json_default)
        model = models[exp_id]
        if result["model"] == "lasso":
            with (exp_dir / "lasso_state.json").open("w", encoding="utf-8") as fh:
                json.dump(model.save_state(), fh, ensure_ascii=False, indent=2)
        else:
            lstm_state = {
                "state_dict": model.model.state_dict() if getattr(model, "model", None) is not None else None,
                "hidden_dim": getattr(model, "hidden_dim", None),
                "num_layers": getattr(model, "num_layers", None),
                "dropout": getattr(model, "dropout", None),
                "learning_rate": getattr(model, "learning_rate", None),
                "trait_keys": list(TRAIT_KEYS),
            }
            torch.save(lstm_state, exp_dir / "lstm_state.pt")

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
        quality = load_ingestion_quality(prepared_path)
        cached_source = str((quality or {}).get("source") or "")
        if cached_source and Path(cached_source).resolve() != Path(pandora_file).resolve():
            logger.info(
                "Prepared PANDORA cache source changed from %s to %s; rebuilding.",
                cached_source,
                pandora_file,
            )
        elif cached_source:
            logger.info("Loading cached prepared PANDORA data from %s.", prepared_path)
            try:
                return load_prepared_cache(prepared_path)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Prepared PANDORA cache is not valid JSON (%s); rebuilding it.",
                    exc,
                )
        elif not quality_sidecar_path(prepared_path).exists():
            logger.info("Prepared PANDORA cache has no source sidecar; rebuilding it.")
        else:
            logger.info("Loading cached prepared PANDORA data from %s.", prepared_path)
            try:
                return load_prepared_cache(prepared_path)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Prepared PANDORA cache is not valid JSON (%s); rebuilding it.",
                    exc,
                )
    return load_pandora_comments(
        pandora_file,
        output_path=prepared_path,
        min_text_length=min_text_length,
        group_by=group_by,  # type: ignore[arg-type]
    )


def _default_pandora_files_by_split() -> Dict[str, List[Path]]:
    root = Path("PANDORA")
    legacy = root / "pandora-big5" / "data"
    return {
        "train": sorted(root.glob("Training dataset*.xlsx")) or sorted(legacy.glob("train-*.parquet")),
        "validation": sorted(root.glob("Validation dataset*.xlsx")) or sorted(legacy.glob("validation-*.parquet")),
        "test": sorted(root.glob("Test dataset*.xlsx")) or sorted(legacy.glob("test-*.parquet")),
    }


def load_file_defined_splits(
    *,
    work_dir: str | Path = "pandora_personality",
    refresh_prepared: bool = False,
    cfg: Optional[ExperimentConfig] = None,
) -> DatasetSplits:
    """Load the repository's train/validation/test Excel or parquet files as true splits."""
    files = _default_pandora_files_by_split()
    missing = [split for split, paths in files.items() if not paths]
    if missing:
        raise FileNotFoundError(f"Missing PANDORA dataset split(s): {', '.join(missing)}")

    work = Path(work_dir)
    data_dir = work / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg or ExperimentConfig()

    split_samples = {}
    for split, paths in files.items():
        prepared_all: List[PreparedUserComments] = []
        for part_no, path in enumerate(paths):
            suffix = f"_{part_no:02d}" if len(paths) > 1 else ""
            prepared_all.extend(load_or_prepare_pandora(
                path,
                data_dir / f"pandora_{split}{suffix}_prepared.json",
                refresh_prepared=refresh_prepared,
                group_by="author",
            ))
        if split == "train":
            split_sample_n = cfg.sample_n_users
        elif split == "validation":
            split_sample_n = max(1, round(cfg.sample_n_users * cfg.val_ratio))
        else:
            split_sample_n = max(1, round(cfg.sample_n_users * cfg.test_ratio))
        split_cfg = ExperimentConfig(
            sample_n_users=split_sample_n,
            min_comments_per_user=cfg.min_comments_per_user,
            seed=cfg.seed,
            top_k=cfg.top_k,
        )
        split_samples[split] = sample_users(prepared_all, split_cfg)

    out = combine_split_samples(
        split_samples["train"],
        split_samples["validation"],
        split_samples["test"],
    )
    out.sources = {split: ";".join(str(path) for path in paths) for split, paths in files.items()}
    return out


def _default_pandora_file() -> Optional[Path]:
    candidates = sorted(Path("PANDORA").glob("Training dataset*.xlsx"))
    if not candidates:
        candidates = sorted(Path("PANDORA").glob("**/train-*.parquet"))
    if not candidates:
        candidates = sorted(Path("PANDORA").glob("**/*.xlsx")) or sorted(Path("PANDORA").glob("**/*.parquet"))
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
        raise SystemExit("No PANDORA dataset file found. Pass --pandora-file path/to/file.xlsx.")

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
