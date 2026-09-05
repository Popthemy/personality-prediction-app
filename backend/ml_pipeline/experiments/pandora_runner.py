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

from backend.ml_pipeline.cleaning.cleaner import CleanedContent, ExtractedSignals
from backend.ml_pipeline.services import metrics_engine as me
from backend.ml_pipeline.services.bert_encoder import BERTEncoder
from backend.ml_pipeline.services.data.pandora import (
    PreparedUserComments,
    UserTraits,
    load_pandora_comments,
)
from backend.ml_pipeline.services.gan_augmenter import GANAugmenter
from backend.ml_pipeline.services.lasso_regressor import LassoTrainer
from backend.ml_pipeline.services.lstm_classifier import (
    LSTMTrainer,
    OCEAN_TRAITS,
    TRAIT_KEYS,
    set_seed,
)
from backend.ml_pipeline.services.qlearning_agent import QLearningAgent, run_training_loop

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

    val_ratio: float = 0.2
    test_ratio: float = 0.2

    synthetic_weight: float = 0.35
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
    return np.clip(unit, 0.0, 1.0), scale


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


def make_split(n_users: int, cfg: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create one participant-level train/validation/test split for all conditions."""
    if n_users < 3:
        raise ValueError("Need at least 3 users for train/validation/test.")
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
    logger.info("Split: train=%d val=%d test=%d.", len(train_idx), len(val_idx), len(test_idx))
    return train_idx, val_idx, test_idx


def _make_gan(embedding_dim: int, cfg: ExperimentConfig) -> GANAugmenter:
    """Build the paired embedding/OCEAN GAN for unit-scale labels."""
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
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: ExperimentConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit GAN on training users only and generate paired synthetic samples."""
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    if len(X_train) < 2:
        logger.warning("Skipping GAN: fewer than 2 training users.")
        return np.empty((0, X_train.shape[1]), dtype=np.float32), np.empty((0, 5), dtype=np.float32)
    gan = _make_gan(X_train.shape[1], cfg)
    gan.fit(X_train, ocean_scores=y_train)
    X_syn, y_syn, _metadata = gan.generate(n_samples=len(X_train))
    return np.asarray(X_syn, dtype=np.float32), np.asarray(y_syn, dtype=np.float32)


def _train_lasso_hint_model(
    features: Features,
    y_train: np.ndarray,
    train_idx: np.ndarray,
    cfg: ExperimentConfig,
) -> Tuple[LassoTrainer, np.ndarray, Dict[str, Any]]:
    """Fit Lasso on training users only and return five auxiliary hint columns."""
    trainer = LassoTrainer(
        alpha=cfg.lasso_alpha,
        max_iter=cfg.lasso_max_iter,
        regularization=cfg.lasso_regularization,
        l1_ratio=cfg.lasso_l1_ratio,
    )
    X_train = features.pooled[train_idx]
    X_train_scaled = trainer.fit_transform_features(X_train)

    train_hints = np.zeros_like(y_train, dtype=np.float32)
    trait_metrics: Dict[str, Any] = {}
    for i, trait in enumerate(OCEAN_TRAITS):
        trait_metrics[trait] = trainer.train_trait_model(X_train_scaled, y_train[:, i], trait)
        train_hints[:, i] = trainer.predict_trait(trait, X_train_scaled)

    sparse_counts = [
        metric["sparse_features"]
        for metric in trait_metrics.values()
        if "sparse_features" in metric
    ]
    total_counts = [
        metric["total_features"]
        for metric in trait_metrics.values()
        if "total_features" in metric
    ]
    total_features = int(total_counts[0]) if total_counts else int(X_train.shape[1])
    mean_sparse = float(np.mean(sparse_counts)) if sparse_counts else 0.0
    metadata = {
        "enabled": True,
        "source": "training-fold Lasso predictions on pooled BERT embeddings",
        "input": "mean-pooled selected-comment BERT embeddings",
        "output": "five normalized OCEAN hint features concatenated to the LSTM pooled state",
        "hint_shape": [int(train_hints.shape[0]), int(train_hints.shape[1])],
        "regularization": cfg.lasso_regularization,
        "alpha": float(trainer.alpha),
        "l1_ratio": float(cfg.lasso_l1_ratio),
        "mean_sparse_features": round(mean_sparse, 4),
        "total_features": total_features,
        "mean_sparsity": round(float(1.0 - (mean_sparse / max(total_features, 1))), 4),
        "trait_metrics": trait_metrics,
    }
    return trainer, train_hints, metadata


def _predict_lasso_hints(trainer: LassoTrainer, pooled_embeddings: np.ndarray) -> np.ndarray:
    """Predict five normalized OCEAN hints from pooled embeddings."""
    X = trainer.transform_features(np.asarray(pooled_embeddings, dtype=np.float32))
    cols = [trainer.predict_trait(trait, X) for trait in OCEAN_TRAITS]
    return np.stack(cols, axis=1).astype(np.float32)


def _condition_sequences(
    features: Features,
    y_train: np.ndarray,
    train_idx: np.ndarray,
    cfg: ExperimentConfig,
    use_gan: bool,
) -> Tuple[List[np.ndarray], np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Return real plus optional synthetic training sequences."""
    train_sequences = [features.sequences[i] for i in train_idx]
    if not use_gan:
        return train_sequences, y_train, None, None, None

    X_syn, y_syn = _paired_gan_samples(features.pooled[train_idx], y_train, cfg)
    if len(X_syn) == 0:
        return train_sequences, y_train, None, None, None

    synthetic_sequences = [row.reshape(1, -1).astype(np.float32) for row in X_syn]
    sample_weights = np.concatenate([
        np.ones(len(train_sequences), dtype=np.float32),
        np.full(len(synthetic_sequences), cfg.synthetic_weight, dtype=np.float32),
    ])
    return (
        train_sequences + synthetic_sequences,
        np.concatenate([y_train, y_syn], axis=0),
        sample_weights,
        X_syn,
        y_syn,
    )


def _condition_training_arrays(
    features: Features,
    y_train: np.ndarray,
    train_idx: np.ndarray,
    cfg: ExperimentConfig,
    use_gan: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Return real plus optional synthetic pooled training arrays."""
    X_train = features.pooled[train_idx]
    if not use_gan:
        return X_train, y_train, None, None, None

    X_syn, y_syn = _paired_gan_samples(X_train, y_train, cfg)
    if len(X_syn) == 0:
        return X_train, y_train, None, None, None

    sample_weights = np.concatenate([
        np.ones(len(X_train), dtype=np.float32),
        np.full(len(X_syn), cfg.synthetic_weight, dtype=np.float32),
    ])
    return (
        np.concatenate([X_train, X_syn], axis=0).astype(np.float32),
        np.concatenate([y_train, y_syn], axis=0).astype(np.float32),
        sample_weights,
        X_syn,
        y_syn,
    )


def _regression_summary(y_true: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
    """Per-trait and aggregate continuous-score metrics."""
    per_trait = {
        trait: me.compute_regression_metrics(y_true[:, i], predictions[:, i])
        for i, trait in enumerate(TRAIT_KEYS)
    }
    keys = ["mae", "mse", "rmse", "r2", "correlation"]
    aggregate = {
        key: round(float(np.mean([metrics[key] for metrics in per_trait.values()])), 4)
        for key in keys
    }
    return {"per_trait": per_trait, "aggregate": aggregate}


def _classification_evidence_rows(
    *,
    sample: Sample,
    exp_id: str,
    spec: Dict[str, Any],
    split: str,
    indices: np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    metrics: Dict[str, Any],
    threshold_field: str,
    ground_truth_cutoff: float,
) -> List[Dict[str, Any]]:
    """Return row-level evidence behind the binary metrics."""
    rows: List[Dict[str, Any]] = []
    for trait_i, trait in enumerate(TRAIT_KEYS):
        trait_metrics = metrics["per_trait"][trait]
        threshold = float(trait_metrics[threshold_field])
        y_binary = (np.asarray(y_true[:, trait_i], dtype=float) >= ground_truth_cutoff).astype(int)
        y_pred = (np.asarray(scores[:, trait_i], dtype=float) >= threshold).astype(int)
        for row_i, sample_i in enumerate(indices):
            truth = int(y_binary[row_i])
            pred = int(y_pred[row_i])
            if truth == 1 and pred == 1:
                outcome = "TP"
            elif truth == 0 and pred == 1:
                outcome = "FP"
            elif truth == 0 and pred == 0:
                outcome = "TN"
            else:
                outcome = "FN"
            rows.append({
                "condition": exp_id,
                "description": spec["label"],
                "split": split,
                "selection": spec["selection"],
                "gan": bool(spec["gan"]),
                "model": spec["model"],
                "user_id": sample.user_ids[int(sample_i)],
                "trait": trait,
                "ground_truth_score": round(float(y_true[row_i, trait_i]), 6),
                "ground_truth_cutoff": float(ground_truth_cutoff),
                "ground_truth_label": "High" if truth else "Low",
                "prediction_score": round(float(scores[row_i, trait_i]), 6),
                "decision_threshold": threshold,
                "predicted_label": "High" if pred else "Low",
                "outcome": outcome,
            })
    return rows


def _run_lasso_condition(
    sample: Sample,
    exp_id: str,
    cfg: ExperimentConfig,
    features: Features,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[Dict[str, Any], LassoTrainer]:
    """Train and evaluate one standalone Lasso final-model condition."""
    spec = EXPERIMENTS[exp_id]
    logger.info("=== %s: %s ===", exp_id, spec["label"])

    y = np.asarray(sample.labels_unit, dtype=np.float32)
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    X_fit, fit_y, sample_weights, X_syn, _y_syn = _condition_training_arrays(
        features,
        y_train,
        train_idx,
        cfg,
        use_gan=bool(spec["gan"]),
    )
    X_val = features.pooled[val_idx]
    X_test = features.pooled[test_idx]

    trainer = LassoTrainer(
        alpha=cfg.lasso_alpha,
        max_iter=cfg.lasso_max_iter,
        regularization=cfg.lasso_regularization,
        l1_ratio=cfg.lasso_l1_ratio,
    )
    X_fit_scaled = trainer.fit_transform_features(X_fit)
    X_val_scaled = trainer.transform_features(X_val)
    X_test_scaled = trainer.transform_features(X_test)

    trait_metrics: Dict[str, Any] = {}
    for i, trait in enumerate(OCEAN_TRAITS):
        trait_metrics[trait] = trainer.train_trait_model(
            X_fit_scaled,
            fit_y[:, i],
            trait,
            validate_X=X_val_scaled,
            validate_y=y_val[:, i],
            sample_weight=sample_weights,
        )

    val_scores = _predict_lasso_hints(trainer, X_val)
    test_scores = _predict_lasso_hints(trainer, X_test)
    validation = me.evaluate_lstm_binary_classifier(
        y_true=y_val,
        probabilities=val_scores,
        trait_names=list(TRAIT_KEYS),
        ground_truth_cutoff=cfg.ground_truth_cutoff,
        candidate_thresholds=list(cfg.candidate_thresholds),
    )
    test = me.evaluate_lstm_binary_with_thresholds(
        y_true=y_test,
        probabilities=test_scores,
        threshold_selection=validation,
        trait_names=list(TRAIT_KEYS),
        ground_truth_cutoff=cfg.ground_truth_cutoff,
        candidate_thresholds=list(cfg.candidate_thresholds),
    )
    regression = _regression_summary(y_test, test_scores)
    evidence = {
        "validation": _classification_evidence_rows(
            sample=sample,
            exp_id=exp_id,
            spec=spec,
            split="validation",
            indices=val_idx,
            y_true=y_val,
            scores=val_scores,
            metrics=validation,
            threshold_field="best_threshold",
            ground_truth_cutoff=cfg.ground_truth_cutoff,
        ),
        "test": _classification_evidence_rows(
            sample=sample,
            exp_id=exp_id,
            spec=spec,
            split="test",
            indices=test_idx,
            y_true=y_test,
            scores=test_scores,
            metrics=test,
            threshold_field="selected_threshold",
            ground_truth_cutoff=cfg.ground_truth_cutoff,
        ),
    }

    result = {
        "experiment": exp_id,
        "label": spec["label"],
        "model": "lasso",
        "selection": spec["selection"],
        "gan": bool(spec["gan"]),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_fit": int(len(X_fit)),
        "mean_comments_selected": float(np.mean(features.n_selected)),
        "training": {
            "regularization": cfg.lasso_regularization,
            "alpha": float(trainer.alpha),
            "l1_ratio": float(cfg.lasso_l1_ratio),
            "sample_weight_used": bool(sample_weights is not None),
            "synthetic_samples_used": int(len(X_syn)) if X_syn is not None else 0,
            "trait_metrics": trait_metrics,
        },
        "validation": validation,
        "test": test,
        "regression": regression,
        "prediction_evidence": evidence,
        "overall": {
            "accuracy": test["aggregate"]["accuracy"],
            "precision": test["aggregate"]["precision"],
            "recall": test["aggregate"]["recall"],
            "f1": test["aggregate"]["f1"],
            "specificity": test["aggregate"]["specificity"],
            "roc_auc": test["aggregate"]["roc_auc"],
            "pr_auc": test["aggregate"]["pr_auc"],
            "mae": regression["aggregate"]["mae"],
            "rmse": regression["aggregate"]["rmse"],
            "r2": regression["aggregate"]["r2"],
            "correlation": regression["aggregate"]["correlation"],
            "val_f1": validation["aggregate"]["f1"],
            "val_accuracy": validation["aggregate"]["accuracy"],
        },
    }
    logger.info(
        "%s done | test accuracy=%s F1=%s specificity=%s",
        exp_id,
        _fmt(result["overall"]["accuracy"]),
        _fmt(result["overall"]["f1"]),
        _fmt(result["overall"]["specificity"]),
    )
    return result, trainer


def _run_lstm_condition(
    sample: Sample,
    exp_id: str,
    cfg: ExperimentConfig,
    features: Features,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[Dict[str, Any], LSTMTrainer]:
    """Train and evaluate one LSTM final-model condition."""
    spec = EXPERIMENTS[exp_id]
    logger.info("=== %s: %s ===", exp_id, spec["label"])

    y = np.asarray(sample.labels_unit, dtype=np.float32)
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    train_sequences, fit_y, sample_weights, X_syn, _y_syn = _condition_sequences(
        features,
        y_train,
        train_idx,
        cfg,
        use_gan=bool(spec["gan"]),
    )
    val_sequences = [features.sequences[i] for i in val_idx]
    test_sequences = [features.sequences[i] for i in test_idx]

    trainer = LSTMTrainer(
        hidden_dim=cfg.lstm_hidden_dim,
        num_layers=cfg.lstm_num_layers,
        dropout=cfg.lstm_dropout,
        learning_rate=cfg.lstm_learning_rate,
    )
    training_history = trainer.train(
        sequences=train_sequences,
        targets=fit_y,
        val_sequences=val_sequences,
        val_targets=y_val,
        epochs=cfg.lstm_epochs,
        batch_size=cfg.lstm_batch_size,
        sample_weights=sample_weights,
        seed=cfg.seed,
        ground_truth_cutoff=cfg.ground_truth_cutoff,
    )

    val_prob = trainer.predict_proba(val_sequences)
    test_prob = trainer.predict_proba(test_sequences)
    validation = me.evaluate_lstm_binary_classifier(
        y_true=y_val,
        probabilities=val_prob,
        trait_names=list(TRAIT_KEYS),
        ground_truth_cutoff=cfg.ground_truth_cutoff,
        candidate_thresholds=list(cfg.candidate_thresholds),
    )
    test = me.evaluate_lstm_binary_with_thresholds(
        y_true=y_test,
        probabilities=test_prob,
        threshold_selection=validation,
        trait_names=list(TRAIT_KEYS),
        ground_truth_cutoff=cfg.ground_truth_cutoff,
        candidate_thresholds=list(cfg.candidate_thresholds),
    )
    evidence = {
        "validation": _classification_evidence_rows(
            sample=sample,
            exp_id=exp_id,
            spec=spec,
            split="validation",
            indices=val_idx,
            y_true=y_val,
            scores=val_prob,
            metrics=validation,
            threshold_field="best_threshold",
            ground_truth_cutoff=cfg.ground_truth_cutoff,
        ),
        "test": _classification_evidence_rows(
            sample=sample,
            exp_id=exp_id,
            spec=spec,
            split="test",
            indices=test_idx,
            y_true=y_test,
            scores=test_prob,
            metrics=test,
            threshold_field="selected_threshold",
            ground_truth_cutoff=cfg.ground_truth_cutoff,
        ),
    }

    result = {
        "experiment": exp_id,
        "label": spec["label"],
        "model": "lstm",
        "selection": spec["selection"],
        "gan": bool(spec["gan"]),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_fit": int(len(train_sequences)),
        "mean_comments_selected": float(np.mean(features.n_selected)),
        "training": {
            **training_history,
            "synthetic_samples_used": int(len(X_syn)) if X_syn is not None else 0,
        },
        "validation": validation,
        "test": test,
        "prediction_evidence": evidence,
        "overall": {
            "accuracy": test["aggregate"]["accuracy"],
            "precision": test["aggregate"]["precision"],
            "recall": test["aggregate"]["recall"],
            "f1": test["aggregate"]["f1"],
            "specificity": test["aggregate"]["specificity"],
            "roc_auc": test["aggregate"]["roc_auc"],
            "pr_auc": test["aggregate"]["pr_auc"],
            "val_f1": validation["aggregate"]["f1"],
            "val_accuracy": validation["aggregate"]["accuracy"],
        },
    }
    logger.info(
        "%s done | test accuracy=%s F1=%s specificity=%s",
        exp_id,
        _fmt(result["overall"]["accuracy"]),
        _fmt(result["overall"]["f1"]),
        _fmt(result["overall"]["specificity"]),
    )
    return result, trainer


def _run_condition(
    sample: Sample,
    exp_id: str,
    cfg: ExperimentConfig,
    features: Features,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[Dict[str, Any], Union[LassoTrainer, LSTMTrainer]]:
    """Train and evaluate one condition from the 2x2x2 Selection/GAN/Model design."""
    model = EXPERIMENTS[exp_id]["model"]
    if model == "lasso":
        return _run_lasso_condition(sample, exp_id, cfg, features, train_idx, val_idx, test_idx)
    if model == "lstm":
        return _run_lstm_condition(sample, exp_id, cfg, features, train_idx, val_idx, test_idx)
    raise ValueError(f"Unknown experiment model: {model!r}")


def run_condition(
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
    """Run a single condition against already-built features and split."""
    result, model = _run_condition(sample, exp_id, cfg, features, train_idx, val_idx, test_idx)
    return (result, model) if return_model else result


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
    train_idx, val_idx, test_idx = make_split(sample.n_users, cfg)

    results: Dict[str, Any] = {}
    models: Dict[str, Union[LassoTrainer, LSTMTrainer]] = {}
    for exp_id, spec in EXPERIMENTS.items():
        result, model = _run_condition(
            sample,
            exp_id,
            cfg,
            features[spec["selection"]],
            train_idx,
            val_idx,
            test_idx,
        )
        results[exp_id] = result
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
            "n_test": int(len(test_idx)),
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
            "test_accuracy": o.get("accuracy"),
            "test_f1": o.get("f1"),
            "test_specificity": o.get("specificity"),
            "test_precision": o.get("precision"),
            "test_recall": o.get("recall"),
            "roc_auc": o.get("roc_auc"),
            "pr_auc": o.get("pr_auc"),
            "mae": o.get("mae"),
            "rmse": o.get("rmse"),
            "r2": o.get("r2"),
            "val_f1": o.get("val_f1"),
            "mean_comments_selected": r.get("mean_comments_selected"),
        })
    return pd.DataFrame(rows)


def presentation_metric_table(results: Dict[str, Any]):
    """Long-form condition x metric table for BI tools and slide charts."""
    import pandas as pd

    rows = []
    for exp_id in EXPERIMENTS:
        if exp_id not in results:
            continue
        r = results[exp_id]
        for split in ("validation", "test"):
            aggregate = r[split]["aggregate"]
            for metric in PRESENTATION_METRICS:
                rows.append({
                    "condition": exp_id,
                    "description": r["label"],
                    "split": split,
                    "selection": r["selection"],
                    "gan": r["gan"],
                    "model": r["model"],
                    "metric": metric,
                    "value": aggregate.get(metric),
                })
    return pd.DataFrame(rows)


def threshold_sweep_table(results: Dict[str, Any]):
    """Long-form five-threshold sweep table used for threshold graphs."""
    import pandas as pd

    rows = []
    for exp_id in EXPERIMENTS:
        if exp_id not in results:
            continue
        r = results[exp_id]
        for split in ("validation", "test"):
            for trait, trait_metrics in r[split]["per_trait"].items():
                selected = trait_metrics.get("best_threshold", trait_metrics.get("selected_threshold"))
                for sweep in trait_metrics.get("threshold_sweep", []):
                    rows.append({
                        "condition": exp_id,
                        "description": r["label"],
                        "split": split,
                        "selection": r["selection"],
                        "gan": r["gan"],
                        "model": r["model"],
                        "trait": trait,
                        "threshold": sweep.get("threshold"),
                        "is_selected_threshold": float(sweep.get("threshold")) == float(selected),
                        "accuracy": sweep.get("accuracy"),
                        "precision": sweep.get("precision"),
                        "recall": sweep.get("recall"),
                        "f1_score": sweep.get("f1_score"),
                        "specificity": sweep.get("specificity"),
                        "tp": sweep.get("tp"),
                        "fp": sweep.get("fp"),
                        "tn": sweep.get("tn"),
                        "fn": sweep.get("fn"),
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


def _find(
    results: Dict[str, Any],
    selection: str,
    gan: bool,
    model: str,
) -> Optional[Dict[str, Any]]:
    for result in results.values():
        if (
            result["selection"] == selection
            and bool(result["gan"]) == bool(gan)
            and result["model"] == model
        ):
            return result
    return None


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
            base = _find(results, "baseline", gan, model)
            qlearn = _find(results, "qlearning", gan, model)
            if base and qlearn:
                q_rows.append({
                    "model": model,
                    "gan": gan,
                    "accuracy_baseline": base["overall"].get("accuracy"),
                    "accuracy_qlearning": qlearn["overall"].get("accuracy"),
                    "delta_accuracy": _delta(qlearn["overall"].get("accuracy"), base["overall"].get("accuracy")),
                    "f1_baseline": base["overall"].get("f1"),
                    "f1_qlearning": qlearn["overall"].get("f1"),
                    "delta_f1": _delta(qlearn["overall"].get("f1"), base["overall"].get("f1")),
                    "delta_specificity": _delta(
                        qlearn["overall"].get("specificity"),
                        base["overall"].get("specificity"),
                    ),
                })

    g_rows = []
    for model in ("lasso", "lstm"):
        for selection in ("baseline", "qlearning"):
            no_gan = _find(results, selection, False, model)
            yes_gan = _find(results, selection, True, model)
            if no_gan and yes_gan:
                g_rows.append({
                    "model": model,
                    "selection": selection,
                    "accuracy_no_gan": no_gan["overall"].get("accuracy"),
                    "accuracy_gan": yes_gan["overall"].get("accuracy"),
                    "delta_accuracy": _delta(yes_gan["overall"].get("accuracy"), no_gan["overall"].get("accuracy")),
                    "f1_no_gan": no_gan["overall"].get("f1"),
                    "f1_gan": yes_gan["overall"].get("f1"),
                    "delta_f1": _delta(yes_gan["overall"].get("f1"), no_gan["overall"].get("f1")),
                    "delta_specificity": _delta(
                        yes_gan["overall"].get("specificity"),
                        no_gan["overall"].get("specificity"),
                    ),
                })

    model_rows = []
    for gan in (False, True):
        for selection in ("baseline", "qlearning"):
            lasso = _find(results, selection, gan, "lasso")
            lstm = _find(results, selection, gan, "lstm")
            if lasso and lstm:
                model_rows.append({
                    "selection": selection,
                    "gan": gan,
                    "accuracy_lasso": lasso["overall"].get("accuracy"),
                    "accuracy_lstm": lstm["overall"].get("accuracy"),
                    "delta_accuracy": _delta(
                        lstm["overall"].get("accuracy"),
                        lasso["overall"].get("accuracy"),
                    ),
                    "f1_lasso": lasso["overall"].get("f1"),
                    "f1_lstm": lstm["overall"].get("f1"),
                    "delta_f1": _delta(
                        lstm["overall"].get("f1"),
                        lasso["overall"].get("f1"),
                    ),
                    "specificity_lasso": lasso["overall"].get("specificity"),
                    "specificity_lstm": lstm["overall"].get("specificity"),
                    "delta_specificity": _delta(
                        lstm["overall"].get("specificity"),
                        lasso["overall"].get("specificity"),
                    ),
                })
    return {
        "qlearning_effect": pd.DataFrame(q_rows),
        "gan_effect": pd.DataFrame(g_rows),
        "model_comparison": pd.DataFrame(model_rows),
    }


def summarize_findings(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate short, JSON-safe headline notes."""
    scored = {k: v for k, v in results.items() if v["overall"].get("f1") is not None}
    best_id = max(
        scored,
        key=lambda key: (
            scored[key]["overall"].get("f1") or -1.0,
            scored[key]["overall"].get("accuracy") or -1.0,
        ),
    ) if scored else None
    best = None
    notes: List[str] = []
    if best_id:
        r = results[best_id]
        best = {
            "condition": best_id,
            "label": r["label"],
            "model": r["model"],
            "accuracy": r["overall"].get("accuracy"),
            "f1": r["overall"].get("f1"),
            "specificity": r["overall"].get("specificity"),
        }
        notes.append(
            f"Best condition: {best_id} ({r['label']}) - "
            f"test F1 {best['f1']:.3f}, accuracy {best['accuracy']:.3f}, "
            f"specificity {best['specificity']:.3f}."
        )

    effects = factor_effects(results)
    for effect_name, metric_name in (
        ("qlearning_effect", "Q-learning selection"),
        ("gan_effect", "GAN augmentation"),
        ("model_comparison", "LSTM versus Lasso"),
    ):
        df = effects[effect_name]
        if len(df) and "delta_f1" in df:
            vals = [v for v in df["delta_f1"].tolist() if v is not None]
            if vals:
                mean_delta = float(np.mean(vals))
                if effect_name == "model_comparison":
                    verdict = "outperforms" if mean_delta > 0 else ("underperforms" if mean_delta < 0 else "matches")
                    notes.append(f"LSTM {verdict} Lasso by {mean_delta:+.3f} mean test F1.")
                else:
                    verdict = "improves" if mean_delta > 0 else ("reduces" if mean_delta < 0 else "does not change")
                    notes.append(f"{metric_name} {verdict} mean test F1 by {mean_delta:+.3f}.")

    notes.append(
        "Official test metrics use thresholds selected on validation, not thresholds chosen on the test labels."
    )
    return {"best_condition": best, "notes": notes}


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


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
