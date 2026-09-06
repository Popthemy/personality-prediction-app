"""
Metrics engine for Big Five personality classification.

The current PANDORA experiment path treats LSTM as the final classifier. The
LSTM emits five probabilities, one P(High) for each OCEAN trait, and this
module selects/applies binary Low/High decision thresholds.

Primary PANDORA evaluation:
- fixed five-threshold sweep: 0.30, 0.40, 0.50, 0.60, 0.70
- validation fold chooses the best threshold per trait
- test fold applies the validation-selected threshold once
- reports Accuracy, Precision, Recall, F1, Specificity, ROC-AUC, and PR-AUC

There is no Medium class in the active PANDORA path. Older regression and
hybrid helpers remain below only for leftover callers outside the new runner.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
)

logger = logging.getLogger('ml_pipeline')

ArrayLike = Union[Sequence[float], np.ndarray]

# Candidate decision thresholds for the active binary LSTM probability sweep.
CANDIDATE_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]

# Explicit High/Low cutoff on the normalized [0, 1] OCEAN scale. Not inferred
# from the evaluation set (that would leak a test-set decision boundary).
DEFAULT_GROUND_TRUTH_CUTOFF = 0.5

# Fine grid for threshold-vs-metric graphs. Predictions are in [0, 1].
DEFAULT_NORMALIZED_THRESHOLDS = [
    round(float(t), 4) for t in np.linspace(0.0, 1.0, 101)
]

_DEFAULT_TRAIT_NAMES = ["O", "C", "E", "A", "N"]
_EXPECTED_N_TRAITS = 5

# ===========================================================================
# Existing functionality -- preserved as-is for backward compatibility.
# These are scale-agnostic (they operate on whatever array they're given),
# so they are reused by the new evaluate() path below rather than
# reimplemented.
# ===========================================================================

def calculate_pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Pearson correlation coefficient r between ground truth and predicted values."""
    if len(y_true) < 2:
        return 0.0
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    if std_true == 0 or std_pred == 0:
        return 0.0
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return float(np.nan_to_num(corr, nan=0.0))


def compute_classification_metrics_at_threshold(
    y_true_binary: np.ndarray,
    probabilities: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Compute binary classification metrics at a specific decision threshold.

    Scale-agnostic: `probabilities` just needs to be compared against
    `threshold` on the same scale (works for [0,1] probabilities or raw
    continuous predictions alike).

    Metrics:
    - Accuracy: (TP + TN) / Total
    - Precision: TP / (TP + FP)
    - F1 Score: Harmonic mean of Precision & Recall
    - Specificity: TN / (TN + FP)
    """
    y_pred_binary = (probabilities >= threshold).astype(int)
    y_true_binary = y_true_binary.astype(int)

    tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
    fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
    tn = np.sum((y_pred_binary == 0) & (y_true_binary == 0))
    fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))

    total = len(y_true_binary)
    accuracy = float((tp + tn) / total) if total > 0 else 0.0

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    if (precision + recall) > 0:
        f1 = float(2 * (precision * recall) / (precision + recall))
    else:
        f1 = 0.0

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        'threshold': float(threshold),
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'specificity': round(specificity, 4),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
    }


# Shared description of compute_model_confidence(). Not a statistical model.
MODEL_CONFIDENCE_SEMANTICS = {
    "name": "model_confidence_heuristic",
    "kind": "certainty_heuristic",
    "calibrated_probability": False,
    "statistical_uncertainty": False,
    "confidence_interval": None,
    "description": (
        "Heuristic blend of how far continuous scores sit from 0.5 and, when "
        "available, a scaled mean absolute error term. It is a model "
        "confidence/certainty score for reporting, not a calibrated "
        "probability and not a statistical uncertainty estimate."
    ),
    "suitable_for": (
        "deterministic interpretation / reporting of output extremity and "
        "error magnitude"
    ),
    "not_suitable_for": (
        "calibrated class probability, confidence intervals, credible "
        "intervals, or p-values"
    ),
    "formula": {
        "certainty": "mean(2 * |score - 0.5|)",
        "error_factor": "clip(1 - mean(|error|) / 2, 0, 1) when errors are supplied",
        "score": "0.6 * certainty + 0.4 * error_factor if errors else certainty",
        "range": "[0, 1]",
    },
}


def _model_confidence_components(
    probabilities: np.ndarray,
    continuous_errors: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Internal parts of the existing confidence heuristic. One formula only."""
    scores = np.asarray(probabilities, dtype=float).ravel()
    if scores.size == 0:
        return {
            "score": 0.50,
            "certainty": 0.50,
            "error_factor": None,
            "used_error_term": False,
        }

    certainty = float(np.mean(2.0 * np.abs(scores - 0.5)))
    used_error = False
    error_factor = None
    if continuous_errors is not None and len(continuous_errors) > 0:
        mean_err = float(np.mean(np.asarray(continuous_errors, dtype=float)))
        error_factor = max(0.0, min(1.0, 1.0 - (mean_err / 2.0)))
        confidence = 0.6 * certainty + 0.4 * error_factor
        used_error = True
    else:
        confidence = certainty

    return {
        "score": round(float(np.clip(confidence, 0.0, 1.0)), 4),
        "certainty": round(float(np.clip(certainty, 0.0, 1.0)), 4),
        "error_factor": (
            round(float(error_factor), 4) if error_factor is not None else None
        ),
        "used_error_term": used_error,
    }


def compute_model_confidence(probabilities: np.ndarray, continuous_errors: Optional[np.ndarray] = None) -> float:
    """
    Model confidence / certainty heuristic from score distance to 0.5, optionally
    blended with a mean-absolute-error factor.

    This is the SINGLE implementation of that heuristic in the pipeline.
    PipelineOrchestrator must not maintain a second copy of this formula --
    call this function (via evaluate(), or directly) instead.

    This is NOT calibrated statistical uncertainty. It does not produce a
    confidence interval, standard error, or well-calibrated probability.
    Scores farther from 0.5 simply increase the certainty term; lower MAE
    increases the optional error factor.

    Formula (unchanged):
    Certainty = mean(2 * |score - 0.5|) in [0.0, 1.0]
    Error Factor = clip(1.0 - mean_mae / 2.0, 0, 1)
    Model Confidence = 0.6 * Certainty + 0.4 * Error Factor
    """
    return float(_model_confidence_components(probabilities, continuous_errors)["score"])


def describe_model_confidence(
    probabilities: np.ndarray,
    continuous_errors: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Same heuristic as ``compute_model_confidence()``, plus semantics for a
    reporting / interpretation layer. Does not invent intervals or a second score.
    """
    parts = _model_confidence_components(probabilities, continuous_errors)
    return {
        "score": parts["score"],
        "certainty": parts["certainty"],
        "error_factor": parts["error_factor"],
        "used_error_term": parts["used_error_term"],
        **MODEL_CONFIDENCE_SEMANTICS,
    }


def evaluate_threshold_sweep(
    y_true_continuous: np.ndarray,
    probabilities: np.ndarray,
    continuous_preds: np.ndarray,
    ground_truth_cutoff: float = 4.0,
    candidate_thresholds: List[float] = CANDIDATE_THRESHOLDS
) -> Dict[str, Any]:
    """
    [PRESERVED - LEGACY PATH] Sweeps classification metrics across candidate
    thresholds against `probabilities` (expected on a [0, 1] scale, e.g. an
    LSTM sigmoid output), blended with continuous-prediction MAE/confidence.

    Kept for any older caller still evaluating an LSTM probability sweep.
    The active PANDORA runner uses evaluate_lstm_binary_classifier() and
    evaluate_lstm_binary_with_thresholds() below.
    """
    y_true_binary = (y_true_continuous >= ground_truth_cutoff).astype(int)
    threshold_results = {}
    best_threshold = 0.50
    best_f1 = -1.0

    for tau in candidate_thresholds:
        metrics = compute_classification_metrics_at_threshold(
            y_true_binary=y_true_binary,
            probabilities=probabilities,
            threshold=tau
        )
        threshold_results[f"{tau:.2f}"] = metrics

        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_threshold = tau

    abs_errors = np.abs(y_true_continuous - continuous_preds)
    mae = float(mean_absolute_error(y_true_continuous, continuous_preds)) if len(y_true_continuous) > 0 else 0.0
    confidence = compute_model_confidence(probabilities, abs_errors)

    optimal_metrics = threshold_results[f"{best_threshold:.2f}"]

    return {
        'optimal_threshold': best_threshold,
        'optimal_metrics': optimal_metrics,
        'threshold_sweep': threshold_results,
        'mae': round(mae, 4),
        'model_confidence': confidence,
    }


def evaluate_hybrid_metrics(
    y_true: np.ndarray,
    y_pred_lasso: np.ndarray,
    y_pred_lstm_continuous: np.ndarray,
    probabilities_lstm: np.ndarray,
    ground_truth_cutoff: float = 4.0,
    candidate_thresholds: List[float] = CANDIDATE_THRESHOLDS
) -> Dict[str, Any]:
    """
    [PRESERVED - LEGACY PATH] Original 8-metric taxonomy, threshold sweep
    applied to LSTM probabilities. Kept for backward compatibility with any
    existing caller. The active PANDORA runner does not use this hybrid path.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred_lasso = np.array(y_pred_lasso, dtype=float)
    y_pred_lstm_continuous = np.array(y_pred_lstm_continuous, dtype=float)
    probabilities_lstm = np.array(probabilities_lstm, dtype=float)

    lasso_mae = float(mean_absolute_error(y_true, y_pred_lasso)) if len(y_true) > 0 else 0.0
    lasso_corr = calculate_pearson_correlation(y_true, y_pred_lasso)
    lasso_r2 = float(r2_score(y_true, y_pred_lasso)) if len(y_true) > 1 else 0.0

    lstm_sweep_eval = evaluate_threshold_sweep(
        y_true_continuous=y_true,
        probabilities=probabilities_lstm,
        continuous_preds=y_pred_lstm_continuous,
        ground_truth_cutoff=ground_truth_cutoff,
        candidate_thresholds=candidate_thresholds
    )

    opt_metrics = lstm_sweep_eval['optimal_metrics']

    return {
        'lasso_r2': round(lasso_r2, 4),
        'lasso_correlation': round(lasso_corr, 4),
        'lasso_mae': round(lasso_mae, 4),
        'lstm_accuracy': opt_metrics['accuracy'],
        'lstm_precision': opt_metrics['precision'],
        'lstm_f1_score': opt_metrics['f1_score'],
        'lstm_specificity': opt_metrics['specificity'],
        'lstm_model_confidence': lstm_sweep_eval['model_confidence'],
        'lstm_mae': lstm_sweep_eval['mae'],
        'optimal_threshold': lstm_sweep_eval['optimal_threshold'],
        'threshold_sweep': lstm_sweep_eval['threshold_sweep'],
    }


# ===========================================================================
# New functionality
# ===========================================================================

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MAE, MSE, RMSE, R^2, Pearson r for one trait's continuous predictions."""
    if len(y_true) == 0:
        return {'mae': 0.0, 'mse': 0.0, 'rmse': 0.0, 'r2': 0.0, 'correlation': 0.0}

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0
    corr = calculate_pearson_correlation(y_true, y_pred)

    return {
        'mae': round(mae, 4),
        'mse': round(mse, 4),
        'rmse': round(rmse, 4),
        'r2': round(r2, 4),
        'correlation': round(corr, 4),
    }


def _multiclass_specificity(cm: np.ndarray) -> float:
    """Macro-averaged one-vs-rest specificity from a confusion matrix."""
    n_classes = cm.shape[0]
    total = cm.sum()
    specs = []
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        specs.append(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0)
    return float(np.mean(specs)) if specs else 0.0


def compute_multiclass_metrics(
    y_true_classes: np.ndarray,
    y_pred_classes: np.ndarray,
    labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """[LEGACY] 3-class metrics. Not used by `evaluate()`."""
    if labels is None:
        labels = sorted(set(np.unique(y_true_classes)) | set(np.unique(y_pred_classes)))

    accuracy = float(accuracy_score(y_true_classes, y_pred_classes))
    precision = float(precision_score(y_true_classes, y_pred_classes, labels=labels, average='macro', zero_division=0))
    recall = float(recall_score(y_true_classes, y_pred_classes, labels=labels, average='macro', zero_division=0))
    f1 = float(f1_score(y_true_classes, y_pred_classes, labels=labels, average='macro', zero_division=0))
    cm = confusion_matrix(y_true_classes, y_pred_classes, labels=labels)
    specificity = _multiclass_specificity(cm)

    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'specificity': round(specificity, 4),
        'confusion_matrix': cm.tolist(),
        'labels': [int(l) for l in labels],
    }


def default_threshold_candidates(values: np.ndarray, n: int = 5) -> List[float]:
    """
    [LEGACY] Data-driven candidate thresholds from the 30th-70th percentile
    of `values`. Not used by `evaluate()` -- that path sweeps the fixed
    [0.00, 1.00] grid so the evaluation set cannot choose its own thresholds.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return list(CANDIDATE_THRESHOLDS)
    pctiles = np.linspace(30, 70, n)
    return [round(float(np.percentile(values, p)), 4) for p in pctiles]


def derive_binary_ground_truth(y_true: np.ndarray, cutoff: Optional[float] = None) -> tuple:
    """
    Binarize continuous ground truth for High/Low evaluation.

    High (1) if y_true >= cutoff, else Low (0). No Medium class.

    Default cutoff is DEFAULT_GROUND_TRUTH_CUTOFF (0.5), the midpoint of the
    normalized [0, 1] OCEAN scale. The cutoff is never inferred from `y_true`
    (that would leak a test-set decision boundary). Callers that learned a
    cutoff on train/validation must pass it explicitly.
    """
    if cutoff is None:
        cutoff = DEFAULT_GROUND_TRUTH_CUTOFF
    y_true_binary = (np.asarray(y_true, dtype=float) >= float(cutoff)).astype(int)
    return y_true_binary, float(cutoff)


def derive_three_class_ground_truth(y_true: np.ndarray, cutoffs: Optional[List[float]] = None) -> tuple:
    """
    [LEGACY] Bucket continuous ground truth into 3 classes (Low=0, Medium=1,
    High=2). Not used by `evaluate()`.
    """
    y_true = np.asarray(y_true, dtype=float)
    if cutoffs is None:
        if len(y_true) == 0:
            cutoffs = [0.0, 0.0]
        else:
            cutoffs = [float(np.percentile(y_true, 33.33)), float(np.percentile(y_true, 66.67))]
    classes = np.digitize(y_true, bins=cutoffs)  # 0, 1, or 2
    return classes.astype(int), list(cutoffs)


def sweep_thresholds_on_scores(
    y_true_binary: np.ndarray,
    scores: np.ndarray,
    candidate_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Sweep candidate thresholds over continuous `scores` (not pre-binarized
    High/Low labels). prediction >= tau -> High (1), else Low (0).

    Reports the complete sweep plus the threshold with the highest F1.
    This reports a post-hoc best-of-sweep; it does not learn or apply a
    decision boundary. The runner must supply a train/validation-selected
    cutoff via `derive_binary_ground_truth` / `evaluate(ground_truth_cutoff=...)`.

    Default candidates are the supervisor-facing five-threshold sweep:
    0.30, 0.40, 0.50, 0.60, 0.70.
    """
    if candidate_thresholds is None:
        candidate_thresholds = list(CANDIDATE_THRESHOLDS)

    results = []
    best = None
    for tau in candidate_thresholds:
        m = compute_classification_metrics_at_threshold(y_true_binary, np.asarray(scores, dtype=float), tau)
        results.append(m)
        if best is None or m['f1_score'] > best['f1_score']:
            best = m

    return {
        'best_threshold': best['threshold'] if best else None,
        'best_f1': best['f1_score'] if best else 0.0,
        'accuracy': best['accuracy'] if best else 0.0,
        'precision': best['precision'] if best else 0.0,
        'recall': best['recall'] if best else 0.0,
        'f1': best['f1_score'] if best else 0.0,
        'specificity': best['specificity'] if best else 0.0,
        'results': results,
    }


def _aggregate_metric(per_trait: Dict[str, Dict[str, Any]], key: str) -> Optional[float]:
    vals = [v.get(key) for v in per_trait.values() if v.get(key) is not None]
    return round(float(np.mean(vals)), 4) if vals else None


def evaluate_lstm_binary_classifier(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    trait_names: Optional[List[str]] = None,
    ground_truth_cutoff: float = DEFAULT_GROUND_TRUTH_CUTOFF,
    candidate_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Select the best Low/High decision threshold for each LSTM trait output.

    Use this on the validation fold. ``probabilities`` must be a ``(N, 5)``
    matrix of sigmoid probabilities where each column is P(High) for one trait.
    """
    y_true_2d = _require_ocean_matrix(y_true, "y_true")
    probs_2d = _require_ocean_matrix(probabilities, "probabilities")
    if probs_2d.shape != y_true_2d.shape:
        raise ValueError(f"probabilities has shape {probs_2d.shape}, expected {y_true_2d.shape}.")

    if trait_names is None:
        trait_names = list(_DEFAULT_TRAIT_NAMES)
    if len(trait_names) != y_true_2d.shape[1]:
        raise ValueError(f"trait_names has {len(trait_names)} entries, expected {y_true_2d.shape[1]}.")
    if candidate_thresholds is None:
        candidate_thresholds = list(CANDIDATE_THRESHOLDS)

    per_trait: Dict[str, Dict[str, Any]] = {}
    for i, trait in enumerate(trait_names):
        y_binary, cutoff = derive_binary_ground_truth(y_true_2d[:, i], ground_truth_cutoff)
        sweep = sweep_thresholds_on_scores(y_binary, probs_2d[:, i], candidate_thresholds)
        roc = compute_roc_curve_metrics(y_binary, probs_2d[:, i])
        pr = compute_precision_recall_curve_metrics(y_binary, probs_2d[:, i])
        labels, counts = np.unique(y_binary, return_counts=True)
        class_balance = {
            BINARY_LABEL: {
                "count": int(counts[list(labels).index(value)]) if value in labels else 0,
                "proportion": (
                    float(counts[list(labels).index(value)] / len(y_binary))
                    if value in labels and len(y_binary) else 0.0
                ),
            }
            for value, BINARY_LABEL in ((0, "Low"), (1, "High"))
        }
        per_trait[trait] = {
            "ground_truth_cutoff": cutoff,
            "candidate_thresholds": [float(t) for t in candidate_thresholds],
            "best_threshold": sweep["best_threshold"],
            "accuracy": sweep["accuracy"],
            "precision": sweep["precision"],
            "recall": sweep["recall"],
            "f1": sweep["f1"],
            "specificity": sweep["specificity"],
            "threshold_sweep": sweep["results"],
            "roc_auc": roc["auc"],
            "pr_auc": pr["average_precision"],
            "class_balance": class_balance,
        }

    return {
        "split": "validation",
        "ground_truth_cutoff": float(ground_truth_cutoff),
        "candidate_thresholds": [float(t) for t in candidate_thresholds],
        "per_trait": per_trait,
        "aggregate": {
            "accuracy": _aggregate_metric(per_trait, "accuracy"),
            "precision": _aggregate_metric(per_trait, "precision"),
            "recall": _aggregate_metric(per_trait, "recall"),
            "f1": _aggregate_metric(per_trait, "f1"),
            "specificity": _aggregate_metric(per_trait, "specificity"),
            "roc_auc": _aggregate_metric(per_trait, "roc_auc"),
            "pr_auc": _aggregate_metric(per_trait, "pr_auc"),
        },
    }


def evaluate_lstm_binary_with_thresholds(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    threshold_selection: Dict[str, Any],
    trait_names: Optional[List[str]] = None,
    ground_truth_cutoff: float = DEFAULT_GROUND_TRUTH_CUTOFF,
    candidate_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Apply validation-selected thresholds to a held-out test fold.

    The test fold may still include the full threshold sweep for analysis, but
    the headline metrics in this return value use only thresholds supplied from
    ``threshold_selection``.
    """
    y_true_2d = _require_ocean_matrix(y_true, "y_true")
    probs_2d = _require_ocean_matrix(probabilities, "probabilities")
    if probs_2d.shape != y_true_2d.shape:
        raise ValueError(f"probabilities has shape {probs_2d.shape}, expected {y_true_2d.shape}.")

    if trait_names is None:
        trait_names = list(_DEFAULT_TRAIT_NAMES)
    if len(trait_names) != y_true_2d.shape[1]:
        raise ValueError(f"trait_names has {len(trait_names)} entries, expected {y_true_2d.shape[1]}.")
    if candidate_thresholds is None:
        candidate_thresholds = list(CANDIDATE_THRESHOLDS)

    selected = threshold_selection.get("per_trait", {})
    per_trait: Dict[str, Dict[str, Any]] = {}
    for i, trait in enumerate(trait_names):
        if trait not in selected:
            raise ValueError(f"Missing validation-selected threshold for trait {trait!r}.")
        tau = float(selected[trait]["best_threshold"])
        y_binary, cutoff = derive_binary_ground_truth(y_true_2d[:, i], ground_truth_cutoff)
        official = compute_classification_metrics_at_threshold(y_binary, probs_2d[:, i], tau)
        sweep = sweep_thresholds_on_scores(y_binary, probs_2d[:, i], candidate_thresholds)
        roc = compute_roc_curve_metrics(y_binary, probs_2d[:, i])
        pr = compute_precision_recall_curve_metrics(y_binary, probs_2d[:, i])
        per_trait[trait] = {
            "ground_truth_cutoff": cutoff,
            "selected_threshold": tau,
            "threshold_source": "validation",
            "accuracy": official["accuracy"],
            "precision": official["precision"],
            "recall": official["recall"],
            "f1": official["f1_score"],
            "specificity": official["specificity"],
            "confusion": {
                "tp": official["tp"],
                "fp": official["fp"],
                "tn": official["tn"],
                "fn": official["fn"],
            },
            "analysis_best_threshold": sweep["best_threshold"],
            "analysis_best_f1": sweep["f1"],
            "threshold_sweep": sweep["results"],
            "roc_auc": roc["auc"],
            "pr_auc": pr["average_precision"],
        }

    return {
        "split": "test",
        "ground_truth_cutoff": float(ground_truth_cutoff),
        "per_trait": per_trait,
        "aggregate": {
            "accuracy": _aggregate_metric(per_trait, "accuracy"),
            "precision": _aggregate_metric(per_trait, "precision"),
            "recall": _aggregate_metric(per_trait, "recall"),
            "f1": _aggregate_metric(per_trait, "f1"),
            "specificity": _aggregate_metric(per_trait, "specificity"),
            "roc_auc": _aggregate_metric(per_trait, "roc_auc"),
            "pr_auc": _aggregate_metric(per_trait, "pr_auc"),
        },
    }


# Backward-compatible aliases used by older tests/callers.
calculate_binary_classification_metrics = compute_classification_metrics_at_threshold
calculate_model_confidence = compute_model_confidence

def compute_roc_curve_metrics(y_true_binary: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    """ROC curve points and ROC-AUC from continuous scores (not High/Low labels)."""
    y_true_binary = np.asarray(y_true_binary).astype(int)
    scores = np.asarray(scores, dtype=float)
    if y_true_binary.size == 0 or len(np.unique(y_true_binary)) < 2:
        return {'fpr': [], 'tpr': [], 'thresholds': [], 'auc': None}

    fpr, tpr, thresholds = roc_curve(y_true_binary, scores)
    return {
        'fpr': [float(x) for x in fpr],
        'tpr': [float(x) for x in tpr],
        'thresholds': [float(x) for x in thresholds],
        'auc': round(float(auc(fpr, tpr)), 4),
    }


def compute_precision_recall_curve_metrics(
    y_true_binary: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, Any]:
    """Precision-Recall curve points and Average Precision from continuous scores."""
    y_true_binary = np.asarray(y_true_binary).astype(int)
    scores = np.asarray(scores, dtype=float)
    if y_true_binary.size == 0 or len(np.unique(y_true_binary)) < 2:
        return {'precision': [], 'recall': [], 'thresholds': [], 'average_precision': None}

    precision, recall, thresholds = precision_recall_curve(y_true_binary, scores)
    ap = float(average_precision_score(y_true_binary, scores))
    return {
        'precision': [float(x) for x in precision],
        'recall': [float(x) for x in recall],
        'thresholds': [float(x) for x in thresholds],
        'average_precision': round(ap, 4),
    }


def _require_ocean_matrix(arr: ArrayLike, name: str) -> np.ndarray:
    """Require a (N, 5) float matrix in O,C,E,A,N order. Never reshape, truncate, or pad."""
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2 or a.shape[1] != _EXPECTED_N_TRAITS:
        raise ValueError(
            f"{name} must have shape (N, 5) in O,C,E,A,N order; got {a.shape}."
        )
    return a


def _resolve_class_labels(arr: ArrayLike, y_true_2d_shape: tuple, name: str) -> np.ndarray:
    """
    Coerce LSTM predictions/labels to (n_samples, n_traits) integer classes.
    If the array has one extra trailing axis matching a plausible number of
    classes (e.g. softmax output), take argmax on that axis. Never silently
    truncates or pads to force a shape match -- raises instead.
    """
    a = np.asarray(arr)
    n_samples, n_traits = y_true_2d_shape

    if a.ndim == 1:
        a2 = a.reshape(-1, 1)
    elif a.ndim == 2:
        a2 = a
    elif a.ndim == 3:
        # (n_samples, n_traits, n_classes) probability/logit output -> argmax
        if a.shape[0] == n_samples and a.shape[1] == n_traits:
            a2 = np.argmax(a, axis=-1)
        else:
            raise ValueError(
                f"{name} has shape {a.shape}, expected (n_samples={n_samples}, "
                f"n_traits={n_traits}, n_classes) for a 3D probability array."
            )
    else:
        raise ValueError(f"{name} must be 1D, 2D, or 3D; got shape {a.shape}")

    if a2.shape != (n_samples, n_traits):
        raise ValueError(
            f"{name} has shape {a2.shape}, expected {(n_samples, n_traits)} "
            f"to match y_true. Provide matching predictions per trait; "
            f"the metrics engine will not reshape/truncate/pad to force a fit."
        )
    return a2.astype(int)


def _threshold_for_trait(
    supplied: Optional[Union[float, Dict[str, float]]],
    trait: str,
) -> Optional[float]:
    if supplied is None:
        return None
    if isinstance(supplied, dict):
        value = supplied.get(trait)
        return None if value is None else float(value)
    return float(supplied)


def evaluate(
    y_true: ArrayLike,
    lasso_predictions: ArrayLike,
    lstm_predictions: ArrayLike,
    lstm_probabilities: Optional[ArrayLike] = None,
    lstm_true_classes: Optional[ArrayLike] = None,
    trait_names: Optional[List[str]] = None,
    class_cutoffs: Optional[Union[List[float], Dict[str, List[float]]]] = None,
    ground_truth_cutoff: Optional[Union[float, Dict[str, float]]] = None,
    candidate_thresholds: Optional[List[float]] = None,
    decision_thresholds: Optional[Union[float, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Legacy continuous evaluator retained for older callers."""
    if lstm_probabilities is not None or lstm_true_classes is not None or class_cutoffs is not None:
        logger.debug(
            "evaluate() ignores lstm_probabilities/lstm_true_classes/class_cutoffs; "
            "both models are scored as continuous (N, 5) OCEAN regressors."
        )

    y_true_2d = _require_ocean_matrix(y_true, "y_true")
    lasso_2d = _require_ocean_matrix(lasso_predictions, "lasso_predictions")
    lstm_2d = _require_ocean_matrix(lstm_predictions, "lstm_predictions")

    if lasso_2d.shape != y_true_2d.shape:
        raise ValueError(
            f"lasso_predictions has shape {lasso_2d.shape}, expected {y_true_2d.shape} to match y_true."
        )
    if lstm_2d.shape != y_true_2d.shape:
        raise ValueError(
            f"lstm_predictions has shape {lstm_2d.shape}, expected {y_true_2d.shape} to match y_true."
        )

    n_traits = y_true_2d.shape[1]
    if trait_names is None:
        trait_names = list(_DEFAULT_TRAIT_NAMES)
    if len(trait_names) != n_traits:
        raise ValueError(f"trait_names has {len(trait_names)} entries, expected {n_traits}.")

    def _mean_keys(per_trait: Dict[str, Dict[str, Any]], keys: List[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key in keys:
            values = [per_trait[t][key] for t in trait_names if per_trait[t].get(key) is not None]
            out[key] = round(float(np.mean(values)), 4) if values else None
        return out

    def _binary_block(
        scores: np.ndarray,
        y_true_binary: np.ndarray,
        sweep: Dict[str, Any],
        roc_block: Dict[str, Any],
        pr_block: Dict[str, Any],
        used_cutoff: float,
        trait: str,
    ) -> Dict[str, Any]:
        caller_tau = _threshold_for_trait(decision_thresholds, trait)
        if caller_tau is not None:
            official = compute_classification_metrics_at_threshold(
                y_true_binary, scores, caller_tau,
            )
            return {
                "ground_truth_cutoff": used_cutoff,
                "threshold_used": official["threshold"],
                "threshold_source": "caller",
                "threshold_is_official": True,
                "accuracy": official["accuracy"],
                "precision": official["precision"],
                "recall": official["recall"],
                "f1": official["f1_score"],
                "roc_auc": roc_block.get("auc"),
                "pr_auc": pr_block.get("average_precision"),
            }
        return {
            "ground_truth_cutoff": used_cutoff,
            "threshold_used": sweep.get("best_threshold"),
            "threshold_source": "analysis_best_f1_on_this_set",
            "threshold_is_official": False,
            "accuracy": sweep.get("accuracy"),
            "precision": sweep.get("precision"),
            "recall": sweep.get("recall"),
            "f1": sweep.get("f1"),
            "roc_auc": roc_block.get("auc"),
            "pr_auc": pr_block.get("average_precision"),
        }

    def _threshold_analysis_block(sweep: Dict[str, Any], trait: str) -> Dict[str, Any]:
        caller_tau = _threshold_for_trait(decision_thresholds, trait)
        metric_values = sweep.get("results") or []
        return {
            "thresholds_tested": [row.get("threshold") for row in metric_values],
            "metric_values": metric_values,
            "selected_threshold": sweep.get("best_threshold"),
            "selected_threshold_rule": "highest_f1_on_this_evaluation_set",
            "selected_metric_values": {
                "accuracy": sweep.get("accuracy"),
                "precision": sweep.get("precision"),
                "recall": sweep.get("recall"),
                "f1": sweep.get("f1"),
            },
            "reported_decision_threshold": (
                caller_tau if caller_tau is not None else sweep.get("best_threshold")
            ),
            "reported_decision_source": (
                "caller" if caller_tau is not None else "analysis_best_f1_on_this_set"
            ),
            "ground_truth_cutoff": sweep.get("ground_truth_cutoff"),
        }

    per_trait_lasso: Dict[str, Dict[str, float]] = {}
    per_trait_lstm: Dict[str, Dict[str, float]] = {}
    per_trait_thr_lasso: Dict[str, Dict[str, Any]] = {}
    per_trait_thr_lstm: Dict[str, Dict[str, Any]] = {}
    per_trait_roc_lasso: Dict[str, Dict[str, Any]] = {}
    per_trait_roc_lstm: Dict[str, Dict[str, Any]] = {}
    per_trait_pr_lasso: Dict[str, Dict[str, Any]] = {}
    per_trait_pr_lstm: Dict[str, Dict[str, Any]] = {}
    per_trait_bin_lasso: Dict[str, Dict[str, Any]] = {}
    per_trait_bin_lstm: Dict[str, Dict[str, Any]] = {}
    per_trait_thr_analysis_lasso: Dict[str, Dict[str, Any]] = {}
    per_trait_thr_analysis_lstm: Dict[str, Dict[str, Any]] = {}
    per_trait_conf_lasso: Dict[str, Dict[str, Any]] = {}
    per_trait_conf_lstm: Dict[str, Dict[str, Any]] = {}

    for i, trait in enumerate(trait_names):
        yt = y_true_2d[:, i]
        lp = lasso_2d[:, i]
        ls = lstm_2d[:, i]

        per_trait_lasso[trait] = compute_regression_metrics(yt, lp)
        per_trait_lstm[trait] = compute_regression_metrics(yt, ls)

        y_true_binary, used_cutoff = derive_binary_ground_truth(
            yt, _threshold_for_trait(ground_truth_cutoff, trait),
        )

        lasso_sweep = sweep_thresholds_on_scores(y_true_binary, lp, candidate_thresholds)
        lasso_sweep['ground_truth_cutoff'] = used_cutoff
        per_trait_thr_lasso[trait] = lasso_sweep

        lstm_sweep = sweep_thresholds_on_scores(y_true_binary, ls, candidate_thresholds)
        lstm_sweep['ground_truth_cutoff'] = used_cutoff
        per_trait_thr_lstm[trait] = lstm_sweep

        per_trait_roc_lasso[trait] = compute_roc_curve_metrics(y_true_binary, lp)
        per_trait_roc_lstm[trait] = compute_roc_curve_metrics(y_true_binary, ls)
        per_trait_pr_lasso[trait] = compute_precision_recall_curve_metrics(y_true_binary, lp)
        per_trait_pr_lstm[trait] = compute_precision_recall_curve_metrics(y_true_binary, ls)

        per_trait_bin_lasso[trait] = _binary_block(
            lp, y_true_binary, lasso_sweep,
            per_trait_roc_lasso[trait], per_trait_pr_lasso[trait],
            used_cutoff, trait,
        )
        per_trait_bin_lstm[trait] = _binary_block(
            ls, y_true_binary, lstm_sweep,
            per_trait_roc_lstm[trait], per_trait_pr_lstm[trait],
            used_cutoff, trait,
        )
        per_trait_thr_analysis_lasso[trait] = _threshold_analysis_block(lasso_sweep, trait)
        per_trait_thr_analysis_lstm[trait] = _threshold_analysis_block(lstm_sweep, trait)
        per_trait_conf_lasso[trait] = describe_model_confidence(lp, np.abs(yt - lp))
        per_trait_conf_lstm[trait] = describe_model_confidence(ls, np.abs(yt - ls))

    reg_keys = ['mae', 'mse', 'rmse', 'r2', 'correlation']
    thr_keys = ['best_f1', 'accuracy', 'precision', 'recall', 'f1']
    bin_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']

    lasso_reg = {
        'per_trait': per_trait_lasso,
        'aggregate': _mean_keys(per_trait_lasso, reg_keys),
    }
    lstm_reg = {
        'per_trait': per_trait_lstm,
        'aggregate': _mean_keys(per_trait_lstm, reg_keys),
    }
    threshold_block = {
        'lasso': {
            'per_trait': per_trait_thr_lasso,
            'aggregate': _mean_keys(per_trait_thr_lasso, thr_keys),
        },
        'lstm': {
            'per_trait': per_trait_thr_lstm,
            'aggregate': _mean_keys(per_trait_thr_lstm, thr_keys),
        },
    }
    roc_block = {
        'lasso': {
            'per_trait': per_trait_roc_lasso,
            'aggregate': _mean_keys(per_trait_roc_lasso, ['auc']),
        },
        'lstm': {
            'per_trait': per_trait_roc_lstm,
            'aggregate': _mean_keys(per_trait_roc_lstm, ['auc']),
        },
    }
    pr_block = {
        'lasso': {
            'per_trait': per_trait_pr_lasso,
            'aggregate': _mean_keys(per_trait_pr_lasso, ['average_precision']),
        },
        'lstm': {
            'per_trait': per_trait_pr_lstm,
            'aggregate': _mean_keys(per_trait_pr_lstm, ['average_precision']),
        },
    }

    return {
        'lasso': lasso_reg,
        'lstm': lstm_reg,
        'threshold': threshold_block,
        'roc': roc_block,
        'precision_recall': pr_block,
        'continuous_regression': {
            'lasso': lasso_reg,
            'lstm': lstm_reg,
        },
        'binary_interpretation': {
            'lasso': {
                'per_trait': per_trait_bin_lasso,
                'aggregate': _mean_keys(per_trait_bin_lasso, bin_keys),
            },
            'lstm': {
                'per_trait': per_trait_bin_lstm,
                'aggregate': _mean_keys(per_trait_bin_lstm, bin_keys),
            },
        },
        'threshold_analysis': {
            'note': (
                "selected_threshold is the highest-F1 point on this evaluation "
                "sweep (descriptive). reported_decision_threshold is the caller "
                "frozen threshold when decision_thresholds is supplied; otherwise "
                "it repeats the descriptive best-F1 point. evaluate() does not "
                "learn a test-set decision boundary."
            ),
            'lasso': {
                'per_trait': per_trait_thr_analysis_lasso,
                'aggregate': threshold_block['lasso']['aggregate'],
            },
            'lstm': {
                'per_trait': per_trait_thr_analysis_lstm,
                'aggregate': threshold_block['lstm']['aggregate'],
            },
        },
        'confidence': {
            'semantics': MODEL_CONFIDENCE_SEMANTICS,
            'lasso': {
                'per_trait': per_trait_conf_lasso,
                'aggregate': {
                    'score': _mean_keys(per_trait_conf_lasso, ['score']).get('score'),
                },
            },
            'lstm': {
                'per_trait': per_trait_conf_lstm,
                'aggregate': {
                    'score': _mean_keys(per_trait_conf_lstm, ['score']).get('score'),
                },
            },
        },
    }
