"""
Metrics Calculation Engine for Big Five Personality Prediction.

Plug-and-play evaluation service. The PANDORA runner and PipelineOrchestrator
call `evaluate()` after Lasso and LSTM produce continuous predictions on the
same held-out participant-level test users. This module has no knowledge of
training, data loading, splitting, experiment identity, GAN, Q-learning, or
BERT.

Primary evaluation (both models are continuous 5-output OCEAN regression)
-------------------------------------------------------------------------
- Per-trait MAE, MSE, RMSE, R^2, Pearson r on normalized [0, 1] scores
- Binary High/Low threshold sweep on continuous scores (no Medium class)
- ROC and Precision-Recall from continuous scores (not binarized labels)
- Trait-specific O/C/E/A/N plus aggregate means

There is no final Low/Medium/High LSTM classification stage. 3-class helpers
below are retained only for leftover legacy callers; `evaluate()` does not
use them.
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

# Candidate decision thresholds for the ORIGINAL [0, 1]-probability sweep
# (e.g. an LSTM sigmoid). Preserved unchanged for backward compatibility --
# do NOT reuse this constant for the new Lasso-on-raw-OCEAN-scale sweep.
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


def compute_model_confidence(probabilities: np.ndarray, continuous_errors: Optional[np.ndarray] = None) -> float:
    """
    Evaluate Model Confidence from output certainty levels (softmax/sigmoid output distance from 0.5)
    blended with continuous prediction error bound.

    This is the SINGLE implementation of model confidence in the pipeline.
    PipelineOrchestrator must not maintain a second copy of this formula --
    call this function (via evaluate(), or directly) instead.

    Formula:
    Certainty = mean(2 * |P - 0.5|) in [0.0, 1.0]
    Error Factor = max(0, 1.0 - mean_mae / 2.0)
    Model Confidence = 0.6 * Certainty + 0.4 * Error Factor
    """
    if len(probabilities) == 0:
        return 0.50

    certainty = float(np.mean(2.0 * np.abs(probabilities - 0.5)))

    if continuous_errors is not None and len(continuous_errors) > 0:
        mean_err = float(np.mean(continuous_errors))
        error_factor = max(0.0, min(1.0, 1.0 - (mean_err / 2.0)))
        confidence = 0.6 * certainty + 0.4 * error_factor
    else:
        confidence = certainty

    return round(float(np.clip(confidence, 0.0, 1.0)), 4)


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

    Kept unchanged for any existing caller still evaluating an LSTM
    probability sweep. The NEW Lasso-on-raw-OCEAN-scale threshold sweep
    used by evaluate() is `sweep_thresholds_on_scores()` below, which does
    not assume a [0, 1] scale and does not blend in confidence/MAE (those
    are separate, explicit outputs in the new interface).
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
    existing caller. New orchestrator code should call `evaluate()` instead,
    which applies the threshold sweep to Lasso's continuous predictions per
    the current project spec and adds MSE/RMSE/multiclass LSTM metrics.
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

    Default candidates are 0.00, 0.01, ..., 1.00 (normalized OCEAN scale).
    """
    if candidate_thresholds is None:
        candidate_thresholds = list(DEFAULT_NORMALIZED_THRESHOLDS)

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
        'results': results,
    }


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
) -> Dict[str, Any]:
    """
    Plug-and-play evaluation of continuous (N, 5) Lasso and LSTM predictions.

    Both models are treated as 5-output OCEAN regressors on the same held-out
    participants. Predictions and ground truth are evaluated on the normalized
    [0, 1] scale. This path does not require or derive Low/Medium/High classes.

    Parameters
    ----------
    y_true, lasso_predictions, lstm_predictions
        Continuous OCEAN arrays with shape (N, 5) in O,C,E,A,N order.
        A mismatch raises ValueError; values are never reshaped, truncated,
        or padded.
    trait_names
        Optional column labels. Defaults to ['O', 'C', 'E', 'A', 'N'].
    ground_truth_cutoff
        Explicit High/Low rule: y >= cutoff -> High (1), else Low (0).
        A single float applies to every trait, or {trait: cutoff}.
        Defaults to 0.5 (midpoint of [0, 1]). Never inferred from the
        evaluation set -- the runner must pass a train/validation-selected
        boundary if one was learned.
    candidate_thresholds
        Thresholds to sweep on continuous predictions. Defaults to
        0.00, 0.01, ..., 1.00. The sweep reports every point plus the
        highest-F1 threshold; that best-of-sweep is descriptive only.

    lstm_probabilities, lstm_true_classes, class_cutoffs
        Accepted for call-site compatibility. Ignored. 3-class evaluation
        is not part of this path.

    Returns
    -------
    dict with keys 'lasso', 'lstm', 'threshold', 'roc', 'precision_recall'.
    Each model block is {'per_trait': {O/C/E/A/N: ...}, 'aggregate': ...}.
    """
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

    def _cutoff_for(trait: str) -> Optional[float]:
        if ground_truth_cutoff is None:
            return None
        if isinstance(ground_truth_cutoff, dict):
            return ground_truth_cutoff.get(trait)
        return ground_truth_cutoff

    def _mean_keys(per_trait: Dict[str, Dict[str, Any]], keys: List[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key in keys:
            values = [per_trait[t][key] for t in trait_names if per_trait[t].get(key) is not None]
            out[key] = round(float(np.mean(values)), 4) if values else None
        return out

    per_trait_lasso: Dict[str, Dict[str, float]] = {}
    per_trait_lstm: Dict[str, Dict[str, float]] = {}
    per_trait_thr_lasso: Dict[str, Dict[str, Any]] = {}
    per_trait_thr_lstm: Dict[str, Dict[str, Any]] = {}
    per_trait_roc_lasso: Dict[str, Dict[str, Any]] = {}
    per_trait_roc_lstm: Dict[str, Dict[str, Any]] = {}
    per_trait_pr_lasso: Dict[str, Dict[str, Any]] = {}
    per_trait_pr_lstm: Dict[str, Dict[str, Any]] = {}

    for i, trait in enumerate(trait_names):
        yt = y_true_2d[:, i]
        lp = lasso_2d[:, i]
        ls = lstm_2d[:, i]

        per_trait_lasso[trait] = compute_regression_metrics(yt, lp)
        per_trait_lstm[trait] = compute_regression_metrics(yt, ls)

        y_true_binary, used_cutoff = derive_binary_ground_truth(yt, _cutoff_for(trait))

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

    reg_keys = ['mae', 'mse', 'rmse', 'r2', 'correlation']
    thr_keys = ['best_f1', 'accuracy', 'precision', 'recall', 'f1']

    return {
        'lasso': {
            'per_trait': per_trait_lasso,
            'aggregate': _mean_keys(per_trait_lasso, reg_keys),
        },
        'lstm': {
            'per_trait': per_trait_lstm,
            'aggregate': _mean_keys(per_trait_lstm, reg_keys),
        },
        'threshold': {
            'lasso': {
                'per_trait': per_trait_thr_lasso,
                'aggregate': _mean_keys(per_trait_thr_lasso, thr_keys),
            },
            'lstm': {
                'per_trait': per_trait_thr_lstm,
                'aggregate': _mean_keys(per_trait_thr_lstm, thr_keys),
            },
        },
        'roc': {
            'lasso': {
                'per_trait': per_trait_roc_lasso,
                'aggregate': _mean_keys(per_trait_roc_lasso, ['auc']),
            },
            'lstm': {
                'per_trait': per_trait_roc_lstm,
                'aggregate': _mean_keys(per_trait_roc_lstm, ['auc']),
            },
        },
        'precision_recall': {
            'lasso': {
                'per_trait': per_trait_pr_lasso,
                'aggregate': _mean_keys(per_trait_pr_lasso, ['average_precision']),
            },
            'lstm': {
                'per_trait': per_trait_pr_lstm,
                'aggregate': _mean_keys(per_trait_pr_lstm, ['average_precision']),
            },
        },
    }