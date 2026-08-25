# """
# Metrics Calculation Engine for Big Five Personality Prediction.

# Provides dynamic 5-threshold evaluation across candidate decision thresholds:
# T = {0.30, 0.40, 0.50, 0.60, 0.70}

# Tracks exact model performance taxonomy:
# - Lasso Regression Metrics: R² Score, Pearson Correlation, MAE
# - LSTM Classification Metrics: Accuracy, Precision, F1 Score, Specificity, Model Confidence, MAE
# - Dynamic threshold optimization for classification performance tuning.
# """

# import numpy as np
# import logging
# from typing import Dict, List, Tuple, Any, Optional
# from sklearn.metrics import mean_absolute_error, r2_score

# logger = logging.getLogger('ml_pipeline')

# # Candidate decision thresholds specified by project requirements
# CANDIDATE_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


# def calculate_pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     """Calculate Pearson correlation coefficient r between ground truth and predicted values."""
#     if len(y_true) < 2:
#         return 0.0
#     std_true = np.std(y_true)
#     std_pred = np.std(y_pred)
#     if std_true == 0 or std_pred == 0:
#         return 0.0
#     corr = np.corrcoef(y_true, y_pred)[0, 1]
#     return float(np.nan_to_num(corr, nan=0.0))


# def compute_classification_metrics_at_threshold(
#     y_true_binary: np.ndarray,
#     probabilities: np.ndarray,
#     threshold: float
# ) -> Dict[str, float]:
#     """
#     Compute binary classification metrics at a specific decision threshold.
    
#     Metrics:
#     - Accuracy: (TP + TN) / Total
#     - Precision: TP / (TP + FP)
#     - F1 Score: Harmonic mean of Precision & Recall
#     - Specificity: TN / (TN + FP)
#     """
#     y_pred_binary = (probabilities >= threshold).astype(int)
#     y_true_binary = y_true_binary.astype(int)

#     tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
#     fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
#     tn = np.sum((y_pred_binary == 0) & (y_true_binary == 0))
#     fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))

#     total = len(y_true_binary)
#     accuracy = float((tp + tn) / total) if total > 0 else 0.0

#     precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
#     recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

#     if (precision + recall) > 0:
#         f1_score = float(2 * (precision * recall) / (precision + recall))
#     else:
#         f1_score = 0.0

#     specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

#     return {
#         'threshold': float(threshold),
#         'accuracy': round(accuracy, 4),
#         'precision': round(precision, 4),
#         'recall': round(recall, 4),
#         'f1_score': round(f1_score, 4),
#         'specificity': round(specificity, 4),
#         'tp': int(tp),
#         'fp': int(fp),
#         'tn': int(tn),
#         'fn': int(fn),
#     }


# def compute_model_confidence(probabilities: np.ndarray, continuous_errors: Optional[np.ndarray] = None) -> float:
#     """
#     Evaluate Model Confidence from output certainty levels (softmax/sigmoid output distance from 0.5)
#     blended with continuous prediction error bound.
    
#     Formula:
#     Certainty = mean(2 * |P - 0.5|) in [0.0, 1.0]
#     Error Factor = max(0, 1.0 - mean_mae / 2.0)
#     Model Confidence = 0.6 * Certainty + 0.4 * Error Factor
#     """
#     if len(probabilities) == 0:
#         return 0.50

#     certainty = float(np.mean(2.0 * np.abs(probabilities - 0.5)))

#     if continuous_errors is not None and len(continuous_errors) > 0:
#         mean_err = float(np.mean(continuous_errors))
#         error_factor = max(0.0, min(1.0, 1.0 - (mean_err / 2.0)))
#         confidence = 0.6 * certainty + 0.4 * error_factor
#     else:
#         confidence = certainty

#     return round(float(np.clip(confidence, 0.0, 1.0)), 4)


# def evaluate_threshold_sweep(
#     y_true_continuous: np.ndarray,
#     probabilities: np.ndarray,
#     continuous_preds: np.ndarray,
#     ground_truth_cutoff: float = 4.0,
#     candidate_thresholds: List[float] = CANDIDATE_THRESHOLDS
# ) -> Dict[str, Any]:
#     """
#     Sweeps classification metrics across candidate thresholds {0.30, 0.40, 0.50, 0.60, 0.70}.
    
#     Returns per-threshold metric dicts, optimal threshold selection, and confidence score.
#     """
#     y_true_binary = (y_true_continuous >= ground_truth_cutoff).astype(int)
#     threshold_results = {}
#     best_threshold = 0.50
#     best_f1 = -1.0

#     for tau in candidate_thresholds:
#         metrics = compute_classification_metrics_at_threshold(
#             y_true_binary=y_true_binary,
#             probabilities=probabilities,
#             threshold=tau
#         )
#         threshold_results[f"{tau:.2f}"] = metrics

#         # Optimization criterion: select threshold with highest F1 (fallback to Accuracy)
#         if metrics['f1_score'] > best_f1:
#             best_f1 = metrics['f1_score']
#             best_threshold = tau

#     abs_errors = np.abs(y_true_continuous - continuous_preds)
#     mae = float(mean_absolute_error(y_true_continuous, continuous_preds)) if len(y_true_continuous) > 0 else 0.0
#     confidence = compute_model_confidence(probabilities, abs_errors)

#     optimal_metrics = threshold_results[f"{best_threshold:.2f}"]

#     return {
#         'optimal_threshold': best_threshold,
#         'optimal_metrics': optimal_metrics,
#         'threshold_sweep': threshold_results,
#         'mae': round(mae, 4),
#         'model_confidence': confidence,
#     }


# def evaluate_hybrid_metrics(
#     y_true: np.ndarray,
#     y_pred_lasso: np.ndarray,
#     y_pred_lstm_continuous: np.ndarray,
#     probabilities_lstm: np.ndarray,
#     ground_truth_cutoff: float = 4.0,
#     candidate_thresholds: List[float] = CANDIDATE_THRESHOLDS
# ) -> Dict[str, Any]:
#     """
#     Compute full 8-metric taxonomy for both Lasso Regression and Stacked LSTM.
    
#     Lasso Metrics:
#     - R² Score
#     - Pearson Correlation
#     - MAE
    
#     LSTM Metrics:
#     - Accuracy (at optimal threshold)
#     - Precision (at optimal threshold)
#     - F1 Score (at optimal threshold)
#     - Specificity (at optimal threshold)
#     - Model Confidence
#     - MAE
    
#     Plus full 5-threshold sweep dictionary for UI charting.
#     """
#     y_true = np.array(y_true, dtype=float)
#     y_pred_lasso = np.array(y_pred_lasso, dtype=float)
#     y_pred_lstm_continuous = np.array(y_pred_lstm_continuous, dtype=float)
#     probabilities_lstm = np.array(probabilities_lstm, dtype=float)

#     # 1. Lasso Regression Metrics
#     lasso_mae = float(mean_absolute_error(y_true, y_pred_lasso)) if len(y_true) > 0 else 0.0
#     lasso_corr = calculate_pearson_correlation(y_true, y_pred_lasso)
#     lasso_r2 = float(r2_score(y_true, y_pred_lasso)) if len(y_true) > 1 else 0.0

#     # 2. LSTM Classification & Regression Metrics + 5-Threshold Sweep
#     lstm_sweep_eval = evaluate_threshold_sweep(
#         y_true_continuous=y_true,
#         probabilities=probabilities_lstm,
#         continuous_preds=y_pred_lstm_continuous,
#         ground_truth_cutoff=ground_truth_cutoff,
#         candidate_thresholds=candidate_thresholds
#     )

#     opt_metrics = lstm_sweep_eval['optimal_metrics']

#     all_8_metrics = {
#         # Lasso specific
#         'lasso_r2': round(lasso_r2, 4),
#         'lasso_correlation': round(lasso_corr, 4),
#         'lasso_mae': round(lasso_mae, 4),
#         # LSTM specific
#         'lstm_accuracy': opt_metrics['accuracy'],
#         'lstm_precision': opt_metrics['precision'],
#         'lstm_f1_score': opt_metrics['f1_score'],
#         'lstm_specificity': opt_metrics['specificity'],
#         'lstm_model_confidence': lstm_sweep_eval['model_confidence'],
#         'lstm_mae': lstm_sweep_eval['mae'],
#         # Shared / Optimization metadata
#         'optimal_threshold': lstm_sweep_eval['optimal_threshold'],
#         'threshold_sweep': lstm_sweep_eval['threshold_sweep'],
#     }

#     return all_8_metrics


"""
Metrics Calculation Engine for Big Five Personality Prediction.

Plug-and-play evaluation service. PipelineOrchestrator calls `evaluate()`
after Lasso and LSTM produce predictions on the held-out test set; this
module has no knowledge of models, training, data loading/cleaning,
experiment identity (E1-E4), GAN, Q-learning, or BERT.

Responsibilities
----------------
- Lasso continuous OCEAN predictions : MAE, MSE, RMSE, R^2, Pearson r
- LSTM 3-class predictions           : Accuracy, Precision, Recall, F1,
                                        Specificity, confusion matrix
- Threshold analysis                 : sweep a threshold over the
                                        *continuous Lasso predictions*,
                                        binarize, compare to ground-truth
                                        class, report accuracy/precision/
                                        recall/F1 per threshold + the best
                                        by F1.

Scale assumption (see module-level docstring note below)
----------------------------------------------------------
The original CANDIDATE_THRESHOLDS = [0.30, ..., 0.70] assumed predictions
already lived on a [0, 1] probability scale (true for the old LSTM-sigmoid
threshold sweep, which is preserved below unchanged for backward
compatibility). PANDORA's OCEAN scores are NOT on that scale: sample rows
inspected in the pandora ingestion work show O/C/E/A/N as 0-100 percentile
values (e.g. 74, 96, 17, 96, 47), so Lasso's raw continuous predictions
will also be ~0-100. The new Lasso threshold sweep therefore derives its
candidate thresholds and its ground-truth cutoff FROM THE ACTUAL DATA
(percentiles of the observed values) rather than reusing the old fixed
[0.3..0.7] list. Callers can still override both explicitly.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

logger = logging.getLogger('ml_pipeline')

ArrayLike = Union[Sequence[float], np.ndarray]

# Candidate decision thresholds for the ORIGINAL [0, 1]-probability sweep
# (e.g. an LSTM sigmoid). Preserved unchanged for backward compatibility --
# do NOT reuse this constant for the new Lasso-on-raw-OCEAN-scale sweep.
CANDIDATE_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]

_DEFAULT_TRAIT_NAMES = ["O", "C", "E", "A", "N"]


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
    """Accuracy, macro Precision/Recall/F1, macro Specificity, confusion matrix for LSTM's 3-class output."""
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
    Data-driven candidate decision thresholds, replacing the old fixed
    [0.30, ..., 0.70] list which assumed a [0,1] probability scale.

    Returns `n` thresholds evenly spaced across the 30th-70th percentile
    of the observed values, so the sweep always lands inside the actual
    data range regardless of whether it's a 0-1 probability, a 1-5 Likert
    score, or a 0-100 percentile trait score.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return list(CANDIDATE_THRESHOLDS)
    pctiles = np.linspace(30, 70, n)
    return [round(float(np.percentile(values, p)), 4) for p in pctiles]


def derive_binary_ground_truth(y_true: np.ndarray, cutoff: Optional[float] = None) -> tuple:
    """
    Binarize continuous ground truth for the Lasso threshold sweep.
    Defaults to the median of the observed values (scale-agnostic) if no
    explicit cutoff is supplied, rather than a hardcoded value like the
    legacy 4.0 (which assumed a 1-5 scale).
    """
    if cutoff is None:
        cutoff = float(np.median(y_true)) if len(y_true) > 0 else 0.0
    y_true_binary = (np.asarray(y_true, dtype=float) >= cutoff).astype(int)
    return y_true_binary, cutoff


def derive_three_class_ground_truth(y_true: np.ndarray, cutoffs: Optional[List[float]] = None) -> tuple:
    """
    Bucket continuous ground truth into 3 classes (Low=0, Medium=1, High=2)
    to compare against LSTM's 3-class predictions.

    Defaults to tertile split (33rd/66th percentile of the observed values)
    if `cutoffs` isn't supplied. This is an explicit, documented,
    overridable assumption -- pass `cutoffs=[c1, c2]` to use the project's
    actual class boundaries instead.
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
    Sweep candidate thresholds over raw continuous `scores` (e.g. Lasso's
    continuous OCEAN prediction), compare the resulting binary prediction
    to `y_true_binary`, and report per-threshold accuracy/precision/
    recall/F1 plus the best threshold by F1.

    Unlike `evaluate_threshold_sweep`, this does not blend in confidence
    or MAE -- those are reported separately by the caller (evaluate()).
    """
    if candidate_thresholds is None:
        candidate_thresholds = default_threshold_candidates(scores)

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


def _as_2d(arr: ArrayLike, name: str) -> np.ndarray:
    """Coerce input to a 2D (n_samples, n_traits) float array without silently reshaping ambiguous cases."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2:
        return a
    raise ValueError(f"{name} must be 1D (n_samples,) or 2D (n_samples, n_traits); got shape {a.shape}")


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
    Single plug-and-play evaluation entry point for PipelineOrchestrator.

    Call this once per experiment (E1-E4) after Lasso and LSTM have
    produced predictions on the HELD-OUT TEST SET. This function does not
    know or care which experiment configuration produced the predictions.

    Parameters
    ----------
    y_true
        Continuous ground-truth OCEAN scores. Shape (n_samples,) for a
        single trait, or (n_samples, n_traits) for multiple (e.g. 5 for
        O/C/E/A/N).
    lasso_predictions
        Lasso's continuous predictions. Must match y_true's shape exactly
        -- no reshaping is attempted; a mismatch raises ValueError.
    lstm_predictions
        LSTM's 3-class predictions. Accepts already-decided class labels
        with shape matching y_true (1D or 2D), or a trailing probability/
        logit axis (n_samples, n_traits, n_classes), in which case argmax
        is taken. Any other shape raises ValueError.
    lstm_probabilities
        Optional. LSTM's raw probability output, used only to compute
        `model_confidence` via the single existing implementation
        (compute_model_confidence). Not required for the classification
        metrics themselves.
    lstm_true_classes
        Optional. Ground-truth class labels for the LSTM comparison, same
        shape as lstm_predictions. If omitted, ground-truth classes are
        DERIVED from `y_true` via `derive_three_class_ground_truth`
        (tertile split by default, or `class_cutoffs` if supplied) -- this
        is a documented assumption, not a silent one. Pass this explicitly
        to bypass the assumption entirely.
    trait_names
        Optional labels for each trait column, e.g. ['O','C','E','A','N'].
        Defaults to that list when n_traits == 5, else generic names.
    class_cutoffs
        Optional override for the 3-class ground-truth boundaries used
        when `lstm_true_classes` is not supplied. Either a single [c1, c2]
        list applied to every trait, or {trait_name: [c1, c2]}.
    ground_truth_cutoff
        Optional override for the binary cutoff used in the Lasso
        threshold sweep. Either a single float applied to every trait, or
        {trait_name: cutoff}. Defaults to each trait's observed median.
    candidate_thresholds
        Optional override for the threshold sweep values. Defaults to
        `default_threshold_candidates()`, which is derived from the actual
        scale of `lasso_predictions` for each trait (NOT the legacy
        [0.30..0.70] list, which assumed a [0,1] probability scale).

    Returns
    -------
    dict with keys 'lasso', 'lstm', 'threshold'.
    Single-trait input -> each key holds a flat metrics dict (matches the
    project's requested example structure exactly).
    Multi-trait input -> each key holds {'per_trait': {...}, 'aggregate': {...}}.

    Raises
    ------
    ValueError
        If any prediction array's shape is incompatible with y_true. This
        function never reshapes, truncates, or fabricates data to force a
        fit.
    """
    y_true_2d = _as_2d(y_true, "y_true")
    lasso_2d = _as_2d(lasso_predictions, "lasso_predictions")
    n_samples, n_traits = y_true_2d.shape

    if lasso_2d.shape != y_true_2d.shape:
        raise ValueError(
            f"lasso_predictions has shape {lasso_2d.shape}, expected {y_true_2d.shape} to match y_true."
        )

    lstm_2d = _resolve_class_labels(lstm_predictions, (n_samples, n_traits), "lstm_predictions")

    if lstm_true_classes is not None:
        lstm_true_2d = _resolve_class_labels(lstm_true_classes, (n_samples, n_traits), "lstm_true_classes")
        derived_cutoffs_by_trait = None
    else:
        lstm_true_2d = np.zeros_like(lstm_2d)
        derived_cutoffs_by_trait = {}

    lstm_probs_2d = None
    if lstm_probabilities is not None:
        lstm_probs_2d = _as_2d(lstm_probabilities, "lstm_probabilities")
        if lstm_probs_2d.shape != y_true_2d.shape:
            raise ValueError(
                f"lstm_probabilities has shape {lstm_probs_2d.shape}, expected {y_true_2d.shape} to match y_true."
            )

    if trait_names is None:
        trait_names = _DEFAULT_TRAIT_NAMES[:n_traits] if n_traits == 5 else [f"trait_{i}" for i in range(n_traits)]
    if len(trait_names) != n_traits:
        raise ValueError(f"trait_names has {len(trait_names)} entries, expected {n_traits}.")

    def _cutoff_for(trait: str, spec: Optional[Union[float, Dict[str, float]]]) -> Optional[float]:
        if spec is None:
            return None
        if isinstance(spec, dict):
            return spec.get(trait)
        return spec

    def _class_cutoffs_for(trait: str) -> Optional[List[float]]:
        if class_cutoffs is None:
            return None
        if isinstance(class_cutoffs, dict):
            return class_cutoffs.get(trait)
        return list(class_cutoffs)

    per_trait_lasso: Dict[str, Dict[str, float]] = {}
    per_trait_lstm: Dict[str, Dict[str, Any]] = {}
    per_trait_threshold: Dict[str, Dict[str, Any]] = {}

    for i, trait in enumerate(trait_names):
        yt = y_true_2d[:, i]
        lp = lasso_2d[:, i]

        # --- Lasso regression metrics ---
        per_trait_lasso[trait] = compute_regression_metrics(yt, lp)

        # --- LSTM 3-class metrics ---
        if lstm_true_classes is None:
            trait_class_cutoffs = _class_cutoffs_for(trait)
            classes, used_cutoffs = derive_three_class_ground_truth(yt, trait_class_cutoffs)
            lstm_true_2d[:, i] = classes
            derived_cutoffs_by_trait[trait] = used_cutoffs
        per_trait_lstm[trait] = compute_multiclass_metrics(lstm_true_2d[:, i], lstm_2d[:, i])
        if lstm_probs_2d is not None:
            per_trait_lstm[trait]['model_confidence'] = compute_model_confidence(
                lstm_probs_2d[:, i], np.abs(yt - lp)
            )

        # --- Threshold sweep on Lasso's continuous predictions ---
        cutoff = _cutoff_for(trait, ground_truth_cutoff)
        y_true_binary, used_cutoff = derive_binary_ground_truth(yt, cutoff)
        sweep = sweep_thresholds_on_scores(y_true_binary, lp, candidate_thresholds)
        sweep['ground_truth_cutoff'] = used_cutoff
        per_trait_threshold[trait] = sweep

    def _aggregate(per_trait: Dict[str, Dict[str, Any]], keys: List[str]) -> Dict[str, float]:
        return {k: round(float(np.mean([per_trait[t][k] for t in trait_names])), 4) for k in keys}

    lasso_agg = _aggregate(per_trait_lasso, ['mae', 'mse', 'rmse', 'r2', 'correlation'])
    lstm_agg = _aggregate(per_trait_lstm, ['accuracy', 'precision', 'recall', 'f1', 'specificity'])
    threshold_agg = _aggregate(per_trait_threshold, ['best_f1', 'accuracy', 'precision', 'recall', 'f1'])

    if n_traits == 1:
        trait = trait_names[0]
        result = {
            'lasso': per_trait_lasso[trait],
            'lstm': per_trait_lstm[trait],
            'threshold': per_trait_threshold[trait],
        }
    else:
        result = {
            'lasso': {'per_trait': per_trait_lasso, 'aggregate': lasso_agg},
            'lstm': {'per_trait': per_trait_lstm, 'aggregate': lstm_agg},
            'threshold': {'per_trait': per_trait_threshold, 'aggregate': threshold_agg},
        }

    if derived_cutoffs_by_trait is not None:
        result['assumptions'] = {
            'lstm_ground_truth_classes': 'derived from y_true (tertile split unless class_cutoffs given)',
            'class_cutoffs_used': derived_cutoffs_by_trait,
        }

    return result