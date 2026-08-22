"""
Metrics Calculation Engine for Big Five Personality Prediction.

Provides dynamic 5-threshold evaluation across candidate decision thresholds:
T = {0.30, 0.40, 0.50, 0.60, 0.70}

Tracks exact model performance taxonomy:
- Lasso Regression Metrics: R² Score, Pearson Correlation, MAE
- LSTM Classification Metrics: Accuracy, Precision, F1 Score, Specificity, Model Confidence, MAE
- Dynamic threshold optimization for classification performance tuning.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger('ml_pipeline')

# Candidate decision thresholds specified by project requirements
CANDIDATE_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


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
        f1_score = float(2 * (precision * recall) / (precision + recall))
    else:
        f1_score = 0.0

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        'threshold': float(threshold),
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1_score, 4),
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
    Sweeps classification metrics across candidate thresholds {0.30, 0.40, 0.50, 0.60, 0.70}.
    
    Returns per-threshold metric dicts, optimal threshold selection, and confidence score.
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

        # Optimization criterion: select threshold with highest F1 (fallback to Accuracy)
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
    Compute full 8-metric taxonomy for both Lasso Regression and Stacked LSTM.
    
    Lasso Metrics:
    - R² Score
    - Pearson Correlation
    - MAE
    
    LSTM Metrics:
    - Accuracy (at optimal threshold)
    - Precision (at optimal threshold)
    - F1 Score (at optimal threshold)
    - Specificity (at optimal threshold)
    - Model Confidence
    - MAE
    
    Plus full 5-threshold sweep dictionary for UI charting.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred_lasso = np.array(y_pred_lasso, dtype=float)
    y_pred_lstm_continuous = np.array(y_pred_lstm_continuous, dtype=float)
    probabilities_lstm = np.array(probabilities_lstm, dtype=float)

    # 1. Lasso Regression Metrics
    lasso_mae = float(mean_absolute_error(y_true, y_pred_lasso)) if len(y_true) > 0 else 0.0
    lasso_corr = calculate_pearson_correlation(y_true, y_pred_lasso)
    lasso_r2 = float(r2_score(y_true, y_pred_lasso)) if len(y_true) > 1 else 0.0

    # 2. LSTM Classification & Regression Metrics + 5-Threshold Sweep
    lstm_sweep_eval = evaluate_threshold_sweep(
        y_true_continuous=y_true,
        probabilities=probabilities_lstm,
        continuous_preds=y_pred_lstm_continuous,
        ground_truth_cutoff=ground_truth_cutoff,
        candidate_thresholds=candidate_thresholds
    )

    opt_metrics = lstm_sweep_eval['optimal_metrics']

    all_8_metrics = {
        # Lasso specific
        'lasso_r2': round(lasso_r2, 4),
        'lasso_correlation': round(lasso_corr, 4),
        'lasso_mae': round(lasso_mae, 4),
        # LSTM specific
        'lstm_accuracy': opt_metrics['accuracy'],
        'lstm_precision': opt_metrics['precision'],
        'lstm_f1_score': opt_metrics['f1_score'],
        'lstm_specificity': opt_metrics['specificity'],
        'lstm_model_confidence': lstm_sweep_eval['model_confidence'],
        'lstm_mae': lstm_sweep_eval['mae'],
        # Shared / Optimization metadata
        'optimal_threshold': lstm_sweep_eval['optimal_threshold'],
        'threshold_sweep': lstm_sweep_eval['threshold_sweep'],
    }

    return all_8_metrics
