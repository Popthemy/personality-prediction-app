"""
Unit & Integration Tests for Hybrid LSTM-Lasso Architecture & 8-Metric Framework.
"""
import numpy as np
import pytest
from django.test import TestCase
import torch

from backend.ml_pipeline.services.lstm_classifier import StackedLSTMClassifier, LSTMTrainer
from backend.ml_pipeline.services.metrics_engine import (
    evaluate_hybrid_metrics,
    calculate_binary_classification_metrics,
    calculate_model_confidence,
    CANDIDATE_THRESHOLDS
)


class TestMetricsEngine(TestCase):
    """Test 8-Metric Framework & 5-Threshold Sweep calculation."""

    def test_candidate_thresholds(self):
        self.assertEqual(CANDIDATE_THRESHOLDS, [0.30, 0.40, 0.50, 0.60, 0.70])

    def test_binary_metrics_perfect(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        res = calculate_binary_classification_metrics(y_true, y_prob, threshold=0.50)
        self.assertEqual(res['accuracy'], 1.0)
        self.assertEqual(res['precision'], 1.0)
        self.assertEqual(res['f1_score'], 1.0)
        self.assertEqual(res['specificity'], 1.0)

    def test_evaluate_hybrid_metrics(self):
        y_true = np.array([4.2, 2.1, 4.8, 1.9, 3.5])
        y_pred_lasso = np.array([4.0, 2.3, 4.5, 2.1, 3.4])
        y_pred_lstm = np.array([4.1, 2.0, 4.6, 2.0, 3.6])
        probs_lstm = np.array([0.85, 0.15, 0.90, 0.10, 0.55])

        metrics = evaluate_hybrid_metrics(
            y_true=y_true,
            y_pred_lasso=y_pred_lasso,
            y_pred_lstm_continuous=y_pred_lstm,
            probabilities_lstm=probs_lstm,
            ground_truth_cutoff=4.0,
            candidate_thresholds=CANDIDATE_THRESHOLDS
        )

        # Check all 8 metrics exist
        self.assertIn('lasso_r2', metrics)
        self.assertIn('lasso_correlation', metrics)
        self.assertIn('lasso_mae', metrics)
        self.assertIn('lstm_accuracy', metrics)
        self.assertIn('lstm_precision', metrics)
        self.assertIn('lstm_f1_score', metrics)
        self.assertIn('lstm_specificity', metrics)
        self.assertIn('lstm_model_confidence', metrics)

        # Check 5-threshold sweep output
        self.assertIn('threshold_sweep', metrics)
        self.assertEqual(len(metrics['threshold_sweep']), 5)
        for t in ['0.3', '0.4', '0.5', '0.6', '0.7']:
            self.assertIn(t, metrics['threshold_sweep'])


class TestStackedLSTMClassifier(TestCase):
    """Test PyTorch Stacked LSTM Neural Network Architecture."""

    def test_model_forward_pass(self):
        model = StackedLSTMClassifier(input_dim=768, hidden_dim=64, num_layers=2)
        batch_size = 4
        seq_len = 10
        dummy_input = torch.randn(batch_size, seq_len, 768)
        probs, reg_out = model(dummy_input)

        self.assertEqual(probs.shape, (batch_size, 1))
        self.assertEqual(reg_out.shape, (batch_size, 1))
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))
        self.assertTrue(torch.all(reg_out >= 1.0) and torch.all(reg_out <= 5.0))

    def test_lstm_trainer_training(self):
        trainer = LSTMTrainer(hidden_dim=32, num_layers=1, dropout=0.0, learning_rate=1e-3)
        # Create synthetic sequence data
        seqs = [np.random.randn(5, 768) for _ in range(6)]
        targets = np.array([4.5, 2.0, 4.2, 1.8, 3.9, 2.5])

        trainer.train_trait_model(
            trait="Openness",
            sequences=seqs,
            targets=targets,
            threshold=4.0,
            epochs=2,
            batch_size=2
        )

        probs, preds = trainer.predict_trait("Openness", seqs[:2])
        self.assertEqual(len(probs), 2)
        self.assertEqual(len(preds), 2)
