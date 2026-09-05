"""Tests for the PANDORA binary LSTM classification path."""

import numpy as np
import torch
from django.test import TestCase

from backend.ml_pipeline.services.lstm_classifier import (
    NUM_TRAITS,
    LSTMTrainer,
    StackedLSTMClassifier,
)
from backend.ml_pipeline.services.metrics_engine import (
    CANDIDATE_THRESHOLDS,
    compute_classification_metrics_at_threshold,
    evaluate_lstm_binary_classifier,
    evaluate_lstm_binary_with_thresholds,
)


class TestBinaryMetricsEngine(TestCase):
    def test_candidate_thresholds(self):
        self.assertEqual(CANDIDATE_THRESHOLDS, [0.30, 0.40, 0.50, 0.60, 0.70])

    def test_binary_metrics_perfect(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        res = compute_classification_metrics_at_threshold(y_true, y_prob, threshold=0.50)
        self.assertEqual(res["accuracy"], 1.0)
        self.assertEqual(res["precision"], 1.0)
        self.assertEqual(res["f1_score"], 1.0)
        self.assertEqual(res["specificity"], 1.0)

    def test_validation_thresholds_are_applied_to_test(self):
        y_val = np.array([
            [0.8, 0.2, 0.9, 0.1, 0.7],
            [0.7, 0.3, 0.8, 0.2, 0.6],
            [0.2, 0.8, 0.1, 0.9, 0.3],
            [0.3, 0.7, 0.2, 0.8, 0.4],
        ])
        p_val = np.array([
            [0.9, 0.2, 0.8, 0.1, 0.7],
            [0.7, 0.4, 0.9, 0.3, 0.6],
            [0.1, 0.8, 0.2, 0.9, 0.2],
            [0.3, 0.7, 0.1, 0.8, 0.4],
        ])

        selected = evaluate_lstm_binary_classifier(y_val, p_val)
        self.assertEqual(selected["split"], "validation")
        self.assertEqual(len(selected["per_trait"]["O"]["threshold_sweep"]), 5)

        scored = evaluate_lstm_binary_with_thresholds(y_val, p_val, selected)
        self.assertEqual(scored["split"], "test")
        self.assertIn("accuracy", scored["aggregate"])
        self.assertIn("f1", scored["aggregate"])
        self.assertIn("specificity", scored["aggregate"])
        self.assertEqual(
            scored["per_trait"]["O"]["selected_threshold"],
            selected["per_trait"]["O"]["best_threshold"],
        )


class TestStackedLSTMClassifier(TestCase):
    def test_model_forward_pass_outputs_five_logits(self):
        model = StackedLSTMClassifier(input_dim=16, hidden_dim=8, num_layers=1, dropout=0.0)
        dummy_input = torch.randn(4, 3, 16)
        logits = model(dummy_input)
        self.assertEqual(logits.shape, (4, NUM_TRAITS))

    def test_model_forward_pass_accepts_lasso_auxiliary_features(self):
        model = StackedLSTMClassifier(
            input_dim=16,
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
            auxiliary_dim=NUM_TRAITS,
        )
        dummy_input = torch.randn(4, 3, 16)
        lasso_hints = torch.rand(4, NUM_TRAITS)
        logits = model(dummy_input, auxiliary_features=lasso_hints)
        self.assertEqual(logits.shape, (4, NUM_TRAITS))

    def test_lstm_trainer_training_and_probabilities(self):
        trainer = LSTMTrainer(hidden_dim=8, num_layers=1, dropout=0.0, learning_rate=1e-3)
        seqs = [np.random.randn(3, 16).astype(np.float32) for _ in range(8)]
        targets = np.array([
            [0.8, 0.2, 0.7, 0.3, 0.9],
            [0.7, 0.3, 0.8, 0.2, 0.6],
            [0.2, 0.8, 0.1, 0.9, 0.4],
            [0.3, 0.7, 0.2, 0.8, 0.1],
            [0.9, 0.1, 0.6, 0.4, 0.8],
            [0.1, 0.9, 0.4, 0.6, 0.2],
            [0.6, 0.4, 0.9, 0.1, 0.7],
            [0.4, 0.6, 0.3, 0.7, 0.3],
        ], dtype=np.float32)

        history = trainer.train(
            sequences=seqs[:6],
            targets=targets[:6],
            val_sequences=seqs[6:],
            val_targets=targets[6:],
            epochs=1,
            batch_size=2,
            seed=123,
        )
        probs = trainer.predict_proba(seqs[6:])
        labels = trainer.predict_labels(seqs[6:], thresholds=[0.5] * NUM_TRAITS)

        self.assertEqual(probs.shape, (2, NUM_TRAITS))
        self.assertEqual(labels.shape, (2, NUM_TRAITS))
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))
        self.assertEqual(history["sample_count"], 6)

    def test_lstm_trainer_training_with_lasso_auxiliary_features(self):
        trainer = LSTMTrainer(
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
            learning_rate=1e-3,
            auxiliary_dim=NUM_TRAITS,
        )
        seqs = [np.random.randn(3, 16).astype(np.float32) for _ in range(8)]
        targets = np.array([
            [0.8, 0.2, 0.7, 0.3, 0.9],
            [0.7, 0.3, 0.8, 0.2, 0.6],
            [0.2, 0.8, 0.1, 0.9, 0.4],
            [0.3, 0.7, 0.2, 0.8, 0.1],
            [0.9, 0.1, 0.6, 0.4, 0.8],
            [0.1, 0.9, 0.4, 0.6, 0.2],
            [0.6, 0.4, 0.9, 0.1, 0.7],
            [0.4, 0.6, 0.3, 0.7, 0.3],
        ], dtype=np.float32)
        aux = np.clip(targets + np.random.normal(0.0, 0.05, targets.shape), 0.0, 1.0).astype(np.float32)

        history = trainer.train(
            sequences=seqs[:6],
            targets=targets[:6],
            val_sequences=seqs[6:],
            val_targets=targets[6:],
            auxiliary_features=aux[:6],
            val_auxiliary_features=aux[6:],
            epochs=1,
            batch_size=2,
            seed=123,
        )
        probs = trainer.predict_proba(seqs[6:], auxiliary_features=aux[6:])

        self.assertEqual(probs.shape, (2, NUM_TRAITS))
        self.assertEqual(history["sample_count"], 6)
