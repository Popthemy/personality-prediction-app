"""PANDORA training service for the shared cohort model.

This service is the PANDORA-first training path used by the existing
``Train`` action. It:

1. Discovers the cloned PANDORA parquet shards automatically.
2. Converts the row-level snapshot into cleaned training records.
3. Splits the data into train / validation / test rows.
4. Encodes text with BERT.
5. Optionally trains a real GAN on the training embeddings only.
6. Trains the Lasso regression heads and LSTM classification heads.
7. Persists the resulting shared cohort artifact for later X-handle
   prediction runs.

The PANDORA export used in this project is row-level text data, so the
trait tuple is treated as a label vector, not as identity.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from backend.core.models import COHORT_MODEL, LASSO_MODEL, VOLUNTEER, User
from backend.ml_pipeline.services.augmentation.gan import GANAugmenter
from backend.ml_pipeline.services.bert_encoder import BERTEncoder
from backend.ml_pipeline.services.data.pandora import PandoraRecord, load_pandora_records
from backend.ml_pipeline.services.lasso_regressor import (
    LassoTrainer,
    denormalize_predictions,
)
from backend.ml_pipeline.services.lstm_classifier import LSTMTrainer
from backend.ml_pipeline.services.metrics_engine import evaluate

logger = logging.getLogger("ml_pipeline")


TRAIT_NAMES = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]


def _score_to_bfi_scale(score: float) -> float:
    """Convert a PANDORA percentile-style score into the 1-5 project scale."""
    if score is None:
        return 0.0
    score = float(score)
    if score > 5.0:
        return float(np.clip(1.0 + score / 25.0, 1.0, 5.0))
    return float(np.clip(score, 1.0, 5.0))


def _record_to_target_vector(record: PandoraRecord) -> np.ndarray:
    return np.array(
        [
            _score_to_bfi_scale(record.traits.O),
            _score_to_bfi_scale(record.traits.C),
            _score_to_bfi_scale(record.traits.E),
            _score_to_bfi_scale(record.traits.A),
            _score_to_bfi_scale(record.traits.N),
        ],
        dtype=np.float32,
    )


def _stack_embeddings(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray(vectors, dtype=np.float32)


class PandoraTrainingService:
    """Train the shared cohort model from the cloned PANDORA dataset."""

    def __init__(
        self,
        researcher: User,
        *,
        volunteer: VOLUNTEER | None = None,
        split_seed: int = 42,
        use_gan: bool = True,
        validation_size: float = 0.15,
        test_size: float = 0.15,
        status_callback: Optional[Callable[[str, int, str], None]] = None,
    ) -> None:
        self.researcher = researcher
        self.volunteer = volunteer
        self.split_seed = split_seed
        self.use_gan = use_gan
        self.validation_size = validation_size
        self.test_size = test_size
        self.status_callback = status_callback
        logger_name = getattr(researcher, "username", None) or getattr(researcher, "email", None) or f"user-{researcher.id}"
        self.logger = logging.getLogger(f"ml_pipeline.pandora.{logger_name}")

    def _report(self, stage: str, progress: int, message: str) -> None:
        self.logger.info("[%s] %s", stage, message)
        if self.status_callback is not None:
            self.status_callback(stage, progress, message)

    def _load_records(self) -> list[PandoraRecord]:
        records = load_pandora_records()
        if not records:
            raise ValueError("No PANDORA records were found in the cloned repository.")
        return records

    def _split_records(self, records: list[PandoraRecord]):
        labels = np.asarray([record.traits.ptype for record in records], dtype=int)
        indices = np.arange(len(records))

        stratify_labels = labels if len(np.unique(labels)) > 1 else None
        try:
            train_idx, test_idx = train_test_split(
                indices,
                test_size=self.test_size,
                random_state=self.split_seed,
                shuffle=True,
                stratify=stratify_labels,
            )
        except ValueError:
            train_idx, test_idx = train_test_split(
                indices,
                test_size=self.test_size,
                random_state=self.split_seed,
                shuffle=True,
                stratify=None,
            )

        train_labels = labels[train_idx]
        val_fraction_of_train = self.validation_size / max(1.0 - self.test_size, 1e-8)

        if len(train_idx) >= 3:
            stratify_train = train_labels if len(np.unique(train_labels)) > 1 else None
            try:
                train_idx, val_idx = train_test_split(
                    train_idx,
                    test_size=val_fraction_of_train,
                    random_state=self.split_seed,
                    shuffle=True,
                    stratify=stratify_train,
                )
            except ValueError:
                train_idx, val_idx = train_test_split(
                    train_idx,
                    test_size=val_fraction_of_train,
                    random_state=self.split_seed,
                    shuffle=True,
                    stratify=None,
                )
        else:
            val_idx = np.array([], dtype=int)

        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]
        test_records = [records[i] for i in test_idx]

        return train_records, val_records, test_records

    def _encode_records(self, records: Sequence[PandoraRecord]) -> np.ndarray:
        encoder = BERTEncoder()
        texts = [record.text for record in records]
        embeddings = []
        total = max(len(texts), 1)
        for index, text in enumerate(texts, start=1):
            result = encoder.encode_text(text)
            embeddings.append(result["embedding"])
            if index == 1 or index == total or index % max(1, total // 4) == 0:
                pct = 20 + int(20 * (index / total))
                self._report(
                    "bert_encoding",
                    min(pct, 39),
                    f"BERT encoded {index}/{total} PANDORA rows",
                )
        return _stack_embeddings(embeddings)

    def _augment_with_gan(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if not self.use_gan or len(X_train) < 2:
            return X_train, y_train, 0

        augmenter = GANAugmenter(
            embedding_dim=X_train.shape[1],
            epochs=50,
            batch_size=min(32, len(X_train)),
            learning_rate=2e-4,
            seed=self.split_seed,
        )
        augmenter.fit(X_train)
        synthetic_X, _, _ = augmenter.generate(n_samples=len(X_train))

        # Assign each synthetic embedding the label of its nearest real
        # training neighbour. This keeps the supervised heads aligned while
        # still letting the GAN expand the representation space.
        nn = NearestNeighbors(n_neighbors=1, metric="cosine")
        nn.fit(X_train)
        _, nearest_idx = nn.kneighbors(synthetic_X)
        synthetic_y = y_train[nearest_idx[:, 0]]

        X_aug = np.vstack([X_train, synthetic_X])
        y_aug = np.vstack([y_train, synthetic_y])
        return X_aug, y_aug, int(len(synthetic_X))

    def _build_sequence_data(self, X: np.ndarray) -> list[np.ndarray]:
        return [row.reshape(1, -1) for row in X]

    def train(self) -> Dict[str, Any]:
        records = self._load_records()
        self._report("pandora_loading", 5, f"Loaded {len(records)} cleaned PANDORA rows")
        train_records, val_records, test_records = self._split_records(records)

        self._report(
            "data_split",
            10,
            f"Split PANDORA into {len(train_records)} train, {len(val_records)} validation, {len(test_records)} test rows",
        )

        self.logger.info(
            "PANDORA split: train=%d val=%d test=%d",
            len(train_records),
            len(val_records),
            len(test_records),
        )

        X_train = self._encode_records(train_records)
        X_val = self._encode_records(val_records) if val_records else np.empty((0, X_train.shape[1]), dtype=np.float32)
        X_test = self._encode_records(test_records) if test_records else np.empty((0, X_train.shape[1]), dtype=np.float32)

        self._report("bert_ready", 40, "BERT encoding completed and embeddings are ready")

        raw_targets_train = np.vstack([_record_to_target_vector(record) for record in train_records])
        raw_targets_val = np.vstack([_record_to_target_vector(record) for record in val_records]) if val_records else np.empty((0, 5), dtype=np.float32)
        raw_targets_test = np.vstack([_record_to_target_vector(record) for record in test_records]) if test_records else np.empty((0, 5), dtype=np.float32)

        lasso_trainer = LassoTrainer()
        lasso_trainer._fit_feature_scaler(X_train)

        X_train_scaled = lasso_trainer.transform_features(X_train)
        X_val_scaled = lasso_trainer.transform_features(X_val) if len(X_val) else X_val
        X_test_scaled = lasso_trainer.transform_features(X_test) if len(X_test) else X_test

        X_train_aug = X_train_scaled
        y_train_aug = raw_targets_train
        synthetic_count = 0
        if self.use_gan:
            self._report("gan_training", 45, "Starting GAN training on the training embeddings only")
            X_train_aug, y_train_aug, synthetic_count = self._augment_with_gan(
                X_train_scaled,
                raw_targets_train,
            )
            self._report(
                "gan_training",
                55,
                f"GAN augmentation completed and produced {synthetic_count} synthetic embeddings",
            )
        else:
            self._report("gan_skipped", 45, "GAN augmentation disabled for this run")

        lstm_trainer = LSTMTrainer()
        trait_artifacts: dict[str, dict[str, Any]] = {}
        model_metrics: dict[str, Any] = {}
        chart_payload: dict[str, Any] = {}

        for trait_index, trait_name in enumerate(TRAIT_NAMES):
            self._report("model_training", 60, f"Training Lasso and LSTM heads for {trait_name}")
            y_train_trait = y_train_aug[:, trait_index]
            y_train_norm = (y_train_trait - 1.0) / 4.0

            y_val_trait = raw_targets_val[:, trait_index] if len(raw_targets_val) else np.array([])
            y_val_norm = (y_val_trait - 1.0) / 4.0 if len(y_val_trait) else None

            y_test_trait = raw_targets_test[:, trait_index] if len(raw_targets_test) else np.array([])
            y_test_norm = (y_test_trait - 1.0) / 4.0 if len(y_test_trait) else None

            lasso_metrics = lasso_trainer.train_trait_model(
                X_train_aug,
                y_train_norm,
                trait_name,
                validate_X=X_val_scaled if len(X_val_scaled) else None,
                validate_y=y_val_norm if y_val_norm is not None and len(y_val_norm) else None,
            )

            lasso_pred_norm = lasso_trainer.predict_trait(trait_name, X_test_scaled) if len(X_test_scaled) else np.array([])
            lasso_pred = denormalize_predictions(lasso_pred_norm) if len(lasso_pred_norm) else np.array([])

            sequence_train = self._build_sequence_data(X_train_aug)
            sequence_val = self._build_sequence_data(X_val_scaled) if len(X_val_scaled) else []
            sequence_test = self._build_sequence_data(X_test_scaled) if len(X_test_scaled) else []

            lstm_trainer.train_trait_model(
                trait_name,
                sequence_train,
                y_train_trait,
                val_sequences=sequence_val if sequence_val else None,
                val_targets=y_val_trait if len(y_val_trait) else None,
                epochs=25,
                batch_size=min(16, max(1, len(sequence_train))),
            )

            lstm_probs = np.array([])
            lstm_preds = np.array([])
            if sequence_test:
                class_probs, predicted_classes, _ = lstm_trainer.predict_trait(
                    trait_name,
                    sequence_test,
                )
                lstm_probs = np.max(class_probs, axis=1)
                lstm_preds = predicted_classes

            trait_eval = evaluate(
                y_true=y_test_trait if len(y_test_trait) else y_train_trait,
                lasso_predictions=lasso_pred if len(lasso_pred) else denormalize_predictions(lasso_trainer.predict_trait(trait_name, X_train_aug)),
                lstm_predictions=lstm_preds if len(lstm_preds) else np.zeros(len(y_train_trait), dtype=int),
                lstm_probabilities=lstm_probs if len(lstm_probs) else None,
                trait_names=[trait_name],
                class_cutoffs=[2.5, 3.5],
                ground_truth_cutoff=4.0,
            )

            threshold_results = trait_eval["threshold"]["results"] if "results" in trait_eval["threshold"] else []
            threshold_chart = {
                "thresholds": [row["threshold"] for row in threshold_results],
                "accuracy": [row["accuracy"] for row in threshold_results],
                "precision": [row["precision"] for row in threshold_results],
                "recall": [row["recall"] for row in threshold_results],
                "f1": [row["f1_score"] for row in threshold_results],
                "specificity": [row["specificity"] for row in threshold_results],
                "best_threshold": trait_eval["threshold"].get("best_threshold"),
                "best_f1": trait_eval["threshold"].get("best_f1"),
            }

            scatter_limit = min(200, len(y_test_trait))
            scatter_indices = np.linspace(0, max(len(y_test_trait) - 1, 0), scatter_limit, dtype=int) if scatter_limit else np.array([], dtype=int)
            regression_scatter = {
                "truth": [float(y_test_trait[i]) for i in scatter_indices] if len(y_test_trait) else [],
                "prediction": [float(lasso_pred[i]) for i in scatter_indices] if len(lasso_pred) else [],
            }

            model, _ = LASSO_MODEL.objects.update_or_create(
                volunteer=self.volunteer,
                trait=trait_name.lower(),
                defaults={
                    "alpha": lasso_trainer.alpha,
                    "coefficients": lasso_trainer.get_all_coefficients(trait_name),
                    "intercept": float(lasso_trainer.models[trait_name].intercept_),
                    "train_mae": lasso_metrics.get("train_mae"),
                    "train_rmse": lasso_metrics.get("train_rmse"),
                    "train_r2": lasso_metrics.get("train_r2"),
                    "validation_mae": lasso_metrics.get("validation_mae"),
                    "validation_rmse": lasso_metrics.get("validation_rmse"),
                    "validation_r2": lasso_metrics.get("validation_r2"),
                    "classification_accuracy": trait_eval["lstm"]["accuracy"],
                    "classification_precision": trait_eval["lstm"]["precision"],
                    "classification_f1_score": trait_eval["lstm"]["f1"],
                    "classification_specificity": trait_eval["lstm"]["specificity"],
                    "training_samples_used": int(len(X_train)),
                    "synthetic_samples_used": int(synthetic_count),
                },
            )
            model.save()

            trait_artifacts[trait_name] = {
                "lasso": lasso_metrics,
                "evaluation": trait_eval,
                "chart_payload": {
                    "regression_scatter": regression_scatter,
                    "threshold_sweep": threshold_chart,
                    "confusion_matrix": {
                        "labels": trait_eval["lstm"].get("confusion_matrix_labels", ["Low", "Medium", "High"]),
                        "matrix": trait_eval["lstm"].get("confusion_matrix", []),
                    },
                },
            }
            model_metrics[trait_name] = trait_eval
            chart_payload[trait_name] = trait_artifacts[trait_name]["chart_payload"]
            self._report("model_training", 75, f"Finished evaluation for {trait_name}")

        trainer_state = lasso_trainer.save_state()
        self._report("saving_model", 90, "Saving trained cohort artifact and metrics")

        cohort_model, _ = COHORT_MODEL.objects.update_or_create(
            name="pandora-cohort-model",
            defaults={
                "version": f"pandora-split-{self.split_seed}",
                "is_active": True,
                "split_seed": self.split_seed,
                "train_ratio": round(len(train_records) / max(len(records), 1), 4),
                "validation_ratio": round(len(val_records) / max(len(records), 1), 4),
                "train_volunteer_ids": [record.sample_id for record in train_records],
                "validation_volunteer_ids": [record.sample_id for record in val_records],
                "train_handles": [record.source_file for record in train_records],
                "validation_handles": [record.source_file for record in val_records],
                "trainer_state": trainer_state,
                "metrics": {
                    "source": "pandora",
                    "use_gan": self.use_gan,
                    "synthetic_samples_generated": synthetic_count,
                    "train_count": len(train_records),
                    "validation_count": len(val_records),
                    "test_count": len(test_records),
                    "per_trait": model_metrics,
                    "chart_payload": chart_payload,
                },
            },
        )

        self._report("completed", 100, "Training complete. The model and chart data are ready.")

        return {
            "status": "success",
            "model_id": cohort_model.id,
            "model_name": cohort_model.name,
            "model_version": cohort_model.version,
            "train_count": len(train_records),
            "validation_count": len(val_records),
            "test_count": len(test_records),
            "synthetic_samples_generated": synthetic_count,
            "trainer_state": trainer_state,
            "metrics": cohort_model.metrics,
            "chart_payload": chart_payload,
            "records_preview": [asdict(record.traits) for record in train_records[:5]],
        }
