"""
Lasso / ElasticNet regression service for personality trait prediction.

Trains one model per OCEAN trait and keeps feature scaling metadata so
predictions remain reproducible across runs.
"""
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger('ml_pipeline')


class LassoTrainer:
    """Regression trainer for OCEAN traits."""

    def __init__(
        self,
        alpha: float = 0.001,
        max_iter: int = 10000,
        regularization: str = 'lasso',
        l1_ratio: float = 0.5,
        trait_alphas: Optional[Dict[str, float]] = None,
        trait_l1_ratios: Optional[Dict[str, float]] = None,
    ):
        self.alpha = alpha
        self.max_iter = max_iter
        self.regularization = regularization.lower().strip()
        self.l1_ratio = l1_ratio
        self.trait_alphas = trait_alphas or {}
        self.trait_l1_ratios = trait_l1_ratios or {}
        self.models = {}
        self.model_metadata = {}
        self.feature_mean = None
        self.feature_scale = None

    def _fit_feature_scaler(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and apply feature standardization on training embeddings."""
        self.feature_mean = embeddings.mean(axis=0)
        self.feature_scale = embeddings.std(axis=0) + 1e-8
        return (embeddings - self.feature_mean) / self.feature_scale

    def transform_features(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings using the fitted training scaler."""
        if self.feature_mean is None or self.feature_scale is None:
            raise ValueError("Feature scaler has not been fitted")
        return (embeddings - self.feature_mean) / self.feature_scale

    def prepare_training_data(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features and normalize labels to 0-1."""
        if len(embeddings) != len(labels):
            raise ValueError("Embeddings and labels must have the same length")

        X = self._fit_feature_scaler(embeddings)
        y = (labels - 1.0) / 4.0
        return X, y

    def get_feature_metadata(self) -> Dict[str, List[float]]:
        """Return fitted feature scaling metadata for persistence."""
        if self.feature_mean is None or self.feature_scale is None:
            return {}

        return {
            'feature_mean': self.feature_mean.tolist(),
            'feature_scale': self.feature_scale.tolist(),
            'regularization': self.regularization,
            'default_alpha': float(self.alpha),
            'default_l1_ratio': float(self.l1_ratio),
        }

    def _resolve_alpha(self, trait: str) -> float:
        return float(self.trait_alphas.get(trait, self.alpha))

    def _resolve_l1_ratio(self, trait: str) -> float:
        return float(self.trait_l1_ratios.get(trait, self.l1_ratio))

    def _create_model(self, trait: str, n_samples: int):
        alpha = self._resolve_alpha(trait)
        l1_ratio = self._resolve_l1_ratio(trait)

        if self.regularization == 'elasticnet':
            if n_samples >= 5:
                return ElasticNetCV(
                    alphas=np.logspace(-4, 0, 25),
                    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                    cv=min(5, n_samples),
                    max_iter=self.max_iter,
                    random_state=42,
                )
            return ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=self.max_iter,
                random_state=42,
            )

        if n_samples >= 5:
            return LassoCV(
                alphas=np.logspace(-4, 0, 25),
                cv=min(5, n_samples),
                max_iter=self.max_iter,
                random_state=42,
            )
        return Lasso(alpha=alpha, max_iter=self.max_iter, random_state=42)

    def _fit_model(self, model, X: np.ndarray, y: np.ndarray, sample_weight=None):
        try:
            if sample_weight is not None:
                model.fit(X, y, sample_weight=sample_weight)
            else:
                model.fit(X, y)
        except TypeError:
            model.fit(X, y)
        return model

    def train_trait_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        trait: str,
        validate_X: np.ndarray = None,
        validate_y: np.ndarray = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Train a single trait model.

        Returns a metrics dict and stores the fitted model on the trainer.
        """
        logger.info(f"Training regression model for {trait}...")

        if len(X) != len(y):
            raise ValueError("Feature and label arrays must have the same length")

        model = self._create_model(trait, len(X))
        model = self._fit_model(model, X, y, sample_weight=sample_weight)

        if isinstance(model, (LassoCV, ElasticNetCV)):
            self.alpha = float(model.alpha_)

        self.models[trait] = model
        self.model_metadata[trait] = {
            'regularization': self.regularization,
            'alpha': float(self._resolve_alpha(trait)),
            'l1_ratio': float(self._resolve_l1_ratio(trait)),
            'sample_weight_used': bool(sample_weight is not None),
            'training_samples': int(len(X)),
        }

        train_pred = model.predict(X)
        train_mae = mean_absolute_error(y, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y, train_pred))
        train_r2 = r2_score(y, train_pred) if len(y) > 1 else 0.0

        metrics = {
            'trait': trait,
            'regularization': self.regularization,
            'alpha': float(self.alpha),
            'l1_ratio': float(self._resolve_l1_ratio(trait)),
            'train_mae': float(train_mae),
            'train_rmse': float(train_rmse),
            'train_r2': float(train_r2),
            'training_samples': int(len(X)),
            'sparse_features': int(np.sum(model.coef_ != 0)),
            'total_features': int(len(model.coef_)),
            'sample_weight_used': bool(sample_weight is not None),
        }

        if validate_X is not None and validate_y is not None and len(validate_X) > 0:
            validate_pred = model.predict(validate_X)
            val_mae = mean_absolute_error(validate_y, validate_pred)
            val_rmse = np.sqrt(mean_squared_error(validate_y, validate_pred))
            val_r2 = r2_score(validate_y, validate_pred) if len(validate_y) > 1 else 0.0

            metrics.update({
                'validation_mae': float(val_mae),
                'validation_rmse': float(val_rmse),
                'validation_r2': float(val_r2),
                'validation_samples': int(len(validate_X)),
            })

        logger.info(
            f"{trait} [{self.regularization}]: Train MAE={train_mae:.4f}, R2={train_r2:.4f}"
        )
        return metrics

    def predict_trait(self, trait: str, X: np.ndarray) -> np.ndarray:
        if trait not in self.models:
            raise ValueError(f"Model for {trait} not trained")

        predictions = self.models[trait].predict(X)
        return np.clip(predictions, 0, 1)

    def predict_all_traits(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        predictions = {}
        for trait in ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']:
            if trait in self.models:
                predictions[trait] = self.predict_trait(trait, X)
        return predictions

    def get_feature_importance(self, trait: str, top_k: int = 10) -> Dict:
        if trait not in self.models:
            raise ValueError(f"Model for {trait} not trained")

        coefficients = self.models[trait].coef_
        top_indices = np.argsort(np.abs(coefficients))[-top_k:][::-1]
        importance = {}
        for idx in top_indices:
            if coefficients[idx] != 0:
                importance[int(idx)] = float(coefficients[idx])
        return importance

    def get_all_coefficients(self, trait: str) -> str:
        if trait not in self.models:
            raise ValueError(f"Model for {trait} not trained")

        coef_dict = {str(i): float(c) for i, c in enumerate(self.models[trait].coef_)}
        coef_dict['_feature_metadata'] = self.get_feature_metadata()
        coef_dict['_model_metadata'] = self.model_metadata.get(trait, {})
        return json.dumps(coef_dict)

    def save_state(self) -> Dict:
        import pickle

        return {
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'regularization': self.regularization,
            'l1_ratio': self.l1_ratio,
            'trait_alphas': self.trait_alphas,
            'trait_l1_ratios': self.trait_l1_ratios,
            'feature_mean': self.feature_mean.tolist() if self.feature_mean is not None else None,
            'feature_scale': self.feature_scale.tolist() if self.feature_scale is not None else None,
            'model_metadata': self.model_metadata,
            'models': {
                trait: pickle.dumps(model).hex()
                for trait, model in self.models.items()
            },
        }

    def load_state(self, state_dict: Dict):
        import pickle

        self.alpha = state_dict.get('alpha', 0.001)
        self.max_iter = state_dict.get('max_iter', 10000)
        self.regularization = state_dict.get('regularization', 'lasso')
        self.l1_ratio = state_dict.get('l1_ratio', 0.5)
        self.trait_alphas = state_dict.get('trait_alphas', {})
        self.trait_l1_ratios = state_dict.get('trait_l1_ratios', {})
        feature_mean = state_dict.get('feature_mean')
        feature_scale = state_dict.get('feature_scale')
        self.feature_mean = np.array(feature_mean) if feature_mean is not None else None
        self.feature_scale = np.array(feature_scale) if feature_scale is not None else None
        self.model_metadata = state_dict.get('model_metadata', {})

        self.models = {}
        for trait, model_hex in state_dict.get('models', {}).items():
            model_bytes = bytes.fromhex(model_hex)
            self.models[trait] = pickle.loads(model_bytes)

        logger.info(f"Loaded {len(self.models)} regression models")


def denormalize_predictions(predictions_normalized: np.ndarray) -> np.ndarray:
    """
    Convert normalized predictions (0-1) back to trait scale (1-5).
    """
    return predictions_normalized * 4.0 + 1.0
