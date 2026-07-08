"""
Lasso Regression Service for personality trait prediction.

Uses scikit-learn's Lasso for sparse, interpretable feature selection.
Trains one model per OCEAN trait.
"""
import logging
from typing import Dict, List, Tuple
import numpy as np
# from sklearn.linear_model import Lasso, LassoCV
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

logger = logging.getLogger('ml_pipeline')


class LassoTrainer:
    """Lasso regression trainer for OCEAN traits."""
    
    def __init__(self, alpha: float = 0.001, max_iter: int = 10000):
        """
        Initialize Lasso trainer.
        
        Args:
            alpha: L1 regularization strength
            max_iter: Maximum iterations
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.models = {}  # {trait: Lasso model}
    
    def prepare_training_data(self, embeddings: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data.
        
        Args:
            embeddings: N x 768 array of BERT embeddings
            labels: N array of trait scores (1-5)
        
        Returns:
            (X, y) prepared for training
        """
        # Normalize embeddings
        X = (embeddings - embeddings.mean(axis=0)) / (embeddings.std(axis=0) + 1e-8)
        
        # Normalize labels to 0-1 range for better Lasso performance
        y = (labels - 1.0) / 4.0  # Convert 1-5 to 0-1
        
        return X, y
    
    def train_trait_model(self, X: np.ndarray, y: np.ndarray, trait: str,
                         validate_X: np.ndarray = None, validate_y: np.ndarray = None) -> Dict:
        """
        Train Lasso model for a single trait.
        
        Args:
            X: Training features (N x 768)
            y: Training labels (N,)
            trait: Trait name
            validate_X: Validation features (optional)
            validate_y: Validation labels (optional)
        
        Returns:
            Dict with model info and metrics
        """
        logger.info(f"Training Lasso for {trait}...")
        
        # Create and train model
        model = Lasso(alpha=self.alpha, max_iter=self.max_iter, random_state=42)
        model.fit(X, y)
        
        # Store model
        self.models[trait] = model
        
        # Training metrics
        train_pred = model.predict(X)
        train_mae = mean_absolute_error(y, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y, train_pred))
        train_r2 = r2_score(y, train_pred)
        
        metrics = {
            'trait': trait,
            'alpha': self.alpha,
            'train_mae': float(train_mae),
            'train_rmse': float(train_rmse),
            'train_r2': float(train_r2),
            'training_samples': len(X),
            'sparse_features': int(np.sum(model.coef_ != 0)),
            'total_features': len(model.coef_),
        }
        
        # Validation metrics (if provided)
        if validate_X is not None and validate_y is not None:
            validate_pred = model.predict(validate_X)
            val_mae = mean_absolute_error(validate_y, validate_pred)
            val_rmse = np.sqrt(mean_squared_error(validate_y, validate_pred))
            val_r2 = r2_score(validate_y, validate_pred)
            
            metrics.update({
                'validation_mae': float(val_mae),
                'validation_rmse': float(val_rmse),
                'validation_r2': float(val_r2),
                'validation_samples': len(validate_X),
            })
        
        logger.info(f"Lasso {trait}: Train MAE={train_mae:.4f}, R²={train_r2:.4f}")
        return metrics
    
    def predict_trait(self, trait: str, X: np.ndarray) -> np.ndarray:
        """
        Predict trait scores using trained model.
        
        Args:
            trait: Trait name
            X: Features (N x 768)
        
        Returns:
            Predicted scores (N,) in range 0-1
        """
        if trait not in self.models:
            raise ValueError(f"Model for {trait} not trained")
        
        predictions = self.models[trait].predict(X)
        # Clip to 0-1 range
        predictions = np.clip(predictions, 0, 1)
        return predictions
    
    def predict_all_traits(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict all OCEAN traits.
        
        Args:
            X: Features (N x 768)
        
        Returns:
            Dict {trait: predictions}
        """
        predictions = {}
        
        for trait in ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']:
            if trait in self.models:
                predictions[trait] = self.predict_trait(trait, X)
        
        return predictions
    
    def get_feature_importance(self, trait: str, top_k: int = 10) -> Dict:
        """
        Get top-k important features for a trait.
        
        Args:
            trait: Trait name
            top_k: Number of top features to return
        
        Returns:
            Dict {feature_idx: coefficient}
        """
        if trait not in self.models:
            raise ValueError(f"Model for {trait} not trained")
        
        coefficients = self.models[trait].coef_
        top_indices = np.argsort(np.abs(coefficients))[-top_k:][::-1]
        
        importance = {}
        for idx in top_indices:
            if coefficients[idx] != 0:  # Only non-zero (sparse)
                importance[int(idx)] = float(coefficients[idx])
        
        return importance
    
    def get_all_coefficients(self, trait: str) -> str:
        """
        Get all coefficients as JSON for storage.
        
        Args:
            trait: Trait name
        
        Returns:
            JSON string of coefficients
        """
        if trait not in self.models:
            raise ValueError(f"Model for {trait} not trained")
        
        coef_dict = {str(i): float(c) for i, c in enumerate(self.models[trait].coef_)}
        return json.dumps(coef_dict)
    
    def save_state(self) -> Dict:
        """Serialize trainer state."""
        import pickle
        
        return {
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'models': {
                trait: pickle.dumps(model).hex()
                for trait, model in self.models.items()
            }
        }
    
    def load_state(self, state_dict: Dict):
        """Load trainer state."""
        import pickle
        
        self.alpha = state_dict.get('alpha', 0.001)
        self.max_iter = state_dict.get('max_iter', 10000)
        
        self.models = {}
        for trait, model_hex in state_dict.get('models', {}).items():
            model_bytes = bytes.fromhex(model_hex)
            self.models[trait] = pickle.loads(model_bytes)
        
        logger.info(f"Loaded {len(self.models)} Lasso models")


def denormalize_predictions(predictions_normalized: np.ndarray) -> np.ndarray:
    """
    Convert normalized predictions (0-1) back to trait scale (1-5).
    
    Args:
        predictions_normalized: Values in [0, 1]
    
    Returns:
        Values in [1, 5]
    """
    return predictions_normalized * 4.0 + 1.0
