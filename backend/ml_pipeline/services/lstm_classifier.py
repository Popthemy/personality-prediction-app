"""
Stacked LSTM Neural Network Service for Personality Classification and Regression.

This module implements a stacked recurrent neural network architecture using PyTorch
to model temporal user post timelines (sequences of 768-dimensional BERT embeddings).

Features:
- Stacked Bidirectional LSTM layers with LayerNorm & Dropout.
- Dual-head architecture:
  1. Classification Head: Outputs probability P(High Trait) in [0, 1].
  2. Regression Head: Outputs continuous predicted OCEAN trait score in [1.0, 5.0].
- Handles sequence padding and dynamic sequence lengths.
- Joint multi-task loss (BCE Loss + Smooth L1 Loss).
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger('ml_pipeline')

# Device configuration (GPU if available, CPU fallback)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class StackedLSTMClassifier(nn.Module):
    """
    Stacked Bidirectional LSTM model for sequence-based personality classification.
    
    Processes sequences of BERT embeddings (Batch, Max_Seq_Len, 768) and outputs
    both classification probabilities and continuous trait predictions.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super(StackedLSTMClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Stacked LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        effective_hidden = hidden_dim * self.num_directions
        self.layer_norm = nn.LayerNorm(effective_hidden)
        self.dropout_layer = nn.Dropout(dropout)

        # Classification Head -> Probabilities P(High Trait)
        self.classifier_head = nn.Sequential(
            nn.Linear(effective_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Regression Head -> Continuous OCEAN Score in [1.0, 5.0]
        self.regression_head = nn.Sequential(
            nn.Linear(effective_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input embeddings of shape (batch_size, seq_len, 768)
            seq_lengths: Optional lengths of each sequence in the batch
            
        Returns:
            Tuple of (class_probs, reg_scores)
            - class_probs: Tensor of shape (batch_size, 1) in range [0, 1]
            - reg_scores: Tensor of shape (batch_size, 1) in range [1.0, 5.0]
        """
        batch_size, seq_len, _ = x.shape

        if seq_lengths is not None:
            # Pack padded sequence for dynamic batching
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (hn, cn) = self.lstm(packed_input)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            lstm_out, (hn, cn) = self.lstm(x)

        # Mean pooling across valid sequence representations
        if seq_lengths is not None:
            mask = torch.arange(seq_len, device=x.device)[None, :] < seq_lengths[:, None]
            mask = mask.unsqueeze(-1).float()
            pooled = (lstm_out * mask).sum(dim=1) / seq_lengths.unsqueeze(-1).float().clamp(min=1.0)
        else:
            pooled = lstm_out.mean(dim=1)

        pooled = self.layer_norm(pooled)
        pooled = self.dropout_layer(pooled)

        class_probs = self.classifier_head(pooled)
        reg_scores = self.regression_head(pooled)
        
        # Clamp continuous predictions to valid Likert scale range [1.0, 5.0]
        reg_scores = torch.clamp(reg_scores, min=1.0, max=5.0)

        return class_probs, reg_scores


class LSTMTrainer:
    """
    Trainer and Inference helper for StackedLSTMClassifier.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.models: Dict[str, StackedLSTMClassifier] = {}

    def _prepare_sequences(
        self,
        sequence_list: List[np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad a list of 2D embedding arrays (num_posts, 768) into a 3D batch tensor.
        """
        seq_lengths = [len(seq) for seq in sequence_list]
        max_len = max(seq_lengths) if seq_lengths else 1
        input_dim = sequence_list[0].shape[1] if (sequence_list and sequence_list[0].ndim > 1) else 768

        padded_tensor = torch.zeros((len(sequence_list), max_len, input_dim), dtype=torch.float32)
        for i, seq in enumerate(sequence_list):
            if seq.ndim == 1:
                seq = seq.reshape(1, -1)
            length = min(len(seq), max_len)
            if length > 0:
                padded_tensor[i, :length, :] = torch.tensor(seq[:length], dtype=torch.float32)

        return padded_tensor, torch.tensor(seq_lengths, dtype=torch.long)

    def train_trait_model(
        self,
        trait: str,
        sequences: List[np.ndarray],
        targets: np.ndarray,
        threshold: float = 4.0,
        epochs: int = 35,
        batch_size: int = 4,
        sample_weights: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train a StackedLSTMClassifier model for a specific OCEAN trait.
        
        Args:
            trait: Name of trait (e.g. 'Openness')
            sequences: List of arrays per volunteer, shape (num_posts, 768)
            targets: Continuous target values in [1.0, 5.0]
            threshold: Likert cutoff threshold for classification (e.g. 4.0 or 3.5)
            epochs: Training epochs
            batch_size: Mini-batch size
            sample_weights: Optional sample weight vector
            
        Returns:
            Dict of training result summary including probabilities, continuous predictions, and loss history.
        """
        if len(sequences) == 0:
            raise ValueError("Cannot train LSTM with empty sequence list.")

        X_padded, seq_lengths = self._prepare_sequences(sequences)
        y_reg = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
        y_cls = (y_reg >= threshold).float()

        weights = torch.tensor(sample_weights, dtype=torch.float32).unsqueeze(-1) if sample_weights is not None else torch.ones_like(y_reg)

        dataset = TensorDataset(X_padded, seq_lengths, y_cls, y_reg, weights)
        dataloader = DataLoader(dataset, batch_size=min(batch_size, len(sequences)), shuffle=True)

        model = StackedLSTMClassifier(
            input_dim=X_padded.shape[2],
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        bce_criterion = nn.BCELoss(reduction='none')
        l1_criterion = nn.SmoothL1Loss(reduction='none')

        loss_history = []
        model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_len, batch_ycls, batch_yreg, batch_w in dataloader:
                batch_x = batch_x.to(device)
                batch_len = batch_len.to(device)
                batch_ycls = batch_ycls.to(device)
                batch_yreg = batch_yreg.to(device)
                batch_w = batch_w.to(device)

                optimizer.zero_grad()
                pred_prob, pred_reg = model(batch_x, batch_len)

                loss_cls = (bce_criterion(pred_prob, batch_ycls) * batch_w).mean()
                loss_reg = (l1_criterion(pred_reg, batch_yreg) * batch_w).mean()
                total_loss = loss_cls + 0.5 * loss_reg

                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += total_loss.item()

            loss_history.append(epoch_loss / max(1, len(dataloader)))

        self.models[trait] = model

        # Run inference on training set to return predictions & probabilities
        model.eval()
        with torch.no_grad():
            X_eval = X_padded.to(device)
            seq_eval = seq_lengths.to(device)
            probs, reg_preds = model(X_eval, seq_eval)

        probabilities = probs.cpu().numpy().flatten()
        continuous_predictions = reg_preds.cpu().numpy().flatten()

        logger.info(f"LSTM training completed for {trait}. Final Loss: {loss_history[-1]:.4f}")

        return {
            'trait': trait,
            'epochs': epochs,
            'final_loss': loss_history[-1],
            'loss_history': loss_history,
            'probabilities': probabilities,
            'continuous_predictions': continuous_predictions,
            'sample_count': len(sequences),
            'threshold': threshold,
        }

    def predict_trait(
        self,
        trait: str,
        sequences: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform inference for a specific trait.
        
        Returns:
            Tuple of (probabilities, continuous_predictions)
        """
        if trait not in self.models:
            # Fallback heuristic if trait model not explicitly trained yet
            num_samples = len(sequences)
            return np.full(num_samples, 0.5), np.full(num_samples, 3.0)

        model = self.models[trait]
        model.eval()
        X_padded, seq_lengths = self._prepare_sequences(sequences)

        with torch.no_grad():
            X_eval = X_padded.to(device)
            seq_eval = seq_lengths.to(device)
            probs, reg_preds = model(X_eval, seq_eval)

        return probs.cpu().numpy().flatten(), reg_preds.cpu().numpy().flatten()
