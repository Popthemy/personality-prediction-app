"""
Binary LSTM classifier for Big Five personality traits.

This module is intentionally focused on the current PANDORA experiment path:

    selected comment BERT embeddings -> stacked bidirectional LSTM
                                      -> optional auxiliary features
                                      -> five sigmoid logits
                                      -> P(High) for O, C, E, A, N

Continuous OCEAN labels are used only to derive supervised Low/High targets
with a fixed ground-truth cutoff. Decision thresholds are selected later by the
metrics engine's validation sweep. There is no Medium class in this module.
"""

from __future__ import annotations

import logging
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from backend.ml_pipeline.services import metrics_engine as me

logger = logging.getLogger("ml_pipeline")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIT_KEYS: Tuple[str, ...] = ("O", "C", "E", "A", "N")
OCEAN_TRAITS: List[str] = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]
BINARY_LABELS = {0: "Low", 1: "High"}
NUM_TRAITS = 5


def set_seed(seed: int = 42) -> None:
    """Fix RNGs used by Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StackedLSTMClassifier(nn.Module):
    """Stacked bidirectional LSTM that emits one High/Low logit per trait."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        num_traits: int = NUM_TRAITS,
        auxiliary_dim: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_traits = num_traits
        self.auxiliary_dim = auxiliary_dim
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        effective_hidden = hidden_dim * self.num_directions
        self.layer_norm = nn.LayerNorm(effective_hidden)
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier_head = nn.Sequential(
            nn.Linear(effective_hidden + auxiliary_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_traits),
        )

    def forward(
        self,
        x: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
        auxiliary_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return raw logits with shape ``(batch, 5)``."""
        _batch_size, seq_len, _ = x.shape

        if seq_lengths is not None:
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x,
                seq_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.lstm(packed_input)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=seq_len,
            )
        else:
            lstm_out, _ = self.lstm(x)

        if seq_lengths is not None:
            mask = torch.arange(seq_len, device=x.device)[None, :] < seq_lengths[:, None]
            mask = mask.unsqueeze(-1).float()
            pooled = (lstm_out * mask).sum(dim=1) / seq_lengths.unsqueeze(-1).float().clamp(min=1.0)
        else:
            pooled = lstm_out.mean(dim=1)

        pooled = self.layer_norm(pooled)
        pooled = self.dropout_layer(pooled)
        if self.auxiliary_dim:
            if auxiliary_features is None:
                auxiliary_features = torch.zeros(
                    (pooled.shape[0], self.auxiliary_dim),
                    dtype=pooled.dtype,
                    device=pooled.device,
                )
            pooled = torch.cat([pooled, auxiliary_features.to(pooled.device).float()], dim=1)
        return self.classifier_head(pooled)


class LSTMTrainer:
    """Train and run the five-output binary LSTM classifier."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        positive_weight: Optional[Sequence[float]] = None,
        auxiliary_dim: int = 0,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.positive_weight = positive_weight
        self.auxiliary_dim = auxiliary_dim
        self.model: Optional[StackedLSTMClassifier] = None
        self.input_dim: Optional[int] = None

    @staticmethod
    def binarize_targets(
        targets: np.ndarray,
        ground_truth_cutoff: float = me.DEFAULT_GROUND_TRUTH_CUTOFF,
    ) -> np.ndarray:
        """Convert continuous ``(N, 5)`` OCEAN scores to Low/High labels."""
        y = np.asarray(targets, dtype=np.float32)
        if y.ndim != 2 or y.shape[1] != NUM_TRAITS:
            raise ValueError(f"targets must have shape (N, 5); got {y.shape}")
        return (y >= float(ground_truth_cutoff)).astype(np.float32)

    def _prepare_sequences(self, sequence_list: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad a list of ``(num_comments, embedding_dim)`` arrays into one batch."""
        if not sequence_list:
            raise ValueError("sequence_list cannot be empty.")

        normalised: List[np.ndarray] = []
        for seq in sequence_list:
            arr = np.asarray(seq, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2:
                raise ValueError(f"Each sequence must be 2-D; got {arr.shape}")
            if arr.shape[0] == 0:
                arr = np.zeros((1, arr.shape[1]), dtype=np.float32)
            normalised.append(arr)

        input_dim = normalised[0].shape[1]
        for arr in normalised:
            if arr.shape[1] != input_dim:
                raise ValueError("All sequences must share the same embedding dimension.")

        seq_lengths = [max(1, len(seq)) for seq in normalised]
        max_len = max(seq_lengths)
        padded = torch.zeros((len(normalised), max_len, input_dim), dtype=torch.float32)
        for i, seq in enumerate(normalised):
            padded[i, : len(seq), :] = torch.from_numpy(seq)

        return padded, torch.tensor(seq_lengths, dtype=torch.long)

    def _prepare_auxiliary(
        self,
        auxiliary_features: Optional[np.ndarray],
        n_samples: int,
    ) -> torch.Tensor:
        """Validate or create the optional auxiliary feature matrix."""
        if self.auxiliary_dim == 0:
            return torch.zeros((n_samples, 0), dtype=torch.float32)
        if auxiliary_features is None:
            raise ValueError(f"Expected auxiliary_features with shape (N, {self.auxiliary_dim}).")
        aux = np.asarray(auxiliary_features, dtype=np.float32)
        if aux.shape != (n_samples, self.auxiliary_dim):
            raise ValueError(
                f"auxiliary_features must have shape {(n_samples, self.auxiliary_dim)}; got {aux.shape}"
            )
        return torch.from_numpy(aux)

    @staticmethod
    def _class_distribution(labels: np.ndarray) -> Dict[str, Any]:
        """Per-trait Low/High counts for diagnostics."""
        out: Dict[str, Any] = {}
        labels = labels.astype(int)
        for i, trait in enumerate(OCEAN_TRAITS):
            col = labels[:, i]
            low = int(np.sum(col == 0))
            high = int(np.sum(col == 1))
            total = int(len(col))
            out[trait] = {
                "Low": {"count": low, "proportion": float(low / total) if total else 0.0},
                "High": {"count": high, "proportion": float(high / total) if total else 0.0},
            }
        return out

    def _validation_score(
        self,
        val_sequences: List[np.ndarray],
        val_targets_binary: np.ndarray,
        criterion: nn.Module,
        val_auxiliary_features: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Return validation loss, mean best F1, and threshold diagnostics."""
        assert self.model is not None
        probs = self.predict_proba(val_sequences, auxiliary_features=val_auxiliary_features)
        X_val, len_val = self._prepare_sequences(val_sequences)
        aux_val = self._prepare_auxiliary(val_auxiliary_features, len(val_sequences))
        with torch.no_grad():
            logits = self.model(X_val.to(device), len_val.to(device), aux_val.to(device))
            target = torch.from_numpy(val_targets_binary).to(device)
            val_loss = float(criterion(logits, target).mean().detach().cpu())

        per_trait: Dict[str, Any] = {}
        f1s: List[float] = []
        for i, trait in enumerate(TRAIT_KEYS):
            sweep = me.sweep_thresholds_on_scores(
                val_targets_binary[:, i],
                probs[:, i],
                candidate_thresholds=list(me.CANDIDATE_THRESHOLDS),
            )
            per_trait[trait] = sweep
            f1s.append(float(sweep["f1"]))
        return val_loss, float(np.mean(f1s)) if f1s else 0.0, per_trait

    def train(
        self,
        sequences: List[np.ndarray],
        targets: np.ndarray,
        val_sequences: Optional[List[np.ndarray]] = None,
        val_targets: Optional[np.ndarray] = None,
        epochs: int = 35,
        batch_size: int = 4,
        sample_weights: Optional[np.ndarray] = None,
        auxiliary_features: Optional[np.ndarray] = None,
        val_auxiliary_features: Optional[np.ndarray] = None,
        seed: int = 42,
        ground_truth_cutoff: float = me.DEFAULT_GROUND_TRUTH_CUTOFF,
    ) -> Dict[str, Any]:
        """Train one five-output binary classifier on participant sequences."""
        if not sequences:
            raise ValueError("Cannot train LSTM with an empty sequence list.")

        set_seed(seed)
        X, seq_lengths = self._prepare_sequences(sequences)
        self.input_dim = int(X.shape[2])

        y_binary = self.binarize_targets(targets, ground_truth_cutoff)
        y_tensor = torch.from_numpy(y_binary)
        aux_tensor = self._prepare_auxiliary(auxiliary_features, len(sequences))

        if sample_weights is None:
            weights = np.ones(len(sequences), dtype=np.float32)
        else:
            weights = np.asarray(sample_weights, dtype=np.float32)
            if weights.shape[0] != len(sequences):
                raise ValueError("sample_weights must have one value per training sequence.")
        weights_tensor = torch.from_numpy(weights)

        has_val = val_sequences is not None and val_targets is not None and len(val_sequences) > 0
        y_val_binary: Optional[np.ndarray] = None
        if has_val:
            y_val_binary = self.binarize_targets(np.asarray(val_targets), ground_truth_cutoff)
            self._prepare_auxiliary(val_auxiliary_features, len(val_sequences or []))

        self.model = StackedLSTMClassifier(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            auxiliary_dim=self.auxiliary_dim,
        ).to(device)

        pos_weight = None
        if self.positive_weight is not None:
            pos_weight = torch.tensor(self.positive_weight, dtype=torch.float32, device=device)

        criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        dataset = TensorDataset(X, seq_lengths, aux_tensor, y_tensor, weights_tensor)
        loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

        train_loss_history: List[float] = []
        val_loss_history: List[float] = []
        val_mean_f1_history: List[float] = []
        best_state = None
        best_val_f1 = -1.0
        best_val_loss = float("inf")
        best_thresholds: Dict[str, Any] = {}

        for _epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_x, batch_len, batch_aux, batch_y, batch_w in loader:
                batch_x = batch_x.to(device)
                batch_len = batch_len.to(device)
                batch_aux = batch_aux.to(device)
                batch_y = batch_y.to(device)
                batch_w = batch_w.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(batch_x, batch_len, batch_aux)
                loss_matrix = criterion(logits, batch_y)
                loss = (loss_matrix.mean(dim=1) * batch_w).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())

            train_loss_history.append(epoch_loss / max(1, len(loader)))

            if has_val and y_val_binary is not None:
                self.model.eval()
                val_loss, val_f1, thresholds = self._validation_score(
                    val_sequences or [],
                    y_val_binary,
                    criterion,
                    val_auxiliary_features,
                )
                val_loss_history.append(val_loss)
                val_mean_f1_history.append(val_f1)
                if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_loss < best_val_loss):
                    best_val_f1 = val_f1
                    best_val_loss = val_loss
                    best_thresholds = thresholds
                    best_state = deepcopy(self.model.state_dict())

        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info("LSTM restored epoch with best validation mean F1 = %.4f", best_val_f1)
        else:
            logger.warning("No validation split supplied; LSTM kept final training epoch.")

        logger.info("LSTM training completed. Final train loss: %.4f", train_loss_history[-1])
        return {
            "epochs": int(epochs),
            "ground_truth_cutoff": float(ground_truth_cutoff),
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history if has_val else None,
            "val_mean_f1_history": val_mean_f1_history if has_val else None,
            "best_val_mean_f1": float(best_val_f1) if has_val else None,
            "best_validation_thresholds": best_thresholds,
            "train_class_distribution": self._class_distribution(y_binary),
            "val_class_distribution": self._class_distribution(y_val_binary) if y_val_binary is not None else None,
            "sample_count": int(len(sequences)),
        }

    def predict_proba(
        self,
        sequences: List[np.ndarray],
        auxiliary_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return ``(N, 5)`` probabilities, one P(High) per trait."""
        if self.model is None:
            raise RuntimeError("No trained LSTM model found. Call train() before prediction.")
        self.model.eval()
        X, seq_lengths = self._prepare_sequences(sequences)
        aux = self._prepare_auxiliary(auxiliary_features, len(sequences))
        with torch.no_grad():
            logits = self.model(X.to(device), seq_lengths.to(device), aux.to(device))
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        return probs.astype(np.float32)

    def predict(self, sequences: List[np.ndarray], auxiliary_features: Optional[np.ndarray] = None) -> np.ndarray:
        """Alias for predict_proba, kept for the experiment runner."""
        return self.predict_proba(sequences, auxiliary_features=auxiliary_features)

    def predict_labels(
        self,
        sequences: List[np.ndarray],
        thresholds: Optional[Sequence[float]] = None,
        auxiliary_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return binary Low/High labels using one threshold or five thresholds."""
        probs = self.predict_proba(sequences, auxiliary_features=auxiliary_features)
        if thresholds is None:
            thresh = np.full(NUM_TRAITS, 0.5, dtype=np.float32)
        else:
            thresh = np.asarray(thresholds, dtype=np.float32)
            if thresh.size == 1:
                thresh = np.full(NUM_TRAITS, float(thresh.item()), dtype=np.float32)
            if thresh.shape != (NUM_TRAITS,):
                raise ValueError(f"thresholds must be scalar or length 5; got {thresh.shape}")
        return (probs >= thresh.reshape(1, -1)).astype(int)

    def save_state(self) -> Dict[str, Any]:
        """Return metadata plus state dict for artifact persistence."""
        if self.model is None:
            raise RuntimeError("Cannot save an untrained LSTM model.")
        return {
            "state_dict": self.model.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "auxiliary_dim": self.auxiliary_dim,
                "num_traits": NUM_TRAITS,
                "trait_keys": list(TRAIT_KEYS),
                "trait_names": list(OCEAN_TRAITS),
            },
        }
