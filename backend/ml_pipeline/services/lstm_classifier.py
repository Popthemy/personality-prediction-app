"""
Stacked LSTM Neural Network Service for Personality Trait Regression.

This module implements a stacked recurrent neural network architecture using PyTorch
to model temporal user post timelines (sequences of pre-computed 768-dimensional
BERT embeddings).

Architecture (LSTM-only — BERT is upstream and not implemented here):

    Pre-computed BERT sequence (batch, seq_len, 768)
                    |
          Stacked Bidirectional LSTM
                    |
        Mean-pooled sequence representation
                    |
                 LayerNorm
                    |
                  Dropout
                    |
             Fully Connected Head
                    |
              5 continuous outputs
                    |
            [O, C, E, A, N] scores

Notes:
- Single regression head outputting 5 continuous OCEAN values directly.
  High/Low thresholding is a downstream evaluation/decision step — it does
  NOT occur inside this module.
- Loss is SmoothL1 (Huber) over 5 continuous outputs, weighted per sample.
- Handles sequence padding and dynamic sequence lengths via
  pack_padded_sequence / pad_packed_sequence.
- Validation model selection uses mean MAE across the 5 OCEAN traits
  (lower is better).  Best-epoch weights are restored after training exactly
  as in the previous implementation.
- `predict()` returns (N, 5) continuous OCEAN predictions.
  `predict_trait()` is retained as a backward-compatible wrapper.
- `evaluate_trait()` delegates metric computation to the common metric
  engine so that MAE/RMSE/R²/Pearson are consistent with the Lasso pathway.
"""

import logging
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

logger = logging.getLogger('ml_pipeline')

# Device configuration (GPU if available, CPU fallback)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# OCEAN trait ordering used throughout this module
OCEAN_TRAITS = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
NUM_TRAITS = len(OCEAN_TRAITS)  # 5


def set_seed(seed: int = 42) -> None:
    """
    Fix all relevant RNGs so training runs are reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_participants(
    participant_ids: List[Any],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Split PARTICIPANT IDs (not individual posts) into train/val/test sets.

    Splits at the participant level so posts from one participant never
    appear in more than one split.  Stratification is not applicable for
    continuous regression targets, so splits are random.

    Deliberately makes NO assumption about how much data is available —
    this project's dataset size will keep changing, so this function must
    not hard-fail just because an exact 80/10/10 split isn't achievable.
    Behavior:

    - test_size / val_size are treated as target proportions, not
      guarantees. Actual counts are derived from whatever `participant_ids`
      it's given, clamped so train/test never go to zero when avoidable.
    - With very few participants (<=2), a full train/val/test split isn't
      meaningful; this returns the best degenerate split possible instead
      of erroring.

    Returns:
        (train_ids, val_ids, test_ids)
    """
    ids = list(participant_ids)
    n = len(ids)

    if n == 0:
        raise ValueError("split_participants received zero participants.")

    if n == 1:
        logger.warning("Only 1 participant available — using it for training only; val/test are empty.")
        return ids, [], []

    if n == 2:
        logger.warning("Only 2 participants available — 1 for training, 1 for test; val is empty.")
        return [ids[0]], [], [ids[1]]

    def _split(ids_in, size, rs):
        try:
            return train_test_split(ids_in, test_size=size, random_state=rs)
        except ValueError as e:
            logger.warning(f"Split failed ({e}); returning unsplit.")
            return ids_in, []

    # --- carve off the test set -------------------------------------------
    test_count = max(1, round(n * test_size))
    test_count = min(test_count, n - 2)  # always leave >=2 for train (+val)

    train_val_ids, test_ids = _split(ids, test_count / n, random_state)

    # --- carve val out of what remains ------------------------------------
    remaining_n = len(train_val_ids)
    if remaining_n < 2 or val_size <= 0:
        if remaining_n < 2:
            logger.warning("Too few participants remain after the test split to also carve out a val set.")
        return train_val_ids, [], test_ids

    remaining_fraction = max(1e-8, 1.0 - test_size)
    val_count = max(1, round(remaining_n * (val_size / remaining_fraction)))
    val_count = min(val_count, remaining_n - 1)

    if val_count <= 0:
        return train_val_ids, [], test_ids

    train_ids, val_ids = _split(train_val_ids, val_count / remaining_n, random_state)

    logger.info(
        f"split_participants: n={n} -> train={len(train_ids)}, "
        f"val={len(val_ids)}, test={len(test_ids)}"
    )
    return train_ids, val_ids, test_ids


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trait_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute regression metrics compatible with the common metric engine.

    Accepts either:
      - (N,) arrays for a single trait, or
      - (N, 5) arrays for all OCEAN traits simultaneously.

    Returns MAE, RMSE, R², and Pearson r for each trait plus overall means.

    Args:
        y_true: Ground-truth continuous OCEAN scores.
        y_pred: Predicted continuous OCEAN scores (same shape as y_true).
        trait_names: Optional list of trait names (defaults to OCEAN_TRAITS).

    Returns:
        Dict with per-trait and mean metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
        y_pred = y_pred[:, np.newaxis]

    n_traits = y_true.shape[1]
    names = trait_names if trait_names is not None else OCEAN_TRAITS[:n_traits]

    per_trait: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        mae = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        r2 = float(r2_score(yt, yp))
        try:
            pearson_r, pearson_p = pearsonr(yt, yp)
        except Exception:
            pearson_r, pearson_p = float('nan'), float('nan')
        per_trait[name] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
        }

    mean_mae = float(np.mean([v['mae'] for v in per_trait.values()]))
    mean_rmse = float(np.mean([v['rmse'] for v in per_trait.values()]))
    mean_r2 = float(np.mean([v['r2'] for v in per_trait.values()]))
    mean_pearson_r = float(np.nanmean([v['pearson_r'] for v in per_trait.values()]))

    return {
        'per_trait': per_trait,
        'mean_mae': mean_mae,
        'mean_rmse': mean_rmse,
        'mean_r2': mean_r2,
        'mean_pearson_r': mean_pearson_r,
    }


class StackedLSTMRegressor(nn.Module):
    """
    Stacked Bidirectional LSTM model for sequence-based OCEAN trait regression.

    Processes sequences of pre-computed BERT embeddings
    (Batch, Max_Seq_Len, 768) and outputs 5 continuous OCEAN scores.
    The final layer has no activation — raw real-valued outputs are produced
    so SmoothL1 loss can be applied directly during training.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_outputs: int = NUM_TRAITS,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super(StackedLSTMRegressor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Stacked Bidirectional LSTM — identical to the previous architecture
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

        # Regression head — 5 continuous OCEAN outputs (no final activation).
        # Structure mirrors the former classifier head; only the output
        # dimension changes from 3 → 5 and the softmax is removed.
        self.regression_head = nn.Sequential(
            nn.Linear(effective_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_outputs)
        )

    def forward(
        self,
        x: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input embeddings of shape (batch_size, seq_len, 768)
            seq_lengths: Optional lengths of each sequence in the batch.

        Returns:
            outputs: Tensor of shape (batch_size, 5) — continuous OCEAN scores.
        """
        batch_size, seq_len, _ = x.shape

        if seq_lengths is not None:
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.lstm(packed_input)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)

        # Mean pooling across valid (non-padded) positions
        if seq_lengths is not None:
            mask = torch.arange(seq_len, device=x.device)[None, :] < seq_lengths[:, None]
            mask = mask.unsqueeze(-1).float()
            pooled = (lstm_out * mask).sum(dim=1) / seq_lengths.unsqueeze(-1).float().clamp(min=1.0)
        else:
            pooled = lstm_out.mean(dim=1)

        pooled = self.layer_norm(pooled)
        pooled = self.dropout_layer(pooled)

        outputs = self.regression_head(pooled)  # (batch_size, 5)
        return outputs


class LSTMTrainer:
    """
    Trainer and Inference helper for StackedLSTMRegressor.

    Trains a single LSTM model jointly on all five OCEAN traits, mapping
    BERT-embedding sequences to (N, 5) continuous OCEAN predictions.

    The model formerly trained one classifier per trait; it now trains one
    shared regression model over all traits simultaneously.  Per-trait
    wrapper methods (`train_trait_model`, `predict_trait`, `evaluate_trait`)
    are retained for backward compatibility.
    """

    # Canonical trait index mapping for the 5-output head
    TRAIT_INDEX = {name: i for i, name in enumerate(OCEAN_TRAITS)}

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
        # Stores the trained model; keyed by 'model' (shared) or a trait
        # name for backward-compat callers that used per-trait storage.
        self.model: Optional[StackedLSTMRegressor] = None
        # Legacy per-trait dict kept so downstream code that checks
        # `trainer.models[trait]` continues to work.
        self.models: Dict[str, StackedLSTMRegressor] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_sequences(
        self,
        sequence_list: List[np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad a list of 2D embedding arrays (num_posts, 768) into a 3D batch tensor.
        Identical to the previous implementation.
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

    # ------------------------------------------------------------------
    # Primary training API
    # ------------------------------------------------------------------

    def train(
        self,
        sequences: List[np.ndarray],
        targets: np.ndarray,
        val_sequences: Optional[List[np.ndarray]] = None,
        val_targets: Optional[np.ndarray] = None,
        epochs: int = 35,
        batch_size: int = 4,
        sample_weights: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Train the LSTM regressor directly on continuous OCEAN targets.

        Args:
            sequences:      List of (num_posts, 768) arrays — one per participant.
            targets:        (N, 5) continuous OCEAN scores [O, C, E, A, N].
            val_sequences:  Held-out validation sequences (strongly recommended).
            val_targets:    (N_val, 5) continuous OCEAN scores for validation.
            epochs:         Number of training epochs.
            batch_size:     Mini-batch size.
            sample_weights: Optional (N,) per-sample loss weights.
            seed:           Random seed for reproducibility.

        Returns:
            Dict containing loss/MAE history and best-epoch metrics.
        """
        if len(sequences) == 0:
            raise ValueError("Cannot train LSTM with an empty sequence list.")

        targets = np.asarray(targets)
        if targets.ndim != 2 or targets.shape[1] != NUM_TRAITS:
            raise ValueError(
                f"targets must be (N, {NUM_TRAITS}); got shape {targets.shape}."
            )
        if targets.shape[0] != len(sequences):
            raise ValueError(
                f"sequences and targets length mismatch: {len(sequences)} vs {targets.shape[0]}."
            )

        set_seed(seed)

        X_padded, seq_lengths = self._prepare_sequences(sequences)
        y_tensor = torch.tensor(targets, dtype=torch.float32)

        has_val = val_sequences is not None and len(val_sequences) > 0
        X_val_padded = val_seq_lengths = y_val_tensor = None
        if has_val:
            val_targets = np.asarray(val_targets)
            if val_targets.ndim != 2 or val_targets.shape[1] != NUM_TRAITS:
                raise ValueError(
                    f"val_targets must be (N_val, {NUM_TRAITS}); got shape {val_targets.shape}."
                )
            X_val_padded, val_seq_lengths = self._prepare_sequences(val_sequences)
            y_val_tensor = torch.tensor(val_targets, dtype=torch.float32)

        weights = (
            torch.tensor(sample_weights, dtype=torch.float32)
            if sample_weights is not None
            else torch.ones(len(sequences), dtype=torch.float32)
        )

        dataset = TensorDataset(X_padded, seq_lengths, y_tensor, weights)
        dataloader = DataLoader(dataset, batch_size=min(batch_size, len(sequences)), shuffle=True)

        model = StackedLSTMRegressor(
            input_dim=X_padded.shape[2],
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_outputs=NUM_TRAITS,
            dropout=self.dropout
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        # SmoothL1 (Huber) loss — robust to outliers, preferred for
        # personality score regression (spec §Loss).
        criterion = nn.SmoothL1Loss(reduction='none')

        train_loss_history: List[float] = []
        val_loss_history: List[float] = []
        val_mae_history: List[float] = []
        best_val_mae = float('inf')
        best_state_dict = None

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0

            for batch_x, batch_len, batch_y, batch_w in dataloader:
                batch_x = batch_x.to(device)
                batch_len = batch_len.to(device)
                batch_y = batch_y.to(device)
                batch_w = batch_w.to(device)

                optimizer.zero_grad()
                preds = model(batch_x, batch_len)  # (B, 5)

                # SmoothL1 returns (B, 5); mean over traits then weight by sample
                per_sample_loss = criterion(preds, batch_y).mean(dim=1)  # (B,)
                loss = (per_sample_loss * batch_w).mean()

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            train_loss_history.append(epoch_loss / max(1, len(dataloader)))

            # --- Validation monitoring ------------------------------------
            if has_val:
                model.eval()
                with torch.no_grad():
                    X_v = X_val_padded.to(device)
                    len_v = val_seq_lengths.to(device)
                    y_v = y_val_tensor.to(device)
                    val_preds = model(X_v, len_v)
                    val_loss = criterion(val_preds, y_v).mean().item()

                val_mae = float(
                    torch.abs(val_preds - y_v).mean().item()
                )
                val_loss_history.append(val_loss)
                val_mae_history.append(val_mae)

                # Model selection: lower validation MAE is better
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        # Restore best-on-validation weights (unchanged logic from prev impl)
        if has_val and best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            logger.info(f"LSTM: restored epoch with best validation MAE = {best_val_mae:.4f}")
        else:
            logger.warning(
                "LSTM: no validation set provided — model selection defaulted to the "
                "final training epoch. Provide val_sequences/val_targets for a real check."
            )

        self.model = model
        # Keep legacy per-trait dict in sync so old callers don't break
        for trait in OCEAN_TRAITS:
            self.models[trait] = model

        # Diagnostic metrics on the TRAINING set (explicitly NOT final performance)
        model.eval()
        with torch.no_grad():
            X_eval = X_padded.to(device)
            seq_eval = seq_lengths.to(device)
            train_preds_np = model(X_eval, seq_eval).cpu().numpy()

        train_diagnostics = compute_regression_metrics(targets, train_preds_np)

        logger.info(f"LSTM training completed. Final train loss: {train_loss_history[-1]:.4f}")

        return {
            'epochs': epochs,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history if has_val else None,
            'val_mae_history': val_mae_history if has_val else None,
            'best_val_mae': best_val_mae if has_val else None,
            'train_diagnostics': train_diagnostics,  # NOT final performance
            'sample_count': len(sequences),
        }

    # ------------------------------------------------------------------
    # Backward-compatible per-trait training wrapper
    # ------------------------------------------------------------------

    def train_trait_model(
        self,
        trait: str,
        sequences: List[np.ndarray],
        targets: np.ndarray,
        val_sequences: Optional[List[np.ndarray]] = None,
        val_targets: Optional[np.ndarray] = None,
        epochs: int = 35,
        batch_size: int = 4,
        sample_weights: Optional[np.ndarray] = None,
        seed: int = 42,
        # Deprecated args kept for call-site compatibility; no longer used
        low_threshold: float = 2.5,
        high_threshold: float = 3.5,
        precomputed_labels: Optional[np.ndarray] = None,
        val_precomputed_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Backward-compatible wrapper around `train()`.

        Previously trained one classifier per trait; now delegates to the
        joint 5-output regression trainer.  `targets` may be passed as
        (N,) for a single trait — in that case all other trait columns are
        zeroed (the model still trains jointly, but only the supplied
        trait's ground-truth is meaningful for that call).  Prefer passing
        full (N, 5) targets to `train()` directly.

        `low_threshold`, `high_threshold`, `precomputed_labels`, and
        `val_precomputed_labels` are accepted but ignored; they were
        classification-only concerns.
        """
        if low_threshold != 2.5 or high_threshold != 3.5:
            logger.warning(
                "train_trait_model: low_threshold/high_threshold are ignored — "
                "the model is now a continuous regressor and does not bin scores."
            )
        if precomputed_labels is not None or val_precomputed_labels is not None:
            logger.warning(
                "train_trait_model: precomputed_labels/val_precomputed_labels are ignored — "
                "the model trains directly on continuous targets."
            )

        targets_arr = np.asarray(targets)
        if targets_arr.ndim == 1:
            # Wrap single-trait targets into (N, 5); zero-fill other traits
            full_targets = np.zeros((len(targets_arr), NUM_TRAITS), dtype=np.float32)
            idx = self.TRAIT_INDEX.get(trait, 0)
            full_targets[:, idx] = targets_arr
            logger.warning(
                f"train_trait_model('{trait}'): received (N,) targets — wrapping into "
                f"(N, 5) with zeros for other traits. Pass full (N, 5) targets for best results."
            )
        else:
            full_targets = targets_arr

        full_val_targets = None
        if val_targets is not None:
            val_arr = np.asarray(val_targets)
            if val_arr.ndim == 1:
                full_val_targets = np.zeros((len(val_arr), NUM_TRAITS), dtype=np.float32)
                idx = self.TRAIT_INDEX.get(trait, 0)
                full_val_targets[:, idx] = val_arr
            else:
                full_val_targets = val_arr

        result = self.train(
            sequences=sequences,
            targets=full_targets,
            val_sequences=val_sequences,
            val_targets=full_val_targets,
            epochs=epochs,
            batch_size=batch_size,
            sample_weights=sample_weights,
            seed=seed,
        )
        result['trait'] = trait
        return result

    # ------------------------------------------------------------------
    # Prediction API
    # ------------------------------------------------------------------

    def predict(
        self,
        sequences: List[np.ndarray],
    ) -> np.ndarray:
        """
        Predict continuous OCEAN scores for N participants.

        Requires a trained model — there is no fallback.

        Args:
            sequences: List of (num_posts, 768) arrays, one per participant.

        Returns:
            predictions: np.ndarray of shape (N, 5) — continuous [O,C,E,A,N] scores.
        """
        if self.model is None:
            raise RuntimeError(
                "No trained LSTM model found. "
                "Call train() or train_trait_model() before requesting predictions."
            )

        self.model.eval()
        X_padded, seq_lengths = self._prepare_sequences(sequences)

        with torch.no_grad():
            X_eval = X_padded.to(device)
            seq_eval = seq_lengths.to(device)
            preds = self.model(X_eval, seq_eval)  # (N, 5)

        return preds.cpu().numpy()

    def predict_trait(
        self,
        trait: str,
        sequences: List[np.ndarray],
    ) -> np.ndarray:
        """
        Backward-compatible prediction for a single named trait.

        Returns (N, 5) continuous predictions — the full OCEAN vector — so
        callers receive the same contract as `predict()`.  Single-trait
        slicing (predictions[:, trait_index]) is left to the caller if needed.

        Previously returned (class_probabilities, predicted_classes, predicted_labels);
        that tuple return value was classification-only and is no longer produced.
        Callers that destructured the old 3-tuple will need to be updated.

        Args:
            trait:     OCEAN trait name (used only for logging/error messages).
            sequences: List of (num_posts, 768) arrays, one per participant.

        Returns:
            predictions: np.ndarray of shape (N, 5) — continuous OCEAN scores.
        """
        if self.model is None and trait not in self.models:
            raise RuntimeError(
                f"No trained LSTM model found for trait '{trait}'. "
                f"Call train() before requesting predictions."
            )
        return self.predict(sequences)

    # ------------------------------------------------------------------
    # Evaluation API
    # ------------------------------------------------------------------

    def evaluate_trait(
        self,
        trait: str,
        sequences: List[np.ndarray],
        targets: np.ndarray,
        # Deprecated args kept for call-site compatibility; no longer used
        low_threshold: float = 2.5,
        high_threshold: float = 3.5,
        precomputed_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model on a held-out (validation or test) split.

        Metrics are computed via `compute_regression_metrics` — the same
        engine used by the Lasso pathway — so results are directly comparable.

        Never call this on participants the model was trained on.

        Args:
            trait:             OCEAN trait name (used for context in returned dict).
            sequences:         Held-out participant sequences.
            targets:           Ground-truth OCEAN scores; may be (N,) for a single
                               trait or (N, 5) for all traits.
            low_threshold:     Ignored (retained for call-site compatibility).
            high_threshold:    Ignored (retained for call-site compatibility).
            precomputed_labels: Ignored (retained for call-site compatibility).

        Returns:
            Dict with per-trait MAE, RMSE, R², Pearson r, plus overall means.
        """
        if precomputed_labels is not None:
            logger.warning(
                "evaluate_trait: precomputed_labels is ignored — evaluation "
                "uses continuous targets only."
            )

        targets_arr = np.asarray(targets)
        preds = self.predict(sequences)  # (N, 5)

        if targets_arr.ndim == 1:
            # Single-trait ground-truth: evaluate only the relevant column
            idx = self.TRAIT_INDEX.get(trait, 0)
            metrics = compute_regression_metrics(
                targets_arr, preds[:, idx], trait_names=[trait]
            )
        else:
            metrics = compute_regression_metrics(targets_arr, preds)

        metrics['trait'] = trait
        return metrics