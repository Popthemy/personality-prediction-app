"""
Stacked LSTM Neural Network Service for Personality Classification.

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
                 3 logits
                    |
                 Softmax
                    |
          Low / Medium / High

Notes:
- Single classification head only. There is no regression head in this module —
  continuous OCEAN score prediction (1.0-5.0) belongs to the separate
  Lasso/ElasticNet pathway, which consumes the same BERT-derived features
  through its own, independent code path.
- Loss is a single CrossEntropyLoss over 3 classes, not a joint
  BCE + SmoothL1 multi-task loss.
- Handles sequence padding and dynamic sequence lengths via
  pack_padded_sequence / pad_packed_sequence.
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
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger('ml_pipeline')

# Device configuration (GPU if available, CPU fallback)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class label mapping used throughout this module
CLASS_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}
NUM_CLASSES = 3


def set_seed(seed: int = 42) -> None:
    """
    Fix all relevant RNGs so training runs are reproducible
    (checklist §7: 'Keep the training process reproducible').
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_participants(
    participant_ids: List[Any],
    labels: Optional[List[int]] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Split PARTICIPANT IDs (not individual posts) into train/val/test sets
    (checklist §8: split at the participant level, so posts from one
    participant never appear in more than one split).

    Args:
        participant_ids: unique identifier per participant.
        labels: optional class label per participant, used to stratify
            the split so class proportions are preserved across splits.
        test_size: fraction of participants held out for the final test set.
        val_size: fraction of the *remaining* (non-test) participants held
            out for validation.
        random_state: fixed seed for reproducibility.

    Returns:
        (train_ids, val_ids, test_ids)
    """
    stratify = labels if labels is not None else None
    train_val_ids, test_ids, train_val_labels, _ = train_test_split(
        participant_ids, labels if labels is not None else participant_ids,
        test_size=test_size, random_state=random_state, stratify=stratify
    )

    stratify_2 = train_val_labels if labels is not None else None
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_size / (1.0 - test_size),
        random_state=random_state, stratify=stratify_2
    )

    return train_ids, val_ids, test_ids


def check_class_distribution(labels: np.ndarray, trait: str = "") -> Dict[str, Any]:
    """
    Report the Low/Medium/High class distribution for a label array and
    flag imbalance (checklist §6: check distribution, avoid/document
    imbalance). This should be run on train, val, and test splits
    separately and logged before evaluation.
    """
    counts = Counter(labels.tolist() if isinstance(labels, np.ndarray) else labels)
    total = sum(counts.values())
    distribution = {
        CLASS_LABELS[c]: {
            'count': counts.get(c, 0),
            'proportion': counts.get(c, 0) / total if total else 0.0
        }
        for c in range(NUM_CLASSES)
    }

    proportions = [distribution[CLASS_LABELS[c]]['proportion'] for c in range(NUM_CLASSES)]
    is_imbalanced = (max(proportions) > 0.6) or (min(proportions) < 0.10) if total else False

    if is_imbalanced:
        logger.warning(f"Class imbalance detected for trait '{trait}': {distribution}")
    else:
        logger.info(f"Class distribution for trait '{trait}': {distribution}")

    return {'trait': trait, 'total': total, 'distribution': distribution, 'is_imbalanced': is_imbalanced}


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Compute the LSTM evaluation metrics required by checklist §9:
    accuracy, macro precision/recall/F1, class-wise breakdown, and
    a confusion matrix. Never report regression metrics here.
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0, labels=[0, 1, 2]
    )
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2]
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    class_wise = {
        CLASS_LABELS[c]: {
            'precision': float(per_class_precision[c]),
            'recall': float(per_class_recall[c]),
            'f1': float(per_class_f1[c]),
            'support': int(per_class_support[c]),
        }
        for c in range(NUM_CLASSES)
    }

    return {
        'accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'class_wise': class_wise,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': [CLASS_LABELS[c] for c in range(NUM_CLASSES)],
        'report_text': classification_report(
            y_true, y_pred, labels=[0, 1, 2],
            target_names=[CLASS_LABELS[c] for c in range(NUM_CLASSES)],
            zero_division=0
        ),
    }


class StackedLSTMClassifier(nn.Module):
    """
    Stacked Bidirectional LSTM model for sequence-based personality trait
    classification into Low / Medium / High.

    Processes sequences of pre-computed BERT embeddings
    (Batch, Max_Seq_Len, 768) and outputs 3-class logits. Does not
    implement or call BERT, and does not perform regression — this model's
    only responsibility is: given a sequence of BERT embeddings, classify
    the trait into one of 3 classes.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super(StackedLSTMClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
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

        # Single classification head -> 3 logits (Low / Medium / High).
        # No Sigmoid here — CrossEntropyLoss expects raw logits and applies
        # log-softmax internally during training. Softmax is applied
        # explicitly at inference time in predict_trait().
        self.classifier_head = nn.Sequential(
            nn.Linear(effective_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
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
            seq_lengths: Optional lengths of each sequence in the batch

        Returns:
            logits: Tensor of shape (batch_size, num_classes) — raw,
                    unnormalized class scores for Low/Medium/High.
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

        logits = self.classifier_head(pooled)  # (batch_size, num_classes)

        return logits


class LSTMTrainer:
    """
    Trainer and Inference helper for StackedLSTMClassifier.

    Trains one LSTM classifier per OCEAN trait, mapping a continuous
    score (or precomputed label) into Low / Medium / High.
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

    @staticmethod
    def _bin_scores_to_classes(
        targets: np.ndarray,
        low_threshold: float,
        high_threshold: float
    ) -> np.ndarray:
        """
        Convert continuous OCEAN scores (e.g. 1.0-5.0) into 3-class labels.

        score <  low_threshold            -> 0 (Low)
        low_threshold <= score < high_threshold -> 1 (Medium)
        score >= high_threshold           -> 2 (High)
        """
        labels = np.ones_like(targets, dtype=np.int64)  # default Medium
        labels[targets < low_threshold] = 0
        labels[targets >= high_threshold] = 2
        return labels

    def train_trait_model(
        self,
        trait: str,
        sequences: List[np.ndarray],
        targets: np.ndarray,
        val_sequences: Optional[List[np.ndarray]] = None,
        val_targets: Optional[np.ndarray] = None,
        low_threshold: float = 2.5,
        high_threshold: float = 3.5,
        precomputed_labels: Optional[np.ndarray] = None,
        val_precomputed_labels: Optional[np.ndarray] = None,
        epochs: int = 35,
        batch_size: int = 4,
        sample_weights: Optional[np.ndarray] = None,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Train a StackedLSTMClassifier model for a specific OCEAN trait.

        `sequences`/`targets` must be TRAINING participants only.
        `val_sequences`/`val_targets` (strongly recommended) must be a
        disjoint set of VALIDATION participants — never the same
        participants as training (checklist §8). Validation performance,
        not training loss, is what should be monitored and reported
        (checklist §7): the dict returned here labels its two blocks
        explicitly as `train_diagnostics` and `validation_metrics` so
        the two are never confused as "the model's performance."

        Args:
            trait: Name of trait (e.g. 'Openness')
            sequences: List of arrays per training participant, shape (num_posts, 768)
            targets: Continuous target values for training participants, used to
                derive Low/Medium/High labels via low_threshold/high_threshold
                unless precomputed_labels is provided.
            val_sequences / val_targets: held-out validation participants,
                used only to monitor generalization each epoch.
            low_threshold / high_threshold: class boundary cutoffs. These
                must be decided and documented BEFORE evaluation (checklist
                §6) and applied identically to train/val/test.
            precomputed_labels / val_precomputed_labels: optional {0,1,2}
                labels to use directly instead of binning targets.
            epochs, batch_size, sample_weights: standard training hyperparameters.
            seed: random seed fixed for reproducibility.

        Returns:
            Dict containing loss/validation history, class-distribution
            checks for both splits, and 'train_diagnostics' — explicitly
            NOT to be reported as final performance (use evaluate_trait
            on the held-out test set for that).
        """
        if len(sequences) == 0:
            raise ValueError("Cannot train LSTM with empty sequence list.")

        set_seed(seed)

        X_padded, seq_lengths = self._prepare_sequences(sequences)

        if precomputed_labels is not None:
            y_cls_np = np.asarray(precomputed_labels)
        else:
            y_cls_np = self._bin_scores_to_classes(np.asarray(targets), low_threshold, high_threshold)
        y_cls = torch.tensor(y_cls_np, dtype=torch.long)

        train_distribution = check_class_distribution(y_cls_np, trait=f"{trait} (train)")

        has_val = val_sequences is not None and len(val_sequences) > 0
        val_distribution = None
        if has_val:
            X_val_padded, val_seq_lengths = self._prepare_sequences(val_sequences)
            if val_precomputed_labels is not None:
                y_val_np = np.asarray(val_precomputed_labels)
            else:
                y_val_np = self._bin_scores_to_classes(np.asarray(val_targets), low_threshold, high_threshold)
            y_val = torch.tensor(y_val_np, dtype=torch.long)
            val_distribution = check_class_distribution(y_val_np, trait=f"{trait} (val)")

        weights = (
            torch.tensor(sample_weights, dtype=torch.float32)
            if sample_weights is not None
            else torch.ones(len(sequences), dtype=torch.float32)
        )

        dataset = TensorDataset(X_padded, seq_lengths, y_cls, weights)
        dataloader = DataLoader(dataset, batch_size=min(batch_size, len(sequences)), shuffle=True)

        model = StackedLSTMClassifier(
            input_dim=X_padded.shape[2],
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_classes=NUM_CLASSES,
            dropout=self.dropout
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        ce_criterion = nn.CrossEntropyLoss(reduction='none')

        train_loss_history = []
        val_loss_history = []
        val_macro_f1_history = []
        best_val_macro_f1 = -1.0
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
                logits = model(batch_x, batch_len)

                per_sample_loss = ce_criterion(logits, batch_y)
                loss = (per_sample_loss * batch_w).mean()

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            train_loss_history.append(epoch_loss / max(1, len(dataloader)))

            # --- Validation monitoring (checklist §7) --------------------
            if has_val:
                model.eval()
                with torch.no_grad():
                    X_v = X_val_padded.to(device)
                    len_v = val_seq_lengths.to(device)
                    y_v = y_val.to(device)
                    val_logits = model(X_v, len_v)
                    val_loss = ce_criterion(val_logits, y_v).mean().item()
                    val_preds = torch.argmax(val_logits, dim=-1).cpu().numpy()

                val_loss_history.append(val_loss)
                _, _, val_macro_f1, _ = precision_recall_fscore_support(
                    y_val_np, val_preds, average='macro', zero_division=0, labels=[0, 1, 2]
                )
                val_macro_f1_history.append(float(val_macro_f1))

                if val_macro_f1 > best_val_macro_f1:
                    best_val_macro_f1 = val_macro_f1
                    best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        # Restore best-on-validation weights if validation was provided;
        # otherwise keep the final-epoch weights (and log that no
        # validation-based model selection occurred).
        if has_val and best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            logger.info(f"'{trait}': restored epoch with best validation macro-F1 = {best_val_macro_f1:.4f}")
        else:
            logger.warning(
                f"'{trait}': no validation set provided — model selection defaulted to the "
                f"final training epoch. Provide val_sequences/val_targets for a real check."
            )

        self.models[trait] = model

        # Diagnostic-only metrics on the training set. Explicitly labeled
        # so these are never mistaken for reportable performance
        # (checklist §7: "Do not use training-set predictions as final
        # performance results.") Use evaluate_trait() on val/test instead.
        model.eval()
        with torch.no_grad():
            X_eval = X_padded.to(device)
            seq_eval = seq_lengths.to(device)
            logits = model(X_eval, seq_eval)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        train_diagnostics = compute_classification_metrics(y_cls_np, preds.cpu().numpy())

        logger.info(f"LSTM training completed for {trait}. Final train loss: {train_loss_history[-1]:.4f}")

        return {
            'trait': trait,
            'epochs': epochs,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history if has_val else None,
            'val_macro_f1_history': val_macro_f1_history if has_val else None,
            'best_val_macro_f1': best_val_macro_f1 if has_val else None,
            'train_class_distribution': train_distribution,
            'val_class_distribution': val_distribution,
            'train_diagnostics': train_diagnostics,  # NOT final performance — see docstring
            'sample_count': len(sequences),
            'low_threshold': low_threshold,
            'high_threshold': high_threshold,
        }

    def predict_trait(
        self,
        trait: str,
        sequences: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Perform inference for a specific trait.

        Requires a trained model for this trait — there is no fallback.
        Research predictions must come from an actual trained LSTM
        (checklist §11: "Require a trained LSTM model before producing
        research predictions").

        Returns:
            Tuple of (class_probabilities, predicted_classes, predicted_labels)
            - class_probabilities: (N, 3) softmax probabilities
            - predicted_classes: (N,) argmax class index {0,1,2}
            - predicted_labels: (N,) list of 'Low'/'Medium'/'High'
        """
        if trait not in self.models:
            raise RuntimeError(
                f"No trained LSTM model found for trait '{trait}'. "
                f"Call train_trait_model('{trait}', ...) before requesting predictions — "
                f"there is no default/fallback prediction for research use."
            )

        model = self.models[trait]
        model.eval()
        X_padded, seq_lengths = self._prepare_sequences(sequences)

        with torch.no_grad():
            X_eval = X_padded.to(device)
            seq_eval = seq_lengths.to(device)
            logits = model(X_eval, seq_eval)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        class_probabilities = probs.cpu().numpy()
        predicted_classes = preds.cpu().numpy()
        predicted_labels = [CLASS_LABELS[c] for c in predicted_classes]

        return class_probabilities, predicted_classes, predicted_labels

    def evaluate_trait(
        self,
        trait: str,
        sequences: List[np.ndarray],
        targets: Optional[np.ndarray] = None,
        low_threshold: float = 2.5,
        high_threshold: float = 3.5,
        precomputed_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model on a held-out (validation or test) split
        and report the metrics required by checklist §9: accuracy, macro
        precision/recall/F1, class-wise breakdown, and confusion matrix.
        Never call this on the participants the model was trained on.
        """
        if precomputed_labels is not None:
            y_true = np.asarray(precomputed_labels)
        elif targets is not None:
            y_true = self._bin_scores_to_classes(np.asarray(targets), low_threshold, high_threshold)
        else:
            raise ValueError("evaluate_trait requires either `targets` or `precomputed_labels`.")

        _, predicted_classes, _ = self.predict_trait(trait, sequences)

        distribution = check_class_distribution(y_true, trait=f"{trait} (eval)")
        metrics = compute_classification_metrics(y_true, predicted_classes)
        metrics['class_distribution'] = distribution
        metrics['trait'] = trait
        return metrics