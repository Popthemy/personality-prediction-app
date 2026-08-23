"""
Conditional GAN augmentation for BERT representation vectors.

This module is intentionally limited to the augmentation layer.

Pipeline position:

    BERT representations
            |
            v
       Conditional GAN
            |
            v
    augmented representations
            |
            v
       ElasticNet / LSTM

The cGAN is trained only on caller-supplied training/development data.
Validation and final-test data are explicitly rejected by fit().

References:
    Goodfellow et al. (2014), Generative Adversarial Nets.
    Mirza & Osindero (2014), Conditional Generative Adversarial Nets.
    Antoniou et al. (2017), Data Augmentation Generative Adversarial Networks.
    Frid-Adar et al. (2018), GAN-based synthetic data augmentation.
    Foster, Generative Deep Learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


@dataclass
class CGANConfig:
    """Configuration for the lightweight vector cGAN."""

    latent_dim: int = 32
    hidden_dim: int = 128

    learning_rate: float = 2e-4
    batch_size: int = 32
    epochs: int = 100

    # Number of samples requested by default if generate() is called
    # without an explicit n_samples.
    n_samples: int = 100

    seed: int = 42

    device: str = "auto"

    # Adam settings commonly used for GAN training.
    beta1: float = 0.5
    beta2: float = 0.999

    # Small epsilon used for numerical protection in diagnostics.
    eps: float = 1e-8

    # Optional gradient clipping for stability on small datasets.
    gradient_clip_norm: Optional[float] = 5.0


@dataclass
class CGANDiagnostics:
    """Training and generated-data diagnostics."""

    generator_loss: List[float] = field(default_factory=list)
    discriminator_loss: List[float] = field(default_factory=list)

    real_samples: int = 0
    synthetic_samples: int = 0

    condition_distribution: Dict[str, int] = field(default_factory=dict)

    real_mean: Optional[np.ndarray] = None
    real_std: Optional[np.ndarray] = None

    generated_mean: Optional[np.ndarray] = None
    generated_std: Optional[np.ndarray] = None

    real_norm_mean: Optional[float] = None
    real_norm_std: Optional[float] = None

    generated_norm_mean: Optional[float] = None
    generated_norm_std: Optional[float] = None

    condition_diagnostics: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )


class Generator(nn.Module):
    """
    Conditional generator.

    Input:
        latent noise z
        personality condition c

    Output:
        synthetic BERT representation x_fake
    """

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(torch.cat([z, condition], dim=1))


class Discriminator(nn.Module):
    """
    Conditional discriminator.

    Input:
        BERT representation x
        personality condition c

    Output:
        single real/fake logit.
    """

    def __init__(
        self,
        embedding_dim: int,
        condition_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim + condition_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        embedding: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(
            torch.cat([embedding, condition], dim=1)
        )


class _ConditionEncoder:
    """
    Encodes personality targets without inventing a new target scheme.

    Categorical targets:
        ["Low", "Medium", "High"]

    Continuous targets:
        shape [n_samples, n_traits]

    Categorical conditions use one-hot vectors.

    Continuous conditions are standardized internally for GAN training.
    The original target values are returned by generate().
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

        self.mode: Optional[str] = None

        self.classes_: Optional[List[Any]] = None
        self.class_to_index_: Dict[Any, int] = {}

        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

        self.condition_dim_: Optional[int] = None

    @property
    def fitted(self) -> bool:
        return self.mode is not None

    def fit(self, y: Any) -> "_ConditionEncoder":
        arr = np.asarray(y)

        if arr.ndim == 1 and (
            arr.dtype.kind in {"U", "S", "O", "b", "i", "u"}
        ):
            self.mode = "categorical"

            values = arr.tolist()

            classes = []
            for value in values:
                if value not in classes:
                    classes.append(value)

            self.classes_ = classes
            self.class_to_index_ = {
                value: i for i, value in enumerate(classes)
            }

            self.condition_dim_ = len(classes)
            return self

        if arr.ndim == 1:
            # Numeric 1-D targets are interpreted as a single continuous
            # target rather than as class IDs.
            arr = arr.reshape(-1, 1)

        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
            self.mode = "continuous"

            arr = arr.astype(np.float32)

            self.mean_ = np.mean(arr, axis=0)
            self.std_ = np.std(arr, axis=0)

            self.std_ = np.where(
                self.std_ < self.eps,
                1.0,
                self.std_,
            )

            self.condition_dim_ = arr.shape[1]
            return self

        raise ValueError(
            "Unsupported target representation. Expected either "
            "categorical labels such as Low/Medium/High or a numeric "
            "continuous target matrix."
        )

    def transform(self, y: Any) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Condition encoder has not been fitted.")

        arr = np.asarray(y)

        if self.mode == "categorical":
            if arr.ndim != 1:
                raise ValueError(
                    "Categorical personality targets must be one-dimensional."
                )

            result = np.zeros(
                (len(arr), self.condition_dim_),
                dtype=np.float32,
            )

            for row, value in enumerate(arr.tolist()):
                if value not in self.class_to_index_:
                    raise ValueError(
                        f"Unknown personality condition: {value!r}. "
                        f"Known conditions: {self.classes_!r}"
                    )

                result[row, self.class_to_index_[value]] = 1.0

            return result

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        arr = arr.astype(np.float32)

        if arr.shape[1] != self.condition_dim_:
            raise ValueError(
                f"Expected {self.condition_dim_} continuous condition "
                f"dimensions, received {arr.shape[1]}."
            )

        return (arr - self.mean_) / self.std_

    def inverse_transform(self, condition: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Condition encoder has not been fitted.")

        condition = np.asarray(condition)

        if self.mode == "categorical":
            indices = np.argmax(condition, axis=1)
            return np.asarray(
                [self.classes_[int(i)] for i in indices],
                dtype=object,
            )

        return condition * self.std_ + self.mean_

    def condition_for_generation(
        self,
        condition: Any,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a requested condition into:
            1. encoded condition used by the generator
            2. original target returned downstream
        """

        if not self.fitted:
            raise RuntimeError("Condition encoder has not been fitted.")

        if n_samples <= 0:
            raise ValueError("n_samples must be greater than zero.")

        if self.mode == "categorical":
            # Example:
            # generate(condition="High", n_samples=100)
            if condition not in self.class_to_index_:
                raise ValueError(
                    f"Unknown condition {condition!r}. "
                    f"Expected one of {self.classes_!r}."
                )

            encoded = np.zeros(
                (n_samples, self.condition_dim_),
                dtype=np.float32,
            )

            encoded[:, self.class_to_index_[condition]] = 1.0

            targets = np.asarray(
                [condition] * n_samples,
                dtype=object,
            )

            return encoded, targets

        # Continuous OCEAN target(s).
        arr = np.asarray(condition, dtype=np.float32)

        if arr.ndim == 0:
            if self.condition_dim_ != 1:
                raise ValueError(
                    f"A scalar condition was supplied, but this project's "
                    f"continuous target has {self.condition_dim_} "
                    f"dimensions. Supply a vector of length "
                    f"{self.condition_dim_} (e.g. the full OCEAN target)."
                )

            arr = arr.reshape(1, 1)

        elif arr.ndim == 1:
            if arr.shape[0] != self.condition_dim_:
                raise ValueError(
                    f"Continuous condition must contain "
                    f"{self.condition_dim_} target values."
                )

            arr = arr.reshape(1, -1)

        elif arr.ndim == 2:
            if arr.shape[1] != self.condition_dim_:
                raise ValueError(
                    f"Continuous condition must have "
                    f"{self.condition_dim_} columns."
                )

        else:
            raise ValueError("Invalid continuous condition shape.")

        if arr.shape[0] == 1:
            targets = np.repeat(arr, n_samples, axis=0)
        elif arr.shape[0] == n_samples:
            targets = arr.copy()
        else:
            raise ValueError(
                "Continuous generation conditions must either contain "
                "one target vector or exactly n_samples target vectors."
            )

        encoded = (targets - self.mean_) / self.std_

        return encoded.astype(np.float32), targets.astype(np.float32)

    def describe(self) -> Dict[str, Any]:
        if self.mode == "categorical":
            return {
                "mode": "categorical",
                "classes": list(self.classes_),
                "condition_dim": self.condition_dim_,
            }

        return {
            "mode": "continuous",
            "condition_dim": self.condition_dim_,
            "mean": self.mean_.copy(),
            "std": self.std_.copy(),
        }


class ConditionalGANAugmenter:
    """
    Plug-and-play conditional GAN augmenter for BERT vectors.

    Typical usage:

        augmenter = ConditionalGANAugmenter(
            embedding_dim=768,
            config=CGANConfig(
                seed=42,
                epochs=100,
                latent_dim=32,
            ),
        )

        augmenter.fit(
            X_train,
            y_train,
            split="train",
        )

        X_syn, y_syn = augmenter.generate(
            condition="High",
            n_samples=100,
        )

    The resulting X_syn/y_syn can be concatenated with the existing
    training arrays before fitting ElasticNet/LSTM.

    Important:
        fit() requires split="train" or split="development".
        "validation" and "test" are rejected.
    """

    ALLOWED_TRAINING_SPLITS = {"train", "development"}
    FORBIDDEN_SPLITS = {"validation", "val", "test", "holdout", "final_test"}

    def __init__(
        self,
        embedding_dim: int,
        config: Optional[CGANConfig] = None,
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero.")

        self.embedding_dim = int(embedding_dim)
        self.config = config or CGANConfig()

        if self.config.latent_dim <= 0:
            raise ValueError("latent_dim must be greater than zero.")

        if self.config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than zero.")

        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if self.config.epochs <= 0:
            raise ValueError("epochs must be greater than zero.")

        self.device = self._resolve_device(self.config.device)

        self.condition_encoder = _ConditionEncoder(
            eps=self.config.eps
        )

        self.generator: Optional[Generator] = None
        self.discriminator: Optional[Discriminator] = None

        self.g_optimizer: Optional[torch.optim.Optimizer] = None
        self.d_optimizer: Optional[torch.optim.Optimizer] = None

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.diagnostics = CGANDiagnostics()

        self._fitted = False
        self._training_split: Optional[str] = None

        self._real_X: Optional[np.ndarray] = None
        self._real_y: Optional[np.ndarray] = None

        self._set_seed(self.config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: ArrayLike,
        y_train: Any,
        *,
        split: str = "train",
    ) -> "ConditionalGANAugmenter":
        """
        Train the cGAN.

        split is deliberately explicit so validation/final-test data cannot
        accidentally be used by a normal call.

        Accepted:
            "train"
            "development"

        Rejected:
            "validation"
            "val"
            "test"
            "final_test"
            "holdout"
        """

        normalized_split = str(split).strip().lower()

        if normalized_split in self.FORBIDDEN_SPLITS:
            raise ValueError(
                f"cGAN fitting on split={split!r} is forbidden. "
                "The cGAN must never be trained on validation or test "
                "participants."
            )

        if normalized_split not in self.ALLOWED_TRAINING_SPLITS:
            raise ValueError(
                "split must explicitly be 'train' or 'development'. "
                "Validation and final-test data are not permitted."
            )

        X = self._validate_embeddings(X_train)
        y = self._validate_targets(y_train, len(X))

        # Fit condition representation exclusively on the supplied
        # training/development data.
        self.condition_encoder.fit(y)

        condition = self.condition_encoder.transform(y)

        if not np.isfinite(condition).all():
            raise ValueError(
                "Personality conditions contain NaN or Inf values."
            )

        self._build_models()

        self._real_X = X.copy()
        self._real_y = self._copy_targets(y)
        self._training_split = normalized_split

        self.diagnostics = CGANDiagnostics(
            real_samples=len(X),
            real_mean=np.mean(X, axis=0),
            real_std=np.std(X, axis=0),
        )

        real_norms = np.linalg.norm(X, axis=1)

        self.diagnostics.real_norm_mean = float(
            np.mean(real_norms)
        )
        self.diagnostics.real_norm_std = float(
            np.std(real_norms)
        )

        self.diagnostics.condition_distribution = (
            self._condition_distribution(y)
        )

        dataset = TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(condition.astype(np.float32)),
        )

        # A small dataset may be smaller than the configured batch size.
        batch_size = min(
            self.config.batch_size,
            max(1, len(dataset)),
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        self._train(loader)

        self._fitted = True

        return self

    def generate(
        self,
        *,
        condition: Any,
        n_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, Any]:
        """
        Generate synthetic embeddings associated with an explicit
        personality condition.

        Categorical example:

            X_syn, y_syn = augmenter.generate(
                condition="High",
                n_samples=100,
            )

        Continuous OCEAN example:

            X_syn, y_syn = augmenter.generate(
                condition=[0.72, 0.55, 0.31, 0.61, 0.80],
                n_samples=100,
            )

        The returned target values are the same condition used to request
        generation. They are never inferred independently by the GAN.
        """

        self._require_fitted()

        n = (
            self.config.n_samples
            if n_samples is None
            else int(n_samples)
        )

        if n <= 0:
            raise ValueError("n_samples must be greater than zero.")

        encoded_condition, targets = (
            self.condition_encoder.condition_for_generation(
                condition,
                n,
            )
        )

        condition_tensor = torch.from_numpy(
            encoded_condition
        ).float().to(self.device)

        self.generator.eval()

        with torch.no_grad():
            z = torch.randn(
                n,
                self.config.latent_dim,
                device=self.device,
            )

            X_syn = self.generator(
                z,
                condition_tensor,
            ).cpu().numpy()

        self._validate_generated_embeddings(X_syn)

        # Store diagnostics without modifying the generated data.
        self._record_generation_diagnostics(
            X_syn,
            targets,
        )

        return X_syn.astype(np.float32), self._copy_targets(targets)

    def generate_dataset(
        self,
        requests: Dict[Any, int],
    ) -> Tuple[np.ndarray, Any, np.ndarray]:
        """
        Generate a complete synthetic training set.

        Example:

            X_syn, y_syn, synthetic = augmenter.generate_dataset({
                "Low": 50,
                "Medium": 50,
                "High": 100,
            })

        Returns:
            X_syn
            y_syn
            synthetic -- boolean array, always True
        """

        if not requests:
            raise ValueError("requests cannot be empty.")

        X_parts = []
        y_parts = []

        for condition, count in requests.items():
            X_part, y_part = self.generate(
                condition=condition,
                n_samples=count,
            )

            X_parts.append(X_part)
            y_parts.append(y_part)

        X_syn = np.concatenate(X_parts, axis=0)

        if self.condition_encoder.mode == "categorical":
            y_syn = np.concatenate(y_parts, axis=0)
        else:
            y_syn = np.concatenate(y_parts, axis=0)

        synthetic = np.ones(
            len(X_syn),
            dtype=bool,
        )

        self.diagnostics.synthetic_samples = len(X_syn)

        return X_syn, y_syn, synthetic

    def augment(
        self,
        X_train: ArrayLike,
        y_train: Any,
        requests: Dict[Any, int],
        *,
        split: str = "train",
    ) -> Tuple[np.ndarray, Any, np.ndarray]:
        """
        Convenience method implementing the augmentation abstraction.

        It trains the cGAN on the supplied training data and returns only
        synthetic samples. Existing downstream code can concatenate them
        with the original training data.
        """

        self.fit(
            X_train,
            y_train,
            split=split,
        )

        return self.generate_dataset(requests)

    def diagnostics_report(self) -> Dict[str, Any]:
        """Return serializable training and quality diagnostics."""

        self._require_fitted()

        return {
            "training_split": self._training_split,
            "embedding_dim": self.embedding_dim,
            "condition": self.condition_encoder.describe(),
            "configuration": {
                "latent_dim": self.config.latent_dim,
                "hidden_dim": self.config.hidden_dim,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "n_samples": self.config.n_samples,
                "seed": self.config.seed,
                "device": str(self.device),
            },
            "real_samples": self.diagnostics.real_samples,
            "synthetic_samples": self.diagnostics.synthetic_samples,
            "condition_distribution": (
                self.diagnostics.condition_distribution
            ),
            "generator_loss": list(
                self.diagnostics.generator_loss
            ),
            "discriminator_loss": list(
                self.diagnostics.discriminator_loss
            ),
            "real_norm_mean": self.diagnostics.real_norm_mean,
            "real_norm_std": self.diagnostics.real_norm_std,
            "generated_norm_mean": (
                self.diagnostics.generated_norm_mean
            ),
            "generated_norm_std": (
                self.diagnostics.generated_norm_std
            ),
            "condition_diagnostics": (
                self.diagnostics.condition_diagnostics
            ),
        }

    # ------------------------------------------------------------------
    # Model construction/training
    # ------------------------------------------------------------------

    def _build_models(self) -> None:
        condition_dim = self.condition_encoder.condition_dim_

        if condition_dim is None:
            raise RuntimeError("Condition encoder has no dimension.")

        self.generator = Generator(
            latent_dim=self.config.latent_dim,
            condition_dim=condition_dim,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)

        self.discriminator = Discriminator(
            embedding_dim=self.embedding_dim,
            condition_dim=condition_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.learning_rate,
            betas=(
                self.config.beta1,
                self.config.beta2,
            ),
        )

        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.learning_rate,
            betas=(
                self.config.beta1,
                self.config.beta2,
            ),
        )

    def _train(self, loader: DataLoader) -> None:
        assert self.generator is not None
        assert self.discriminator is not None
        assert self.g_optimizer is not None
        assert self.d_optimizer is not None

        self.generator.train()
        self.discriminator.train()

        for _epoch in range(self.config.epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            batches = 0

            for real_X, condition in loader:
                real_X = real_X.to(self.device)
                condition = condition.to(self.device)

                batch_size = real_X.shape[0]

                # ------------------------------------------------------
                # 1. Discriminator
                # ------------------------------------------------------
                self.d_optimizer.zero_grad(set_to_none=True)

                z = torch.randn(
                    batch_size,
                    self.config.latent_dim,
                    device=self.device,
                )

                fake_X = self.generator(
                    z,
                    condition,
                )

                real_logits = self.discriminator(
                    real_X,
                    condition,
                )

                fake_logits = self.discriminator(
                    fake_X.detach(),
                    condition,
                )

                real_targets = torch.ones_like(real_logits)
                fake_targets = torch.zeros_like(fake_logits)

                d_real_loss = self.loss_fn(
                    real_logits,
                    real_targets,
                )

                d_fake_loss = self.loss_fn(
                    fake_logits,
                    fake_targets,
                )

                d_loss = (
                    d_real_loss + d_fake_loss
                ) / 2.0

                d_loss.backward()

                if self.config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(),
                        self.config.gradient_clip_norm,
                    )

                self.d_optimizer.step()

                # ------------------------------------------------------
                # 2. Generator
                # ------------------------------------------------------
                self.g_optimizer.zero_grad(set_to_none=True)

                z = torch.randn(
                    batch_size,
                    self.config.latent_dim,
                    device=self.device,
                )

                fake_X = self.generator(
                    z,
                    condition,
                )

                fake_logits = self.discriminator(
                    fake_X,
                    condition,
                )

                # Generator tries to make conditional fake examples
                # classified as real.
                g_targets = torch.ones_like(fake_logits)

                g_loss = self.loss_fn(
                    fake_logits,
                    g_targets,
                )

                g_loss.backward()

                if self.config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.generator.parameters(),
                        self.config.gradient_clip_norm,
                    )

                self.g_optimizer.step()

                epoch_d_loss += float(d_loss.item())
                epoch_g_loss += float(g_loss.item())
                batches += 1

            self.diagnostics.discriminator_loss.append(
                epoch_d_loss / max(batches, 1)
            )

            self.diagnostics.generator_loss.append(
                epoch_g_loss / max(batches, 1)
            )

    # ------------------------------------------------------------------
    # Validation and safety checks
    # ------------------------------------------------------------------

    def _validate_embeddings(
        self,
        X: ArrayLike,
    ) -> np.ndarray:
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                "BERT representations must be a 2-D matrix "
                "[n_samples, embedding_dim]."
            )

        if X.shape[1] != self.embedding_dim:
            raise ValueError(
                f"BERT embedding dimension mismatch: expected "
                f"{self.embedding_dim}, received {X.shape[1]}."
            )

        if len(X) == 0:
            raise ValueError(
                "cGAN cannot be trained with zero samples."
            )

        if not np.issubdtype(X.dtype, np.number):
            raise ValueError(
                "BERT representations must contain numeric values."
            )

        X = X.astype(np.float32)

        if not np.isfinite(X).all():
            raise ValueError(
                "BERT representations contain NaN or Inf values."
            )

        return X

    def _validate_targets(
        self,
        y: Any,
        n_samples: int,
    ) -> np.ndarray:
        arr = np.asarray(y)

        if len(arr) != n_samples:
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        if arr.ndim == 1:
            return arr.copy()

        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
            arr = arr.astype(np.float32)

            if not np.isfinite(arr).all():
                raise ValueError(
                    "Continuous personality targets contain NaN or Inf."
                )

            return arr

        raise ValueError(
            "Targets must be categorical 1-D labels or a numeric "
            "2-D continuous target matrix."
        )

    def _validate_generated_embeddings(
        self,
        X_syn: np.ndarray,
    ) -> None:
        if X_syn.ndim != 2:
            raise RuntimeError(
                "Generator returned an invalid tensor rank."
            )

        if X_syn.shape[1] != self.embedding_dim:
            raise RuntimeError(
                f"Generator output dimension {X_syn.shape[1]} does not "
                f"match BERT dimension {self.embedding_dim}."
            )

        if not np.isfinite(X_syn).all():
            raise RuntimeError(
                "Generator produced NaN or Inf values."
            )

        norms = np.linalg.norm(X_syn, axis=1)

        if not np.isfinite(norms).all():
            raise RuntimeError(
                "Generator produced invalid embedding norms."
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _record_generation_diagnostics(
        self,
        X_syn: np.ndarray,
        y_syn: Any,
    ) -> None:
        generated_norms = np.linalg.norm(X_syn, axis=1)

        self.diagnostics.generated_mean = np.mean(
            X_syn,
            axis=0,
        )

        self.diagnostics.generated_std = np.std(
            X_syn,
            axis=0,
        )

        self.diagnostics.generated_norm_mean = float(
            np.mean(generated_norms)
        )

        self.diagnostics.generated_norm_std = float(
            np.std(generated_norms)
        )

        # Basic global distribution diagnostics.
        if self._real_X is not None:
            real_mean = np.mean(self._real_X, axis=0)
            real_std = np.std(self._real_X, axis=0)

            mean_error = float(
                np.mean(
                    np.abs(
                        self.diagnostics.generated_mean
                        - real_mean
                    )
                )
            )

            std_error = float(
                np.mean(
                    np.abs(
                        self.diagnostics.generated_std
                        - real_std
                    )
                )
            )

            self.diagnostics.condition_diagnostics[
                "global"
            ] = {
                "mean_absolute_mean_difference": mean_error,
                "mean_absolute_std_difference": std_error,
                "real_norm_mean": self.diagnostics.real_norm_mean,
                "generated_norm_mean": (
                    self.diagnostics.generated_norm_mean
                ),
            }

        # Condition-specific diagnostics.
        if self.condition_encoder.mode == "categorical":
            for label in self.condition_encoder.classes_:
                mask = np.asarray(y_syn) == label

                if not np.any(mask):
                    continue

                X_condition = X_syn[mask]

                self.diagnostics.condition_diagnostics[
                    str(label)
                ] = {
                    "synthetic_samples": int(np.sum(mask)),
                    "mean": np.mean(
                        X_condition,
                        axis=0,
                    ),
                    "std": np.std(
                        X_condition,
                        axis=0,
                    ),
                    "norm_mean": float(
                        np.mean(
                            np.linalg.norm(
                                X_condition,
                                axis=1,
                            )
                        )
                    ),
                }

    def _condition_distribution(
        self,
        y: np.ndarray,
    ) -> Dict[str, int]:
        if self.condition_encoder.mode == "categorical":
            unique, counts = np.unique(
                y,
                return_counts=True,
            )

            return {
                str(label): int(count)
                for label, count in zip(unique, counts)
            }

        return {
            "continuous_target_matrix": int(len(y)),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "The cGAN must be fitted before generation."
            )

    @staticmethod
    def _copy_targets(y: Any) -> Any:
        if isinstance(y, np.ndarray):
            return y.copy()

        return np.asarray(y).copy()

    @staticmethod
    def _set_seed(seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Deterministic settings make experiments more reproducible.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")

            return torch.device("cpu")

        resolved = torch.device(device)

        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return resolved
