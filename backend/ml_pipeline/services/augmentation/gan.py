"""
Standalone GAN-based augmentation for participant-level BERT embeddings.

This module intentionally operates only on embeddings supplied by the caller.
It does not access BERT, ElasticNet, LSTM, validation data, or test data.

Typical usage:

    augmenter = GANAugmenter(
        embedding_dim=X_train.shape[1],
        latent_dim=64,
        epochs=200,
        batch_size=32,
        learning_rate=2e-4,
        seed=42,
    )

    augmenter.fit(X_train)

    X_syn, y_syn, metadata = augmenter.generate(
        n_samples=500,
        targets=None,
    )

    X_train_aug = np.concatenate([X_train, X_syn])
    y_train_aug = np.concatenate([y_train, y_syn])

Only X_train is passed to fit().
Validation/test data must never be passed to fit().
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GANConfig:
    """Configuration for the standalone GAN augmenter."""

    latent_dim: int = 64
    hidden_dim: int = 128

    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 2e-4
    betas: tuple[float, float] = (0.5, 0.999)

    seed: int = 42
    device: str = "auto"

    # Numerical safeguards.
    gradient_clip: Optional[float] = 5.0

    # Prevent pathological generated magnitudes.
    plausibility_std_multiplier: float = 4.0

    # If True, generated samples are clipped to the observed real range.
    # This is a safety check, not Gaussian augmentation.
    clip_to_real_range: bool = True


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """
    Lightweight MLP generator.

    latent noise z -> synthetic BERT representation
    """

    def __init__(
        self,
        latent_dim: int,
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """
    Lightweight MLP discriminator.

    BERT representation -> probability that representation is real.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).view(-1)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class GANDiagnostics:
    generator_loss: list[float]
    discriminator_loss: list[float]

    real_samples: int
    synthetic_samples: int

    embedding_dim: int

    real_mean: float
    real_std: float
    synthetic_mean: float
    synthetic_std: float

    real_mean_norm: float
    synthetic_mean_norm: float

    real_has_nan: bool
    real_has_inf: bool
    synthetic_has_nan: bool
    synthetic_has_inf: bool

    configuration: dict


@dataclass
class SyntheticMetadata:
    """Metadata accompanying generated samples."""

    is_synthetic: np.ndarray
    source: np.ndarray
    generator_seed: int


# ---------------------------------------------------------------------------
# GAN augmenter
# ---------------------------------------------------------------------------

class GANAugmenter:
    """
    Standalone GAN augmentation service for BERT embeddings.

    Important data-boundary rule:

        fit(X_train)

    must receive TRAINING participants only.

    The class has no access to validation or test datasets and therefore
    cannot accidentally train on them unless the caller explicitly passes
    them to fit().
    """

    def __init__(
        self,
        embedding_dim: Optional[int] = None,
        config: Optional[GANConfig] = None,
        **config_overrides,
    ) -> None:

        self.config = config or GANConfig()

        for key, value in config_overrides.items():
            if not hasattr(self.config, key):
                raise TypeError(f"Unknown GAN configuration parameter: {key}")
            setattr(self.config, key, value)

        if self.config.latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")

        if self.config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")

        if self.config.epochs <= 0:
            raise ValueError("epochs must be > 0")

        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if self.config.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

        self.embedding_dim = embedding_dim

        self.generator: Optional[Generator] = None
        self.discriminator: Optional[Discriminator] = None

        self.generator_losses: list[float] = []
        self.discriminator_losses: list[float] = []

        self._fitted = False
        self._real_min: Optional[np.ndarray] = None
        self._real_max: Optional[np.ndarray] = None
        self._real_mean: Optional[np.ndarray] = None
        self._real_std: Optional[np.ndarray] = None
        self._real_embeddings_count: int = 0

        self._device = self._resolve_device()

        self._set_seed(self.config.seed)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _resolve_device(self) -> torch.device:
        requested = self.config.device.lower()

        if requested == "auto":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but CUDA is unavailable.")

        return torch.device(requested)

    @staticmethod
    def _set_seed(seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_embeddings(
        self,
        embeddings: ArrayLike,
        *,
        require_2d: bool = True,
    ) -> np.ndarray:

        X = np.asarray(embeddings, dtype=np.float32)

        if require_2d and X.ndim != 2:
            raise ValueError(
                f"Expected 2-D embeddings [samples, dimensions], "
                f"received shape {X.shape}"
            )

        if X.shape[0] == 0:
            raise ValueError("Embeddings contain zero samples.")

        if not np.isfinite(X).all():
            raise ValueError(
                "Embeddings contain NaN or Inf values."
            )

        if self.embedding_dim is not None:
            if X.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"Expected embedding dimension "
                    f"{self.embedding_dim}, got {X.shape[1]}."
                )

        return X

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        real_embeddings: ArrayLike,
        targets=None,
    ) -> "GANAugmenter":
        """
        Train the GAN on supplied training embeddings.

        Parameters
        ----------
        real_embeddings:
            TRAINING participant embeddings only.

        targets:
            Accepted for API compatibility. The default GAN is unconditional
            and therefore does not use targets.

        Notes
        -----
        Validation and test data must not be supplied here.
        """

        del targets  # Unconditional GAN.

        X = self._validate_embeddings(real_embeddings)

        if self.embedding_dim is None:
            self.embedding_dim = X.shape[1]

        if X.shape[0] < 2:
            raise ValueError(
                "GAN training requires at least two real training samples."
            )

        # Store real-data statistics for sanity checks and optional clipping.
        self._real_min = X.min(axis=0)
        self._real_max = X.max(axis=0)
        self._real_mean = X.mean(axis=0)
        self._real_std = X.std(axis=0)

        # Avoid zero-width numerical ranges.
        self._real_std = np.maximum(self._real_std, 1e-6)

        self._real_embeddings_count = len(X)

        self.generator = Generator(
            latent_dim=self.config.latent_dim,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self._device)

        self.discriminator = Discriminator(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self._device)

        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(
            dataset,
            batch_size=min(self.config.batch_size, len(dataset)),
            shuffle=True,
            drop_last=False,
        )

        g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
        )

        d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
        )

        criterion = nn.BCEWithLogitsLoss()

        self.generator_losses = []
        self.discriminator_losses = []

        self._set_seed(self.config.seed)

        self.generator.train()
        self.discriminator.train()

        for _epoch in range(self.config.epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            batches = 0

            for (real_batch,) in loader:
                real_batch = real_batch.to(self._device)
                batch_size = real_batch.shape[0]

                # ------------------------------------------------------
                # Train discriminator
                # ------------------------------------------------------

                d_optimizer.zero_grad(set_to_none=True)

                real_logits = self.discriminator(real_batch)

                z = torch.randn(
                    batch_size,
                    self.config.latent_dim,
                    device=self._device,
                )

                fake_batch = self.generator(z).detach()
                fake_logits = self.discriminator(fake_batch)

                real_labels = torch.ones(
                    batch_size,
                    device=self._device,
                )

                fake_labels = torch.zeros(
                    batch_size,
                    device=self._device,
                )

                d_real_loss = criterion(real_logits, real_labels)
                d_fake_loss = criterion(fake_logits, fake_labels)

                d_loss = 0.5 * (d_real_loss + d_fake_loss)

                d_loss.backward()

                if self.config.gradient_clip is not None:
                    nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(),
                        self.config.gradient_clip,
                    )

                d_optimizer.step()

                # ------------------------------------------------------
                # Train generator
                # ------------------------------------------------------

                g_optimizer.zero_grad(set_to_none=True)

                z = torch.randn(
                    batch_size,
                    self.config.latent_dim,
                    device=self._device,
                )

                generated = self.generator(z)
                generated_logits = self.discriminator(generated)

                # Generator wants discriminator to classify generated
                # representations as real.
                g_loss = criterion(
                    generated_logits,
                    real_labels,
                )

                g_loss.backward()

                if self.config.gradient_clip is not None:
                    nn.utils.clip_grad_norm_(
                        self.generator.parameters(),
                        self.config.gradient_clip,
                    )

                g_optimizer.step()

                epoch_d_loss += float(d_loss.detach().cpu())
                epoch_g_loss += float(g_loss.detach().cpu())
                batches += 1

            self.discriminator_losses.append(
                epoch_d_loss / max(batches, 1)
            )
            self.generator_losses.append(
                epoch_g_loss / max(batches, 1)
            )

        self._fitted = True

        return self

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        n_samples: int,
        targets=None,
    ):
        """
        Generate synthetic BERT representations.

        Returns
        -------
        synthetic_embeddings:
            ndarray of shape [n_samples, embedding_dim]

        synthetic_targets:
            Targets supplied by caller, or None.

        metadata:
            SyntheticMetadata identifying all samples as synthetic.
        """

        if not self._fitted:
            raise RuntimeError(
                "GANAugmenter must be fitted on training data before generation."
            )

        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")

        if self.generator is None:
            raise RuntimeError("Generator has not been initialized.")

        self.generator.eval()

        # Generation gets a deterministic stream derived from the configured
        # seed without altering the training process.
        rng = np.random.default_rng(self.config.seed)

        z_np = rng.standard_normal(
            size=(n_samples, self.config.latent_dim)
        ).astype(np.float32)

        z = torch.from_numpy(z_np).to(self._device)

        with torch.no_grad():
            synthetic = self.generator(z).cpu().numpy()

        synthetic = self._apply_sanity_constraints(synthetic)

        self._validate_generated(synthetic)

        if targets is not None:
            synthetic_targets = np.asarray(targets)

            if len(synthetic_targets) != n_samples:
                raise ValueError(
                    "Number of target values must equal n_samples."
                )
        else:
            synthetic_targets = None

        metadata = SyntheticMetadata(
            is_synthetic=np.ones(n_samples, dtype=bool),
            source=np.full(
                n_samples,
                "GAN",
                dtype=object,
            ),
            generator_seed=self.config.seed,
        )

        return synthetic, synthetic_targets, metadata

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def augment(
        self,
        real_embeddings: ArrayLike,
        targets,
        n_samples: int,
    ):
        """
        Convenience method.

        The GAN must already have been fitted using training embeddings.

        Returns the original real samples followed by synthetic samples.
        """

        real = self._validate_embeddings(real_embeddings)

        if len(real) != len(targets):
            raise ValueError(
                "real_embeddings and targets must contain the same number "
                "of samples."
            )

        synthetic, synthetic_targets, synthetic_metadata = self.generate(
            n_samples=n_samples,
            targets=None,
        )

        # The default unconditional GAN cannot infer labels. For labelled
        # experiments, callers should provide a target-generation policy.
        #
        # We deliberately do not invent target values.
        if targets is not None:
            raise ValueError(
                "This unconditional GAN does not infer target values. "
                "Use generate() and explicitly assign synthetic targets "
                "according to the experiment's target policy."
            )

        return real, synthetic, synthetic_metadata

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------

    def _apply_sanity_constraints(
        self,
        synthetic: np.ndarray,
    ) -> np.ndarray:

        if not np.isfinite(synthetic).all():
            raise RuntimeError(
                "GAN generated NaN or Inf values."
            )

        if self._real_mean is None or self._real_std is None:
            return synthetic

        # Reject extreme values relative to the observed training
        # distribution. This is a safety mechanism, not a replacement
        # for empirical downstream evaluation.
        lower = (
            self._real_mean
            - self.config.plausibility_std_multiplier * self._real_std
        )

        upper = (
            self._real_mean
            + self.config.plausibility_std_multiplier * self._real_std
        )

        synthetic = np.clip(synthetic, lower, upper)

        if self.config.clip_to_real_range:
            synthetic = np.clip(
                synthetic,
                self._real_min,
                self._real_max,
            )

        return synthetic.astype(np.float32)

    def _validate_generated(
        self,
        synthetic: np.ndarray,
    ) -> None:

        if synthetic.ndim != 2:
            raise RuntimeError(
                "Generated embeddings are not two-dimensional."
            )

        if synthetic.shape[1] != self.embedding_dim:
            raise RuntimeError(
                f"Generated embedding dimension "
                f"{synthetic.shape[1]} does not match "
                f"real embedding dimension {self.embedding_dim}."
            )

        if not np.isfinite(synthetic).all():
            raise RuntimeError(
                "Generated embeddings contain NaN or Inf values."
            )

    def sanity_check(
        self,
        synthetic_embeddings: ArrayLike,
    ) -> dict:
        """
        Compare generated representations with training representations.

        No validation/test data is required or accepted.
        """

        synthetic = np.asarray(
            synthetic_embeddings,
            dtype=np.float32,
        )

        self._validate_generated(synthetic)

        if self._real_mean is None:
            raise RuntimeError("GAN has not been fitted.")

        real_mean_scalar = float(self._real_mean.mean())
        real_std_scalar = float(self._real_std.mean())

        synthetic_mean_scalar = float(synthetic.mean())
        synthetic_std_scalar = float(synthetic.std())

        real_norm = float(
            np.linalg.norm(self._real_mean)
        )

        synthetic_norms = np.linalg.norm(
            synthetic,
            axis=1,
        )

        return {
            "embedding_dim": self.embedding_dim,
            "real_samples": self._real_embeddings_count,
            "synthetic_samples": len(synthetic),

            "real_mean": real_mean_scalar,
            "real_std": real_std_scalar,

            "synthetic_mean": synthetic_mean_scalar,
            "synthetic_std": synthetic_std_scalar,

            "real_mean_norm": real_norm,
            "synthetic_mean_norm": float(
                synthetic_norms.mean()
            ),

            "synthetic_min": float(synthetic.min()),
            "synthetic_max": float(synthetic.max()),

            "has_nan": bool(np.isnan(synthetic).any()),
            "has_inf": bool(np.isinf(synthetic).any()),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(
        self,
        synthetic_embeddings: Optional[ArrayLike] = None,
    ) -> GANDiagnostics:

        if not self._fitted:
            raise RuntimeError("GAN has not been fitted.")

        if synthetic_embeddings is None:
            synthetic_embeddings = np.empty(
                (0, self.embedding_dim),
                dtype=np.float32,
            )

        synthetic = np.asarray(
            synthetic_embeddings,
            dtype=np.float32,
        )

        if len(synthetic):
            self._validate_generated(synthetic)

            synthetic_mean = float(synthetic.mean())
            synthetic_std = float(synthetic.std())
            synthetic_norm = float(
                np.linalg.norm(synthetic, axis=1).mean()
            )
            synthetic_nan = bool(np.isnan(synthetic).any())
            synthetic_inf = bool(np.isinf(synthetic).any())
        else:
            synthetic_mean = float("nan")
            synthetic_std = float("nan")
            synthetic_norm = float("nan")
            synthetic_nan = False
            synthetic_inf = False

        return GANDiagnostics(
            generator_loss=self.generator_losses.copy(),
            discriminator_loss=self.discriminator_losses.copy(),

            real_samples=self._real_embeddings_count,
            synthetic_samples=len(synthetic),

            embedding_dim=self.embedding_dim,

            real_mean=float(self._real_mean.mean()),
            real_std=float(self._real_std.mean()),

            synthetic_mean=synthetic_mean,
            synthetic_std=synthetic_std,

            real_mean_norm=float(
                np.linalg.norm(self._real_mean)
            ),
            synthetic_mean_norm=synthetic_norm,

            real_has_nan=False,
            real_has_inf=False,

            synthetic_has_nan=synthetic_nan,
            synthetic_has_inf=synthetic_inf,

            configuration=asdict(self.config),
        )
