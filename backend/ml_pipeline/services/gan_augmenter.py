"""
Standalone GAN-based augmentation for participant-level BERT embeddings,
producing paired supervised training samples.

This module intentionally operates only on data supplied by the caller.
It does not access BERT, ElasticNet, LSTM, validation data, or test data.

WHY THIS IS A JOINT (EMBEDDING, OCEAN) GAN, NOT AN UNCONDITIONAL ONE
----------------------------------------------------------------------
Lasso/LSTM training needs paired (X, y) samples: X = BERT embedding,
y = 5-dim OCEAN vector. An unconditional GAN over embeddings alone can
only ever produce X — it has no principled way to attach a y to each
generated sample. Randomly assigning a label, copying a real
participant's label, or reusing one fixed label for every synthetic
sample would all be fabrication, not augmentation.

Instead, the Generator has two output heads sharing one trunk:

    z (noise) -> shared trunk -> [embedding_head]  -> synthetic embedding
                              -> [ocean_head]      -> synthetic OCEAN vector

and the Discriminator judges the *concatenated* (embedding, OCEAN) pair
as real or fake. Adversarial training therefore only rewards the
generator when its generated embedding and generated OCEAN vector look
jointly plausible together — i.e. the network has to learn the
embedding<->OCEAN relationship present in the real training pairs. The
resulting synthetic OCEAN score is not copied or randomly assigned; it
is produced by the same generator forward pass as its paired embedding.

Typical usage:

    augmenter = GANAugmenter(
        embedding_dim=X_train.shape[1],
        latent_dim=64,
        epochs=200,
        batch_size=32,
        learning_rate=2e-4,
        seed=42,
    )

    augmenter.fit(X_train, ocean_scores=y_train)  # y_train: (N, 5)

    X_syn, y_syn, metadata = augmenter.generate(n_samples=500)

    X_train_aug = np.concatenate([X_train, X_syn])
    y_train_aug = np.concatenate([y_train, y_syn])

Only the TRAINING fold (X_train, y_train) is ever passed to fit().
Validation/test data must never be passed to fit() or used to derive
synthetic targets.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]

# Fixed OCEAN target width: [Openness, Conscientiousness, Extraversion,
# Agreeableness, Neuroticism]. This is a hard contract, not configurable —
# every paired sample this module produces has exactly 5 target values.
OCEAN_DIM = 5
OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


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

    # Prevent pathological generated magnitudes (applied to the embedding
    # part and, separately, the OCEAN part of each generated pair).
    plausibility_std_multiplier: float = 4.0

    # If True, generated samples are clipped to the observed real range.
    # This is a safety check, not Gaussian augmentation.
    clip_to_real_range: bool = True

    # If True, the generated OCEAN part is additionally clipped to the
    # known Likert-scale domain bound [1.0, 5.0], on top of whatever the
    # observed real-range clip already does. Safe regardless of what
    # scale the caller's y happens to be in, as long as it's 1-5.
    clip_ocean_to_domain_bounds: bool = True
    ocean_domain_min: float = 1.0
    ocean_domain_max: float = 5.0


# ---------------------------------------------------------------------------
# Generator — shared trunk, two output heads (embedding, OCEAN)
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """
    Lightweight MLP generator with two heads.

    latent noise z -> shared trunk -> (synthetic embedding, synthetic OCEAN vector)

    Both heads come off the same trunk representation, which is what
    lets the generator learn to couple the two outputs rather than
    producing them independently.
    """

    def __init__(
        self,
        latent_dim: int,
        embedding_dim: int,
        ocean_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.embedding_head = nn.Linear(hidden_dim, embedding_dim)
        self.ocean_head = nn.Linear(hidden_dim, ocean_dim)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(z)
        embedding = self.embedding_head(h)
        ocean = self.ocean_head(h)
        return embedding, ocean


# ---------------------------------------------------------------------------
# Discriminator — judges the concatenated (embedding, OCEAN) pair
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """
    Lightweight MLP discriminator.

    concatenated [embedding | OCEAN] pair -> probability the PAIR is real.

    Judging the pair jointly (rather than the embedding alone) is what
    forces the generator to make its OCEAN output consistent with its
    embedding output, instead of the two being generated independently.
    """

    def __init__(
        self,
        embedding_dim: int,
        ocean_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim + ocean_dim, hidden_dim),
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
    ocean_dim: int

    real_mean: float
    real_std: float
    synthetic_mean: float
    synthetic_std: float

    real_mean_norm: float
    synthetic_mean_norm: float

    real_ocean_mean: list[float]
    real_ocean_std: list[float]
    synthetic_ocean_mean: list[float]
    synthetic_ocean_std: list[float]

    real_has_nan: bool
    real_has_inf: bool
    synthetic_has_nan: bool
    synthetic_has_inf: bool

    configuration: dict


@dataclass
class SyntheticMetadata:
    """Metadata accompanying generated (embedding, OCEAN) pairs."""

    is_synthetic: np.ndarray
    source: np.ndarray
    generator_seed: int


# ---------------------------------------------------------------------------
# GAN augmenter
# ---------------------------------------------------------------------------

class GANAugmenter:
    """
    Standalone GAN augmentation service producing paired
    (embedding, OCEAN) supervised samples.

    Important data-boundary rule:

        fit(X_train, ocean_scores=y_train)

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
        self.ocean_dim = OCEAN_DIM

        self.generator: Optional[Generator] = None
        self.discriminator: Optional[Discriminator] = None

        self.generator_losses: list[float] = []
        self.discriminator_losses: list[float] = []

        self._fitted = False

        self._real_min: Optional[np.ndarray] = None
        self._real_max: Optional[np.ndarray] = None
        self._real_mean: Optional[np.ndarray] = None
        self._real_std: Optional[np.ndarray] = None

        self._real_ocean_min: Optional[np.ndarray] = None
        self._real_ocean_max: Optional[np.ndarray] = None
        self._real_ocean_mean: Optional[np.ndarray] = None
        self._real_ocean_std: Optional[np.ndarray] = None

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

    def _validate_ocean_scores(self, ocean_scores: ArrayLike) -> np.ndarray:
        """
        Validate a (N, 5) OCEAN target array. This is a hard contract —
        exactly 5 columns, in the fixed [O, C, E, A, N] order — since
        downstream LassoTrainer/LSTMTrainer both expect that shape.
        """
        y = np.asarray(ocean_scores, dtype=np.float32)

        if y.ndim != 2:
            raise ValueError(
                f"Expected 2-D OCEAN scores [samples, {OCEAN_DIM}], "
                f"received shape {y.shape}"
            )

        if y.shape[1] != OCEAN_DIM:
            raise ValueError(
                f"Expected exactly {OCEAN_DIM} OCEAN targets per sample "
                f"({OCEAN_TRAITS}), got {y.shape[1]}."
            )

        if y.shape[0] == 0:
            raise ValueError("OCEAN scores contain zero samples.")

        if not np.isfinite(y).all():
            raise ValueError("OCEAN scores contain NaN or Inf values.")

        return y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        real_embeddings: ArrayLike,
        ocean_scores: Optional[ArrayLike] = None,
        targets: Optional[ArrayLike] = None,
    ) -> "GANAugmenter":
        """
        Train the joint (embedding, OCEAN) GAN on paired training data.

        Parameters
        ----------
        real_embeddings:
            TRAINING participant embeddings only, shape (N, embedding_dim).

        ocean_scores:
            TRAINING participant OCEAN targets, shape (N, 5), paired
            row-for-row with real_embeddings. Required — the GAN now
            learns the joint distribution of (embedding, OCEAN), so it
            cannot be fit on embeddings alone.

        targets:
            Deprecated alias for ocean_scores, kept for call-site
            backward compatibility. If both are given, ocean_scores wins.

        Notes
        -----
        Validation and test data must not be supplied here — this is the
        caller's responsibility; this method has no way to distinguish
        a training participant from a held-out one.
        """
        if ocean_scores is None and targets is not None:
            warnings.warn(
                "GANAugmenter.fit(targets=...) is deprecated; use "
                "fit(ocean_scores=...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            ocean_scores = targets

        if ocean_scores is None:
            raise ValueError(
                "fit() requires ocean_scores: this GAN learns the joint "
                "(embedding, OCEAN) distribution and cannot produce "
                "paired synthetic samples without real paired training data."
            )

        X = self._validate_embeddings(real_embeddings)
        y = self._validate_ocean_scores(ocean_scores)

        if len(X) != len(y):
            raise ValueError(
                f"real_embeddings and ocean_scores must have the same number "
                f"of rows (got {len(X)} and {len(y)})."
            )

        if self.embedding_dim is None:
            self.embedding_dim = X.shape[1]

        if X.shape[0] < 2:
            raise ValueError(
                "GAN training requires at least two real paired training samples."
            )

        # Store real-data statistics for sanity checks and optional clipping.
        self._real_min = X.min(axis=0)
        self._real_max = X.max(axis=0)
        self._real_mean = X.mean(axis=0)
        self._real_std = np.maximum(X.std(axis=0), 1e-6)

        self._real_ocean_min = y.min(axis=0)
        self._real_ocean_max = y.max(axis=0)
        self._real_ocean_mean = y.mean(axis=0)
        self._real_ocean_std = np.maximum(y.std(axis=0), 1e-6)

        self._real_embeddings_count = len(X)

        self.generator = Generator(
            latent_dim=self.config.latent_dim,
            embedding_dim=self.embedding_dim,
            ocean_dim=self.ocean_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self._device)

        self.discriminator = Discriminator(
            embedding_dim=self.embedding_dim,
            ocean_dim=self.ocean_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self._device)

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
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

            for real_emb_batch, real_ocean_batch in loader:
                real_emb_batch = real_emb_batch.to(self._device)
                real_ocean_batch = real_ocean_batch.to(self._device)
                batch_size = real_emb_batch.shape[0]

                real_pair = torch.cat([real_emb_batch, real_ocean_batch], dim=1)

                # ------------------------------------------------------
                # Train discriminator on real vs. fake PAIRS
                # ------------------------------------------------------

                d_optimizer.zero_grad(set_to_none=True)

                real_logits = self.discriminator(real_pair)

                z = torch.randn(
                    batch_size,
                    self.config.latent_dim,
                    device=self._device,
                )

                fake_emb, fake_ocean = self.generator(z)
                fake_pair = torch.cat([fake_emb, fake_ocean], dim=1).detach()
                fake_logits = self.discriminator(fake_pair)

                real_labels = torch.ones(batch_size, device=self._device)
                fake_labels = torch.zeros(batch_size, device=self._device)

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
                # Train generator — wants the discriminator to judge its
                # generated (embedding, OCEAN) PAIR as real, which is
                # what forces the two outputs to become jointly plausible
                # rather than independently generated.
                # ------------------------------------------------------

                g_optimizer.zero_grad(set_to_none=True)

                z = torch.randn(
                    batch_size,
                    self.config.latent_dim,
                    device=self._device,
                )

                gen_emb, gen_ocean = self.generator(z)
                gen_pair = torch.cat([gen_emb, gen_ocean], dim=1)
                generated_logits = self.discriminator(gen_pair)

                g_loss = criterion(generated_logits, real_labels)

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

            self.discriminator_losses.append(epoch_d_loss / max(batches, 1))
            self.generator_losses.append(epoch_g_loss / max(batches, 1))

        self._fitted = True

        return self

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        n_samples: int,
    ) -> tuple[np.ndarray, np.ndarray, SyntheticMetadata]:
        """
        Generate synthetic (embedding, OCEAN) pairs.

        Each row of the two returned arrays is one synthetic supervised
        sample, produced from the same generator forward pass — the
        OCEAN target is not assigned or copied afterward.

        Returns
        -------
        synthetic_embeddings:
            ndarray of shape [n_samples, embedding_dim]

        synthetic_ocean_scores:
            ndarray of shape [n_samples, 5], paired row-for-row with
            synthetic_embeddings.

        metadata:
            SyntheticMetadata identifying all samples as synthetic.
        """

        if not self._fitted:
            raise RuntimeError(
                "GANAugmenter must be fitted on paired training data "
                "(fit(X_train, ocean_scores=y_train)) before generation."
            )

        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")

        if self.generator is None:
            raise RuntimeError("Generator has not been initialized.")

        self.generator.eval()

        # Generation gets a deterministic stream derived from the
        # configured seed, so repeated calls with the same seed and
        # fitted weights reproduce the same synthetic pairs.
        rng = np.random.default_rng(self.config.seed)

        z_np = rng.standard_normal(
            size=(n_samples, self.config.latent_dim)
        ).astype(np.float32)

        z = torch.from_numpy(z_np).to(self._device)

        with torch.no_grad():
            synthetic_emb, synthetic_ocean = self.generator(z)
            synthetic_emb = synthetic_emb.cpu().numpy()
            synthetic_ocean = synthetic_ocean.cpu().numpy()

        synthetic_emb = self._apply_sanity_constraints(synthetic_emb)
        synthetic_ocean = self._apply_ocean_sanity_constraints(synthetic_ocean)

        self._validate_generated(synthetic_emb, synthetic_ocean)

        metadata = SyntheticMetadata(
            is_synthetic=np.ones(n_samples, dtype=bool),
            source=np.full(n_samples, "GAN", dtype=object),
            generator_seed=self.config.seed,
        )

        return synthetic_emb, synthetic_ocean, metadata

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def augment(
        self,
        real_embeddings: ArrayLike,
        real_ocean_scores: ArrayLike,
        n_samples: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SyntheticMetadata]:
        """
        Convenience method. The GAN must already have been fitted using
        training embeddings + OCEAN scores (see fit()).

        Returns the validated real pair arrays and freshly generated
        synthetic pair arrays, ready for the caller to concatenate:

            real_X, real_y, syn_X, syn_y, meta = augmenter.augment(...)
            X_train_aug = np.concatenate([real_X, syn_X])
            y_train_aug = np.concatenate([real_y, syn_y])

        Only real training-fold data should ever be passed in here —
        this method does not fit anything, it only validates and
        generates.
        """

        real_X = self._validate_embeddings(real_embeddings)
        real_y = self._validate_ocean_scores(real_ocean_scores)

        if len(real_X) != len(real_y):
            raise ValueError(
                "real_embeddings and real_ocean_scores must contain the "
                "same number of samples."
            )

        synthetic_X, synthetic_y, synthetic_metadata = self.generate(n_samples=n_samples)

        return real_X, real_y, synthetic_X, synthetic_y, synthetic_metadata

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------

    def _apply_sanity_constraints(self, synthetic: np.ndarray) -> np.ndarray:

        if not np.isfinite(synthetic).all():
            raise RuntimeError("GAN generated NaN or Inf embedding values.")

        if self._real_mean is None or self._real_std is None:
            return synthetic

        lower = self._real_mean - self.config.plausibility_std_multiplier * self._real_std
        upper = self._real_mean + self.config.plausibility_std_multiplier * self._real_std

        synthetic = np.clip(synthetic, lower, upper)

        if self.config.clip_to_real_range:
            synthetic = np.clip(synthetic, self._real_min, self._real_max)

        return synthetic.astype(np.float32)

    def _apply_ocean_sanity_constraints(self, synthetic_ocean: np.ndarray) -> np.ndarray:

        if not np.isfinite(synthetic_ocean).all():
            raise RuntimeError("GAN generated NaN or Inf OCEAN values.")

        if self._real_ocean_mean is None or self._real_ocean_std is None:
            return synthetic_ocean

        lower = self._real_ocean_mean - self.config.plausibility_std_multiplier * self._real_ocean_std
        upper = self._real_ocean_mean + self.config.plausibility_std_multiplier * self._real_ocean_std

        synthetic_ocean = np.clip(synthetic_ocean, lower, upper)

        if self.config.clip_to_real_range:
            synthetic_ocean = np.clip(synthetic_ocean, self._real_ocean_min, self._real_ocean_max)

        if self.config.clip_ocean_to_domain_bounds:
            synthetic_ocean = np.clip(
                synthetic_ocean,
                self.config.ocean_domain_min,
                self.config.ocean_domain_max,
            )

        return synthetic_ocean.astype(np.float32)

    def _validate_generated(self, synthetic_emb: np.ndarray, synthetic_ocean: np.ndarray) -> None:

        if synthetic_emb.ndim != 2:
            raise RuntimeError("Generated embeddings are not two-dimensional.")

        if synthetic_emb.shape[1] != self.embedding_dim:
            raise RuntimeError(
                f"Generated embedding dimension {synthetic_emb.shape[1]} "
                f"does not match real embedding dimension {self.embedding_dim}."
            )

        if not np.isfinite(synthetic_emb).all():
            raise RuntimeError("Generated embeddings contain NaN or Inf values.")

        if synthetic_ocean.ndim != 2:
            raise RuntimeError("Generated OCEAN scores are not two-dimensional.")

        if synthetic_ocean.shape[1] != OCEAN_DIM:
            raise RuntimeError(
                f"Generated OCEAN dimension {synthetic_ocean.shape[1]} "
                f"does not equal the required {OCEAN_DIM}."
            )

        if synthetic_ocean.shape[0] != synthetic_emb.shape[0]:
            raise RuntimeError(
                "Generated embeddings and generated OCEAN scores have "
                "mismatched sample counts — they must stay paired."
            )

        if not np.isfinite(synthetic_ocean).all():
            raise RuntimeError("Generated OCEAN scores contain NaN or Inf values.")

    def sanity_check(
        self,
        synthetic_embeddings: ArrayLike,
        synthetic_ocean_scores: ArrayLike,
    ) -> dict:
        """
        Compare generated (embedding, OCEAN) pairs with the real training
        pairs the GAN was fitted on. No validation/test data is required
        or accepted.
        """

        synthetic_emb = np.asarray(synthetic_embeddings, dtype=np.float32)
        synthetic_ocean = np.asarray(synthetic_ocean_scores, dtype=np.float32)

        self._validate_generated(synthetic_emb, synthetic_ocean)

        if self._real_mean is None or self._real_ocean_mean is None:
            raise RuntimeError("GAN has not been fitted.")

        real_norm = float(np.linalg.norm(self._real_mean))
        synthetic_norms = np.linalg.norm(synthetic_emb, axis=1)

        return {
            "embedding_dim": self.embedding_dim,
            "ocean_dim": self.ocean_dim,
            "real_samples": self._real_embeddings_count,
            "synthetic_samples": len(synthetic_emb),

            "real_mean": float(self._real_mean.mean()),
            "real_std": float(self._real_std.mean()),
            "synthetic_mean": float(synthetic_emb.mean()),
            "synthetic_std": float(synthetic_emb.std()),

            "real_mean_norm": real_norm,
            "synthetic_mean_norm": float(synthetic_norms.mean()),

            "synthetic_min": float(synthetic_emb.min()),
            "synthetic_max": float(synthetic_emb.max()),

            "real_ocean_mean": self._real_ocean_mean.tolist(),
            "real_ocean_std": self._real_ocean_std.tolist(),
            "synthetic_ocean_mean": synthetic_ocean.mean(axis=0).tolist(),
            "synthetic_ocean_std": synthetic_ocean.std(axis=0).tolist(),
            "synthetic_ocean_min": synthetic_ocean.min(axis=0).tolist(),
            "synthetic_ocean_max": synthetic_ocean.max(axis=0).tolist(),

            "has_nan": bool(np.isnan(synthetic_emb).any() or np.isnan(synthetic_ocean).any()),
            "has_inf": bool(np.isinf(synthetic_emb).any() or np.isinf(synthetic_ocean).any()),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(
        self,
        synthetic_embeddings: Optional[ArrayLike] = None,
        synthetic_ocean_scores: Optional[ArrayLike] = None,
    ) -> GANDiagnostics:

        if not self._fitted:
            raise RuntimeError("GAN has not been fitted.")

        if synthetic_embeddings is None:
            synthetic_embeddings = np.empty((0, self.embedding_dim), dtype=np.float32)
        if synthetic_ocean_scores is None:
            synthetic_ocean_scores = np.empty((0, self.ocean_dim), dtype=np.float32)

        synthetic_emb = np.asarray(synthetic_embeddings, dtype=np.float32)
        synthetic_ocean = np.asarray(synthetic_ocean_scores, dtype=np.float32)

        if len(synthetic_emb):
            self._validate_generated(synthetic_emb, synthetic_ocean)

            synthetic_mean = float(synthetic_emb.mean())
            synthetic_std = float(synthetic_emb.std())
            synthetic_norm = float(np.linalg.norm(synthetic_emb, axis=1).mean())
            synthetic_nan = bool(np.isnan(synthetic_emb).any())
            synthetic_inf = bool(np.isinf(synthetic_emb).any())
            synthetic_ocean_mean = synthetic_ocean.mean(axis=0).tolist()
            synthetic_ocean_std = synthetic_ocean.std(axis=0).tolist()
        else:
            synthetic_mean = float("nan")
            synthetic_std = float("nan")
            synthetic_norm = float("nan")
            synthetic_nan = False
            synthetic_inf = False
            synthetic_ocean_mean = [float("nan")] * self.ocean_dim
            synthetic_ocean_std = [float("nan")] * self.ocean_dim

        return GANDiagnostics(
            generator_loss=self.generator_losses.copy(),
            discriminator_loss=self.discriminator_losses.copy(),

            real_samples=self._real_embeddings_count,
            synthetic_samples=len(synthetic_emb),

            embedding_dim=self.embedding_dim,
            ocean_dim=self.ocean_dim,

            real_mean=float(self._real_mean.mean()),
            real_std=float(self._real_std.mean()),

            synthetic_mean=synthetic_mean,
            synthetic_std=synthetic_std,

            real_mean_norm=float(np.linalg.norm(self._real_mean)),
            synthetic_mean_norm=synthetic_norm,

            real_ocean_mean=self._real_ocean_mean.tolist(),
            real_ocean_std=self._real_ocean_std.tolist(),
            synthetic_ocean_mean=synthetic_ocean_mean,
            synthetic_ocean_std=synthetic_ocean_std,

            real_has_nan=False,
            real_has_inf=False,

            synthetic_has_nan=synthetic_nan,
            synthetic_has_inf=synthetic_inf,

            configuration=asdict(self.config),
        )
    