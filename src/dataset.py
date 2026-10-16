"""
dataset.py - Controlled Synthetic Data Generation for Simplicity Bias Falsification

Constructs a synthetic binary classification task with disentangled feature subspaces:
  - x_S (Simple): Structurally linear predictive feature (Gaussian cluster along diagonal)
  - x_C (Complex): Structurally non-linear 2D interaction feature (XOR quadrant manifold)
  - x_N (Noise): Pure uninformative Gaussian distractor features

Mathematical Specification & Data-Generating Process:
  1. Latent Target:
     z in {-1, +1} sampled with equal prior probability P(z = 1) = P(z = -1) = 0.5.
  2. Simple Linear Subspace (x_S in R^{d_S}):
     Latent feature label z_S is flipped relative to z with probability eta_S:
       P(z_S = z) = 1 - eta_S,  P(z_S = -z) = eta_S.
     Given z_S, coordinates are generated via:
       x_S = z_S * mu_S + epsilon_S,
     where mu_S = [1, ..., 1]^T / sqrt(d_S) and epsilon_S ~ N(0, noise_std^2 * I_{d_S}).
     The latent label reliability is 1 - eta_S. For well-separated clusters (||mu_S|| / noise_std >> 1),
     the Bayes-optimal accuracy on z given x_S is approximately 1 - eta_S.
  3. Complex Non-linear Subspace (x_C in R^{d_C}):
     Latent feature label z_C is flipped relative to z with probability eta_C:
       P(z_C = z) = 1 - eta_C,  P(z_C = -z) = eta_C.
     Coordinates (u_1, u_2) are generated within the quadrant defined by z_C:
       r_1, r_2 ~ Uniform([0.2, 1.5]),
       s_1 ~ Rademacher({-1, +1}),
       s_2 = z_C * s_1,
       u_1 = s_1 * r_1 + epsilon_{C,1},
       u_2 = s_2 * r_2 + epsilon_{C,2},
     where epsilon_{C} ~ N(0, (noise_std * 0.2)^2).
     Because s_1 is independent of z, E[u_1 | z] = 0 and E[u_2 | z] = 0, ensuring exactly zero
     marginal linear correlation with z. The non-linear interaction sign(u_1 * u_2) matches z_C
     with negligible jitter crossing probability (boundary margin >= 0.2 > 4 * sigma).
  4. Noise Subspace (x_N in R^{d_N}):
     epsilon_N ~ N(0, I_{d_N}), completely independent of z.
  5. Strict Split Isolation:
     Standardization (z-score: mean = 0, std = 1) is fitted strictly on the training partition
     and applied to validation and test partitions, preventing data leakage.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from typing import Dict, Tuple, Any

class SyntheticFalsificationDataset:
    def __init__(
        self,
        n_samples: int = 4000,
        eta_s: float = 0.05,
        eta_c: float = 0.05,
        d_s: int = 2,
        d_c: int = 2,
        d_noise: int = 4,
        noise_std: float = 0.2,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        """
        Parameters:
            n_samples: Total number of synthetic instances.
            eta_s: Label noise rate for simple linear feature S (0 <= eta_s < 0.5).
            eta_c: Label noise rate for complex XOR feature C (0 <= eta_c < 0.5).
            d_s: Dimensionality of simple feature subspace (default 2).
            d_c: Dimensionality of complex feature subspace (default 2).
            d_noise: Dimensionality of pure noise subspace (default 4).
            noise_std: Standard deviation of additive Gaussian noise.
            train_ratio: Proportion of samples assigned to training split (default 0.70).
            val_ratio: Proportion of samples assigned to validation split (default 0.15).
            test_ratio: Proportion of samples assigned to test split (default 0.15).
            seed: Random seed for reproducible generation.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Splits must sum to 1.0"
        self.n_samples = n_samples
        self.eta_s = eta_s
        self.eta_c = eta_c
        self.d_s = d_s
        self.d_c = d_c
        self.d_noise = d_noise
        self.noise_std = noise_std
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self._generate_data()

    def _generate_data(self):
        # 1. Ground truth target label z in {-1, +1} with balanced 50/50 prior
        half = self.n_samples // 2
        z = np.array([-1] * half + [1] * (self.n_samples - half), dtype=np.int64)
        self.rng.shuffle(z)

        # 2. Simple Linear Subspace x_S
        # Latent label z_s reflects z with error probability eta_s
        flip_s = self.rng.rand(self.n_samples) < self.eta_s
        z_s = np.where(flip_s, -z, z)

        mu_s = np.ones(self.d_s) / np.sqrt(self.d_s)
        noise_s = self.rng.randn(self.n_samples, self.d_s) * self.noise_std
        x_s_raw = np.outer(z_s, mu_s) + noise_s

        # 3. Complex Non-linear Subspace x_C (XOR Quadrant Manifold)
        # Latent label z_c reflects z with error probability eta_c
        flip_c = self.rng.rand(self.n_samples) < self.eta_c
        z_c = np.where(flip_c, -z, z)

        # Sample magnitudes with explicit margin away from coordinate axes: [0.2, 1.5]
        r1 = self.rng.uniform(0.2, 1.5, size=self.n_samples)
        r2 = self.rng.uniform(0.2, 1.5, size=self.n_samples)

        # Random sign for first coordinate: s1 in {-1, +1} with p=0.5
        s1 = self.rng.choice([-1, 1], size=self.n_samples)
        # Set s2 such that sign(s1 * s2) = z_c
        s2 = z_c * s1

        # Coordinate jitter standard deviation (0.04 ensures points stay inside quadrant: margin 0.2 = 5 * sigma)
        jitter_std = self.noise_std * 0.2
        u1 = s1 * r1 + self.rng.randn(self.n_samples) * jitter_std
        u2 = s2 * r2 + self.rng.randn(self.n_samples) * jitter_std

        if self.d_c == 2:
            x_c_raw = np.column_stack([u1, u2])
        else:
            extra_c = self.rng.randn(self.n_samples, self.d_c - 2) * self.noise_std
            x_c_raw = np.column_stack([u1, u2, extra_c])

        # 4. Pure Noise Subspace x_N
        x_n_raw = self.rng.randn(self.n_samples, self.d_noise)

        # 5. Partition indices into train / val / test before normalization
        n_train = int(self.train_ratio * self.n_samples)
        n_val = int(self.val_ratio * self.n_samples)

        indices = np.arange(self.n_samples)
        self.rng.shuffle(indices)

        self.train_idx = indices[:n_train]
        self.val_idx = indices[n_train:n_train + n_val]
        self.test_idx = indices[n_train + n_val:]

        # 6. Strict Split Isolation: fit mean and std ONLY on training partition
        mean_s = np.mean(x_s_raw[self.train_idx], axis=0)
        std_s = np.std(x_s_raw[self.train_idx], axis=0) + 1e-8

        mean_c = np.mean(x_c_raw[self.train_idx], axis=0)
        std_c = np.std(x_c_raw[self.train_idx], axis=0) + 1e-8

        mean_n = np.mean(x_n_raw[self.train_idx], axis=0)
        std_n = np.std(x_n_raw[self.train_idx], axis=0) + 1e-8

        x_s = (x_s_raw - mean_s) / std_s
        x_c = (x_c_raw - mean_c) / std_c
        x_n = (x_n_raw - mean_n) / std_n

        # Full concatenated input feature vector x in R^8
        x_full = np.concatenate([x_s, x_c, x_n], axis=1)
        y = ((z + 1) // 2).astype(np.int64)  # map {-1, +1} -> {0, 1}

        self.x_s = x_s.astype(np.float32)
        self.x_c = x_c.astype(np.float32)
        self.x_n = x_n.astype(np.float32)
        self.x_full = x_full.astype(np.float32)
        self.y = y
        self.z = z
        self.z_s = z_s
        self.z_c = z_c

    def get_splits(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Returns train, val, test dicts with x_full, x_s, x_c, x_n, y."""
        def make_split(idx):
            return {
                "x_full": self.x_full[idx],
                "x_s": self.x_s[idx],
                "x_c": self.x_c[idx],
                "x_n": self.x_n[idx],
                "y": self.y[idx],
                "z": self.z[idx],
            }
        return make_split(self.train_idx), make_split(self.val_idx), make_split(self.test_idx)

    def get_dataloaders(self, batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_data, val_data, test_data = self.get_splits()

        def make_loader(d, shuffle=True):
            tensor_x = torch.from_numpy(d["x_full"])
            tensor_y = torch.from_numpy(d["y"])
            ds = TensorDataset(tensor_x, tensor_y)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        return (
            make_loader(train_data, shuffle=True),
            make_loader(val_data, shuffle=False),
            make_loader(test_data, shuffle=False),
        )

    def evaluate_feature_separability(self) -> Dict[str, float]:
        """
        Empirically validates single-feature predictive separability:
          - Linear Logistic Regression on S should achieve accuracy ~ 1 - eta_s.
          - Pure linear separator (through origin) on C operates strictly at chance (~50%).
          - Affine linear model on C (with intercept) demonstrates inability to systematically solve XOR.
        """
        train_d, _, test_d = self.get_splits()

        lr_s = LogisticRegression(max_iter=500)
        lr_s.fit(train_d["x_s"], train_d["y"])
        acc_s_linear = accuracy_score(test_d["y"], lr_s.predict(test_d["x_s"]))

        # Pure linear separator through origin (b = 0)
        lr_c_origin = LogisticRegression(fit_intercept=False, max_iter=500)
        lr_c_origin.fit(train_d["x_c"], train_d["y"])
        acc_c_linear_origin = accuracy_score(test_d["y"], lr_c_origin.predict(test_d["x_c"]))

        # Affine separator with intercept
        lr_c_affine = LogisticRegression(fit_intercept=True, max_iter=500)
        lr_c_affine.fit(train_d["x_c"], train_d["y"])
        acc_c_affine = accuracy_score(test_d["y"], lr_c_affine.predict(test_d["x_c"]))

        return {
            "linear_acc_on_S": float(acc_s_linear),
            "linear_acc_on_C_origin": float(acc_c_linear_origin),
            "affine_acc_on_C": float(acc_c_affine),
            "latent_S_reliability": float(1.0 - self.eta_s),
            "latent_C_reliability": float(1.0 - self.eta_c),
        }

if __name__ == "__main__":
    ds = SyntheticFalsificationDataset(n_samples=4000, eta_s=0.05, eta_c=0.05, seed=123)
    metrics = ds.evaluate_feature_separability()
    print("Audited Dataset Validation Test:")
    print(f"  Linear model on S: {metrics['linear_acc_on_S']:.4f} (Latent Reliability: {metrics['latent_S_reliability']:.2f})")
    print(f"  Linear model on C: {metrics['linear_acc_on_C']:.4f} (Latent Reliability: {metrics['latent_C_reliability']:.2f})")
