"""
dataset.py - Controlled Synthetic Data Generation for Simplicity Bias Falsification

Constructs a synthetic binary classification task with disentangled feature subspaces:
  - x_S (Simple): Structurally linear predictive feature
  - x_C (Complex): Structurally nonlinear 2D interaction feature (XOR quadrant manifold)
  - x_N (Noise): Pure uninformative Gaussian distractor features

All features are normalized to unit variance to eliminate scale confounders.
Predictive strength is precisely modulated via independent label-flip noise rates eta_S and eta_C.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
        seed: int = 42,
    ):
        """
        Parameters:
            n_samples: Total number of samples generated.
            eta_s: Label noise rate for simple feature S (0 <= eta_s < 0.5).
            eta_c: Label noise rate for complex feature C (0 <= eta_c < 0.5).
            d_s: Dimensionality of simple feature subspace.
            d_c: Dimensionality of complex feature subspace.
            d_noise: Dimensionality of pure noise subspace.
            noise_std: Standard deviation of additive Gaussian noise.
            seed: Random seed for reproducible generation.
        """
        self.n_samples = n_samples
        self.eta_s = eta_s
        self.eta_c = eta_c
        self.d_s = d_s
        self.d_c = d_c
        self.d_noise = d_noise
        self.noise_std = noise_std
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self._generate_data()

    def _generate_data(self):
        # 1. Ground truth target label z in {-1, +1} with balanced 50/50 prior
        half = self.n_samples // 2
        z = np.array([-1] * half + [1] * (self.n_samples - half))
        self.rng.shuffle(z)

        # 2. Simple Linear Signal generation:
        # z_s is an imperfect reflection of z with error rate eta_s
        flip_s = self.rng.rand(self.n_samples) < self.eta_s
        z_s = np.where(flip_s, -z, z)

        # x_S is placed along a linear direction mu_S
        mu_s = np.ones(self.d_s) / np.sqrt(self.d_s)
        x_s = np.outer(z_s, mu_s) + self.rng.randn(self.n_samples, self.d_s) * self.noise_std

        # 3. Complex Non-linear Signal generation:
        # We use an XOR 2D quadrant manifold: sign(u1 * u2)
        # Latent coordinates u1, u2 uniformly distributed on [-1.5, 1.5]
        u1 = self.rng.uniform(-1.5, 1.5, size=self.n_samples)
        u2 = self.rng.uniform(-1.5, 1.5, size=self.n_samples)
        # Ensure a minimum margin from axes to prevent ambiguity
        u1 = np.where(np.abs(u1) < 0.1, np.sign(u1 + 1e-5) * 0.25, u1)
        u2 = np.where(np.abs(u2) < 0.1, np.sign(u2 + 1e-5) * 0.25, u2)

        # Base XOR label
        xor_label = np.sign(u1 * u2)

        # We conditionally construct (u1, u2) such that sign(u1 * u2) aligns with z with noise eta_c
        flip_c = self.rng.rand(self.n_samples) < self.eta_c
        target_c_label = np.where(flip_c, -z, z)

        # If xor_label != target_c_label, reflect u1 to flip sign(u1 * u2)
        mismatch = (xor_label != target_c_label)
        u1[mismatch] = -u1[mismatch]

        # Stack into complex subspace x_C
        if self.d_c == 2:
            x_c = np.column_stack([u1, u2])
        else:
            extra_c = self.rng.randn(self.n_samples, self.d_c - 2) * self.noise_std
            x_c = np.column_stack([u1, u2, extra_c])

        # Add Gaussian jitter to x_c
        x_c += self.rng.randn(self.n_samples, self.d_c) * (self.noise_std * 0.5)

        # 4. Pure Noise Subspace x_N
        x_n = self.rng.randn(self.n_samples, self.d_noise)

        # 5. Strict Normalization: standard scale all subspaces to zero mean and unit variance
        x_s = (x_s - np.mean(x_s, axis=0)) / (np.std(x_s, axis=0) + 1e-8)
        x_c = (x_c - np.mean(x_c, axis=0)) / (np.std(x_c, axis=0) + 1e-8)
        x_n = (x_n - np.mean(x_n, axis=0)) / (np.std(x_n, axis=0) + 1e-8)

        # Full concatenated feature representation
        x_full = np.concatenate([x_s, x_c, x_n], axis=1)
        y = ((z + 1) // 2).astype(np.int64)  # map {-1, 1} to {0, 1}

        self.x_s = x_s.astype(np.float32)
        self.x_c = x_c.astype(np.float32)
        self.x_n = x_n.astype(np.float32)
        self.x_full = x_full.astype(np.float32)
        self.y = y

        # Train / Val / Test split (70% / 15% / 15%)
        n_train = int(0.70 * self.n_samples)
        n_val = int(0.15 * self.n_samples)

        indices = np.arange(self.n_samples)
        self.rng.shuffle(indices)

        self.train_idx = indices[:n_train]
        self.val_idx = indices[n_train:n_train + n_val]
        self.test_idx = indices[n_train + n_val:]

    def get_splits(self):
        """Returns train, val, test dicts with x_full, x_s, x_c, x_n, y."""
        def make_split(idx):
            return {
                "x_full": self.x_full[idx],
                "x_s": self.x_s[idx],
                "x_c": self.x_c[idx],
                "x_n": self.x_n[idx],
                "y": self.y[idx],
            }
        return make_split(self.train_idx), make_split(self.val_idx), make_split(self.test_idx)

    def get_dataloaders(self, batch_size: int = 64):
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

    def evaluate_feature_separability(self):
        """
        Empirically validates single-feature predictive separability using
        Linear Logistic Regression on S and C separately.
        Linear model should achieve high accuracy on S (~ 1 - eta_s),
        and strictly chance accuracy (~ 50%) on C, proving structural nonlinearity.
        """
        train_d, _, test_d = self.get_splits()

        lr_s = LogisticRegression(max_iter=500)
        lr_s.fit(train_d["x_s"], train_d["y"])
        acc_s_linear = accuracy_score(test_d["y"], lr_s.predict(test_d["x_s"]))

        lr_c = LogisticRegression(max_iter=500)
        lr_c.fit(train_d["x_c"], train_d["y"])
        acc_c_linear = accuracy_score(test_d["y"], lr_c.predict(test_d["x_c"]))

        return {
            "linear_acc_on_S": float(acc_s_linear),
            "linear_acc_on_C": float(acc_c_linear),
            "theoretical_S_ceiling": float(1.0 - self.eta_s),
            "theoretical_C_ceiling": float(1.0 - self.eta_c),
        }

if __name__ == "__main__":
    ds = SyntheticFalsificationDataset(n_samples=2000, eta_s=0.05, eta_c=0.05, seed=123)
    metrics = ds.evaluate_feature_separability()
    print("Dataset Validation Test:")
    print(f"  Linear model on S: {metrics['linear_acc_on_S']:.4f} (Ceiling: {metrics['theoretical_S_ceiling']:.2f})")
    print(f"  Linear model on C: {metrics['linear_acc_on_C']:.4f} (Ceiling: {metrics['theoretical_C_ceiling']:.2f})")
