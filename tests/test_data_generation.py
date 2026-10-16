"""
test_data_generation.py - Automated Unit Tests for Synthetic Falsification Dataset

Validates statistical properties, joint distribution fidelity, and strict split isolation:
  1. Class balance: 50/50 prior distribution of z in {-1, +1}.
  2. Split isolation: No overlap in train/val/test splits, and normalization fitted only on train.
  3. Latent flip rates: z_S and z_C flip rates match eta_S and eta_C.
  4. Coordinate marginal correlations: E[x_C,i * z] == 0, E[x_N,i * z] == 0, E[x_S,i * z] >> 0.
  5. Feature separability: Logistic regression solves x_S (~95%) and performs at chance on x_C (~50%).
"""

import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.dataset import SyntheticFalsificationDataset

def test_class_balance():
    ds = SyntheticFalsificationDataset(n_samples=4000, seed=42)
    # Check overall target balance
    assert len(ds.y) == 4000
    assert np.mean(ds.y == 1) == 0.50
    assert np.mean(ds.y == 0) == 0.50

    # Check split sizes and balance
    train_d, val_d, test_d = ds.get_splits()
    assert len(train_d["y"]) == 2800  # 70%
    assert len(val_d["y"]) == 600     # 15%
    assert len(test_d["y"]) == 600    # 15%

    # Split class balance should be within binomial margin
    assert abs(np.mean(train_d["y"]) - 0.50) < 0.03
    assert abs(np.mean(val_d["y"]) - 0.50) < 0.05
    assert abs(np.mean(test_d["y"]) - 0.50) < 0.05

def test_split_isolation_and_leakage():
    ds = SyntheticFalsificationDataset(n_samples=4000, seed=123)
    # Check disjoint index sets
    train_set = set(ds.train_idx)
    val_set = set(ds.val_idx)
    test_set = set(ds.test_idx)

    assert len(train_set & val_set) == 0, "Train and Val splits overlap!"
    assert len(train_set & test_set) == 0, "Train and Test splits overlap!"
    assert len(val_set & test_set) == 0, "Val and Test splits overlap!"
    assert len(train_set | val_set | test_set) == 4000

    # Training partition should have empirical mean ~ 0 and std ~ 1
    train_d, val_d, test_d = ds.get_splits()
    np.testing.assert_allclose(np.mean(train_d["x_full"], axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.std(train_d["x_full"], axis=0), 1.0, atol=1e-5)

    # Test partition should NOT have exact 0.0 mean or 1.0 std (proves it wasn't standardized on full data)
    test_mean = np.mean(test_d["x_full"], axis=0)
    assert not np.allclose(test_mean, 0.0, atol=1e-4), "Test split exhibits suspicious exact zero mean (leakage check)"

def test_feature_shapes():
    ds = SyntheticFalsificationDataset(n_samples=2000, d_s=2, d_c=2, d_noise=4, seed=99)
    assert ds.x_s.shape == (2000, 2)
    assert ds.x_c.shape == (2000, 2)
    assert ds.x_n.shape == (2000, 4)
    assert ds.x_full.shape == (2000, 8)
    assert ds.y.shape == (2000,)

def test_empirical_flip_rates():
    # Large sample test to verify convergence of empirical flip rates
    ds = SyntheticFalsificationDataset(n_samples=20000, eta_s=0.05, eta_c=0.25, seed=777)
    emp_flip_s = np.mean(ds.z_s != ds.z)
    emp_flip_c = np.mean(ds.z_c != ds.z)

    assert abs(emp_flip_s - 0.05) < 0.008, f"Empirical flip S {emp_flip_s} deviates from eta_s=0.05"
    assert abs(emp_flip_c - 0.25) < 0.010, f"Empirical flip C {emp_flip_c} deviates from eta_c=0.25"

def test_marginal_correlations():
    ds = SyntheticFalsificationDataset(n_samples=10000, eta_s=0.05, eta_c=0.05, seed=101)
    z = ds.z

    # Linear feature S must have strong positive correlation with z
    for col in range(ds.d_s):
        corr_s = np.corrcoef(ds.x_s[:, col], z)[0, 1]
        assert corr_s > 0.85, f"Linear feature coordinate {col} has low correlation: {corr_s}"

    # Non-linear feature C coordinates must have near-zero marginal linear correlation with z
    for col in range(ds.d_c):
        corr_c = np.corrcoef(ds.x_c[:, col], z)[0, 1]
        assert abs(corr_c) < 0.03, f"XOR coordinate {col} has non-negligible marginal correlation: {corr_c}"

    # Noise subspace N must have near-zero correlation with z
    for col in range(ds.d_noise):
        corr_n = np.corrcoef(ds.x_n[:, col], z)[0, 1]
        assert abs(corr_n) < 0.03, f"Noise coordinate {col} has non-negligible correlation: {corr_n}"

def test_linear_separability():
    ds = SyntheticFalsificationDataset(n_samples=4000, eta_s=0.05, eta_c=0.05, seed=42)
    metrics = ds.evaluate_feature_separability()

    # Linear model on S should easily reach ~ 1 - eta_s (~95%)
    assert metrics["linear_acc_on_S"] >= 0.93, f"Linear model failed to separate S: {metrics['linear_acc_on_S']}"

    # Pure linear separator through origin on C MUST perform at chance level (~50% +/- 3%)
    assert abs(metrics["linear_acc_on_C_origin"] - 0.50) < 0.04, f"Linear model through origin succeeded on C: {metrics['linear_acc_on_C_origin']}"

    # Even with intercept, linear classifier cannot solve the XOR problem (must be << 90%)
    assert metrics["affine_acc_on_C"] < 0.75, f"Affine linear model unexpectedly solved XOR: {metrics['affine_acc_on_C']}"
