"""
train.py - Controlled Training and Counterfactual Evaluation Engine

Executes training loops with rigorous tracking:
  - Joint performance on full multimodal feature representation.
  - Counterfactual feature reliance via marginal feature neutralization (permutation).
  - Time-to-threshold metric calculation (T_S, T_C) and Delta = T_C - T_S.
  - Single-feature oracle baselines.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from typing import Dict, Any, Tuple

# Add repository root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import SyntheticFalsificationDataset
from src.models import create_model

def evaluate_counterfactual(
    model: nn.Module,
    test_data: Dict[str, np.ndarray],
    d_s: int = 2,
    d_c: int = 2,
    d_n: int = 4,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluates model performance under counterfactual feature interventions:
      1. Joint: unperturbed inputs [x_s, x_c, x_n]
      2. S-only: x_c and x_n permuted across the test set (marginalizing out C and N)
      3. C-only: x_s and x_n permuted across the test set (marginalizing out S and N)
      4. N-only: x_s and x_c permuted (null control)
    """
    model.eval()
    x = test_data["x_full"]
    y = test_data["y"]
    n = len(x)

    with torch.no_grad():
        # 1. Joint evaluation
        logits_joint = model(torch.from_numpy(x).to(device))
        preds_joint = torch.argmax(logits_joint, dim=1).cpu().numpy()
        acc_joint = accuracy_score(y, preds_joint)

        # 2. S-only: keep x[:, :d_s], shuffle the rest
        perm_c = np.random.permutation(n)
        perm_n = np.random.permutation(n)
        x_s_only = np.zeros_like(x)
        x_s_only[:, :d_s] = x[:, :d_s]
        x_s_only[:, d_s:d_s + d_c] = x[perm_c, d_s:d_s + d_c]
        x_s_only[:, d_s + d_c:] = x[perm_n, d_s + d_c:]

        logits_s = model(torch.from_numpy(x_s_only).to(device))
        preds_s = torch.argmax(logits_s, dim=1).cpu().numpy()
        acc_s_only = accuracy_score(y, preds_s)

        # 3. C-only: keep x[:, d_s:d_s+d_c], shuffle the rest
        perm_s = np.random.permutation(n)
        x_c_only = np.zeros_like(x)
        x_c_only[:, :d_s] = x[perm_s, :d_s]
        x_c_only[:, d_s:d_s + d_c] = x[:, d_s:d_s + d_c]
        x_c_only[:, d_s + d_c:] = x[perm_n, d_s + d_c:]

        logits_c = model(torch.from_numpy(x_c_only).to(device))
        preds_c = torch.argmax(logits_c, dim=1).cpu().numpy()
        acc_c_only = accuracy_score(y, preds_c)

        # 4. N-only: shuffle both S and C (should yield chance accuracy ~ 0.50)
        x_n_only = np.zeros_like(x)
        x_n_only[:, :d_s] = x[perm_s, :d_s]
        x_n_only[:, d_s:d_s + d_c] = x[perm_c, d_s:d_s + d_c]
        x_n_only[:, d_s + d_c:] = x[:, d_s + d_c:]

        logits_n = model(torch.from_numpy(x_n_only).to(device))
        preds_n = torch.argmax(logits_n, dim=1).cpu().numpy()
        acc_n_only = accuracy_score(y, preds_n)

    return {
        "acc_joint": float(acc_joint),
        "acc_s_only": float(acc_s_only),
        "acc_c_only": float(acc_c_only),
        "acc_n_only": float(acc_n_only),
    }

def train_single_feature_baseline(
    feature_type: str,
    dataset: SyntheticFalsificationDataset,
    arch: str = "StandardMLP",
    lr: float = 0.01,
    optimizer_name: str = "AdamW",
    epochs: int = 40,
    seed: int = 42,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Trains a model using solely feature S or feature C to prove intrinsic learnability.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_d, val_d, test_d = dataset.get_splits()
    feat_key = "x_s" if feature_type == "S" else "x_c"
    in_dim = train_d[feat_key].shape[1]

    model = create_model(arch, in_features=in_dim).to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    x_train_t = torch.from_numpy(train_d[feat_key]).to(device)
    y_train_t = torch.from_numpy(train_d["y"]).to(device)
    x_test_t = torch.from_numpy(test_d[feat_key]).to(device)
    y_test_t = test_d["y"]

    batch_size = 64
    n_train = len(train_d["y"])
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = x_train_t[indices], y_train_t[indices]

            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(x_test_t), dim=1).cpu().numpy()
            test_acc = accuracy_score(y_test_t, preds)
        history.append(float(test_acc))

    return {
        "feature": feature_type,
        "final_acc": history[-1],
        "max_acc": max(history),
        "history": history
    }

def train_joint_experiment(
    dataset: SyntheticFalsificationDataset,
    arch: str = "StandardMLP",
    lr: float = 0.01,
    optimizer_name: str = "AdamW",
    epochs: int = 50,
    seed: int = 42,
    threshold: float = 0.80,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Main joint training loop tracking S vs C learning dynamics and time-to-threshold.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_d, val_d, test_d = dataset.get_splits()
    in_dim = train_d["x_full"].shape[1]

    model = create_model(arch, in_features=in_dim).to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    x_train_t = torch.from_numpy(train_d["x_full"]).to(device)
    y_train_t = torch.from_numpy(train_d["y"]).to(device)
    n_train = len(y_train_t)
    batch_size = 64

    dynamics = []
    t_s = None
    t_c = None

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        ep_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            bx, by = x_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            n_batches += 1

        avg_loss = ep_loss / max(n_batches, 1)

        # Counterfactual evaluation at this epoch
        eval_res = evaluate_counterfactual(
            model,
            test_d,
            d_s=dataset.d_s,
            d_c=dataset.d_c,
            d_n=dataset.d_noise,
            device=device
        )
        eval_res["epoch"] = ep
        eval_res["train_loss"] = float(avg_loss)
        dynamics.append(eval_res)

        # Check threshold crossings
        if t_s is None and eval_res["acc_s_only"] >= threshold:
            t_s = ep
        if t_c is None and eval_res["acc_c_only"] >= threshold:
            t_c = ep

    # If threshold was never reached, set to epochs + 1 (censored value)
    t_s_val = t_s if t_s is not None else (epochs + 1)
    t_c_val = t_c if t_c is not None else (epochs + 1)
    delta = t_c_val - t_s_val

    # Final feature reliance score: R = max(0, Acc - 0.5)
    final_res = dynamics[-1]
    rel_s = max(0.0, final_res["acc_s_only"] - 0.5)
    rel_c = max(0.0, final_res["acc_c_only"] - 0.5)
    total_rel = rel_s + rel_c + 1e-8
    ratio_s = rel_s / total_rel

    return {
        "arch": arch,
        "optimizer": optimizer_name,
        "lr": lr,
        "seed": seed,
        "threshold": threshold,
        "T_S": int(t_s_val),
        "T_C": int(t_c_val),
        "Delta": int(delta),
        "final_acc_joint": float(final_res["acc_joint"]),
        "final_acc_s_only": float(final_res["acc_s_only"]),
        "final_acc_c_only": float(final_res["acc_c_only"]),
        "final_acc_n_only": float(final_res["acc_n_only"]),
        "reliance_ratio_S": float(ratio_s),
        "dynamics": dynamics,
    }
