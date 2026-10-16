"""
train.py - Controlled Training and Counterfactual Evaluation Engine

Executes training loops with rigorous tracking:
  - Exact optimizer-step accounting (steps_per_epoch, total_optimizer_steps).
  - Counterfactual feature reliance via reproducible multi-trial marginal feature permutation.
  - Transparent threshold-crossing tracking with explicit right-censoring indicators.
  - Single-feature oracle baselines.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from typing import Dict, Any, Tuple, Optional, List

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
    device: str = "cpu",
    n_permutations: int = 5,
    rng: Optional[np.random.RandomState] = None,
) -> Dict[str, float]:
    """
    Evaluates model performance under counterfactual feature interventions:
      1. Joint: unperturbed inputs [x_s, x_c, x_n]
      2. S-only: x_c and x_n permuted across the test set (marginalizing out C and N)
      3. C-only: x_s and x_n permuted across the test set (marginalizing out S and N)
      4. N-only: x_s and x_c permuted across the test set (null control, should be ~0.50)

    Averages over n_permutations independent shuffles to eliminate Monte Carlo permutation noise.
    """
    model.eval()
    x = test_data["x_full"]
    y = test_data["y"]
    n = len(x)

    if rng is None:
        rng = np.random.RandomState(42)

    with torch.no_grad():
        # 1. Joint evaluation (deterministic on test set)
        logits_joint = model(torch.from_numpy(x).to(device))
        preds_joint = torch.argmax(logits_joint, dim=1).cpu().numpy()
        acc_joint = float(accuracy_score(y, preds_joint))

        acc_s_list: List[float] = []
        acc_c_list: List[float] = []
        acc_n_list: List[float] = []

        for _ in range(n_permutations):
            perm_s = rng.permutation(n)
            perm_c = rng.permutation(n)
            perm_n = rng.permutation(n)

            # S-only: keep S, shuffle C and N
            x_s_only = np.zeros_like(x)
            x_s_only[:, :d_s] = x[:, :d_s]
            x_s_only[:, d_s:d_s + d_c] = x[perm_c, d_s:d_s + d_c]
            x_s_only[:, d_s + d_c:] = x[perm_n, d_s + d_c:]

            logits_s = model(torch.from_numpy(x_s_only).to(device))
            preds_s = torch.argmax(logits_s, dim=1).cpu().numpy()
            acc_s_list.append(float(accuracy_score(y, preds_s)))

            # C-only: keep C, shuffle S and N
            x_c_only = np.zeros_like(x)
            x_c_only[:, :d_s] = x[perm_s, :d_s]
            x_c_only[:, d_s:d_s + d_c] = x[:, d_s:d_s + d_c]
            x_c_only[:, d_s + d_c:] = x[perm_n, d_s + d_c:]

            logits_c = model(torch.from_numpy(x_c_only).to(device))
            preds_c = torch.argmax(logits_c, dim=1).cpu().numpy()
            acc_c_list.append(float(accuracy_score(y, preds_c)))

            # N-only: shuffle S and C, keep N
            x_n_only = np.zeros_like(x)
            x_n_only[:, :d_s] = x[perm_s, :d_s]
            x_n_only[:, d_s:d_s + d_c] = x[perm_c, d_s:d_s + d_c]
            x_n_only[:, d_s + d_c:] = x[:, d_s + d_c:]

            logits_n = model(torch.from_numpy(x_n_only).to(device))
            preds_n = torch.argmax(logits_n, dim=1).cpu().numpy()
            acc_n_list.append(float(accuracy_score(y, preds_n)))

    return {
        "acc_joint": acc_joint,
        "acc_s_only": float(np.mean(acc_s_list)),
        "acc_c_only": float(np.mean(acc_c_list)),
        "acc_n_only": float(np.mean(acc_n_list)),
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
    Trains a model using solely feature S or feature C to prove intrinsic single-feature learnability.
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
    total_steps = 0

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
            total_steps += 1

        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(x_test_t), dim=1).cpu().numpy()
            test_acc = accuracy_score(y_test_t, preds)
        history.append(float(test_acc))

    return {
        "feature": feature_type,
        "final_acc": history[-1],
        "max_acc": max(history),
        "total_optimizer_steps": total_steps,
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
    monitoring_split: str = "validation",
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Main joint training loop implementing clean split separation:
      - Training split (train_d): Model optimization only.
      - Validation split (val_d): Per-epoch monitoring and threshold-crossing detection.
      - Test split (test_d): Held-out out-of-sample evaluation at termination.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    eval_rng = np.random.RandomState(seed + 9999)

    train_d, val_d, test_d = dataset.get_splits()
    in_dim = train_d["x_full"].shape[1]

    # Select split for in-loop monitoring
    if monitoring_split == "validation":
        monitor_d = val_d
    elif monitoring_split == "test":
        monitor_d = test_d
    else:
        raise ValueError(f"Unsupported monitoring_split: {monitoring_split}")

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

    # Calculate exact steps per epoch
    steps_per_epoch = int(np.ceil(n_train / batch_size))
    total_optimizer_steps = 0

    val_dynamics = []
    t_s: Optional[int] = None
    t_c: Optional[int] = None
    test_ep1_res: Optional[Dict[str, float]] = None

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
            total_optimizer_steps += 1

            ep_loss += loss.item()
            n_batches += 1

        avg_loss = ep_loss / max(n_batches, 1)

        # Per-epoch counterfactual evaluation on monitoring split (validation)
        val_eval_res = evaluate_counterfactual(
            model,
            monitor_d,
            d_s=dataset.d_s,
            d_c=dataset.d_c,
            d_n=dataset.d_noise,
            device=device,
            n_permutations=5,
            rng=eval_rng,
        )
        val_eval_res["epoch"] = ep
        val_eval_res["train_loss"] = float(avg_loss)
        val_dynamics.append(val_eval_res)

        # Check threshold crossings on monitoring split
        if t_s is None and val_eval_res["acc_s_only"] >= threshold:
            t_s = ep
        if t_c is None and val_eval_res["acc_c_only"] >= threshold:
            t_c = ep

        # Record test evaluation at epoch 1 as a held-out check
        if ep == 1:
            test_ep1_res = evaluate_counterfactual(
                model,
                test_d,
                d_s=dataset.d_s,
                d_c=dataset.d_c,
                d_n=dataset.d_noise,
                device=device,
                n_permutations=5,
                rng=eval_rng,
            )

    # Final out-of-sample evaluation on held-out TEST split (epoch 50)
    final_test_res = evaluate_counterfactual(
        model,
        test_d,
        d_s=dataset.d_s,
        d_c=dataset.d_c,
        d_n=dataset.d_noise,
        device=device,
        n_permutations=5,
        rng=eval_rng,
    )

    # Explicit censoring accounting (Task 2):
    t_s_censored = (t_s is None)
    t_c_censored = (t_c is None)

    s_acquisition = {
        "crossed_threshold": bool(not t_s_censored),
        "observed_epoch": int(t_s) if t_s is not None else None,
        "censoring_limit": int(epochs),
        "monitoring_split": monitoring_split,
    }
    c_acquisition = {
        "crossed_threshold": bool(not t_c_censored),
        "observed_epoch": int(t_c) if t_c is not None else None,
        "censoring_limit": int(epochs),
        "monitoring_split": monitoring_split,
    }

    # Tracking values preserved for schema compatibility:
    t_s_val = t_s if t_s is not None else (epochs + 1)
    t_c_val = t_c if t_c is not None else (epochs + 1)
    delta = t_c_val - t_s_val

    # Final feature reliance scores
    final_val_res = val_dynamics[-1]
    rel_s_test = max(0.0, final_test_res["acc_s_only"] - 0.5)
    rel_c_test = max(0.0, final_test_res["acc_c_only"] - 0.5)
    total_rel_test = rel_s_test + rel_c_test + 1e-8
    ratio_s_test = rel_s_test / total_rel_test

    rel_s_val = max(0.0, final_val_res["acc_s_only"] - 0.5)
    rel_c_val = max(0.0, final_val_res["acc_c_only"] - 0.5)
    total_rel_val = rel_s_val + rel_c_val + 1e-8
    ratio_s_val = rel_s_val / total_rel_val

    return {
        "arch": arch,
        "optimizer": optimizer_name,
        "lr": lr,
        "seed": seed,
        "threshold": threshold,
        "epochs": epochs,
        "monitoring_split": monitoring_split,
        "batch_size": batch_size,
        "n_train_samples": n_train,
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,
        "S_acquisition": s_acquisition,
        "C_acquisition": c_acquisition,
        "T_S_observed": t_s,
        "T_C_observed": t_c,
        "T_S_censored": t_s_censored,
        "T_C_censored": t_c_censored,
        "T_S": int(t_s_val),
        "T_C": int(t_c_val),
        "Delta": int(delta),
        # Held-out test split metrics (primary evaluation):
        "final_test_metrics": final_test_res,
        "epoch_1_test_metrics": test_ep1_res,
        "final_acc_joint": float(final_test_res["acc_joint"]),
        "final_acc_s_only": float(final_test_res["acc_s_only"]),
        "final_acc_c_only": float(final_test_res["acc_c_only"]),
        "final_acc_n_only": float(final_test_res["acc_n_only"]),
        "reliance_ratio_S": float(ratio_s_test),
        # Validation monitoring metrics:
        "final_val_metrics": final_val_res,
        "final_val_acc_joint": float(final_val_res["acc_joint"]),
        "final_val_acc_s_only": float(final_val_res["acc_s_only"]),
        "final_val_acc_c_only": float(final_val_res["acc_c_only"]),
        "final_val_reliance_ratio_S": float(ratio_s_val),
        "dynamics": val_dynamics,
    }
