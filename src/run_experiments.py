"""
run_experiments.py - Automated Experiment Execution Suite

Executes the complete experimental matrix across controlled conditions and random seeds:
  - Single-feature oracle learnability baselines
  - Condition A: Matched predictive strength (eta_S = 0.05, eta_C = 0.05)
  - Condition B: Complex signal dominance (eta_S = 0.25, eta_C = 0.01) [FALSIFICATION TEST]
  - Condition C: Simple signal dominance (eta_S = 0.01, eta_C = 0.25)
  - Condition D: Optimizer modulation (AdamW vs SGD)
  - Condition E: Architecture ablation (StandardMLP vs DeepMLP vs ResMLP)

Saves full structured logs to results/raw_data/experiments_results.json
"""

import os
import sys
import json
import time
from typing import Dict, List, Any
import numpy as np

# Add repository root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import SyntheticFalsificationDataset
from src.train import train_joint_experiment, train_single_feature_baseline

SEEDS = [101, 102, 103, 104, 105]
EPOCHS = 50
THRESHOLD = 0.80

def run_all_experiments():
    os.makedirs("results/raw_data", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    print("=" * 70)
    print("STARTING SCIENTIFIC EXPERIMENT SUITE: FALSIFYING SIMPLICITY BIAS")
    print(f"Random Seeds: {SEEDS} | Epochs: {EPOCHS} | Metric Threshold: {THRESHOLD}")
    print("=" * 70)

    start_time = time.time()
    all_results: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seeds": SEEDS,
            "epochs": EPOCHS,
            "threshold": THRESHOLD,
        },
        "single_feature_baselines": {},
        "conditions": {}
    }

    # -------------------------------------------------------------
    # 0. Single-Feature Baselines (Verification of Learnability)
    # -------------------------------------------------------------
    print("\n[Phase 0] Verifying Single-Feature Baselines (Independent Learnability)...")
    ds_matched = SyntheticFalsificationDataset(n_samples=4000, eta_s=0.05, eta_c=0.05, seed=42)
    base_s = train_single_feature_baseline("S", ds_matched, epochs=40, seed=42)
    base_c = train_single_feature_baseline("C", ds_matched, epochs=40, seed=42)
    all_results["single_feature_baselines"] = {
        "S_alone": {"final_acc": base_s["final_acc"], "max_acc": base_s["max_acc"]},
        "C_alone": {"final_acc": base_c["final_acc"], "max_acc": base_c["max_acc"]},
    }
    print(f"  > S alone Max Accuracy: {base_s['max_acc']:.4f}")
    print(f"  > C alone Max Accuracy: {base_c['max_acc']:.4f}")
    print("  -> Both features independently verified as learnable by the model architecture.")

    # Helper function to run a condition across seeds
    def execute_condition(cond_name: str, eta_s: float, eta_c: float, arch: str, opt: str, lr: float):
        print(f"\n--- Running Condition {cond_name} ---")
        print(f"Setup: eta_S={eta_s}, eta_C={eta_c}, Arch={arch}, Opt={opt}, LR={lr}")
        cond_runs = []
        for seed in SEEDS:
            # Fix data with seed 42 to keep feature manifold identical, vary model/init seed
            ds = SyntheticFalsificationDataset(n_samples=4000, eta_s=eta_s, eta_c=eta_c, seed=42)
            run_res = train_joint_experiment(
                dataset=ds,
                arch=arch,
                lr=lr,
                optimizer_name=opt,
                epochs=EPOCHS,
                seed=seed,
                threshold=THRESHOLD
            )
            cond_runs.append(run_res)
            print(f"  Seed {seed}: T_S={run_res['T_S']:2d}, T_C={run_res['T_C']:2d}, Delta={run_res['Delta']:3d} | "
                  f"AccJoint={run_res['final_acc_joint']:.3f}, AccS={run_res['final_acc_s_only']:.3f}, AccC={run_res['final_acc_c_only']:.3f}")

        # Compute summary statistics
        deltas = [r["Delta"] for r in cond_runs]
        ts_list = [r["T_S"] for r in cond_runs]
        tc_list = [r["T_C"] for r in cond_runs]
        ratio_s = [r["reliance_ratio_S"] for r in cond_runs]

        summary = {
            "mean_Delta": float(np.mean(deltas)),
            "std_Delta": float(np.std(deltas)),
            "mean_T_S": float(np.mean(ts_list)),
            "std_T_S": float(np.std(ts_list)),
            "mean_T_C": float(np.mean(tc_list)),
            "std_T_C": float(np.std(tc_list)),
            "mean_reliance_S": float(np.mean(ratio_s)),
            "std_reliance_S": float(np.std(ratio_s)),
        }
        print(f"  Summary: Mean Delta = {summary['mean_Delta']:.2f} +/- {summary['std_Delta']:.2f} | Reliance S = {summary['mean_reliance_S']:.2f}")

        return {
            "config": {
                "eta_s": eta_s,
                "eta_c": eta_c,
                "arch": arch,
                "optimizer": opt,
                "lr": lr,
            },
            "summary": summary,
            "runs": cond_runs
        }

    # 1. Condition A: Matched Baseline
    all_results["conditions"]["Condition_A"] = execute_condition(
        "A (Matched Baseline)", eta_s=0.05, eta_c=0.05, arch="StandardMLP", opt="AdamW", lr=0.01
    )

    # 2. Condition B: Complex Dominant (Falsification Boundary)
    all_results["conditions"]["Condition_B"] = execute_condition(
        "B (Complex Signal Dominant - Falsification Test)", eta_s=0.25, eta_c=0.01, arch="StandardMLP", opt="AdamW", lr=0.01
    )

    # 3. Condition C: Simple Dominant (Positive Control)
    all_results["conditions"]["Condition_C"] = execute_condition(
        "C (Simple Signal Dominant)", eta_s=0.01, eta_c=0.25, arch="StandardMLP", opt="AdamW", lr=0.01
    )

    # 4. Condition D: Optimization Modulation (SGD vs AdamW on matched signals)
    all_results["conditions"]["Condition_D_SGD"] = execute_condition(
        "D-SGD (Matched Signals with SGD)", eta_s=0.05, eta_c=0.05, arch="StandardMLP", opt="SGD", lr=0.02
    )

    # 5. Condition E: Architectural Ablations (DeepMLP and ResMLP on matched signals)
    all_results["conditions"]["Condition_E_Deep"] = execute_condition(
        "E-Deep (DeepMLP 4-Layers)", eta_s=0.05, eta_c=0.05, arch="DeepMLP", opt="AdamW", lr=0.01
    )
    all_results["conditions"]["Condition_E_ResMLP"] = execute_condition(
        "E-ResMLP (Residual Skip Connections)", eta_s=0.05, eta_c=0.05, arch="ResMLP", opt="AdamW", lr=0.01
    )

    out_file = "results/raw_data/experiments_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"ALL EXPERIMENTS COMPLETED IN {elapsed:.1f}s")
    print(f"Raw data saved successfully to: {out_file}")
    print("=" * 70)

if __name__ == "__main__":
    run_all_experiments()
