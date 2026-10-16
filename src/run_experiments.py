"""
run_experiments.py - Automated Experiment Execution Suite

Executes the complete experimental matrix across controlled conditions and random seeds:
  - Single-feature oracle learnability baselines (S alone, C alone)
  - Condition A: Matched predictive strength (eta_S = 0.05, eta_C = 0.05)
  - Condition B: Complex signal dominance (eta_S = 0.25, eta_C = 0.01) [FALSIFICATION TEST]
  - Condition C: Simple signal dominance (eta_S = 0.01, eta_C = 0.25) [POSITIVE CONTROL]
  - Condition D: Optimizer modulation (AdamW vs SGD on matched signals)
  - Condition E1: Architecture ablation (DeepMLP 4-Layer on matched signals)
  - Condition E2: Architecture ablation (ResMLP on matched signals)

Saves full structured logs to results/raw_data/experiments_results.json
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional
import numpy as np

# Add repository root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import SyntheticFalsificationDataset
from src.models import create_model
from src.train import train_joint_experiment, train_single_feature_baseline

import argparse

SEEDS = [101, 102, 103, 104, 105]
DATA_SEED = 42
EPOCHS = 50
THRESHOLD = 0.80

def get_param_count(model_name: str, in_features: int = 8) -> int:
    model = create_model(model_name, in_features=in_features)
    return sum(p.numel() for p in model.parameters())

def run_all_experiments(
    protocol: str = "corrected",
    monitoring_split: str = "validation",
    out_file: Optional[str] = None
):
    if out_file is None:
        if protocol == "corrected":
            out_file = "results/raw_data/corrected_protocol/experiments_results_corrected.json"
        else:
            out_file = "results/raw_data/historical_run/experiments_results_historical_test_monitored.json"

    out_dir = os.path.dirname(out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    print("=" * 70)
    print(f"STARTING AUDITED EXPERIMENT SUITE ({protocol.upper()} PROTOCOL)")
    print(f"Monitoring Split: {monitoring_split} | Out File: {out_file}")
    print(f"Data Seed: {DATA_SEED} | Model Seeds: {SEEDS} | Epochs: {EPOCHS} | Threshold: {THRESHOLD}")
    print("=" * 70)

    start_time = time.time()
    all_results: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "protocol": protocol,
            "monitoring_split": monitoring_split,
            "data_seed": DATA_SEED,
            "seeds": SEEDS,
            "epochs": EPOCHS,
            "threshold": THRESHOLD,
            "sample_size": 4000,
            "train_samples": 2800,
            "val_samples": 600,
            "test_samples": 600,
            "batch_size": 64,
            "steps_per_epoch": 44,
            "total_optimizer_steps_per_run": 2200,
            "architectures_parameter_counts": {
                "StandardMLP": get_param_count("StandardMLP"),
                "DeepMLP": get_param_count("DeepMLP"),
                "ResMLP": get_param_count("ResMLP"),
            }
        },
        "single_feature_baselines": {},
        "conditions": {}
    }

    # -------------------------------------------------------------
    # 0. Single-Feature Baselines (Verification of Independent Learnability)
    # -------------------------------------------------------------
    print("\n[Phase 0] Verifying Single-Feature Baselines (Independent Learnability)...")
    ds_matched = SyntheticFalsificationDataset(n_samples=4000, eta_s=0.05, eta_c=0.05, seed=DATA_SEED)
    sep_metrics = ds_matched.evaluate_feature_separability()
    base_s = train_single_feature_baseline("S", ds_matched, epochs=40, seed=DATA_SEED)
    base_c = train_single_feature_baseline("C", ds_matched, epochs=40, seed=DATA_SEED)

    all_results["single_feature_baselines"] = {
        "S_alone": {
            "final_acc": base_s["final_acc"],
            "max_acc": base_s["max_acc"],
            "steps": base_s["total_optimizer_steps"]
        },
        "C_alone": {
            "final_acc": base_c["final_acc"],
            "max_acc": base_c["max_acc"],
            "steps": base_c["total_optimizer_steps"]
        },
        "logistic_regression": sep_metrics
    }
    print(f"  > S alone Max Accuracy: {base_s['max_acc']:.4f}")
    print(f"  > C alone Max Accuracy: {base_c['max_acc']:.4f}")
    print(f"  > Linear LR on S: {sep_metrics['linear_acc_on_S']:.4f}")
    print(f"  > Linear LR on C (origin): {sep_metrics['linear_acc_on_C_origin']:.4f}")
    print(f"  > Affine LR on C: {sep_metrics['affine_acc_on_C']:.4f}")

    # Helper function to execute a condition across seeds
    def execute_condition(cond_id: str, display_name: str, eta_s: float, eta_c: float, arch: str, opt: str, lr: float):
        print(f"\n--- Running Condition {cond_id}: {display_name} ---")
        param_count = get_param_count(arch)
        print(f"Setup: eta_S={eta_s}, eta_C={eta_c}, Arch={arch} ({param_count} params), Opt={opt}, LR={lr}")
        cond_runs = []
        for seed in SEEDS:
            ds = SyntheticFalsificationDataset(n_samples=4000, eta_s=eta_s, eta_c=eta_c, seed=DATA_SEED)
            run_res = train_joint_experiment(
                dataset=ds,
                arch=arch,
                lr=lr,
                optimizer_name=opt,
                epochs=EPOCHS,
                seed=seed,
                threshold=THRESHOLD,
                monitoring_split=monitoring_split,
            )
            cond_runs.append(run_res)
            c_tag = "crossed" if not run_res["T_C_censored"] else "censored(>50)"
            s_tag = "crossed" if not run_res["T_S_censored"] else "censored(>50)"
            print(f"  Seed {seed}: T_S={run_res['T_S']:2d} ({s_tag}), T_C={run_res['T_C']:2d} ({c_tag}), Delta={run_res['Delta']:3d} | "
                  f"Joint={run_res['final_acc_joint']:.3f}, AccS={run_res['final_acc_s_only']:.3f}, AccC={run_res['final_acc_c_only']:.3f}")

        # Compute summary statistics (Task 2: No continuous calculations on censored 51)
        ratio_s = [r["reliance_ratio_S"] for r in cond_runs]
        joint_accs = [r["final_acc_joint"] for r in cond_runs]
        s_accs = [r["final_acc_s_only"] for r in cond_runs]
        c_accs = [r["final_acc_c_only"] for r in cond_runs]
        n_accs = [r["final_acc_n_only"] for r in cond_runs]

        # Extract observed crossing epochs strictly from runs that crossed
        s_obs_epochs = [r["S_acquisition"]["observed_epoch"] for r in cond_runs if r["S_acquisition"]["crossed_threshold"]]
        c_obs_epochs = [r["C_acquisition"]["observed_epoch"] for r in cond_runs if r["C_acquisition"]["crossed_threshold"]]

        s_censored_count = sum(1 for r in cond_runs if not r["S_acquisition"]["crossed_threshold"])
        c_censored_count = sum(1 for r in cond_runs if not r["C_acquisition"]["crossed_threshold"])

        s_cross_rate = len(s_obs_epochs) / len(cond_runs)
        c_cross_rate = len(c_obs_epochs) / len(cond_runs)

        mean_t_s_obs = float(np.mean(s_obs_epochs)) if s_obs_epochs else None
        std_t_s_obs = float(np.std(s_obs_epochs, ddof=1)) if len(s_obs_epochs) > 1 else 0.0
        mean_t_c_obs = float(np.mean(c_obs_epochs)) if c_obs_epochs else None
        std_t_c_obs = float(np.std(c_obs_epochs, ddof=1)) if len(c_obs_epochs) > 1 else 0.0

        summary = {
            "S_crossing_rate": float(s_cross_rate),
            "S_crossing_count": f"{len(s_obs_epochs)}/{len(cond_runs)}",
            "S_observed_crossing_epochs": s_obs_epochs,
            "S_mean_observed_epoch": mean_t_s_obs,
            "S_std_observed_epoch": std_t_s_obs,
            "S_censored_count": s_censored_count,
            "C_crossing_rate": float(c_cross_rate),
            "C_crossing_count": f"{len(c_obs_epochs)}/{len(cond_runs)}",
            "C_observed_crossing_epochs": c_obs_epochs,
            "C_mean_observed_epoch": mean_t_c_obs,
            "C_std_observed_epoch": std_t_c_obs,
            "C_censored_count": c_censored_count,
            "mean_joint_acc": float(np.mean(joint_accs)),
            "std_joint_acc": float(np.std(joint_accs)),
            "mean_acc_s_only": float(np.mean(s_accs)),
            "std_acc_s_only": float(np.std(s_accs)),
            "mean_acc_c_only": float(np.mean(c_accs)),
            "std_acc_c_only": float(np.std(c_accs)),
            "mean_acc_n_only": float(np.mean(n_accs)),
            "std_acc_n_only": float(np.std(n_accs)),
            "mean_reliance_S": float(np.mean(ratio_s)),
            "std_reliance_S": float(np.std(ratio_s)),
        }
        print(f"  Summary: Crossing Rates: S={summary['S_crossing_rate']:.0%} ({summary['S_crossing_count']}), "
              f"C={summary['C_crossing_rate']:.0%} ({summary['C_crossing_count']}) | "
              f"Observed T_S={s_obs_epochs}, Observed T_C={c_obs_epochs} | Reliance S = {summary['mean_reliance_S']:.2f}")

        return {
            "condition_id": cond_id,
            "display_name": display_name,
            "config": {
                "eta_s": eta_s,
                "eta_c": eta_c,
                "arch": arch,
                "parameter_count": param_count,
                "optimizer": opt,
                "lr": lr,
            },
            "summary": summary,
            "runs": cond_runs
        }

    # 1. Condition A: Matched Baseline
    all_results["conditions"]["Condition_A"] = execute_condition(
        "A", "Matched Baseline (eta_S=0.05, eta_C=0.05)", eta_s=0.05, eta_c=0.05, arch="StandardMLP", opt="AdamW", lr=0.01
    )

    # 2. Condition B: Complex Dominant (Falsification Boundary)
    all_results["conditions"]["Condition_B"] = execute_condition(
        "B", "Complex Signal Dominant (eta_S=0.25, eta_C=0.01)", eta_s=0.25, eta_c=0.01, arch="StandardMLP", opt="AdamW", lr=0.01
    )

    # 3. Condition C: Simple Dominant (Positive Control)
    all_results["conditions"]["Condition_C"] = execute_condition(
        "C", "Simple Signal Dominant (eta_S=0.01, eta_C=0.25)", eta_s=0.01, eta_c=0.25, arch="StandardMLP", opt="AdamW", lr=0.01
    )

    # 4. Condition D: Optimization Modulation (SGD vs AdamW on matched signals)
    all_results["conditions"]["Condition_D_SGD"] = execute_condition(
        "D", "Matched Signals with SGD (eta_S=0.05, eta_C=0.05)", eta_s=0.05, eta_c=0.05, arch="StandardMLP", opt="SGD", lr=0.02
    )

    # 5. Condition E1: Architecture Ablation (DeepMLP 4-Layers on matched signals)
    all_results["conditions"]["Condition_E1_Deep"] = execute_condition(
        "E1", "DeepMLP 4-Layers (eta_S=0.05, eta_C=0.05)", eta_s=0.05, eta_c=0.05, arch="DeepMLP", opt="AdamW", lr=0.01
    )

    # 6. Condition E2: Architecture Ablation (ResMLP on matched signals)
    all_results["conditions"]["Condition_E2_ResMLP"] = execute_condition(
        "E2", "ResMLP Residual Connections (eta_S=0.05, eta_C=0.05)", eta_s=0.05, eta_c=0.05, arch="ResMLP", opt="AdamW", lr=0.01
    )

    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"ALL EXPERIMENTS COMPLETED IN {elapsed:.1f}s")
    print(f"Raw verified data saved successfully to: {out_file}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run audited PARITY experiments")
    parser.add_argument("--protocol", choices=["corrected", "historical"], default="corrected",
                        help="Protocol: 'corrected' uses validation-set monitoring; 'historical' uses test-set monitoring")
    parser.add_argument("--monitoring-split", choices=["validation", "test"], default=None,
                        help="Override monitoring split explicitly ('validation' or 'test')")
    parser.add_argument("--output-file", default=None,
                        help="Explicit destination JSON path")
    args = parser.parse_args()

    split = args.monitoring_split or ("validation" if args.protocol == "corrected" else "test")
    run_all_experiments(
        protocol=args.protocol,
        monitoring_split=split,
        out_file=args.output_file,
    )
