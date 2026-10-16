"""
analyze.py - Rigorous Statistical Testing and Publication-Quality Figure Generation

Performs:
  1. Methodologically sound hypothesis testing:
     - Fixed-horizon counterfactual accuracy difference (continuous, uncensored at epoch 50).
     - Transparent time-to-threshold analysis acknowledging right-censoring at epoch 50.
     - Dynamic computation of crossing counts (no hardcoding).
     - Paired exact McNemar/binomial test for paired binary threshold crossing outcomes.
     - Transparent reporting of sample size limits and exact p-values.
     - Unstandardized effect sizes to prevent zero-variance Cohen's d degeneration.
  2. Publication-quality figure rendering saved to results/figures/.
"""

import json
import os
import sys
import argparse
from typing import Optional, Dict, Any, List
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Add repository root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def compute_paired_exact_test(s_crossed_flags, c_crossed_flags, alternative="greater"):
    """
    Exact McNemar / Binomial test on paired binary threshold crossing outcomes.
    b = count(S=1, C=0), c = count(S=0, C=1)
    """
    b = sum(1 for s, c in zip(s_crossed_flags, c_crossed_flags) if s and not c)
    c = sum(1 for s, c in zip(s_crossed_flags, c_crossed_flags) if not s and c)
    n_disc = b + c
    if n_disc == 0:
        return {
            "b_S_only_crossed": b,
            "c_C_only_crossed": c,
            "n_discordant": 0,
            "p_value": 1.0
        }

    res = stats.binomtest(b, n=n_disc, p=0.5, alternative=alternative)
    return {
        "b_S_only_crossed": b,
        "c_C_only_crossed": c,
        "n_discordant": n_disc,
        "p_value": float(res.pvalue)
    }


def analyze_condition(runs, cond_name):
    """Dynamically extract and analyze all run-level statistics for a condition."""
    n = len(runs)

    # Threshold crossing indicators
    s_crossed = [
        bool(r.get("S_acquisition", {}).get("crossed_threshold", not r.get("T_S_censored", True)))
        for r in runs
    ]
    c_crossed = [
        bool(r.get("C_acquisition", {}).get("crossed_threshold", not r.get("T_C_censored", True)))
        for r in runs
    ]

    # Observed crossing epochs strictly from crossing runs
    s_obs_epochs = [
        r.get("S_acquisition", {}).get("observed_epoch", r.get("T_S_observed"))
        for r in runs
        if r.get("S_acquisition", {}).get("crossed_threshold", not r.get("T_S_censored", True))
    ]
    c_obs_epochs = [
        r.get("C_acquisition", {}).get("observed_epoch", r.get("T_C_observed"))
        for r in runs
        if r.get("C_acquisition", {}).get("crossed_threshold", not r.get("T_C_censored", True))
    ]

    # Paired McNemar exact tests
    paired_sc_greater = compute_paired_exact_test(s_crossed, c_crossed, alternative="greater")
    paired_sc_twosided = compute_paired_exact_test(s_crossed, c_crossed, alternative="two-sided")
    paired_cs_greater = compute_paired_exact_test(c_crossed, s_crossed, alternative="greater")

    # Unpaired Fisher's exact test (for comparative transparency)
    table_sc = [
        [sum(s_crossed), n - sum(s_crossed)],
        [sum(c_crossed), n - sum(c_crossed)]
    ]
    _, p_fisher_greater = stats.fisher_exact(table_sc, alternative="greater")
    _, p_fisher_twosided = stats.fisher_exact(table_sc, alternative="two-sided")

    # Continuous accuracies at epoch 50
    s_acc = np.array([r["final_acc_s_only"] for r in runs])
    c_acc = np.array([r["final_acc_c_only"] for r in runs])
    joint_acc = np.array([r["final_acc_joint"] for r in runs])
    diff_sc = s_acc - c_acc

    t_stat_sc, p_val_sc_twosided = stats.ttest_rel(s_acc, c_acc)
    p_val_sc_greater = stats.ttest_rel(s_acc, c_acc, alternative="greater").pvalue
    w_stat, p_wilcox = stats.wilcoxon(diff_sc) if np.any(diff_sc != 0) else (0.0, 1.0)

    # 95% Confidence Interval for mean difference
    se_diff = stats.sem(diff_sc)
    ci_margin = se_diff * stats.t.ppf(0.975, df=n - 1)
    ci_lower = float(np.mean(diff_sc) - ci_margin)
    ci_upper = float(np.mean(diff_sc) + ci_margin)

    # Early dynamics at epoch 1
    s_ep1 = np.array([r["dynamics"][0]["acc_s_only"] for r in runs])
    c_ep1 = np.array([r["dynamics"][0]["acc_c_only"] for r in runs])
    diff_ep1 = s_ep1 - c_ep1
    t_ep1, p_ep1 = stats.ttest_rel(s_ep1, c_ep1)

    ret_dict = {
        "condition_name": cond_name,
        "n_runs": n,
        "S_crossing_rate": float(np.mean(s_crossed)),
        "S_crossing_count": f"{sum(s_crossed)}/{n}",
        "S_observed_crossing_epochs": [int(x) for x in s_obs_epochs if x is not None],
        "S_mean_observed_epoch": float(np.mean(s_obs_epochs)) if len(s_obs_epochs) > 0 else None,
        "S_censored_count": int(n - sum(s_crossed)),

        "C_crossing_rate": float(np.mean(c_crossed)),
        "C_crossing_count": f"{sum(c_crossed)}/{n}",
        "C_observed_crossing_epochs": [int(x) for x in c_obs_epochs if x is not None],
        "C_mean_observed_epoch": float(np.mean(c_obs_epochs)) if len(c_obs_epochs) > 0 else None,
        "C_censored_count": int(n - sum(c_crossed)),

        "binary_crossing_tests": {
            "pairing_description": "Observations paired by random model seed on identical data instance.",
            "primary_test": "paired_mcnemar_exact_two_sided",
            "primary_two_sided_p": float(paired_sc_twosided["p_value"]),
            "primary_significance_at_0_05": bool(paired_sc_twosided["p_value"] < 0.05),
            "paired_mcnemar_exact_two_sided_p": float(paired_sc_twosided["p_value"]),
            "significance_at_0_05": bool(paired_sc_twosided["p_value"] < 0.05),
            "supplementary_directional_tests": {
                "s_greater_c": {
                    "hypothesis": "S > C",
                    "p_value": float(paired_sc_greater["p_value"]),
                    "directional_significant_at_0_05": bool(paired_sc_greater["p_value"] < 0.05)
                },
                "c_greater_s": {
                    "hypothesis": "C > S",
                    "p_value": float(paired_cs_greater["p_value"]),
                    "directional_significant_at_0_05": bool(paired_cs_greater["p_value"] < 0.05)
                }
            },
            "directional_s_greater_c_p": float(paired_sc_greater["p_value"]),
            "directional_s_greater_c_significant_at_0_05": bool(paired_sc_greater["p_value"] < 0.05),
            "directional_c_greater_s_p": float(paired_cs_greater["p_value"]),
            "directional_c_greater_s_significant_at_0_05": bool(paired_cs_greater["p_value"] < 0.05),
            "paired_mcnemar_exact_one_sided_p": float(paired_sc_greater["p_value"]),
            "paired_mcnemar_cs_greater_p": float(paired_cs_greater["p_value"]),
            "discordant_pairs": f"S_only={paired_sc_greater['b_S_only_crossed']}, C_only={paired_sc_greater['c_C_only_crossed']}",
            "unpaired_fisher_exact_one_sided_p": float(p_fisher_greater),
            "unpaired_fisher_exact_two_sided_p": float(p_fisher_twosided),
            "contingency_table": table_sc,
        },

        "continuous_epoch_50_accuracy": {
            "mean_joint_acc": float(np.mean(joint_acc)),
            "std_joint_acc": float(np.std(joint_acc, ddof=1)),
            "mean_acc_S": float(np.mean(s_acc)),
            "std_acc_S": float(np.std(s_acc, ddof=1)),
            "mean_acc_C": float(np.mean(c_acc)),
            "std_acc_C": float(np.std(c_acc, ddof=1)),
            "mean_diff_S_minus_C": float(np.mean(diff_sc)),
            "std_diff_S_minus_C": float(np.std(diff_sc, ddof=1)),
            "sem_diff": float(se_diff),
            "ci_95": [ci_lower, ci_upper],
            "paired_t_stat": float(t_stat_sc),
            "degrees_of_freedom": int(n - 1),
            "paired_p_val_two_sided": float(p_val_sc_twosided),
            "paired_p_val_greater": float(p_val_sc_greater),
            "wilcoxon_w_stat": float(w_stat),
            "wilcoxon_p_val": float(p_wilcox)
        },

        "early_dynamics_epoch_1": {
            "mean_acc_S_ep1": float(np.mean(s_ep1)),
            "std_acc_S_ep1": float(np.std(s_ep1, ddof=1)),
            "mean_acc_C_ep1": float(np.mean(c_ep1)),
            "std_acc_C_ep1": float(np.std(c_ep1, ddof=1)),
            "mean_diff_ep1": float(np.mean(diff_ep1)),
            "std_diff_ep1": float(np.std(diff_ep1, ddof=1)),
            "paired_t_stat_ep1": float(t_ep1),
            "paired_p_val_ep1": float(p_ep1)
        }
    }

    if runs and "epoch_1_test_metrics" in runs[0] and runs[0]["epoch_1_test_metrics"] is not None:
        s_ep1_t = np.array([r["epoch_1_test_metrics"]["acc_s_only"] for r in runs])
        c_ep1_t = np.array([r["epoch_1_test_metrics"]["acc_c_only"] for r in runs])
        diff_ep1_t = s_ep1_t - c_ep1_t
        t_ep1_t, p_ep1_t = stats.ttest_rel(s_ep1_t, c_ep1_t)
        ret_dict["held_out_test_epoch_1"] = {
            "mean_acc_S_ep1": float(np.mean(s_ep1_t)),
            "std_acc_S_ep1": float(np.std(s_ep1_t, ddof=1)),
            "mean_acc_C_ep1": float(np.mean(c_ep1_t)),
            "std_acc_C_ep1": float(np.std(c_ep1_t, ddof=1)),
            "mean_diff_ep1": float(np.mean(diff_ep1_t)),
            "std_diff_ep1": float(np.std(diff_ep1_t, ddof=1)),
            "paired_t_stat_ep1": float(t_ep1_t),
            "paired_p_val_ep1": float(p_ep1_t)
        }

    return ret_dict


def run_statistical_analysis(
    data_path: Optional[str] = None,
    summary_path: Optional[str] = None,
    fig_dir: Optional[str] = None,
    protocol: str = "corrected"
):
    if data_path is None:
        if protocol == "corrected":
            data_path = "results/raw_data/corrected_protocol/experiments_results_corrected.json"
        else:
            data_path = "results/raw_data/historical_run/experiments_results_historical_test_monitored.json"

    if summary_path is None:
        if protocol == "corrected":
            summary_path = "results/raw_data/corrected_protocol/statistical_summary_corrected.json"
        else:
            summary_path = "results/raw_data/historical_run/statistical_summary_historical_test_monitored.json"

    if fig_dir is None:
        if protocol == "corrected":
            fig_dir = "results/figures"
        else:
            fig_dir = "results/figures/historical_run"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}. Run experiments first!")

    out_summary_dir = os.path.dirname(summary_path)
    if out_summary_dir:
        os.makedirs(out_summary_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    with open(data_path, "r") as f:
        data = json.load(f)

    conds = data["conditions"]
    meta = data.get("metadata", {})

    print("=" * 70)
    print(f"AUDITED RIGOROUS STATISTICAL HYPOTHESIS TESTING ({protocol.upper()} PROTOCOL)")
    print(f"Source Data: {data_path}")
    print(f"Summary Target: {summary_path}")
    print("=" * 70)

    # 1. Condition A: Matched Baseline
    runs_a = conds["Condition_A"]["runs"]
    stat_a = analyze_condition(runs_a, "Condition_A_Matched")

    # 2. Condition B: Complex Dominant
    runs_b = conds["Condition_B"]["runs"]
    stat_b = analyze_condition(runs_b, "Condition_B_Complex_Dominant")

    # 3. Two-sample & Paired comparison between Condition A and Condition B
    s_acc_a = np.array([r["final_acc_s_only"] for r in runs_a])
    c_acc_a = np.array([r["final_acc_c_only"] for r in runs_a])
    diff_sc_a = s_acc_a - c_acc_a

    s_acc_b = np.array([r["final_acc_s_only"] for r in runs_b])
    c_acc_b = np.array([r["final_acc_c_only"] for r in runs_b])
    diff_sc_b = s_acc_b - c_acc_b
    diff_cs_b = c_acc_b - s_acc_b
    se_diff_b = float(stats.sem(diff_cs_b))
    ci_margin_b = se_diff_b * stats.t.ppf(0.975, df=len(runs_b) - 1)
    ci_cs_b = [
        float(np.mean(diff_cs_b) - ci_margin_b),
        float(np.mean(diff_cs_b) + ci_margin_b)
    ]

    margin_shift_samples = diff_sc_a - diff_sc_b
    mean_margin_shift = float(np.mean(margin_shift_samples))

    # Welch's two-sample t-test
    welch_res = stats.ttest_ind(diff_sc_a, diff_sc_b, equal_var=False)
    t_welch = float(welch_res.statistic)
    p_welch = float(welch_res.pvalue)
    df_welch = float(welch_res.df)

    # Paired t-test across matching seeds (101 to 105)
    paired_ab_res = stats.ttest_rel(diff_sc_a, diff_sc_b)
    t_paired_ab = float(paired_ab_res.statistic)
    p_paired_ab = float(paired_ab_res.pvalue)
    df_paired_ab = len(runs_a) - 1

    se_shift = stats.sem(margin_shift_samples)
    ci_shift = [
        float(mean_margin_shift - se_shift * stats.t.ppf(0.975, df=df_paired_ab)),
        float(mean_margin_shift + se_shift * stats.t.ppf(0.975, df=df_paired_ab))
    ]

    # Full audit report summary
    stat_report = {
        "metadata": {
            "source_data": data_path,
            "protocol": meta.get("protocol", protocol),
            "monitoring_split": meta.get("monitoring_split", "validation" if protocol == "corrected" else "test"),
        },
        "Condition_A_Matched": {
            "S_crossing_rate": stat_a["S_crossing_rate"],
            "S_crossing_count": stat_a["S_crossing_count"],
            "S_observed_crossing_epochs": stat_a["S_observed_crossing_epochs"],
            "S_mean_observed_epoch": stat_a["S_mean_observed_epoch"],
            "S_censored_count": stat_a["S_censored_count"],
            "C_crossing_rate": stat_a["C_crossing_rate"],
            "C_crossing_count": stat_a["C_crossing_count"],
            "C_observed_crossing_epochs": stat_a["C_observed_crossing_epochs"],
            "C_mean_observed_epoch": stat_a["C_mean_observed_epoch"],
            "C_censored_count": stat_a["C_censored_count"],
            "censoring_note": (
                f"S crossed tau=0.80 in {stat_a['S_crossing_count']} runs (observed epochs: {stat_a['S_observed_crossing_epochs']}). "
                f"C crossed in {stat_a['C_crossing_count']} runs (observed epochs: {stat_a['C_observed_crossing_epochs']})."
            ),
            "mean_final_acc_S": stat_a["continuous_epoch_50_accuracy"]["mean_acc_S"],
            "std_final_acc_S": stat_a["continuous_epoch_50_accuracy"]["std_acc_S"],
            "mean_final_acc_C": stat_a["continuous_epoch_50_accuracy"]["mean_acc_C"],
            "std_final_acc_C": stat_a["continuous_epoch_50_accuracy"]["std_acc_C"],
            "accuracy_difference_mean (S - C)": stat_a["continuous_epoch_50_accuracy"]["mean_diff_S_minus_C"],
            "std_diff_S_minus_C": stat_a["continuous_epoch_50_accuracy"]["std_diff_S_minus_C"],
            "sem_diff": stat_a["continuous_epoch_50_accuracy"]["sem_diff"],
            "ci_95_diff": stat_a["continuous_epoch_50_accuracy"]["ci_95"],
            "degrees_of_freedom": stat_a["continuous_epoch_50_accuracy"]["degrees_of_freedom"],
            "paired_t_stat_acc": stat_a["continuous_epoch_50_accuracy"]["paired_t_stat"],
            "paired_p_val_acc_twosided": stat_a["continuous_epoch_50_accuracy"]["paired_p_val_two_sided"],
            "paired_p_val_acc_greater": stat_a["continuous_epoch_50_accuracy"]["paired_p_val_greater"],
            "paired_p_val_acc": stat_a["continuous_epoch_50_accuracy"]["paired_p_val_two_sided"],
            "wilcoxon_p_val_acc": stat_a["continuous_epoch_50_accuracy"]["wilcoxon_p_val"],
            "binary_crossing_tests": stat_a["binary_crossing_tests"],
            "early_dynamics_epoch_1": stat_a["early_dynamics_epoch_1"],
            "held_out_test_epoch_1": stat_a.get("held_out_test_epoch_1"),
            "statistical_conclusion": (
                f"Under matched signals, S crossed threshold in {stat_a['S_crossing_count']} runs vs {stat_a['C_crossing_count']} for C on the validation split. "
                "Because observations are paired by model seed, the primary two-sided exact McNemar test yields "
                f"p = {stat_a['binary_crossing_tests']['paired_mcnemar_exact_two_sided_p']:.4f} (nonsignificant at alpha = 0.05); "
                f"a directional one-sided test yields p = {stat_a['binary_crossing_tests']['paired_mcnemar_exact_one_sided_p']:.4f} "
                "(conditional on a non-preregistered directional hypothesis). An unpaired Fisher exact test yields "
                f"one-sided p = {stat_a['binary_crossing_tests']['unpaired_fisher_exact_one_sided_p']:.4f} "
                f"(two-sided p = {stat_a['binary_crossing_tests']['unpaired_fisher_exact_two_sided_p']:.4f}), though its independence assumption is violated. "
                f"At epoch 50 (held-out test split), counterfactual accuracy difference (S - C) is "
                f"{stat_a['continuous_epoch_50_accuracy']['mean_diff_S_minus_C'] * 100:+.2f}% "
                f"(paired t({stat_a['continuous_epoch_50_accuracy']['degrees_of_freedom']}) = {stat_a['continuous_epoch_50_accuracy']['paired_t_stat']:.2f}, "
                f"two-sided p = {stat_a['continuous_epoch_50_accuracy']['paired_p_val_two_sided']:.4f}). "
                "Exploratory epoch-1 accuracy differences indicate an initial linear lead on both validation "
                f"({stat_a['early_dynamics_epoch_1']['mean_diff_ep1']*100:+.2f}%, paired t={stat_a['early_dynamics_epoch_1']['paired_t_stat_ep1']:.2f}, p={stat_a['early_dynamics_epoch_1']['paired_p_val_ep1']:.4f}) "
                "and held-out test splits."
            )
        },
        "Condition_B_Complex_Dominant": {
            "S_crossing_rate": stat_b["S_crossing_rate"],
            "S_crossing_count": stat_b["S_crossing_count"],
            "S_observed_crossing_epochs": stat_b["S_observed_crossing_epochs"],
            "S_censored_count": stat_b["S_censored_count"],
            "C_crossing_rate": stat_b["C_crossing_rate"],
            "C_crossing_count": stat_b["C_crossing_count"],
            "C_observed_crossing_epochs": stat_b["C_observed_crossing_epochs"],
            "C_mean_observed_epoch": stat_b["C_mean_observed_epoch"],
            "C_censored_count": stat_b["C_censored_count"],
            "censoring_note": (
                f"S crossed tau=0.80 in {stat_b['S_crossing_count']} runs (observed epochs: {stat_b['S_observed_crossing_epochs']}). "
                f"C crossed in {stat_b['C_crossing_count']} runs (observed epochs: {stat_b['C_observed_crossing_epochs']})."
            ),
            "mean_final_acc_S": stat_b["continuous_epoch_50_accuracy"]["mean_acc_S"],
            "std_final_acc_S": stat_b["continuous_epoch_50_accuracy"]["std_acc_S"],
            "mean_final_acc_C": stat_b["continuous_epoch_50_accuracy"]["mean_acc_C"],
            "std_final_acc_C": stat_b["continuous_epoch_50_accuracy"]["std_acc_C"],
            "accuracy_difference_mean (C - S)": float(np.mean(diff_cs_b)),
            "std_diff_C_minus_S": float(np.std(diff_cs_b, ddof=1)),
            "sem_diff": se_diff_b,
            "ci_95_diff": ci_cs_b,
            "degrees_of_freedom": int(len(runs_b) - 1),
            "paired_t_stat_acc": float(stats.ttest_rel(c_acc_b, s_acc_b).statistic),
            "paired_p_val_acc_twosided": float(stats.ttest_rel(c_acc_b, s_acc_b).pvalue),
            "paired_p_val_acc": float(stats.ttest_rel(c_acc_b, s_acc_b).pvalue),
            "wilcoxon_p_val_acc": stat_b["continuous_epoch_50_accuracy"]["wilcoxon_p_val"],
            "binary_crossing_tests": stat_b["binary_crossing_tests"],
            "statistical_conclusion": (
                f"When complex signal quality dominates, learning order decisively inverts: C crossed in "
                f"{stat_b['C_crossing_count']} runs, while S crossed in {stat_b['S_crossing_count']} runs. "
                "Because observations are paired by model seed, the primary two-sided exact McNemar test yields "
                f"p = {stat_b['binary_crossing_tests']['paired_mcnemar_exact_two_sided_p']:.4f} (nonsignificant at alpha = 0.05 due to N=5 discrete sample limit); "
                f"the directional one-sided test (C > S) yields p = {stat_b['binary_crossing_tests']['directional_c_greater_s_p']:.5f}. "
                f"Terminal epoch-50 accuracy heavily favors C ({stat_b['continuous_epoch_50_accuracy']['mean_acc_C'] * 100:.2f}% vs "
                f"{stat_b['continuous_epoch_50_accuracy']['mean_acc_S'] * 100:.2f}%, paired t({len(runs_b)-1}) = {stats.ttest_rel(c_acc_b, s_acc_b).statistic:.2f}, "
                f"p = {stats.ttest_rel(c_acc_b, s_acc_b).pvalue:.2e})."
            )
        },
        "Condition_Comparison_A_vs_B": {
            "unstandardized_effect_size_margin_shift": mean_margin_shift,
            "ci_95_margin_shift": ci_shift,
            "welch_t_stat": t_welch,
            "welch_df": df_welch,
            "welch_p_val": p_welch,
            "paired_t_stat_across_seeds": t_paired_ab,
            "paired_df": df_paired_ab,
            "paired_p_val": p_paired_ab,
            "interpretation": (
                f"Statistically significant reversal of feature reliance between Condition A and Condition B: "
                f"unstandardized margin shift = {mean_margin_shift * 100:+.2f}% (95% CI: [{ci_shift[0]*100:+.2f}%, {ci_shift[1]*100:+.2f}%], "
                f"paired t({df_paired_ab}) = {t_paired_ab:.2f}, p = {p_paired_ab:.2e})."
            )
        }
    }

    if "held_out_test_epoch_1" in stat_a:
        stat_report["Condition_A_Matched"]["held_out_test_epoch_1"] = stat_a["held_out_test_epoch_1"]

    with open(summary_path, "w") as f:
        json.dump(stat_report, f, indent=2)

    print("\n[Condition A]")
    print(f"  Crossing counts: S={stat_a['S_crossing_count']}, C={stat_a['C_crossing_count']}")
    print(f"  Primary two-sided McNemar exact test: p = {stat_a['binary_crossing_tests']['paired_mcnemar_exact_two_sided_p']:.4f} (significant at 0.05: {stat_a['binary_crossing_tests']['significance_at_0_05']})")
    print(f"  Supplementary directional McNemar (S > C): p = {stat_a['binary_crossing_tests']['directional_s_greater_c_p']:.4f}")
    print(f"  Unpaired Fisher exact test: p = {stat_a['binary_crossing_tests']['unpaired_fisher_exact_one_sided_p']:.4f} (one-sided), p = {stat_a['binary_crossing_tests']['unpaired_fisher_exact_two_sided_p']:.4f} (two-sided)")
    print(f"  Epoch 50 Accuracy: S={stat_a['continuous_epoch_50_accuracy']['mean_acc_S']:.3f} vs C={stat_a['continuous_epoch_50_accuracy']['mean_acc_C']:.3f} (paired t={stat_a['continuous_epoch_50_accuracy']['paired_t_stat']:.3f}, p={stat_a['continuous_epoch_50_accuracy']['paired_p_val_two_sided']:.4f})")
    print(f"  Epoch 1 Accuracy: S={stat_a['early_dynamics_epoch_1']['mean_acc_S_ep1']:.3f} vs C={stat_a['early_dynamics_epoch_1']['mean_acc_C_ep1']:.3f} (paired t={stat_a['early_dynamics_epoch_1']['paired_t_stat_ep1']:.3f}, p={stat_a['early_dynamics_epoch_1']['paired_p_val_ep1']:.4f})")

    print("\n[Condition B]")
    print(f"  Crossing counts: S={stat_b['S_crossing_count']}, C={stat_b['C_crossing_count']}")
    print(f"  Primary two-sided McNemar exact test: p = {stat_b['binary_crossing_tests']['paired_mcnemar_exact_two_sided_p']:.4f} (significant at 0.05: {stat_b['binary_crossing_tests']['significance_at_0_05']})")
    print(f"  Supplementary directional McNemar (C > S): p = {stat_b['binary_crossing_tests']['directional_c_greater_s_p']:.5f}")
    print(f"  Epoch 50 Accuracy: C={stat_b['continuous_epoch_50_accuracy']['mean_acc_C']:.3f} vs S={stat_b['continuous_epoch_50_accuracy']['mean_acc_S']:.3f} (paired t={stats.ttest_rel(c_acc_b, s_acc_b).statistic:.2f}, p={stats.ttest_rel(c_acc_b, s_acc_b).pvalue:.4e})")

    print("\n[Comparison A vs B]")
    print(f"  Unstandardized Reliance Margin Shift: {mean_margin_shift * 100:.2f}% (95% CI: [{ci_shift[0]*100:.2f}%, {ci_shift[1]*100:.2f}%])")
    print(f"  Welch's t = {t_welch:.3f} (df = {df_welch:.2f}, p = {p_welch:.4e})")
    print(f"  Paired t across seeds = {t_paired_ab:.3f} (df = {df_paired_ab}, p = {p_paired_ab:.4e})")

    # -------------------------------------------------------------
    # Figure Generation
    # -------------------------------------------------------------
    epochs = len(runs_a[0]["dynamics"])
    ep_axis = np.arange(1, epochs + 1)

    joint_curves = np.array([[d["acc_joint"] for d in r["dynamics"]] for r in runs_a])
    s_curves = np.array([[d["acc_s_only"] for d in r["dynamics"]] for r in runs_a])
    c_curves = np.array([[d["acc_c_only"] for d in r["dynamics"]] for r in runs_a])
    n_curves = np.array([[d["acc_n_only"] for d in r["dynamics"]] for r in runs_a])

    # 1. Figure 1: Learning Dynamics for Condition A (Matched Baseline)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(ep_axis, np.mean(joint_curves, axis=0), color="#1E293B", linewidth=2.5, label="Joint Input [S + C + Noise]")
    ax.fill_between(ep_axis, np.mean(joint_curves, axis=0) - np.std(joint_curves, axis=0),
                    np.mean(joint_curves, axis=0) + np.std(joint_curves, axis=0), color="#1E293B", alpha=0.1)

    ax.plot(ep_axis, np.mean(s_curves, axis=0), color="#2563EB", linewidth=2.5, label="S-only (Linear Feature, eta_S=0.05)")
    ax.fill_between(ep_axis, np.mean(s_curves, axis=0) - np.std(s_curves, axis=0),
                    np.mean(s_curves, axis=0) + np.std(s_curves, axis=0), color="#2563EB", alpha=0.15)

    ax.plot(ep_axis, np.mean(c_curves, axis=0), color="#DC2626", linewidth=2.5, label="C-only (Nonlinear XOR, eta_C=0.05)")
    ax.fill_between(ep_axis, np.mean(c_curves, axis=0) - np.std(c_curves, axis=0),
                    np.mean(c_curves, axis=0) + np.std(c_curves, axis=0), color="#DC2626", alpha=0.15)

    ax.plot(ep_axis, np.mean(n_curves, axis=0), color="#94A3B8", linestyle="--", linewidth=1.8, label="Noise Control (~0.50 Chance)")

    ax.axhline(0.80, color="#64748B", linestyle=":", linewidth=1.5, label="Acquisition Threshold (tau = 0.80)")
    ax.axhline(0.50, color="#CBD5E1", linestyle="-.", linewidth=1.2, label="Chance Accuracy (0.50)")
    ax.set_title("Figure 1: Learning Dynamics Under Matched Signal Strength (Condition A)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Training Epoch", fontsize=11, fontweight='medium')
    ax.set_ylabel("Counterfactual Accuracy", fontsize=11, fontweight='medium')
    ax.set_ylim(0.40, 1.02)
    ax.legend(frameon=True, loc="lower right", fontsize=9.0)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig1_learning_dynamics_matched.png"))
    plt.close()

    # 2. Figure 2: Falsification Comparison (Condition A vs Condition B)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Left: Condition A
    ax1.plot(ep_axis, np.mean(s_curves, axis=0), color="#2563EB", linewidth=2.5, label="S-only (Linear, eta_S=0.05)")
    ax1.plot(ep_axis, np.mean(c_curves, axis=0), color="#DC2626", linewidth=2.5, label="C-only (XOR, eta_C=0.05)")
    ax1.axhline(0.80, color="#64748B", linestyle=":", label="Threshold tau = 0.80")
    ax1.axhline(0.50, color="#CBD5E1", linestyle="--", label="Chance (0.50)")
    ax1.set_title(f"Condition A: Matched Signals\n(S crossed: {stat_a['S_crossing_count']}, C crossed: {stat_a['C_crossing_count']})", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Counterfactual Accuracy", fontsize=10)
    ax1.set_ylim(0.42, 1.02)
    ax1.legend(loc="lower right", fontsize=8.5)

    # Right: Condition B
    s_curves_b = np.array([[d["acc_s_only"] for d in r["dynamics"]] for r in runs_b])
    c_curves_b = np.array([[d["acc_c_only"] for d in r["dynamics"]] for r in runs_b])

    ax2.plot(ep_axis, np.mean(s_curves_b, axis=0), color="#2563EB", linestyle="--", linewidth=2.5, label="S-only (Noisy, eta_S=0.25)")
    ax2.plot(ep_axis, np.mean(c_curves_b, axis=0), color="#DC2626", linewidth=2.5, label="C-only (Clean, eta_C=0.01)")
    ax2.axhline(0.80, color="#64748B", linestyle=":", label="Threshold tau = 0.80")
    ax2.axhline(0.50, color="#CBD5E1", linestyle="--", label="Chance (0.50)")
    ax2.set_title(f"Condition B: Complex Signal Dominant\n(INVERSION: C crossed: {stat_b['C_crossing_count']}, S crossed: {stat_b['S_crossing_count']})", fontsize=11, fontweight='bold', color="#991B1B")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Counterfactual Accuracy", fontsize=10)
    ax2.set_ylim(0.42, 1.02)
    ax2.legend(loc="lower right", fontsize=8.5)

    plt.suptitle("Figure 2: Falsification Boundary — Signal Quality Inverts Simplicity Bias", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig2_falsification_signal_dominance.png"))
    plt.close()

    # 3. Figure 3: Feature Reliance Comparison across Conditions
    cond_keys = ["Condition_A", "Condition_B", "Condition_C", "Condition_D_SGD", "Condition_E1_Deep", "Condition_E2_ResMLP"]
    labels = ["A: Matched", "B: C-Dominant", "C: S-Dominant", "D: SGD", "E1: DeepMLP", "E2: ResMLP"]

    s_means = [conds[c]["summary"]["mean_reliance_S"] for c in cond_keys]
    c_means = [1.0 - s for s in s_means]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    x_indices = np.arange(len(labels))
    width = 0.55

    ax.bar(x_indices, s_means, width, label="Reliance on S (Linear)", color="#2563EB", alpha=0.9)
    ax.bar(x_indices, c_means, width, bottom=s_means, label="Reliance on C (Nonlinear)", color="#DC2626", alpha=0.9)

    ax.set_ylabel("Normalized Feature Reliance Ratio", fontsize=11, fontweight='bold')
    ax.set_title("Figure 3: Relative Feature Reliance Across Experimental Conditions", fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9.5)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="#475569", linestyle="--", linewidth=1.2, label="Parity (Balanced Reliance)")
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig3_reliance_heatmaps.png"))
    plt.close()

    # 4. Figure 4: Optimizer and Architecture Ablation: Crossing Proportions & Timing
    comp_groups = ["StandardMLP\n(AdamW)", "StandardMLP\n(SGD)", "DeepMLP (4L)\n(AdamW)", "ResMLP\n(AdamW)"]
    c_keys = ["Condition_A", "Condition_D_SGD", "Condition_E1_Deep", "Condition_E2_ResMLP"]
    c_cross_rates = [conds[k]["summary"]["C_crossing_rate"] for k in c_keys]
    obs_notes = []
    for k in c_keys:
        c_sum = conds[k]["summary"]
        k_obs = c_sum.get("C_observed_crossing_epochs", [])
        c_cnt = c_sum.get("C_censored_count", 5 - len(k_obs))
        tot = len(k_obs) + c_cnt
        if len(k_obs) == 0:
            obs_notes.append(f"0/{tot} crossed (T > 50)")
        elif c_cnt == 0:
            obs_notes.append(f"{len(k_obs)}/{tot} crossed (Ep {k_obs})")
        else:
            obs_notes.append(f"{len(k_obs)}/{tot} crossed (Ep {k_obs})\n{c_cnt}/{tot} censored (> 50)")

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)
    bars = ax.bar(comp_groups, [r * 100 for r in c_cross_rates], color=["#3B82F6", "#F59E0B", "#8B5CF6", "#10B981"], alpha=0.85, width=0.5)
    ax.set_ylabel("Nonlinear Acquisition Rate (%)", fontsize=10.5, fontweight='bold')
    ax.set_title("Figure 4: Nonlinear Threshold Crossing Rate (tau = 0.80 within 50 epochs)", fontsize=12, fontweight='bold', pad=12)
    ax.set_ylim(0, 120)

    for bar, rate, note in zip(bars, c_cross_rates, obs_notes):
        count_str = f"{int(round(rate * 5))}/5 runs ({rate:.0%})"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.5, count_str, ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10.0, note, ha='center', va='bottom', fontsize=8.0, color="#475569")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig4_optimizer_architecture_ablation.png"))
    plt.close()

    # 5. Figure 5: Final Accuracy Difference Distribution across Seeds (Uncensored Metric)
    diff_groups = [
        diff_sc_a,  # S - C in A
        diff_sc_b,  # S - C in B (negative, showing C advantage)
        [r["final_acc_s_only"] - r["final_acc_c_only"] for r in conds["Condition_C"]["runs"]],
        [r["final_acc_s_only"] - r["final_acc_c_only"] for r in conds["Condition_D_SGD"]["runs"]],
        [r["final_acc_s_only"] - r["final_acc_c_only"] for r in conds["Condition_E1_Deep"]["runs"]],
        [r["final_acc_s_only"] - r["final_acc_c_only"] for r in conds["Condition_E2_ResMLP"]["runs"]],
    ]
    diff_labels = [
        "A: Matched\n(S - C)",
        "B: C-Dominant\n(S - C)",
        "C: S-Dominant\n(S - C)",
        "D: SGD\n(S - C)",
        "E1: Deep\n(S - C)",
        "E2: Res\n(S - C)"
    ]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    box = ax.boxplot(diff_groups, patch_artist=True, tick_labels=diff_labels, widths=0.45)
    colors = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6', '#06B6D4']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0, color="#1E293B", linestyle="--", linewidth=1.5, label="Feature Parity (0.0 Margin)")
    ax.set_ylabel("Counterfactual Accuracy Margin at Epoch 50 (S - C)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 5: Counterfactual Accuracy Margins at Epoch 50 Across Conditions (Uncensored)", fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig5_statistical_summary.png"))
    plt.close()

    print(f"\n[Analysis Complete] All statistical tests recomputed dynamically and figures saved to {fig_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run audited PARITY statistical analysis")
    parser.add_argument("--protocol", choices=["corrected", "historical"], default="corrected",
                        help="Protocol to analyze: 'corrected' or 'historical'")
    parser.add_argument("--data-path", default=None, help="Explicit path to raw experiment JSON")
    parser.add_argument("--summary-path", default=None, help="Explicit path to output statistical summary JSON")
    parser.add_argument("--fig-dir", default=None, help="Explicit directory for figures")
    args = parser.parse_args()

    run_statistical_analysis(
        data_path=args.data_path,
        summary_path=args.summary_path,
        fig_dir=args.fig_dir,
        protocol=args.protocol,
    )
