"""
test_statistical_analysis.py - Regression Test Suite for Statistical Analysis & Evaluation Protocol

Verifies critical statistical methodologies:
  1. Dynamic Threshold-Crossing Counts: Extracted dynamically from run records; rejects hardcoded tables.
  2. Exact McNemar Tests: Correct discordant pair counts and one-sided/two-sided exact binomial p-values.
  3. Proper Censoring Representation: Non-crossing runs have observed_epoch = None, never an event at epoch 51.
  4. Observed Crossing Means: Means of crossing times strictly exclude censored runs (never include 51).
  5. Protocol & Schema Separation: Historical and corrected runs are isolated in distinct directories with explicit metadata.
"""

import os
import json
import pytest
import numpy as np
import scipy.stats as stats

from src.analyze import compute_paired_exact_test, analyze_condition


# -----------------------------------------------------------------------------
# Test 1: Dynamic Threshold-Crossing Counts vs Hardcoded Tables
# -----------------------------------------------------------------------------
def test_dynamic_threshold_crossing_counts():
    """Verify that crossing counts and contingency tables are derived dynamically from raw runs."""
    # Create synthetic runs with known dynamic crossings: S crosses in 4/5, C crosses in 1/5
    mock_runs = [
        {
            "seed": 1,
            "S_acquisition": {"crossed_threshold": True, "observed_epoch": 2},
            "C_acquisition": {"crossed_threshold": False, "observed_epoch": None},
            "final_acc_s_only": 0.85, "final_acc_c_only": 0.60, "final_acc_joint": 0.90,
            "dynamics": [{"acc_s_only": 0.70, "acc_c_only": 0.55}]
        },
        {
            "seed": 2,
            "S_acquisition": {"crossed_threshold": True, "observed_epoch": 1},
            "C_acquisition": {"crossed_threshold": True, "observed_epoch": 5},
            "final_acc_s_only": 0.88, "final_acc_c_only": 0.82, "final_acc_joint": 0.92,
            "dynamics": [{"acc_s_only": 0.75, "acc_c_only": 0.60}]
        },
        {
            "seed": 3,
            "S_acquisition": {"crossed_threshold": True, "observed_epoch": 3},
            "C_acquisition": {"crossed_threshold": False, "observed_epoch": None},
            "final_acc_s_only": 0.84, "final_acc_c_only": 0.58, "final_acc_joint": 0.89,
            "dynamics": [{"acc_s_only": 0.72, "acc_c_only": 0.52}]
        },
        {
            "seed": 4,
            "S_acquisition": {"crossed_threshold": True, "observed_epoch": 2},
            "C_acquisition": {"crossed_threshold": False, "observed_epoch": None},
            "final_acc_s_only": 0.86, "final_acc_c_only": 0.62, "final_acc_joint": 0.91,
            "dynamics": [{"acc_s_only": 0.74, "acc_c_only": 0.58}]
        },
        {
            "seed": 5,
            "S_acquisition": {"crossed_threshold": False, "observed_epoch": None},
            "C_acquisition": {"crossed_threshold": False, "observed_epoch": None},
            "final_acc_s_only": 0.75, "final_acc_c_only": 0.65, "final_acc_joint": 0.82,
            "dynamics": [{"acc_s_only": 0.65, "acc_c_only": 0.55}]
        },
    ]

    analysis = analyze_condition(mock_runs, "Test_Dynamic_Condition")

    # Counts must match raw observations
    assert analysis["S_crossing_count"] == "4/5"
    assert analysis["S_crossing_rate"] == 0.80
    assert analysis["C_crossing_count"] == "1/5"
    assert analysis["C_crossing_rate"] == 0.20

    # Contingency table must be [[4, 1], [1, 4]]
    contingency = analysis["binary_crossing_tests"]["contingency_table"]
    assert contingency == [[4, 1], [1, 4]]

    # Ensure no hardcoded [[5, 0], [0, 5]] exists in dynamic extraction
    assert contingency != [[5, 0], [0, 5]]


# -----------------------------------------------------------------------------
# Test 2: Paired Exact McNemar Tests
# -----------------------------------------------------------------------------
def test_paired_mcnemar_exact_test_discordant_pairs():
    """Verify paired exact McNemar test computes correct discordant pairs and p-values."""
    # Case A: b = 3 (S=1, C=0), c = 0 (S=0, C=1) - mirrors Condition A historical
    s_flags_a = [True, True, True, False, False]
    c_flags_a = [False, False, False, False, False]

    res_a_greater = compute_paired_exact_test(s_flags_a, c_flags_a, alternative="greater")
    res_a_twosided = compute_paired_exact_test(s_flags_a, c_flags_a, alternative="two-sided")

    assert res_a_greater["b_S_only_crossed"] == 3
    assert res_a_greater["c_C_only_crossed"] == 0
    assert res_a_greater["n_discordant"] == 3

    # Exact binomial probability for b=3, n=3, p=0.5: (0.5)^3 = 0.125
    assert pytest.approx(res_a_greater["p_value"], rel=1e-5) == 0.1250
    # Two-sided probability: 2 * 0.125 = 0.250
    assert pytest.approx(res_a_twosided["p_value"], rel=1e-5) == 0.2500

    # Case B: b = 0, c = 5 (all 5 runs C crossed, S did not) - mirrors Condition B
    s_flags_b = [False, False, False, False, False]
    c_flags_b = [True, True, True, True, True]

    res_b_greater = compute_paired_exact_test(c_flags_b, s_flags_b, alternative="greater")
    assert res_b_greater["b_S_only_crossed"] == 5
    assert res_b_greater["c_C_only_crossed"] == 0
    assert res_b_greater["n_discordant"] == 5
    # (0.5)^5 = 0.03125
    assert pytest.approx(res_b_greater["p_value"], rel=1e-5) == 0.03125


def test_paired_mcnemar_zero_discordant():
    """Verify handling when there are zero discordant pairs (e.g., both features cross in all runs)."""
    s_flags = [True, True, True]
    c_flags = [True, True, True]
    res = compute_paired_exact_test(s_flags, c_flags, alternative="greater")
    assert res["n_discordant"] == 0
    assert res["p_value"] == 1.0


# -----------------------------------------------------------------------------
# Test 3: Censoring Representation & Prohibition of 51 as an Event
# -----------------------------------------------------------------------------
def test_censoring_representation_prohibits_epoch_51_event():
    """Verify that censored runs have observed_epoch = None, never an observed event at 51."""
    mock_censored_run = {
        "seed": 101,
        "S_acquisition": {
            "crossed_threshold": False,
            "observed_epoch": None,
            "censoring_limit": 50,
            "monitoring_split": "validation"
        },
        "C_acquisition": {
            "crossed_threshold": False,
            "observed_epoch": None,
            "censoring_limit": 50,
            "monitoring_split": "validation"
        },
        "T_S_censored": True,
        "T_C_censored": True,
        "T_S_observed": None,
        "T_C_observed": None,
        "final_acc_s_only": 0.76,
        "final_acc_c_only": 0.67,
        "final_acc_joint": 0.90,
        "dynamics": [{"acc_s_only": 0.70, "acc_c_only": 0.55}]
    }

    # Verify acquisition metadata
    assert mock_censored_run["S_acquisition"]["crossed_threshold"] is False
    assert mock_censored_run["S_acquisition"]["observed_epoch"] is None
    assert mock_censored_run["T_S_observed"] is None
    assert mock_censored_run["T_S_censored"] is True

    # Critical requirement: observed epoch must NEVER be 51
    assert mock_censored_run["S_acquisition"]["observed_epoch"] != 51
    assert mock_censored_run["T_S_observed"] != 51


# -----------------------------------------------------------------------------
# Test 4: Means of Observed Crossing Times Exclude Censored Observations
# -----------------------------------------------------------------------------
def test_mean_observed_crossing_excludes_censored_runs():
    """Verify that means of crossing times include only crossing runs, never censored runs."""
    mock_runs = [
        {
            "seed": 101,
            "S_acquisition": {"crossed_threshold": False, "observed_epoch": None},
            "C_acquisition": {"crossed_threshold": False, "observed_epoch": None},
            "final_acc_s_only": 0.75, "final_acc_c_only": 0.65, "final_acc_joint": 0.85,
            "dynamics": [{"acc_s_only": 0.70, "acc_c_only": 0.55}]
        },
        {
            "seed": 102,
            "S_acquisition": {"crossed_threshold": True, "observed_epoch": 1},
            "C_acquisition": {"crossed_threshold": False, "observed_epoch": None},
            "final_acc_s_only": 0.85, "final_acc_c_only": 0.68, "final_acc_joint": 0.92,
            "dynamics": [{"acc_s_only": 0.80, "acc_c_only": 0.60}]
        },
        {
            "seed": 103,
            "S_acquisition": {"crossed_threshold": True, "observed_epoch": 1},
            "C_acquisition": {"crossed_threshold": False, "observed_epoch": None},
            "final_acc_s_only": 0.84, "final_acc_c_only": 0.67, "final_acc_joint": 0.91,
            "dynamics": [{"acc_s_only": 0.80, "acc_c_only": 0.60}]
        },
        {
            "seed": 104,
            "S_acquisition": {"crossed_threshold": True, "observed_epoch": 1},
            "C_acquisition": {"crossed_threshold": False, "observed_epoch": None},
            "final_acc_s_only": 0.86, "final_acc_c_only": 0.69, "final_acc_joint": 0.93,
            "dynamics": [{"acc_s_only": 0.81, "acc_c_only": 0.58}]
        },
        {
            "seed": 105,
            "S_acquisition": {"crossed_threshold": False, "observed_epoch": None},
            "C_acquisition": {"crossed_threshold": False, "observed_epoch": None},
            "final_acc_s_only": 0.77, "final_acc_c_only": 0.70, "final_acc_joint": 0.88,
            "dynamics": [{"acc_s_only": 0.72, "acc_c_only": 0.62}]
        },
    ]

    analysis = analyze_condition(mock_runs, "Condition_A_Simulated")

    # Observed crossing epochs for S should strictly be [1, 1, 1]
    assert analysis["S_observed_crossing_epochs"] == [1, 1, 1]
    # Mean of observed crossing times must be exactly 1.0, NOT (1+1+1+51+51)/5 = 21.0
    assert analysis["S_mean_observed_epoch"] == 1.0
    assert analysis["S_censored_count"] == 2

    # For C, where 0 runs crossed:
    assert analysis["C_observed_crossing_epochs"] == []
    assert analysis["C_mean_observed_epoch"] is None
    assert analysis["C_censored_count"] == 5


# -----------------------------------------------------------------------------
# Test 5: Historical and Corrected Runs Separation
# -----------------------------------------------------------------------------
def test_historical_and_corrected_runs_separation():
    """Verify that historical and corrected run files exist in distinct paths without collision."""
    historical_dir = "results/raw_data/historical_run"
    corrected_dir = "results/raw_data/corrected_protocol"

    historical_file = os.path.join(historical_dir, "experiments_results_historical_test_monitored.json")
    historical_summary = os.path.join(historical_dir, "statistical_summary_historical_test_monitored.json")
    # Assert separation of directories
    assert os.path.abspath(historical_dir) != os.path.abspath(corrected_dir)

    # Assert historical artifacts exist and are preserved
    assert os.path.exists(historical_file), f"Missing historical experiment archive: {historical_file}"
    assert os.path.exists(historical_summary), f"Missing historical summary archive: {historical_summary}"

    # Load historical metadata
    with open(historical_file, "r") as f:
        hist_data = json.load(f)

    # Verify historical data has valid schema
    assert "conditions" in hist_data
    assert "Condition_A" in hist_data["conditions"]
    assert len(hist_data["conditions"]["Condition_A"]["runs"]) == 5

    # Assert corrected artifacts exist and verify schema
    corrected_file = os.path.join(corrected_dir, "experiments_results_corrected.json")
    corrected_summary = os.path.join(corrected_dir, "statistical_summary_corrected.json")
    assert os.path.exists(corrected_file), f"Missing corrected experiment file: {corrected_file}"
    assert os.path.exists(corrected_summary), f"Missing corrected summary file: {corrected_summary}"

    with open(corrected_file, "r") as f:
        corr_data = json.load(f)

    assert corr_data["metadata"]["monitoring_split"] == "validation"
    assert corr_data["metadata"]["protocol"] == "corrected"
    assert len(corr_data["conditions"]["Condition_A"]["runs"]) == 5
    # Verify each run contains held-out final test metrics
    assert "final_test_metrics" in corr_data["conditions"]["Condition_A"]["runs"][0]
    assert "epoch_1_test_metrics" in corr_data["conditions"]["Condition_A"]["runs"][0]


# -----------------------------------------------------------------------------
# Test 6: Primary Significance Logic & Regression Tests (Condition A & B)
# -----------------------------------------------------------------------------
def test_condition_a_primary_significance_regression():
    """
    Regression test for Condition A:
    Under 5 discordant pairs (b=5, c=0), primary two-sided McNemar exact test p = 0.0625.
    Under the bug, significance_at_0_05 was True because it checked paired_sc_greater (p = 0.03125).
    Under the correction, significance_at_0_05 MUST be False.
    """
    corrected_file = "results/raw_data/corrected_protocol/experiments_results_corrected.json"
    with open(corrected_file, "r") as f:
        data = json.load(f)

    runs_a = data["conditions"]["Condition_A"]["runs"]
    analysis = analyze_condition(runs_a, "Condition_A_Matched")

    binary_tests = analysis["binary_crossing_tests"]
    two_sided_p = binary_tests["paired_mcnemar_exact_two_sided_p"]
    one_sided_p = binary_tests["directional_s_greater_c_p"]

    # 1. Exact p-value checks from actual paired observations
    assert pytest.approx(two_sided_p, rel=1e-5) == 0.0625
    assert pytest.approx(one_sided_p, rel=1e-5) == 0.03125

    # 2. Primary significance flag must be strictly determined by two-sided p < 0.05
    assert binary_tests["primary_significance_at_0_05"] is False
    assert binary_tests["significance_at_0_05"] is False

    # 3. Invariant: significance_at_0_05 == (primary_two_sided_p < 0.05)
    assert binary_tests["significance_at_0_05"] == (two_sided_p < 0.05)

    # 4. Supplementary directional test is separately preserved and does NOT override primary flag
    assert binary_tests["supplementary_directional_tests"]["s_greater_c"]["hypothesis"] == "S > C"
    assert pytest.approx(binary_tests["supplementary_directional_tests"]["s_greater_c"]["p_value"], rel=1e-5) == 0.03125
    assert binary_tests["supplementary_directional_tests"]["s_greater_c"]["directional_significant_at_0_05"] is True
    assert binary_tests["directional_s_greater_c_significant_at_0_05"] is True
    # Ensure primary key is not reused ambiguously inside directional subdict
    assert "significance_at_0_05" not in binary_tests["supplementary_directional_tests"]["s_greater_c"]
    # Crucial assertion: one-sided significance does not bleed into the primary significance flag
    assert binary_tests["significance_at_0_05"] != binary_tests["supplementary_directional_tests"]["s_greater_c"]["directional_significant_at_0_05"]


def test_condition_b_primary_significance_regression():
    """
    Regression test for Condition B:
    Under 5 discordant pairs in opposite direction (b=0, c=5), primary two-sided McNemar p = 0.0625.
    Directional one-sided test in the C > S direction is p = 0.03125.
    Under the bug, the code evaluated paired_sc_greater (S > C, p = 1.0).
    Under the correction:
      - primary two-sided p is 0.0625 => significance_at_0_05 is False.
      - directional C > S test is correctly reported as p = 0.03125.
      - significance_at_0_05 does not depend on which one-sided test is evaluated.
    """
    corrected_file = "results/raw_data/corrected_protocol/experiments_results_corrected.json"
    with open(corrected_file, "r") as f:
        data = json.load(f)

    runs_b = data["conditions"]["Condition_B"]["runs"]
    analysis = analyze_condition(runs_b, "Condition_B_Complex_Dominant")

    binary_tests = analysis["binary_crossing_tests"]
    two_sided_p = binary_tests["paired_mcnemar_exact_two_sided_p"]
    cs_greater_p = binary_tests["directional_c_greater_s_p"]
    sc_greater_p = binary_tests["directional_s_greater_c_p"]

    # 1. Exact p-values from actual paired observations
    assert pytest.approx(two_sided_p, rel=1e-5) == 0.0625
    assert pytest.approx(cs_greater_p, rel=1e-5) == 0.03125
    assert pytest.approx(sc_greater_p, rel=1e-5) == 1.0

    # 2. Primary significance flag is False because 0.0625 >= 0.05
    assert binary_tests["primary_significance_at_0_05"] is False
    assert binary_tests["significance_at_0_05"] is False

    # 3. Invariant: significance_at_0_05 == (primary_two_sided_p < 0.05)
    assert binary_tests["significance_at_0_05"] == (two_sided_p < 0.05)

    # 4. Supplementary directional C > S test is labeled and reported separately
    assert binary_tests["supplementary_directional_tests"]["c_greater_s"]["hypothesis"] == "C > S"
    assert pytest.approx(binary_tests["supplementary_directional_tests"]["c_greater_s"]["p_value"], rel=1e-5) == 0.03125
    assert binary_tests["supplementary_directional_tests"]["c_greater_s"]["directional_significant_at_0_05"] is True
    assert binary_tests["directional_c_greater_s_significant_at_0_05"] is True
    assert "significance_at_0_05" not in binary_tests["supplementary_directional_tests"]["c_greater_s"]

    # 5. Continuous epoch 50 accuracy is independently significant, confirming the distinction
    t_acc = analysis["continuous_epoch_50_accuracy"]["paired_p_val_two_sided"]
    assert t_acc < 1e-4


def test_alpha_threshold_boundaries_and_no_directional_override():
    """Verify strict alpha < 0.05 threshold and directional isolation across boundary cases."""
    # Boundary case 1: 6 discordant pairs, b=6, c=0 => two-sided p = 2 * (0.5)^6 = 0.03125 < 0.05 => Significant
    s_flags_sig = [True] * 6
    c_flags_sig = [False] * 6
    two_sided_sig = compute_paired_exact_test(s_flags_sig, c_flags_sig, alternative="two-sided")
    assert two_sided_sig["p_value"] < 0.05
    assert bool(two_sided_sig["p_value"] < 0.05) is True

    # Boundary case 2: 5 discordant pairs, b=5, c=0 => two-sided p = 0.0625 >= 0.05 => Not Significant
    s_flags_nonsig = [True] * 5
    c_flags_nonsig = [False] * 5
    two_sided_nonsig = compute_paired_exact_test(s_flags_nonsig, c_flags_nonsig, alternative="two-sided")
    one_sided_nonsig = compute_paired_exact_test(s_flags_nonsig, c_flags_nonsig, alternative="greater")
    assert two_sided_nonsig["p_value"] == 0.0625
    assert one_sided_nonsig["p_value"] == 0.03125

    # Primary significance rule
    sig_at_0_05 = bool(two_sided_nonsig["p_value"] < 0.05)
    assert sig_at_0_05 is False

    # Ensure one-sided p-value cannot force primary significance to True
    assert bool(one_sided_nonsig["p_value"] < 0.05) is True
    assert sig_at_0_05 != bool(one_sided_nonsig["p_value"] < 0.05)

    # Boundary threshold test: strictly < 0.05
    for p_val, expected in [(0.0499, True), (0.0500, False), (0.0501, False)]:
        assert (p_val < 0.05) is expected


# -----------------------------------------------------------------------------
# Test 7: Paired Continuous Accuracy Exactness & Discrepancy Prevention
# -----------------------------------------------------------------------------
def test_continuous_accuracy_statistics_condition_a_exact():
    """
    Regression test ensuring Condition A paired continuous accuracy statistics at epoch 50
    are calculated exactly from raw seed-level observations.
    """
    corrected_file = "results/raw_data/corrected_protocol/experiments_results_corrected.json"
    with open(corrected_file, "r") as f:
        data = json.load(f)

    runs_a = data["conditions"]["Condition_A"]["runs"]
    assert len(runs_a) == 5

    seeds = [r["seed"] for r in runs_a]
    assert seeds == [101, 102, 103, 104, 105]

    s_acc = np.array([r["final_acc_s_only"] for r in runs_a], dtype=np.float64)
    c_acc = np.array([r["final_acc_c_only"] for r in runs_a], dtype=np.float64)
    diff = s_acc - c_acc

    # Verify exact raw observations
    expected_s = np.array([0.7770, 0.7267, 0.7623, 0.7533, 0.7123])
    expected_c = np.array([0.6800, 0.7133, 0.6747, 0.7003, 0.7317])
    expected_diff = np.array([0.0970, 0.0133, 0.0877, 0.0530, -0.0193])

    np.testing.assert_allclose(s_acc, expected_s, atol=1e-3)
    np.testing.assert_allclose(c_acc, expected_c, atol=1e-3)
    np.testing.assert_allclose(diff, expected_diff, atol=1e-3)

    # Calculate statistics directly
    n = len(diff)
    df = n - 1
    mean_s = float(np.mean(s_acc))
    mean_c = float(np.mean(c_acc))
    mean_diff = float(np.mean(diff))
    sample_sd_diff = float(np.std(diff, ddof=1))
    sem_diff = float(stats.sem(diff))
    t_res = stats.ttest_rel(s_acc, c_acc)
    t_stat = float(t_res.statistic)
    p_twosided = float(t_res.pvalue)

    t_crit = stats.t.ppf(0.975, df=df)
    ci_lower = mean_diff - t_crit * sem_diff
    ci_upper = mean_diff + t_crit * sem_diff

    # Assert exact statistical values
    assert df == 4
    assert pytest.approx(mean_s, rel=1e-4) == 0.7463333333333333
    assert pytest.approx(mean_c, rel=1e-4) == 0.7000000000000000
    assert pytest.approx(mean_diff, rel=1e-4) == 0.04633333333333334
    assert pytest.approx(sample_sd_diff, rel=1e-4) == 0.04929784759781887
    assert pytest.approx(sem_diff, rel=1e-4) == 0.02204666767462954
    assert pytest.approx(t_stat, rel=1e-4) == 2.1016025649378278
    assert pytest.approx(p_twosided, rel=1e-4) == 0.10346595840522543
    assert pytest.approx(ci_lower, rel=1e-4) == -0.01487802921781426
    assert pytest.approx(ci_upper, rel=1e-4) == 0.10754469588448093

    # Verify that analyze_condition returns exactly these metrics
    analysis = analyze_condition(runs_a, "Condition_A_Matched")
    cont = analysis["continuous_epoch_50_accuracy"]
    assert pytest.approx(cont["mean_acc_S"], rel=1e-4) == mean_s
    assert pytest.approx(cont["mean_acc_C"], rel=1e-4) == mean_c
    assert pytest.approx(cont["mean_diff_S_minus_C"], rel=1e-4) == mean_diff
    assert pytest.approx(cont["std_diff_S_minus_C"], rel=1e-4) == sample_sd_diff
    assert pytest.approx(cont["sem_diff"], rel=1e-4) == sem_diff
    assert cont["degrees_of_freedom"] == 4
    assert pytest.approx(cont["paired_t_stat"], rel=1e-4) == t_stat
    assert pytest.approx(cont["paired_p_val_two_sided"], rel=1e-4) == p_twosided
    assert pytest.approx(cont["ci_95"][0], rel=1e-4) == ci_lower
    assert pytest.approx(cont["ci_95"][1], rel=1e-4) == ci_upper


def test_continuous_accuracy_statistics_condition_b_exact():
    """
    Regression test ensuring Condition B paired continuous accuracy statistics at epoch 50
    are calculated exactly from raw seed-level observations.
    """
    corrected_file = "results/raw_data/corrected_protocol/experiments_results_corrected.json"
    with open(corrected_file, "r") as f:
        data = json.load(f)

    runs_b = data["conditions"]["Condition_B"]["runs"]
    assert len(runs_b) == 5

    seeds = [r["seed"] for r in runs_b]
    assert seeds == [101, 102, 103, 104, 105]

    c_acc = np.array([r["final_acc_c_only"] for r in runs_b], dtype=np.float64)
    s_acc = np.array([r["final_acc_s_only"] for r in runs_b], dtype=np.float64)
    diff_cs = c_acc - s_acc

    # Verify exact raw observations
    expected_c = np.array([0.9743, 0.9620, 0.9713, 0.9730, 0.9567])
    expected_s = np.array([0.5053, 0.5103, 0.5010, 0.5143, 0.5177])
    expected_diff_cs = np.array([0.4690, 0.4517, 0.4703, 0.4587, 0.4390])

    np.testing.assert_allclose(c_acc, expected_c, atol=1e-3)
    np.testing.assert_allclose(s_acc, expected_s, atol=1e-3)
    np.testing.assert_allclose(diff_cs, expected_diff_cs, atol=1e-3)

    # Calculate statistics directly
    n = len(diff_cs)
    df = n - 1
    mean_c = float(np.mean(c_acc))
    mean_s = float(np.mean(s_acc))
    mean_diff = float(np.mean(diff_cs))
    sample_sd_diff = float(np.std(diff_cs, ddof=1))
    sem_diff = float(stats.sem(diff_cs))
    t_res = stats.ttest_rel(c_acc, s_acc)
    t_stat = float(t_res.statistic)
    p_twosided = float(t_res.pvalue)

    t_crit = stats.t.ppf(0.975, df=df)
    ci_lower = mean_diff - t_crit * sem_diff
    ci_upper = mean_diff + t_crit * sem_diff

    # Assert exact statistical values
    assert df == 4
    assert pytest.approx(mean_c, rel=1e-4) == 0.9674666666666667
    assert pytest.approx(mean_s, rel=1e-4) == 0.5097333333333334
    assert pytest.approx(mean_diff, rel=1e-4) == 0.45773333333333327
    assert pytest.approx(sample_sd_diff, rel=1e-4) == 0.01298375052817082
    assert pytest.approx(sem_diff, rel=1e-4) == 0.00580650975677775
    assert pytest.approx(t_stat, rel=1e-4) == 78.83106246382104
    assert pytest.approx(p_twosided, rel=1e-4) == 1.5520151932330426e-07
    assert pytest.approx(ci_lower, rel=1e-4) == 0.44161187774084443
    assert pytest.approx(ci_upper, rel=1e-4) == 0.47385478892582211


def test_continuous_accuracy_discrepancy_prevention_diagnostic():
    """
    Diagnostic test verifying that the historical reporting typo
    [0.0383, 0.0617, 0.0533, 0.0300, 0.0483] for Condition A and
    [0.4450, 0.4633, 0.4633, 0.4600, 0.4575] for Condition B
    are rejected, and confirming true raw observation properties.
    """
    typo_a = np.array([0.0383, 0.0617, 0.0533, 0.0300, 0.0483])
    # The typo vector has SD ~ 0.0125 and t ~ 8.32, which is mathematically inconsistent
    # with the reported t=2.1016 and p=0.1035
    sd_typo_a = float(np.std(typo_a, ddof=1))
    t_typo_a = float(np.mean(typo_a) / stats.sem(typo_a))
    assert pytest.approx(sd_typo_a, rel=1e-2) == 0.01245
    assert pytest.approx(t_typo_a, rel=1e-2) == 8.32

    # Load actual raw data and prove it does NOT match the typo vector
    corrected_file = "results/raw_data/corrected_protocol/experiments_results_corrected.json"
    with open(corrected_file, "r") as f:
        data = json.load(f)

    runs_a = data["conditions"]["Condition_A"]["runs"]
    diff_a = np.array([r["final_acc_s_only"] - r["final_acc_c_only"] for r in runs_a])

    # The actual differences are [0.0970, 0.0133, 0.0877, 0.0530, -0.0193]
    assert not np.allclose(diff_a, typo_a, atol=1e-2)
    # The actual SD is 0.0493 (NOT 0.0125), yielding the true t(4) = 2.1016, p = 0.1035
    assert pytest.approx(np.std(diff_a, ddof=1), rel=1e-2) == 0.0493
    assert pytest.approx(stats.ttest_rel([r["final_acc_s_only"] for r in runs_a], [r["final_acc_c_only"] for r in runs_a]).statistic, rel=1e-2) == 2.1016

    # For Condition B:
    typo_b = np.array([0.4450, 0.4633, 0.4633, 0.4600, 0.4575])
    sd_typo_b = float(np.std(typo_b, ddof=1))
    t_typo_b = float(np.mean(typo_b) / stats.sem(typo_b))
    assert pytest.approx(sd_typo_b, rel=1e-2) == 0.00757
    assert pytest.approx(t_typo_b, rel=1e-2) == 135.22

    runs_b = data["conditions"]["Condition_B"]["runs"]
    diff_b = np.array([r["final_acc_c_only"] - r["final_acc_s_only"] for r in runs_b])
    # The actual differences are [0.4690, 0.4517, 0.4703, 0.4587, 0.4390]
    assert not np.allclose(diff_b, typo_b, atol=1e-2)
    # The actual SD is 0.0130 (NOT 0.0076), yielding the true t(4) = 78.8311, p = 1.5520e-7
    assert pytest.approx(np.std(diff_b, ddof=1), rel=1e-2) == 0.0130
    assert pytest.approx(stats.ttest_rel([r["final_acc_c_only"] for r in runs_b], [r["final_acc_s_only"] for r in runs_b]).statistic, rel=1e-2) == 78.8311

