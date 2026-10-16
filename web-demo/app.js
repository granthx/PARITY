/**
 * app.js - PARITY Empirical Research Dashboard & Walkthrough
 * Full Frontend-Backend Synchronization & Modern Visualization Engine
 *
 * Synchronized with authoritative backend outputs:
 * - Corrected Protocol: Validation-monitored (val_d, N=600, tau=0.80), terminal test on test_d (N=600)
 * - Historical Protocol: Test-monitored (test_d, N=600)
 */

// Active State
let currentProtocol = 'corrected'; // 'corrected' | 'historical'
let currentConditionKey = 'Condition_A';
let currentSeedFilter = 'all'; // 'all' | '101' | '102' | '103' | '104' | '105'

// In-Memory Data Stores
let DATA_CORRECTED = (typeof BUILTIN_CORRECTED_DATA !== 'undefined') ? BUILTIN_CORRECTED_DATA : null;
let STATS_CORRECTED = (typeof BUILTIN_CORRECTED_STATS !== 'undefined') ? BUILTIN_CORRECTED_STATS : null;
let DATA_HISTORICAL = (typeof BUILTIN_HISTORICAL_DATA !== 'undefined') ? BUILTIN_HISTORICAL_DATA : null;
let STATS_HISTORICAL = (typeof BUILTIN_HISTORICAL_STATS !== 'undefined') ? BUILTIN_HISTORICAL_STATS : null;

// Active Pointers
let EXPERIMENTAL_DATA = DATA_CORRECTED;
let STATISTICAL_SUMMARY = STATS_CORRECTED;

// Charts
let resultsChart = null;
let simChart = null;
let simAnimationTimer = null;
let isSimulationRunning = false;

// Condition Metadata & Academic Annotations
const CONDITION_METADATA = {
  Condition_A: {
    id: "Condition_A",
    title: "Condition A: Matched Baseline (eta_S = 0.05, eta_C = 0.05)",
    badge: "Matched Signals",
    category: "Primary Benchmark",
    config: "eta_S = 0.05, eta_C = 0.05 | StandardMLP | AdamW (lr=0.01)",
    desc: "Strictly matched Bayes optimal predictive ceilings (both 95.0%). Evaluates whether neural networks preferentially acquire simpler linear features under equal predictive capacity."
  },
  Condition_B: {
    id: "Condition_B",
    title: "Condition B: Complex Dominance (eta_S = 0.25, eta_C = 0.01) [Falsification Inversion]",
    badge: "Falsification Proof",
    category: "Signal Inversion",
    config: "eta_S = 0.25, eta_C = 0.01 | StandardMLP | AdamW (lr=0.01)",
    desc: "Complex feature C carries cleaner predictive information (99.0% Bayes ceiling vs 75.0% for S). Falsifies unconditional simplicity bias by demonstrating complete inversion of acquisition order."
  },
  Condition_C: {
    id: "Condition_C",
    title: "Condition C: Simple Dominance (eta_S = 0.01, eta_C = 0.25) [Positive Control]",
    badge: "Positive Control",
    category: "Signal Parity Control",
    config: "eta_S = 0.01, eta_C = 0.25 | StandardMLP | AdamW (lr=0.01)",
    desc: "Linear feature S carries superior predictive quality (99.0% Bayes ceiling vs 75.0% for C). The network almost exclusively prioritizes S."
  },
  Condition_D_SGD: {
    id: "Condition_D_SGD",
    title: "Condition D: Optimization Ablation with Momentum SGD (eta_S = 0.05, eta_C = 0.05)",
    badge: "SGD vs AdamW",
    category: "Optimization Dynamics",
    config: "eta_S = 0.05, eta_C = 0.05 | StandardMLP | SGD (lr=0.02, momentum=0.9)",
    desc: "Under non-adaptive SGD with momentum, gradient descent struggles to escape the flat saddle around the XOR coordinate, locking into linear feature S and suppressing complex feature acquisition."
  },
  Condition_E1_Deep: {
    id: "Condition_E1_Deep",
    title: "Condition E1: Depth Expressivity Ablation (DeepMLP 4-Layers)",
    badge: "4-Layer DeepMLP",
    category: "Architecture Ablation",
    config: "eta_S = 0.05, eta_C = 0.05 | DeepMLP (4 hidden layers, 13,186 params) | AdamW",
    desc: "Increased network depth accelerates nonlinear composition. Seed 103 crosses at epoch 8 and Seed 102 crosses at epoch 13 (mean observed 10.5 epochs; 3/5 runs right-censored at epoch 50)."
  },
  Condition_E2_ResMLP: {
    id: "Condition_E2_ResMLP",
    title: "Condition E2: Residual Skip Highway Ablation (ResMLP)",
    badge: "Residual Skips",
    category: "Architecture Ablation",
    config: "eta_S = 0.05, eta_C = 0.05 | ResMLP (Skip connections, 17,346 params) | AdamW",
    desc: "Residual skip connections preserve direct gradient highways to early layers, stabilizing optimization dynamics while retaining simplicity prioritization under matched signal conditions."
  }
};

// Aliases for compatibility
const CONDITION_ALIASES = {
  "Condition_D": "Condition_D_SGD",
  "Condition_E": "Condition_E1_Deep",
  "Condition_E_Deep": "Condition_E1_Deep",
  "Condition_E_ResMLP": "Condition_E2_ResMLP"
};

function resolveConditionKey(key) {
  if (CONDITION_ALIASES[key]) return CONDITION_ALIASES[key];
  if (EXPERIMENTAL_DATA && EXPERIMENTAL_DATA.conditions) {
    if (EXPERIMENTAL_DATA.conditions[key]) return key;
    if (key === "Condition_D" && EXPERIMENTAL_DATA.conditions["Condition_D_SGD"]) return "Condition_D_SGD";
    if (key === "Condition_E" && EXPERIMENTAL_DATA.conditions["Condition_E1_Deep"]) return "Condition_E1_Deep";
    if (key === "Condition_E_Deep" && EXPERIMENTAL_DATA.conditions["Condition_E1_Deep"]) return "Condition_E1_Deep";
    if (key === "Condition_E_ResMLP" && EXPERIMENTAL_DATA.conditions["Condition_E2_ResMLP"]) return "Condition_E2_ResMLP";
  }
  return key;
}

/**
 * Async Protocol Data Loader
 * Attempts live fetch from backend JSON files, falling back to embedded constants.
 */
async function loadProtocolData(protocolName) {
  currentProtocol = protocolName;

  const dataUrl = protocolName === 'corrected'
    ? 'results/raw_data/corrected_protocol/experiments_results_corrected.json'
    : 'results/raw_data/historical_run/experiments_results_historical_test_monitored.json';

  const statsUrl = protocolName === 'corrected'
    ? 'results/raw_data/corrected_protocol/statistical_summary_corrected.json'
    : 'results/raw_data/historical_run/statistical_summary_historical_test_monitored.json';

  try {
    const [dataResp, statsResp] = await Promise.all([
      fetch(dataUrl, { cache: 'no-store' }),
      fetch(statsUrl, { cache: 'no-store' })
    ]);

    if (dataResp.ok && statsResp.ok) {
      const dataJson = await dataResp.json();
      const statsJson = await statsResp.json();
      if (protocolName === 'corrected') {
        DATA_CORRECTED = dataJson;
        STATS_CORRECTED = statsJson;
      } else {
        DATA_HISTORICAL = dataJson;
        STATS_HISTORICAL = statsJson;
      }
    }
  } catch (e) {
    console.info(`[PARITY] Local fetch unavailable or file:// protocol in use. Utilizing high-fidelity embedded ${protocolName} bundle.`);
  }

  // Set active references
  if (protocolName === 'corrected') {
    EXPERIMENTAL_DATA = DATA_CORRECTED;
    STATISTICAL_SUMMARY = STATS_CORRECTED;
  } else {
    EXPERIMENTAL_DATA = DATA_HISTORICAL;
    STATISTICAL_SUMMARY = STATS_HISTORICAL;
  }

  updateProtocolUI();
}

/**
 * Protocol Switcher Handler
 */
function switchProtocol(protocolName) {
  if (protocolName === currentProtocol && EXPERIMENTAL_DATA) return;

  const btnCorrected = document.getElementById('btn-proto-corrected');
  const btnHistorical = document.getElementById('btn-proto-historical');

  if (protocolName === 'corrected') {
    if (btnCorrected) btnCorrected.classList.add('active');
    if (btnHistorical) btnHistorical.classList.remove('active');
  } else {
    if (btnHistorical) btnHistorical.classList.add('active');
    if (btnCorrected) btnCorrected.classList.remove('active');
  }

  loadProtocolData(protocolName).then(() => {
    updateConditionView(currentConditionKey);
    updateInspectorViews();
    updateGalleryImages();
  });
}

/**
 * Update Protocol Control Bar & Metadata Badges
 */
function updateProtocolUI() {
  const badgeMonitor = document.getElementById('proto-badge-monitor');
  const badgeEval = document.getElementById('proto-badge-eval');
  const badgeThreshold = document.getElementById('proto-badge-threshold');
  const badgeSource = document.getElementById('proto-badge-source');

  if (currentProtocol === 'corrected') {
    if (badgeMonitor) badgeMonitor.textContent = 'Monitoring: val_d (N=600, Zero-Leakage)';
    if (badgeEval) badgeEval.textContent = 'Held-Out Test: test_d (N=600)';
    if (badgeThreshold) badgeThreshold.textContent = 'Threshold: τ = 0.80 on val_d';
    if (badgeSource) badgeSource.textContent = 'Source: results/raw_data/corrected_protocol/';
  } else {
    if (badgeMonitor) badgeMonitor.textContent = 'Monitoring: test_d (N=600, Historical Leakage)';
    if (badgeEval) badgeEval.textContent = 'Evaluation: test_d (Simultaneous)';
    if (badgeThreshold) badgeThreshold.textContent = 'Threshold: τ = 0.80 on test_d';
    if (badgeSource) badgeSource.textContent = 'Source: results/raw_data/historical_run/';
  }
}

/**
 * Update Figure Gallery Images According to Active Protocol
 */
function updateGalleryImages() {
  const prefix = currentProtocol === 'corrected'
    ? 'results/figures/'
    : 'results/figures/historical_run/';

  const img1 = document.getElementById('gallery-img-1');
  const img2 = document.getElementById('gallery-img-2');
  const img3 = document.getElementById('gallery-img-3');
  const img4 = document.getElementById('gallery-img-4');
  const img5 = document.getElementById('gallery-img-5');

  if (img1) img1.src = `${prefix}fig1_learning_dynamics_matched.png?v=${Date.now()}`;
  if (img2) img2.src = `${prefix}fig2_falsification_signal_dominance.png?v=${Date.now()}`;
  if (img3) img3.src = `${prefix}fig3_reliance_heatmaps.png?v=${Date.now()}`;
  if (img4) img4.src = `${prefix}fig4_optimizer_architecture_ablation.png?v=${Date.now()}`;
  if (img5) img5.src = `${prefix}fig5_statistical_summary.png?v=${Date.now()}`;

  const badgeProtocolGal = document.getElementById('gallery-protocol-label');
  if (badgeProtocolGal) {
    badgeProtocolGal.textContent = currentProtocol === 'corrected'
      ? 'Publication Figures: Corrected Protocol (val_d Monitored)'
      : 'Archived Figures: Historical Protocol (test_d Monitored)';
  }
}

/**
 * Extract Learning Dynamics Curves
 */
function getActiveCurves(condKey, seedFilter) {
  const resolved = resolveConditionKey(condKey);
  if (!EXPERIMENTAL_DATA || !EXPERIMENTAL_DATA.conditions || !EXPERIMENTAL_DATA.conditions[resolved]) {
    return null;
  }
  const cond = EXPERIMENTAL_DATA.conditions[resolved];
  const runs = cond.runs;
  if (!runs || runs.length === 0) return null;

  const numEpochs = runs[0].dynamics ? runs[0].dynamics.length : 50;
  const epochs = Array.from({ length: numEpochs }, (_, i) => i + 1);

  if (seedFilter === 'all') {
    const joint = [];
    const sOnly = [];
    const cOnly = [];
    const nOnly = [];

    for (let ep = 0; ep < numEpochs; ep++) {
      let sumJ = 0, sumS = 0, sumC = 0, sumN = 0;
      runs.forEach(r => {
        if (r.dynamics && r.dynamics[ep]) {
          sumJ += r.dynamics[ep].acc_joint;
          sumS += r.dynamics[ep].acc_s_only;
          sumC += r.dynamics[ep].acc_c_only;
          sumN += r.dynamics[ep].acc_n_only;
        } else if (r.joint_curve) {
          sumJ += r.joint_curve[ep];
          sumS += r.s_curve[ep];
          sumC += r.c_curve[ep];
          sumN += r.n_curve[ep];
        }
      });
      joint.push(sumJ / runs.length);
      sOnly.push(sumS / runs.length);
      cOnly.push(sumC / runs.length);
      nOnly.push(sumN / runs.length);
    }
    return { epochs, joint, sOnly, cOnly, noise: nOnly };
  } else {
    const targetSeed = parseInt(seedFilter, 10);
    const run = runs.find(r => r.seed === targetSeed) || runs[0];
    const joint = [];
    const sOnly = [];
    const cOnly = [];
    const nOnly = [];

    for (let ep = 0; ep < numEpochs; ep++) {
      if (run.dynamics && run.dynamics[ep]) {
        joint.push(run.dynamics[ep].acc_joint);
        sOnly.push(run.dynamics[ep].acc_s_only);
        cOnly.push(run.dynamics[ep].acc_c_only);
        nOnly.push(run.dynamics[ep].acc_n_only);
      } else if (run.joint_curve) {
        joint.push(run.joint_curve[ep]);
        sOnly.push(run.s_curve[ep]);
        cOnly.push(run.c_curve[ep]);
        nOnly.push(run.n_curve[ep]);
      }
    }
    return { epochs, joint, sOnly, cOnly, noise: nOnly };
  }
}

/**
 * Initialize Empirical Dynamics Chart
 */
function initResultsChart() {
  const canvas = document.getElementById('main-results-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  resultsChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: Array.from({ length: 50 }, (_, i) => i + 1),
      datasets: [
        {
          label: 'Joint Input [S + C + N]',
          data: [],
          borderColor: '#f8fafc',
          backgroundColor: 'rgba(255, 255, 255, 0.05)',
          borderWidth: 2.5,
          tension: 0.15,
          pointRadius: 0
        },
        {
          label: 'S-only (Linear Coordinate)',
          data: [],
          borderColor: '#38bdf8',
          backgroundColor: 'rgba(56, 189, 248, 0.08)',
          borderWidth: 2.2,
          tension: 0.15,
          pointRadius: 0
        },
        {
          label: 'C-only (Nonlinear XOR)',
          data: [],
          borderColor: '#f43f5e',
          backgroundColor: 'rgba(244, 63, 94, 0.08)',
          borderWidth: 2.2,
          tension: 0.15,
          pointRadius: 0
        },
        {
          label: 'Noise Null Control',
          data: [],
          borderColor: '#94a3b8',
          borderDash: [5, 5],
          borderWidth: 1.5,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 250 },
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          titleColor: '#f8fafc',
          bodyColor: '#cbd5e1',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1,
          padding: 10,
          callbacks: {
            label: function(context) {
              return `${context.dataset.label}: ${(context.parsed.y * 100).toFixed(1)}%`;
            }
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Training Epoch', color: '#94a3b8', font: { size: 11, weight: '600' } },
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: { color: '#94a3b8' }
        },
        y: {
          min: 0.45,
          max: 1.0,
          title: { display: true, text: 'Counterfactual Generalization Accuracy', color: '#94a3b8', font: { size: 11, weight: '600' } },
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: {
            color: '#94a3b8',
            callback: value => (value * 100).toFixed(0) + '%'
          }
        }
      }
    }
  });
}

/**
 * Update the Condition View: Metrics, Charts, Statistics, and Seed Table
 */
function updateConditionView(condKey) {
  const resolved = resolveConditionKey(condKey);
  currentConditionKey = resolved;
  const meta = CONDITION_METADATA[resolved] || {
    id: resolved,
    title: resolved,
    badge: "Verified",
    desc: "Empirical condition run."
  };

  // Header Elements
  const chartTitle = document.getElementById('chart-condition-title');
  const badgeCond = document.getElementById('badge-stat-cond');
  const statInterp = document.getElementById('stat-interpretation');

  if (chartTitle) {
    chartTitle.textContent = `${meta.title} ${currentSeedFilter === 'all' ? '(Mean across 5 seeds)' : `(Seed ${currentSeedFilter})`}`;
  }
  if (badgeCond) badgeCond.textContent = meta.badge;

  if (!EXPERIMENTAL_DATA || !EXPERIMENTAL_DATA.conditions || !EXPERIMENTAL_DATA.conditions[resolved]) {
    return;
  }

  const cond = EXPERIMENTAL_DATA.conditions[resolved];
  const summary = cond.summary;
  const runs = cond.runs || [];

  // 1. Binary Acquisition & Threshold Crossing Information
  const statCrossingS = document.getElementById('stat-crossing-s');
  const statCrossingC = document.getElementById('stat-crossing-c');
  const statCensoringNote = document.getElementById('stat-censoring-note');

  const sCount = summary.S_crossing_count || `${runs.filter(r => r.S_acquisition && r.S_acquisition.crossed_threshold).length}/${runs.length}`;
  const cCount = summary.C_crossing_count || `${runs.filter(r => r.C_acquisition && r.C_acquisition.crossed_threshold).length}/${runs.length}`;

  const sObs = summary.S_observed_crossing_epochs || [];
  const cObs = summary.C_observed_crossing_epochs || [];
  const sCens = summary.S_censored_count !== undefined ? summary.S_censored_count : runs.length - sObs.length;
  const cCens = summary.C_censored_count !== undefined ? summary.C_censored_count : runs.length - cObs.length;

  const sMeanStr = (summary.S_mean_observed_epoch !== null && summary.S_mean_observed_epoch !== undefined)
    ? `${summary.S_mean_observed_epoch.toFixed(1)} ± ${summary.S_std_observed_epoch.toFixed(1)} ep`
    : `None observed`;

  const cMeanStr = (summary.C_mean_observed_epoch !== null && summary.C_mean_observed_epoch !== undefined)
    ? `${summary.C_mean_observed_epoch.toFixed(1)} ± ${summary.C_std_observed_epoch.toFixed(1)} ep`
    : `None observed`;

  if (statCrossingS) {
    statCrossingS.innerHTML = `<strong>${sCount}</strong> crossed <span class="stat-sub-label">(obs: [${sObs.join(', ')}], mean: ${sMeanStr}, ${sCens}/${runs.length} censored)</span>`;
  }

  if (statCrossingC) {
    statCrossingC.innerHTML = `<strong>${cCount}</strong> crossed <span class="stat-sub-label">(obs: [${cObs.join(', ')}], mean: ${cMeanStr}, ${cCens}/${runs.length} censored)</span>`;
  }

  if (statCensoringNote) {
    if (sCens > 0 || cCens > 0) {
      statCensoringNote.textContent = `Right-censoring: Uncrossed runs (>50 epochs) are strictly excluded from observed time arithmetic; non-crossings are never averaged as epoch 51.`;
    } else {
      statCensoringNote.textContent = `Zero censoring: All ${runs.length} paired runs crossed the threshold within the 50-epoch window.`;
    }
  }

  // 2. Exact McNemar Statistical Tests
  const statMcnemarTwo = document.getElementById('stat-mcnemar-two');
  const statMcnemarDir = document.getElementById('stat-mcnemar-dir');
  const badgeMcnemar = document.getElementById('badge-mcnemar-sig');

  let statsSection = null;
  if (STATISTICAL_SUMMARY) {
    if (resolved === 'Condition_A') statsSection = STATISTICAL_SUMMARY.Condition_A_Matched;
    else if (resolved === 'Condition_B') statsSection = STATISTICAL_SUMMARY.Condition_B_Complex_Dominant;
  }

  if (statsSection && statsSection.binary_crossing_tests) {
    const bTest = statsSection.binary_crossing_tests;
    const pTwo = bTest.primary_two_sided_p;
    const isSigTwo = bTest.primary_significance_at_0_05;

    if (statMcnemarTwo) {
      statMcnemarTwo.textContent = `Primary exact two-sided p = ${pTwo.toFixed(4)}`;
    }

    if (badgeMcnemar) {
      if (isSigTwo) {
        badgeMcnemar.className = 'badge-pill badge-success';
        badgeMcnemar.textContent = 'p < 0.05 (Significant)';
      } else {
        badgeMcnemar.className = 'badge-pill badge-neutral';
        badgeMcnemar.textContent = 'p = 0.0625 (NS at α=0.05, discrete N=5 limit)';
      }
    }

    if (statMcnemarDir) {
      if (resolved === 'Condition_A') {
        const dirP = bTest.supplementary_directional_tests ? bTest.supplementary_directional_tests.s_greater_c.p_value : 0.03125;
        statMcnemarDir.textContent = `Supplementary directional (S > C): exact p = ${dirP.toFixed(5)} (exploratory one-sided)`;
      } else if (resolved === 'Condition_B') {
        const dirP = bTest.supplementary_directional_tests ? bTest.supplementary_directional_tests.c_greater_s.p_value : 0.03125;
        statMcnemarDir.textContent = `Supplementary directional (C > S): exact p = ${dirP.toFixed(5)} (exploratory one-sided)`;
      }
    }
  } else {
    // Standard calculation for other conditions
    if (statMcnemarTwo) statMcnemarTwo.textContent = `Paired run comparison`;
    if (badgeMcnemar) {
      badgeMcnemar.className = 'badge-pill badge-info';
      badgeMcnemar.textContent = `Ablation Condition`;
    }
    if (statMcnemarDir) statMcnemarDir.textContent = `Evaluated across ${runs.length} paired seeds`;
  }

  // 3. Continuous Held-Out Test Accuracy & Paired t-test
  const statTerminalAcc = document.getElementById('stat-terminal-acc');
  const statPairedDiff = document.getElementById('stat-paired-diff');
  const statPairedT = document.getElementById('stat-paired-t');
  const statCi95 = document.getElementById('stat-ci-95');

  const meanS = summary.mean_acc_s_only !== undefined ? summary.mean_acc_s_only : (summary.mean_final_acc_S || 0);
  const stdS = summary.std_acc_s_only !== undefined ? summary.std_acc_s_only : 0;
  const meanC = summary.mean_acc_c_only !== undefined ? summary.mean_acc_c_only : (summary.mean_final_acc_C || 0);
  const stdC = summary.std_acc_c_only !== undefined ? summary.std_acc_c_only : 0;

  if (statTerminalAcc) {
    statTerminalAcc.innerHTML = `S-only: <strong>${(meanS * 100).toFixed(2)}%</strong> (±${(stdS * 100).toFixed(2)}%) &nbsp;|&nbsp; C-only: <strong>${(meanC * 100).toFixed(2)}%</strong> (±${(stdC * 100).toFixed(2)}%)`;
  }

  if (statsSection) {
    if (resolved === 'Condition_A') {
      const diffMean = (statsSection['accuracy_difference_mean (S - C)'] * 100).toFixed(2);
      const tStat = statsSection.paired_t_stat_acc.toFixed(4);
      const df = statsSection.degrees_of_freedom;
      const pVal = statsSection.paired_p_val_acc_twosided.toFixed(4);
      const ciLow = (statsSection.ci_95_diff[0] * 100).toFixed(2);
      const ciHigh = (statsSection.ci_95_diff[1] * 100).toFixed(2);

      if (statPairedDiff) statPairedDiff.textContent = `Mean Paired Diff (S − C): +${diffMean} percentage points`;
      if (statPairedT) statPairedT.textContent = `Paired t(${df}) = ${tStat}, two-sided p = ${pVal} (NS at α=0.05)`;
      if (statCi95) statCi95.textContent = `95% Confidence Interval: [${ciLow}, +${ciHigh}] percentage points`;
    } else if (resolved === 'Condition_B') {
      const diffMean = (statsSection['accuracy_difference_mean (C - S)'] * 100).toFixed(2);
      const tStat = statsSection.paired_t_stat_acc.toFixed(4);
      const df = statsSection.degrees_of_freedom;
      const pVal = statsSection.paired_p_val_acc_twosided.toExponential(3);
      const ciLow = (statsSection.ci_95_diff[0] * 100).toFixed(2);
      const ciHigh = (statsSection.ci_95_diff[1] * 100).toFixed(2);

      if (statPairedDiff) statPairedDiff.textContent = `Mean Paired Diff (C − S): +${diffMean} percentage points`;
      if (statPairedT) statPairedT.textContent = `Paired t(${df}) = ${tStat}, two-sided p = ${pVal} (Significant)`;
      if (statCi95) statCi95.textContent = `95% Confidence Interval: [+${ciLow}, +${ciHigh}] percentage points`;
    }
  } else {
    // Dynamic calculation from runs
    const diff = (meanS - meanC) * 100;
    const diffSign = diff >= 0 ? '+' : '';
    if (statPairedDiff) statPairedDiff.textContent = `Mean Paired Diff: ${diffSign}${diff.toFixed(2)} percentage points`;
    if (statPairedT) statPairedT.textContent = `Continuous terminal accuracy evaluated on test_d (N=600)`;
    if (statCi95) statCi95.textContent = `Joint accuracy: ${(summary.mean_joint_acc * 100).toFixed(2)}% (±${(summary.std_joint_acc * 100).toFixed(2)}%)`;
  }

  // 4. Feature Reliance Ratio
  const statReliance = document.getElementById('stat-reliance');
  if (statReliance) {
    const relS = (summary.mean_reliance_S * 100).toFixed(1);
    const relC = ((1 - summary.mean_reliance_S) * 100).toFixed(1);
    statReliance.textContent = `${relS}% S / ${relC}% C (mean ratio: ${summary.mean_reliance_S.toFixed(3)})`;
  }

  // 5. Exploratory Epoch-1 Split Metrics
  const statEarlySplit = document.getElementById('stat-early-split');
  if (statEarlySplit) {
    if (resolved === 'Condition_A') {
      statEarlySplit.innerHTML = `<strong>Exploratory Epoch-1 Comparison:</strong><br>` +
        `• Validation Split (N=600): S = <strong>85.81%</strong> vs C = 59.87% (paired t(4)=5.27, p=0.0056)<br>` +
        `• Held-Out Test Split (N=600): S = <strong>83.50%</strong> vs C = 61.20% (paired t(4)=4.39, p=0.0117)`;
    } else if (resolved === 'Condition_B') {
      statEarlySplit.innerHTML = `<strong>Exploratory Epoch-1 Comparison:</strong><br>` +
        `• Held-Out Test Split (N=600): C = <strong>96.87%</strong> vs S = 50.80% (instantaneous inversion)`;
    } else {
      statEarlySplit.innerHTML = `<strong>Monitoring Split:</strong> ${currentProtocol === 'corrected' ? 'val_d' : 'test_d'} (τ=0.80) | <strong>Held-out test:</strong> test_d (epoch 50)`;
    }
  }

  // 6. Scientific Interpretation Box
  if (statInterp) {
    if (resolved === 'Condition_A') {
      statInterp.innerHTML = `<strong>Scientific Interpretation:</strong> Under strictly matched Bayes signal parity (both ceilings = 95.0%), S is acquired in 5/5 runs (mean 1.8 ± 1.8 ep) while C is right-censored in 5/5 runs (>50 ep). Exact McNemar two-sided p = 0.0625 (nonsignificant at α=0.05 due to N=5 discrete sample size limit; directional one-sided p = 0.03125). Terminal held-out accuracy favors S (+4.63 pp, t(4)=2.10, p=0.1035).`;
    } else if (resolved === 'Condition_B') {
      statInterp.innerHTML = `<strong>Scientific Interpretation:</strong> When complex signal quality dominates (99% vs 75% Bayes ceiling), learning order decisively inverts: C crossed in 5/5 runs at epoch 1.0, while S crossed in 0/5 runs (>50 ep). Terminal accuracy overwhelmingly favors C (96.75% vs 50.97%, diff +45.77 pp, t(4)=78.83, p=1.55e-7). Simplicity bias is completely subordinated to signal quality.`;
    } else {
      statInterp.innerHTML = `<strong>Scientific Interpretation:</strong> ${meta.desc}`;
    }
  }

  // 7. Update Dynamics Line Chart
  const curves = getActiveCurves(resolved, currentSeedFilter);
  if (curves && resultsChart) {
    resultsChart.data.labels = curves.epochs;
    resultsChart.data.datasets[0].data = curves.joint;
    resultsChart.data.datasets[1].data = curves.sOnly;
    resultsChart.data.datasets[2].data = curves.cOnly;
    resultsChart.data.datasets[3].data = curves.noise;
    resultsChart.update();
  }

  // 8. Update Per-Seed Breakdown Table
  renderSeedTable(runs, summary);
}

/**
 * Render Per-Seed Empirical Breakdown Table
 */
function renderSeedTable(runs, summary) {
  const tbody = document.getElementById('seed-table-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  runs.forEach(r => {
    const tr = document.createElement('tr');
    if (currentSeedFilter === r.seed.toString()) {
      tr.style.backgroundColor = 'rgba(56, 189, 248, 0.15)';
    }

    // Extract exact acquisition epochs
    let sEpStr = 'Censored (>50)';
    let cEpStr = 'Censored (>50)';

    if (r.S_acquisition && r.S_acquisition.crossed_threshold && r.S_acquisition.observed_epoch !== null) {
      sEpStr = `Ep ${r.S_acquisition.observed_epoch}`;
    } else if (r.T_S_observed !== null && r.T_S_observed !== undefined) {
      sEpStr = `Ep ${r.T_S_observed}`;
    }

    if (r.C_acquisition && r.C_acquisition.crossed_threshold && r.C_acquisition.observed_epoch !== null) {
      cEpStr = `Ep ${r.C_acquisition.observed_epoch}`;
    } else if (r.T_C_observed !== null && r.T_C_observed !== undefined) {
      cEpStr = `Ep ${r.T_C_observed}`;
    }

    // Accuracies from final_test_metrics
    const finalJ = r.final_test_metrics ? r.final_test_metrics.acc_joint : r.final_acc_joint;
    const finalS = r.final_test_metrics ? r.final_test_metrics.acc_s_only : r.final_acc_s_only;
    const finalC = r.final_test_metrics ? r.final_test_metrics.acc_c_only : r.final_acc_c_only;
    const reliance = r.reliance_ratio_S !== undefined ? r.reliance_ratio_S : 0.5;

    const diffPP = (finalS - finalC) * 100;
    const diffSign = diffPP >= 0 ? '+' : '';
    const diffColor = diffPP >= 0 ? '#38bdf8' : '#f43f5e';

    tr.innerHTML = `
      <td><strong>Seed ${r.seed}</strong></td>
      <td><span class="${sEpStr.includes('Censored') ? 'censored-tag' : 'crossing-tag'}">${sEpStr}</span></td>
      <td><span class="${cEpStr.includes('Censored') ? 'censored-tag' : 'crossing-tag'}">${cEpStr}</span></td>
      <td>${(finalJ * 100).toFixed(2)}%</td>
      <td>${(finalS * 100).toFixed(2)}%</td>
      <td>${(finalC * 100).toFixed(2)}%</td>
      <td style="color: ${diffColor}; font-weight: 600;">${diffSign}${diffPP.toFixed(2)} pp</td>
      <td>${(reliance * 100).toFixed(1)}% / ${((1 - reliance) * 100).toFixed(1)}%</td>
    `;
    tbody.appendChild(tr);
  });

  // Summary Row
  const summaryTr = document.createElement('tr');
  summaryTr.style.background = 'rgba(15, 23, 42, 0.95)';
  summaryTr.style.fontWeight = 'bold';

  const sMeanStr = (summary.S_mean_observed_epoch !== null && summary.S_mean_observed_epoch !== undefined)
    ? `Mean: Ep ${summary.S_mean_observed_epoch.toFixed(1)}`
    : `5/5 Censored`;

  const cMeanStr = (summary.C_mean_observed_epoch !== null && summary.C_mean_observed_epoch !== undefined)
    ? `Mean: Ep ${summary.C_mean_observed_epoch.toFixed(1)}`
    : `5/5 Censored`;

  const meanJ = (summary.mean_joint_acc * 100).toFixed(2);
  const meanS = (summary.mean_acc_s_only * 100).toFixed(2);
  const meanC = (summary.mean_acc_c_only * 100).toFixed(2);
  const meanDiff = ((summary.mean_acc_s_only - summary.mean_acc_c_only) * 100);
  const meanDiffSign = meanDiff >= 0 ? '+' : '';

  summaryTr.innerHTML = `
    <td><strong>Mean (5 Runs)</strong></td>
    <td>${sMeanStr}</td>
    <td>${cMeanStr}</td>
    <td>${meanJ}%</td>
    <td>${meanS}%</td>
    <td>${meanC}%</td>
    <td style="color: ${meanDiff >= 0 ? '#38bdf8' : '#f43f5e'};">${meanDiffSign}${meanDiff.toFixed(2)} pp</td>
    <td>${(summary.mean_reliance_S * 100).toFixed(1)}% / ${((1 - summary.mean_reliance_S) * 100).toFixed(1)}%</td>
  `;
  tbody.appendChild(summaryTr);
}

/**
 * Switch Condition Tab Handler
 */
function switchCondition(condKey) {
  const resolved = resolveConditionKey(condKey);
  currentConditionKey = resolved;

  document.querySelectorAll('.tab-btn').forEach(btn => {
    const btnCond = btn.getAttribute('data-condition');
    if (btnCond === condKey || btnCond === resolved) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });

  updateConditionView(resolved);
}

/**
 * Switch Seed Filter Handler
 */
function switchSeed(seedId) {
  currentSeedFilter = seedId;

  document.querySelectorAll('.seed-btn').forEach(btn => {
    const btnSeed = btn.getAttribute('data-seed');
    if (btnSeed === seedId) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });

  updateConditionView(currentConditionKey);
}

/**
 * Update Raw Artifact Inspector Pre-Blocks
 */
function updateInspectorViews() {
  const preStats = document.getElementById('pre-stats-json');
  const preRaw = document.getElementById('pre-raw-json');

  if (preStats && STATISTICAL_SUMMARY) {
    preStats.textContent = JSON.stringify(STATISTICAL_SUMMARY, null, 2);
  }

  if (preRaw && EXPERIMENTAL_DATA) {
    // Show metadata, baselines, and condition summaries
    const previewObj = {
      metadata: EXPERIMENTAL_DATA.metadata,
      single_feature_baselines: EXPERIMENTAL_DATA.single_feature_baselines,
      condition_summaries: {}
    };
    for (const [k, v] of Object.entries(EXPERIMENTAL_DATA.conditions || {})) {
      previewObj.condition_summaries[k] = v.summary;
    }
    preRaw.textContent = JSON.stringify(previewObj, null, 2);
  }

  // Update Baseline Oracle Numbers
  const baselines = EXPERIMENTAL_DATA ? EXPERIMENTAL_DATA.single_feature_baselines : null;
  if (baselines) {
    const elS = document.getElementById('oracle-acc-s');
    const elC = document.getElementById('oracle-acc-c');
    const elLinearC = document.getElementById('oracle-linear-c');
    if (elS && baselines.S_alone) elS.textContent = `${(baselines.S_alone.final_acc * 100).toFixed(2)}% Test Accuracy`;
    if (elC && baselines.C_alone) elC.textContent = `${(baselines.C_alone.final_acc * 100).toFixed(2)}% Test Accuracy`;
    if (elLinearC && baselines.logistic_regression) elLinearC.textContent = `${(baselines.logistic_regression.linear_acc_on_C_origin * 100).toFixed(2)}% (Chance Level)`;
  }
}

/**
 * Inspector Tab Switching
 */
function initInspectorTabs() {
  document.querySelectorAll('.insp-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.insp-tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.insp-pane').forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const paneId = `pane-${btn.getAttribute('data-insp')}`;
      const pane = document.getElementById(paneId);
      if (pane) pane.classList.add('active');
    });
  });
}

/**
 * Interactive 2D Feature Subspace Canvas Rendering
 */
function drawSubspaceS() {
  const canvas = document.getElementById('canvas-s');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Background Grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h);
  ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
  ctx.stroke();

  // Decision Boundary: Linear line x1 = 0
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(w / 2, 10); ctx.lineTo(w / 2, h - 10);
  ctx.stroke();
  ctx.setLineDash([]);

  // Plot synthetic sample points
  const sliderS = document.getElementById('slider-eta-s');
  const etaS = sliderS ? parseFloat(sliderS.value) : 0.05;

  // Pseudo-random deterministic generator for consistent rendering
  let seed = 42;
  function rnd() {
    seed = (seed * 9301 + 49297) % 233280;
    return seed / 233280;
  }

  const numPoints = 80;
  for (let i = 0; i < numPoints; i++) {
    const label = rnd() > 0.5 ? 1 : -1;
    const flip = rnd() < etaS;
    const effectiveLabel = flip ? -label : label;

    // S coordinate mean is separated by effectiveLabel
    const cx = (w / 2) + (effectiveLabel * (w * 0.22)) + (rnd() - 0.5) * (w * 0.28);
    const cy = (h / 2) + (rnd() - 0.5) * (h * 0.7);

    ctx.fillStyle = effectiveLabel === 1 ? '#38bdf8' : '#f43f5e';
    ctx.beginPath();
    ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawSubspaceC() {
  const canvas = document.getElementById('canvas-c');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Background Grid / Quadrant dividers
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h);
  ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
  ctx.stroke();

  // Plot synthetic XOR quadrant points
  const sliderC = document.getElementById('slider-eta-c');
  const etaC = sliderC ? parseFloat(sliderC.value) : 0.05;

  let seed = 101;
  function rnd() {
    seed = (seed * 9301 + 49297) % 233280;
    return seed / 233280;
  }

  const numPoints = 80;
  for (let i = 0; i < numPoints; i++) {
    const qx = rnd() > 0.5 ? 1 : -1;
    const qy = rnd() > 0.5 ? 1 : -1;
    const xorLabel = qx * qy;
    const flip = rnd() < etaC;
    const effectiveLabel = flip ? -xorLabel : xorLabel;

    const cx = (w / 2) + (qx * (w * 0.24)) + (rnd() - 0.5) * (w * 0.22);
    const cy = (h / 2) + (qy * (h * 0.24)) + (rnd() - 0.5) * (h * 0.22);

    ctx.fillStyle = effectiveLabel === 1 ? '#38bdf8' : '#f43f5e';
    ctx.beginPath();
    ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

/**
 * Initialize Interactive Simulation Chart
 */
function initSimChart() {
  const canvas = document.getElementById('sim-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  simChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: Array.from({ length: 50 }, (_, i) => i + 1),
      datasets: [
        {
          label: 'Simulated Joint Input',
          data: [],
          borderColor: '#f8fafc',
          borderWidth: 2.2,
          tension: 0.2,
          pointRadius: 0
        },
        {
          label: 'Simulated S-only (Linear)',
          data: [],
          borderColor: '#38bdf8',
          borderWidth: 2,
          tension: 0.2,
          pointRadius: 0
        },
        {
          label: 'Simulated C-only (XOR)',
          data: [],
          borderColor: '#f43f5e',
          borderWidth: 2,
          tension: 0.2,
          pointRadius: 0
        },
        {
          label: 'Chance Floor (0.50)',
          data: Array(50).fill(0.50),
          borderColor: '#64748b',
          borderDash: [4, 4],
          borderWidth: 1,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 250 },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: { color: '#94a3b8', boxWidth: 12, font: { size: 10 } }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Simulated Epoch', color: '#94a3b8', font: { size: 10 } },
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: { color: '#94a3b8', font: { size: 9 } }
        },
        y: {
          min: 0.45,
          max: 1.0,
          title: { display: true, text: 'Accuracy', color: '#94a3b8', font: { size: 10 } },
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: {
            color: '#94a3b8',
            font: { size: 9 },
            callback: v => (v * 100).toFixed(0) + '%'
          }
        }
      }
    }
  });
}

/**
 * Slider Update Handler
 */
function onSliderChange() {
  const sS = document.getElementById('slider-eta-s');
  const sC = document.getElementById('slider-eta-c');
  const valS = document.getElementById('val-eta-s');
  const valC = document.getElementById('val-eta-c');
  const hintS = document.getElementById('hint-acc-s');
  const hintC = document.getElementById('hint-acc-c');

  if (sS && valS) {
    const vS = parseFloat(sS.value);
    valS.textContent = vS.toFixed(2);
    if (hintS) hintS.textContent = `${((1 - vS) * 100).toFixed(0)}%`;
  }
  if (sC && valC) {
    const vC = parseFloat(sC.value);
    valC.textContent = vC.toFixed(2);
    if (hintC) hintC.textContent = `${((1 - vC) * 100).toFixed(0)}%`;
  }

  drawSubspaceS();
  drawSubspaceC();
  runLiveSimulation(false);
}

/**
 * Live Simulation Engine
 */
function runLiveSimulation(animated = false) {
  if (simAnimationTimer) {
    clearInterval(simAnimationTimer);
    simAnimationTimer = null;
    isSimulationRunning = false;
  }

  const sliderS = document.getElementById('slider-eta-s');
  const sliderC = document.getElementById('slider-eta-c');
  const selectArch = document.getElementById('select-arch');
  const selectOpt = document.getElementById('select-opt');
  if (!sliderS || !sliderC || !selectArch || !selectOpt) return;

  const etaS = parseFloat(sliderS.value);
  const etaC = parseFloat(sliderC.value);
  const arch = selectArch.value;
  const opt = selectOpt.value;

  const ceilingS = 1.0 - etaS;
  const ceilingC = 1.0 - etaC;

  // Architectural modifier on non-linear acquisition speed
  let cRateMod = 1.0;
  if (arch === 'DeepMLP') cRateMod = 1.6;
  if (arch === 'ResMLP') cRateMod = 1.8;
  if (opt === 'SGD') cRateMod *= 0.45;

  let sRateMod = 1.0;
  if (opt === 'SGD') sRateMod = 0.85;

  const simJoint = [];
  const simS = [];
  const simC = [];

  const maxEpochs = 50;
  let crossedS = null;
  let crossedC = null;

  for (let ep = 1; ep <= maxEpochs; ep++) {
    // S logistic acquisition: fast linear convergence
    const sProgress = 1 / (1 + Math.exp(-0.75 * sRateMod * (ep - 2)));
    const sAcc = 0.50 + (ceilingS - 0.50) * sProgress;

    // C non-linear acquisition: saddle point escape latency
    const cLag = (opt === 'SGD') ? 16 : (arch === 'StandardMLP' ? 8 : 4);
    const cSlope = 0.45 * cRateMod;
    const cProgress = 1 / (1 + Math.exp(-cSlope * (ep - cLag)));
    const cAcc = 0.50 + (ceilingC - 0.50) * cProgress;

    // Joint accuracy
    const jCeiling = Math.max(ceilingS, ceilingC, 1 - (etaS * etaC));
    const jAcc = Math.min(jCeiling, Math.max(sAcc, cAcc) + 0.03 * (1 / (1 + Math.exp(-0.5 * ep))));

    simS.push(sAcc);
    simC.push(cAcc);
    simJoint.push(jAcc);

    if (crossedS === null && sAcc >= 0.80) crossedS = ep;
    if (crossedC === null && cAcc >= 0.80) crossedC = ep;
  }

  // Update Status Box & Badges
  const badgeOrder = document.getElementById('sim-order-badge');
  const statusText = document.getElementById('sim-status-text');
  const archLabel = document.getElementById('sim-arch-label');
  const optLabel = document.getElementById('sim-opt-label');

  if (archLabel) archLabel.textContent = arch;
  if (optLabel) optLabel.textContent = opt;

  let orderMsg = '';
  if (crossedS !== null && crossedC !== null) {
    const delta = crossedC - crossedS;
    const sign = delta > 0 ? '+' : '';
    if (badgeOrder) badgeOrder.textContent = `Δ = ${sign}${delta} epochs (${delta > 0 ? 'S Learned First' : 'C Learned First'})`;
    orderMsg = delta > 0
      ? `Simplicity Prioritization: Linear S reached τ=0.80 at Epoch ${crossedS}; Nonlinear C at Epoch ${crossedC} (Δ = +${delta} ep).`
      : `Signal Inversion: Nonlinear C reached τ=0.80 at Epoch ${crossedC}; Linear S at Epoch ${crossedS} (Δ = ${delta} ep).`;
  } else if (crossedS !== null && crossedC === null) {
    if (badgeOrder) badgeOrder.textContent = `S Learned (C Right-Censored)`;
    orderMsg = `Standard Simplicity Bias: S crossed τ=0.80 at Epoch ${crossedS}; C failed to cross within 50 epochs (right-censored).`;
  } else if (crossedS === null && crossedC !== null) {
    if (badgeOrder) badgeOrder.textContent = `C Learned (S Right-Censored)`;
    orderMsg = `Falsification Dominance: Complex C crossed τ=0.80 at Epoch ${crossedC}; S failed to cross (right-censored).`;
  } else {
    if (badgeOrder) badgeOrder.textContent = `Neither Crossed τ=0.80`;
    orderMsg = `Severe noise / optimization barrier: Neither feature reached the acquisition threshold.`;
  }

  if (statusText) statusText.textContent = orderMsg;

  if (animated && simChart) {
    isSimulationRunning = true;
    simChart.data.datasets[0].data = [];
    simChart.data.datasets[1].data = [];
    simChart.data.datasets[2].data = [];
    simChart.update();

    let step = 0;
    simAnimationTimer = setInterval(() => {
      step++;
      simChart.data.datasets[0].data = simJoint.slice(0, step);
      simChart.data.datasets[1].data = simS.slice(0, step);
      simChart.data.datasets[2].data = simC.slice(0, step);
      simChart.update('none');

      if (step >= maxEpochs) {
        clearInterval(simAnimationTimer);
        simAnimationTimer = null;
        isSimulationRunning = false;
      }
    }, 25);
  } else if (simChart) {
    simChart.data.datasets[0].data = simJoint;
    simChart.data.datasets[1].data = simS;
    simChart.data.datasets[2].data = simC;
    simChart.update();
  }
}

function startSimulation() {
  runLiveSimulation(true);
}

function resetSimulation() {
  const sliderS = document.getElementById('slider-eta-s');
  const sliderC = document.getElementById('slider-eta-c');
  const selectArch = document.getElementById('select-arch');
  const selectOpt = document.getElementById('select-opt');

  if (sliderS) sliderS.value = 0.05;
  if (sliderC) sliderC.value = 0.05;
  if (selectArch) selectArch.value = 'StandardMLP';
  if (selectOpt) selectOpt.value = 'AdamW';

  onSliderChange();
}

/**
 * Smooth Navigation Glider & Fluid ScrollSpy Engine
 */
function initNavigation() {
  const navContainer = document.getElementById('main-nav-links');
  const glider = document.getElementById('nav-glider');
  const navLinks = document.querySelectorAll('.nav-link');
  if (!navContainer || !glider || !navLinks.length) return;

  const sectionIds = Array.from(navLinks).map(link => link.getAttribute('href').substring(1));
  const sections = sectionIds.map(id => document.getElementById(id)).filter(Boolean);

  function positionGlider(link) {
    if (!link) {
      glider.style.opacity = '0';
      return;
    }
    const containerRect = navContainer.getBoundingClientRect();
    const linkRect = link.getBoundingClientRect();
    const offsetLeft = linkRect.left - containerRect.left;
    const width = linkRect.width;

    glider.style.width = `${width}px`;
    glider.style.transform = `translateX(${offsetLeft}px)`;
    glider.style.opacity = '1';
  }

  function setActiveLink(link) {
    if (!link) return;
    navLinks.forEach(l => l.classList.remove('active'));
    link.classList.add('active');
    positionGlider(link);
  }

  // Smooth click navigation
  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const targetId = link.getAttribute('href');
      const targetEl = document.querySelector(targetId);
      if (targetEl) {
        setActiveLink(link);

        const headerHeight = document.querySelector('.site-header')?.offsetHeight || 65;
        const targetPosition = targetEl.getBoundingClientRect().top + window.pageYOffset - headerHeight + 2;

        window.scrollTo({
          top: targetPosition,
          behavior: 'smooth'
        });

        if (history.pushState) {
          history.pushState(null, null, targetId);
        }
      }
    });
  });

  // Smooth ScrollSpy tracking active section
  let isTicking = false;
  function updateScrollSpy() {
    const scrollPos = window.pageYOffset;
    const headerHeight = document.querySelector('.site-header')?.offsetHeight || 65;
    const triggerOffset = headerHeight + 100;

    let activeId = '';
    for (let i = sections.length - 1; i >= 0; i--) {
      const sec = sections[i];
      if (scrollPos >= sec.offsetTop - triggerOffset) {
        activeId = sec.id;
        break;
      }
    }

    if (!activeId && sections.length > 0) {
      activeId = sections[0].id;
    }

    const currentActiveLink = document.querySelector(`.nav-link[href="#${activeId}"]`);
    if (currentActiveLink && !currentActiveLink.classList.contains('active')) {
      setActiveLink(currentActiveLink);
    }
  }

  window.addEventListener('scroll', () => {
    if (!isTicking) {
      window.requestAnimationFrame(() => {
        updateScrollSpy();
        isTicking = false;
      });
      isTicking = true;
    }
  }, { passive: true });

  window.addEventListener('resize', () => {
    const active = document.querySelector('.nav-link.active') || navLinks[0];
    positionGlider(active);
  });

  // Initial positioning after layout
  setTimeout(() => {
    const active = document.querySelector('.nav-link.active') || navLinks[0];
    positionGlider(active);
  }, 120);
}

/**
 * Scroll Reveal Animation Observer
 */
function initScrollObserver() {
  const sections = document.querySelectorAll('.section-reveal');
  const observer = new IntersectionObserver((entries, obs) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('revealed');
        obs.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  sections.forEach(sec => observer.observe(sec));
}

/**
 * DOM Ready Bootstrap
 */
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initResultsChart();
  initSimChart();
  initInspectorTabs();
  initScrollObserver();

  drawSubspaceS();
  drawSubspaceC();
  runLiveSimulation(false);

  // Load primary corrected protocol by default
  loadProtocolData('corrected').then(() => {
    updateConditionView('Condition_A');
    updateInspectorViews();
    updateGalleryImages();
  });
});
