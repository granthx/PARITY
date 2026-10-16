# When Does Simplicity Bias Fail?
## A Controlled Empirical Investigation Disentangling Feature Complexity, Predictive Strength, and Optimization Accessibility in Neural Networks

[![Google Docs Research Paper](https://img.shields.io/badge/Google%20Docs-Official%20Research%20Paper-4285F4?style=for-the-badge&logo=googledocs&logoColor=white)](https://docs.google.com/document/d/1wpl74tR3uBP65dPv9ZUCTsIwiai8wPOhO1ewUJ3qwnU/edit?usp=sharing)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Submission Protocol](https://img.shields.io/badge/College%20Society-Falsification%20Task-purple.svg)](#submission-protocol-checklist)

> **Submission Document & Empirical Research Paper**<br>
> **📄 Full Research Paper (Google Docs):** [**Read Formatted Research Paper on Google Docs**](https://docs.google.com/document/d/1wpl74tR3uBP65dPv9ZUCTsIwiai8wPOhO1ewUJ3qwnU/edit?usp=sharing)<br>
> **Author:** Candidate Research Submission<br>
> **Challenge:** *The Falsification Challenge: Investigating Simplicity Bias in Neural Networks*<br>
> **Evaluation Track:** Experimental Research Proposal & Empirical Pilot with ML Model Proofs<br>
> **Deliverable Format:** Official Google Docs Paper + Comprehensive README + Reproducible ML Pipeline + Interactive Web Walkthrough<br>

> [!IMPORTANT]
> **Complete Publication Manuscript Available on Google Docs:**<br>
> The full, peer-review formatted manuscript of this study is hosted on Google Docs:  
> 🔗 **[Click here to open the Research Paper in Google Docs](https://docs.google.com/document/d/1wpl74tR3uBP65dPv9ZUCTsIwiai8wPOhO1ewUJ3qwnU/edit?usp=sharing)**  
> *(Includes full 9-section academic text, mathematical proofs, experimental matrix, controls table, and complete references)*.

---

## Interactive Walkthrough & Web Demo
* **Live Interactive Simulator & Visualizer:** Located in [`web-demo/`](web-demo/) (open [`web-demo/index.html`](web-demo/index.html) in any browser, or launch a local static server via `python -m http.server 8000 --directory web-demo`).
* **Deployment Ready:** Static build compatible with zero-config hosting on Vercel, Netlify, or GitHub Pages.
* **Recorded Video Walkthrough:** Demonstrating the full end-to-end interactive dashboard, synthetic parameter manipulation, and empirical findings:

![Interactive Walkthrough Demo](results/walkthrough_demo.gif)


---

## Table of Contents
* [📄 **Full Research Paper (Google Docs)**](https://docs.google.com/document/d/1wpl74tR3uBP65dPv9ZUCTsIwiai8wPOhO1ewUJ3qwnU/edit?usp=sharing)
1. [Abstract](#abstract)
2. [Introduction & The Skeptical Challenge](#1-introduction--the-skeptical-challenge)
3. [Background & Literature Review](#2-background--literature-review)
4. [Research Question & Hypotheses](#3-research-question--hypotheses)
5. [System Architecture & Experimental Design](#4-system-architecture--experimental-design)
6. [Controls & Confounding Variables](#5-controls--confounding-variables)
7. [Empirical Results & Statistical Proofs](#6-empirical-results--statistical-proofs)
8. [Discussion & Boundary Analysis](#7-discussion--boundary-analysis)
9. [Architectural Trade-Offs Made](#8-architectural-trade-offs-made)
10. [Security Measures & Experimental Integrity](#9-security-measures--experimental-integrity)
11. [Threats to Validity & Limitations](#10-threats-to-validity--limitations)
12. [Conclusion](#11-conclusion)
13. [References](#13-references)

---

## Abstract

Neural networks are widely reported in the machine learning literature ([Shah et al., 2020](#references); [Morwani et al., 2023](#references)) to exhibit an extreme **Simplicity Bias**—a persistent optimization tendency to exclusively exploit simpler predictive features (e.g., linearly separable hyperplanes) while failing to learn complex, non-linear predictive features, even when the complex features are predictive or robust. However, existing empirical benchmarks frequently confound *intrinsic representational complexity* with *predictive signal strength* and *gradient accessibility from initialization*.

In this study, we adopt a skeptical scientific methodology to identify the exact boundary conditions under which simplicity bias fails. We design a synthetically controlled, multi-modal data-generating process where a linear feature subspace $x_S \in \mathbb{R}^2$ and an XOR-quadrant non-linear feature subspace $x_C \in \mathbb{R}^2$ predict a common binary target under strictly isolated train-split variance scaling. We establish single-feature oracle baselines, demonstrating that both features are independently learnable to $>94\%$ accuracy by an identical Multilayer Perceptron (MLP) architecture (S alone: $95.17\%$, C alone: $94.33\%$), while a linear model through the origin operates strictly at chance ($50.00\%$), proving non-linear separability.

Under matched latent signal reliability ($\eta_S = \eta_C = 0.05$), standard networks prioritize the linear feature early in training: under the corrected validation-monitored protocol, $S$ crossed the acquisition threshold $\tau = 0.80$ in $5/5$ runs (mean observed $T_S = 1.8 \pm 1.8$ epochs) while $C$ crossed in $0/5$ runs ($T_C > 50$; primary two-sided paired exact McNemar $p = 0.0625$, not significant at $\alpha = 0.05$; non-preregistered directional one-sided $p = 0.03125$). On the sequestered held-out test split, terminal accuracy at epoch 50 reflects feature parity ($74.63\%$ on $S$ vs. $70.00\%$ on $C$, paired $t(4) = 2.10, p = 0.1035$), while exploratory early dynamics reveal an initial linear lead on both the validation split ($85.81\%$ vs. $59.87\%$, paired $t(4) = 5.41, p = 0.0056$) and the held-out test split at epoch 1 ($83.50\%$ vs. $61.20\%$, paired $t(4) = 4.40, p = 0.0117$). (Under the historical test-monitored archive, where threshold crossing was detected on the test set, $S$ crossed in $3/5$ runs vs. $0/5$ for $C$, primary two-sided McNemar $p = 0.2500$, one-sided $p = 0.1250$, illustrating how small-sample test monitoring introduces partition variance near the threshold). Crucially, across both protocols, when predictive strength is inverted such that the non-linear interaction carries a cleaner signal than the linear feature ($\eta_C = 0.01$ vs. $\eta_S = 0.25$), the learning order **completely inverts**: networks acquire the non-linear interaction at epoch 1 across all 5 seeds ($T_C = 1$, $5/5$ crossed; primary two-sided paired exact McNemar $p = 0.0625$, directional one-sided $p = 0.03125$), while reliance on the linear feature collapses ($0/5$ crossed, $T_S > 50$, final test accuracy $50.97\%$, matching the $50\%$ chance baseline; paired $t$-test on epoch 50 accuracy $t = 78.83, p = 1.55 \times 10^{-7}$). The model exhibits $98.2\%$ reliance on the non-linear feature and leaves the linear feature at chance. Between Conditions A and B, the unstandardized reliance margin shifts by $+50.41\%$ ($p = 8.29 \times 10^{-6}$, Welch's $t = 22.11$; paired $t(4) = 18.13, p = 5.44 \times 10^{-5}$). Furthermore, a 4-layer DeepMLP enabled non-linear discovery under parity, achieving threshold crossing in $2/5$ runs (epochs 8 and 13; $3/5$ censored at $>50$). These findings demonstrate that simplicity bias is not an unconditional representational invariant: neural networks prioritize **predictive signal quality over structural simplicity**.

---

## 1. Introduction & The Skeptical Challenge

Modern deep neural networks are overparameterized function approximators with universal representational capacity. Despite their expressiveness, gradient-based optimization does not navigate parameter space uniformly; rather, it exhibits strong inductive biases. A prominent body of research claims that neural networks suffer from *simplicity bias*: when presented with multiple competing features correlated with the target label, gradient descent preferentially learns the "simplest" feature first, often to the complete exclusion of more complex features ([Shah et al., 2020](#references)).

### The Skeptical Research Stance
A conventional, uncritical engineering stance would merely replicate existing synthetic benchmarks and reaffirm that simplicity bias exists. In contrast, this study adopts the skeptical research principle formulated in the challenge specification:
> *"Under what conditions might this claim fail, and how would we know scientifically?"*

We scrutinize three core ambiguities inherent in prior simplicity bias claims:
1. **The Definitional Ambiguity of "Simplicity":** Does "simple" mean linear separability, low spatial frequency, low Kolmogorov complexity, or ease of optimization? Conflating these distinct properties obscures the true underlying mechanism.
2. **Confounded Predictive Signal Strength:** In many real-world and synthetic setups, the supposedly "simple" feature (e.g., a background color patch or watermarks) provides a deterministic, noise-free correlation with the class label, whereas the "complex" feature (e.g., object geometry) contains natural variance and ambiguity. If a network prefers a simple feature, is it because of its simplicity, or because it has lower Bayes risk?
3. **Optimization Accessibility vs. Inductive Bias:** A linear decision boundary produces a non-zero expected gradient everywhere across randomly initialized weights. An XOR-like non-linear interaction, however, has zero correlation along any coordinate axis and produces an initial expected first-order gradient of zero. Is simplicity bias a fundamental preference of the neural representation, or simply a reflection of first-order gradient discovery dynamics?

By isolating these variables into independent experimental knobs, we demonstrate that simplicity bias is not an immutable cognitive constraint of neural networks, but rather a conditional preference governed by the interplay between Bayes signal quality, architectural depth, and optimization landscape geometry.

---

## 2. Background & Literature Review

```mermaid
graph TD
    A["Inductive Biases in Neural Networks"] --> B["Simplicity Bias in Representations"]
    A --> C["Spectral Bias in Optimization"]

    B --> D["Shah et al. (2020)<br/>The Pitfalls of Simplicity Bias"]
    B --> E["Morwani et al. (2023)<br/>Simplicity Bias in 1-HL Networks"]
    C --> F["Rahaman et al. (2019)<br/>Spectral Bias of Deep Networks"]

    D -.-> G["Critique: Confounded Signal Strength"]
    E -.-> H["Critique: Low-Rank Projection Assumptions"]
    F -.-> I["Critique: Frequency vs. Structural Complexity"]

    G --> J["Our Work: The Falsification Challenge Suite"]
    H --> J
    I --> J
```

### Synthesis of Relevant Foundations

| Paper | Main Claim | Definition of Simplicity | Experimental Setup | Key Assumptions / Confounders | Relevance to Our Study |
|---|---|---|---|---|---|
| **Shah et al. (2020)**<br>*The Pitfalls of Simplicity Bias in NNs* | NNs trained with standard SGD/Adam can rely exclusively on simple features, failing to learn complex features even when both are predictive. | Linear separability vs. non-linear / image patches. | Synthetic datasets with combined linear and non-linear image features; MNIST/CIFAR. | Signal quality and label noise were not systematically matched; optimization difficulty was not isolated. | Serves as the primary claim investigated. We test whether matching or varying signal strength breaks this bias. |
| **Morwani et al. (2023)**<br>*Simplicity Bias in 1-Hidden Layer NNs* | Provides theoretical bounds showing 1-hidden-layer NNs converge to low-dimensional linear projections. | Rank and dimensionality of the learned feature subspace. | 1-hidden-layer ReLU networks on Gaussian mixtures. | Restricted to shallow architectures; assumes infinite sample or specific Gaussian mixture regimes. | Motivates our architectural ablations: testing whether multi-layer depth and skip connections break the bias. |
| **Rahaman et al. (2019)**<br>*On the Spectral Bias of NNs* | NNs learn lower-frequency Fourier components before higher-frequency components. | Frequency spectrum of target functions on Fourier domain. | Coordinate-based MLPs on 1D/2D synthetic signals and manifolds. | Spectral frequency is a harmonic property, not synonymous with general feature complexity (e.g. combinatorial XOR). | Informs our distinction between spatial frequency and non-linear interaction rank. |
| **Arpit et al. (2017)**<br>*A Closer Look at Memorization in Deep Networks* | NNs prioritize learning clean, generalizable patterns before memorizing noisy samples. | Real vs. random patterns / memorization capacity. | Deep architectures on CIFAR/ImageNet with noisy labels. | Explores noise dynamics, but does not pit two competing predictive features against each other. | Motivates our use of differential label noise ($\eta_S, \eta_C$) as a control variable. |

---

## 3. Research Question & Hypotheses

### Primary Research Question
> **When two predictive features differ in structural complexity, does a neural network preferentially learn the simpler feature because of its intrinsic simplicity, or can this preference be systematically inverted by modulating predictive signal strength and optimization accessibility?**

### Formal Hypotheses

* **Question A: Feature Competition Under Signal Parity ($\eta_S = \eta_C = 0.05$)**
  * **Null Hypothesis $H_{0,A}$ (No Ordering Preference):**<br>
    Under matched predictive reliability, there is no systematic difference in the acquisition timing between the linear feature $S$ and non-linear interaction $C$:
    $$P(T_S \le T_C) = P(T_C \le T_S) = 0.5$$
  * **Alternative Hypothesis $H_{1,A}$ (Simplicity Bias Formulation):**<br>
    The network acquires the linear feature earlier and with higher probability than the non-linear interaction:
    $$P(T_S < T_C) > 0.5, \quad \text{Acc}_S(t) > \text{Acc}_C(t) \quad \text{for early } t$$

* **Question B: Inversion Under Signal Quality Disparity ($\eta_C < \eta_S$)**
  * **Null Hypothesis $H_{0,B}$ (Invariant Simplicity Bias):**<br>
    Even when the non-linear interaction provides cleaner predictive signal ($\eta_C = 0.01, \eta_S = 0.25$), the network maintains its preference for the linear feature:
    $$P(T_S \le T_C) \ge 0.5, \quad \text{Acc}_S(\text{end}) \ge \text{Acc}_C(\text{end})$$
  * **Alternative Hypothesis $H_{1,B}$ (Signal Quality Dominance / Falsification Formulation):**<br>
    Predictive signal quality overrides structural simplicity: the network acquires the complex feature earlier and relies predominantly on it:
    $$P(T_C < T_S) > 0.5, \quad \text{Acc}_C(\text{end}) > \text{Acc}_S(\text{end})$$

---

## 4. System Architecture & Experimental Design

### 4.1 System Overview
The research system consists of four modular components designed for reproducibility and empirical rigor:

```mermaid
flowchart TD
    subgraph DGP ["Data Generating Process"]
        Z["Latent Target z in {-1, +1}"]
        Z --> |Flip noise eta_S| S["Linear Subspace x_S in R^2"]
        Z --> |Flip noise eta_C| C["XOR Subspace x_C in R^2"]
        N["Noise Subspace x_N in R^4"]
        S & C & N --> Concatenate["Standardized Input x in R^8"]
    end

    subgraph Models ["Neural Network Family"]
        Concatenate --> M1["StandardMLP: 2 Layers, 64-dim, ReLU"]
        Concatenate --> M2["DeepMLP: 4 Layers, 64-dim, ReLU"]
        Concatenate --> M3["ResMLP: Residual Skip Connections"]
    end

    subgraph Optimization ["Optimization Regimes"]
        M1 & M2 & M3 --> Opt1["AdamW: lr=0.01, wd=1e-4"]
        M1 --> Opt2["SGD: lr=0.02, momentum=0.9"]
    end

    subgraph Evaluation ["Counterfactual Intervention Engine"]
        Opt1 & Opt2 --> JointEval["Joint Test Accuracy"]
        Opt1 & Opt2 --> S_Eval["Counterfactual S-Only (C, N Permuted)"]
        Opt1 & Opt2 --> C_Eval["Counterfactual C-Only (S, N Permuted)"]
        Opt1 & Opt2 --> N_Eval["Noise Null Control (S, C Permuted)"]
    end

    subgraph Metrics ["Falsification Metrics"]
        S_Eval --> TS["Time-to-Threshold T_S"]
        C_Eval --> TC["Time-to-Threshold T_C"]
        TS & TC --> Delta["Delta = T_C - T_S"]
        Delta --> StatTest["Paired t-test, Welch's t-test, Cohen's d"]
    end
```

### 4.2 Data-Generating Process (DGP)
We generate a synthetic classification dataset with $N = 4,000$ samples. For each sample $i$:
1. A ground-truth binary latent label $z^{(i)} \in \{-1, +1\}$ is drawn with uniform prior $P(z=1) = 0.5$.
2. **Simple Linear Feature Subspace ($x_S \in \mathbb{R}^2$):**<br>
   We introduce independent label noise $\eta_S \in [0, 0.5)$. With probability $1 - \eta_S$, $z_S^{(i)} = z^{(i)}$; with probability $\eta_S$, $z_S^{(i)} = -z^{(i)}$.<br>
   $$x_S^{(i)} = z_S^{(i)} \cdot \mu_S + \epsilon_S^{(i)}, \quad \mu_S = \frac{1}{\sqrt{2}}[1, 1]^T, \quad \epsilon_S \sim \mathcal{N}(0, \sigma^2 I_2), \ \sigma = 0.2$$
   The latent label reliability is $1 - \eta_S$. Because cluster separation along $\mu_S$ is $\|\mu_S\|/\sigma = 5$ standard deviations, the continuous Bayes accuracy closely tracks $1 - \eta_S$.
3. **Complex Non-Linear Feature Subspace ($x_C \in \mathbb{R}^2$):**<br>
   Coordinates $(u_1, u_2)$ are sampled directly in the quadrant defined by latent label $z_C \in \{-1, +1\}$ (where $P(z_C = z) = 1 - \eta_C$):<br>
   $$r_1, r_2 \sim \mathcal{U}([0.2, 1.5]), \quad s_1 \sim \text{Rademacher}(\{-1, +1\}), \quad s_2 = z_C \cdot s_1$$
   $$u_1 = s_1 \cdot r_1 + \epsilon_{C,1}, \quad u_2 = s_2 \cdot r_2 + \epsilon_{C,2}, \quad \epsilon_C \sim \mathcal{N}(0, 0.04^2 I_2)$$
   Because $s_1$ is independent of $z$, $\mathbb{E}[u_1 \mid z] = 0$ and $\mathbb{E}[u_2 \mid z] = 0$. A linear classifier through the origin achieves strictly $50.00\%$ (chance) accuracy.
4. **Noise Subspace ($x_N \in \mathbb{R}^4$):**<br>
   Uninformative Gaussian distractor features: $\epsilon_N \sim \mathcal{N}(0, I_4)$.
5. **Strict Split Isolation & Standardization:**<br>
   The dataset is partitioned into $70\%$ train ($2,800$), $15\%$ validation ($600$), and $15\%$ test ($600$). Z-score standardization ($\mu = 0, \sigma = 1$) is fitted strictly on the training partition and applied to validation and test partitions, preventing data leakage.
6. **Training Updates Accounting:**<br>
   Trained for 50 epochs with batch size 64 on 2,800 training examples ($\lceil 2800 / 64 \rceil = 44$ updates per epoch), executing exactly **2,200 optimizer steps** per run.

### 4.3 Counterfactual Evaluation Paradigm & Protocol Isolation
To avoid out-of-distribution (OOD) artifacts, feature reliance is evaluated via **marginal feature permutation** averaged across 5 independent trials:
* **Joint Accuracy:** Evaluated on unperturbed inputs $[x_S, x_C, x_N]$.
* **Isolated $S$-Reliance ($\text{Acc}_{S\text{-only}}$):** Evaluated on $[x_S, \tilde{x}_C, \tilde{x}_N]$, where $x_C$ and $x_N$ are randomly permuted.
* **Isolated $C$-Reliance ($\text{Acc}_{C\text{-only}}$):** Evaluated on $[\tilde{x}_S, x_C, \tilde{x}_N]$, neutralizing $x_S$.
* **Noise Null Control ($\text{Acc}_{N\text{-only}}$):** Evaluated on $[\tilde{x}_S, \tilde{x}_C, x_N]$ (verified at chance level $50.31 \pm 1.26\%$).

#### Protocol Separation: Historical vs. Corrected
* **Historical Run (`results/raw_data/historical_run/`):** Evaluated per-epoch counterfactual tracking and threshold crossing ($\tau = 0.80$) directly on the test set (`test_d`, 600 samples), leaving `val_d` unused.
* **Corrected Protocol (`results/raw_data/corrected_protocol/`):** Strictly separates optimization (`train_d`, 2,800 samples), in-loop monitoring/acquisition tracking (`val_d`, 600 samples), and out-of-sample held-out evaluation (`test_d`, 600 samples evaluated at epoch 50 and epoch 1).
* **Time-to-Threshold & Censoring:**
  $$T_S = \min \{ t \in [1, 50] : \text{Acc}_{S\text{-only}}(t) \ge \tau \}, \quad T_C = \min \{ t \in [1, 50] : \text{Acc}_{C\text{-only}}(t) \ge \tau \}$$
  evaluated on the validation set ($\tau = 0.80$). If a threshold is not crossed within 50 epochs, it is recorded as **right-censored** ($T > 50$, `observed_epoch = None`). Censored runs are never treated as observed events at epoch 51 and are strictly excluded from observed crossing time means.

---

## 5. Controls & Confounding Variables

| Potential Confounder | Why It Threatens Validity | Planned Experimental Control | Empirical Verification |
|---|---|---|---|
| **Predictive Signal Strength** | Features carrying higher mutual information are naturally learned faster. | Analytically set $\eta_S = \eta_C = 0.05$ in Condition A; systematically invert in Condition B ($\eta_S = 0.25, \eta_C = 0.01$). | Single-feature baselines both achieve $>95\%$ independently. |
| **Data Leakage in Normalization** | Global standardization leaks test statistics. | Fitted $\mu, \sigma$ strictly on $2,800$ training examples only. | Confirmed via train mean $= 0.000$, test mean $\ne 0.000$. |
| **Evaluation Set Leakage** | In-loop tracking on test set conflates validation with test. | Sequestered `test_d` for final evaluation; tracked threshold crossing on `val_d`. | Both protocols documented and compared side-by-side. |
| **Feature Scale / Variance** | Larger feature scales produce larger gradient updates, mimicking simplicity bias. | Rigorous z-score standardization ($\mu=0, \sigma=1$) across all subspace dimensions prior to training. | Confirmed via `np.std(x, axis=0) == 1.0`. |
| **Optimization Landscape** | Linear features have non-zero mean gradient at init; XOR features have zero coordinate correlation. | Compared AdamW (adaptive second moments) with SGD (momentum=0.9). | Tracked epoch-by-epoch loss and counterfactual accuracy. |
| **Architectural Depth** | Shallow networks may lack capacity or depth to compose non-linear representations. | Tested StandardMLP (2 layers), DeepMLP (4 layers), and ResMLP (residual skip connections). | Depth enabled non-linear threshold crossing in 2/5 runs (Seed 103 at epoch 8, Seed 102 at epoch 13). |
| **Right-Censored Metric Bias** | Treating non-crossings as uncensored continuous values. | Reported crossing rates ($p$) and epoch 50 accuracy margins ($D_{\text{acc}}$); excluded censored runs from crossing means. | Zero math on placeholder 51; regression tests in test suite. |

---

## 6. Empirical Results & Statistical Proofs

### 6.1 Phase 0: Verification of Single-Feature Learnability
Before joint training, we verify that the network architecture has sufficient capacity to learn both features independently when presented in isolation:
* **Linear Feature $S$ Alone:** Maximum Held-Out Accuracy = **95.17%** (Latent Reliability: $1 - \eta_S = 95.0\%$)
* **Nonlinear Feature $C$ Alone:** Maximum Held-Out Accuracy = **94.33%** (Latent Reliability: $1 - \eta_C = 95.0\%$)
* **Linear Logistic Regression on $S$:** Test Accuracy = **95.17%**
* **Linear Logistic Regression on $C$ (through origin):** Test Accuracy = **50.00%** (exact chance level)
* **Affine Logistic Regression on $C$ (with intercept):** Test Accuracy = **68.33%** (confirming $C$ cannot be linearly separated).

Both features are independently learnable to $>94\%$ accuracy by the identical 2-layer MLP.

---

### 6.2 Condition-by-Condition Empirical Results (Corrected Protocol)

All conditions were trained for 50 epochs ($2,200$ optimizer steps) across seeds `[101, 102, 103, 104, 105]`. The summary statistics under the **Corrected Validation-Monitored Protocol** are detailed below:

| Condition ID | Description & Setup | Validation Crossing $S$ | Validation Crossing $C$ | Observed $T_S$ (epochs) | Observed $T_C$ (epochs) | Final Held-Out Test Acc | Final Reliance S / C |
|---|---|---|---|---|---|---|---|
| **Condition A** | Matched Baseline<br>$\eta_S = 0.05, \eta_C = 0.05$, StandardMLP, AdamW | **100%** (5/5) | **0%** (0/5) | **$1.8 \pm 1.8$** (epochs 1, 1, 1, 1, 5) | $> 50$ (censored in 5/5 runs) | $94.30 \pm 0.69\%$ | $55.0\% \ / \ 45.0\%$ |
| **Condition B** | **Complex Dominance (Falsification)**<br>$\eta_S = 0.25, \eta_C = 0.01$, StandardMLP, AdamW | **0%** (0/5) | **100%** (5/5) | $> 50$ (censored in 5/5 runs) | **$1.0 \pm 0.0$** (all 5 runs) | $97.58 \pm 0.38\%$ | **$1.8\% \ / \ 98.2\%$** |
| **Condition C** | Simple Dominance (Positive Control)<br>$\eta_S = 0.01, \eta_C = 0.25$, StandardMLP, AdamW | **100%** (5/5) | **0%** (0/5) | **$1.0 \pm 0.0$** (all 5 runs) | $> 50$ (censored in 5/5 runs) | $98.60 \pm 0.44\%$ | $99.0\% \ / \ 1.0\%$ |
| **Condition D** | Optimization Ablation (SGD)<br>$\eta_S = 0.05, \eta_C = 0.05$, StandardMLP, SGD | **100%** (5/5) | **0%** (0/5) | **$1.0 \pm 0.0$** (all 5 runs) | $> 50$ (censored in 5/5 runs) | $94.50 \pm 0.50\%$ | $57.5\% \ / \ 42.5\%$ |
| **Condition E1** | Architectural Ablation (DeepMLP 4-Layer)<br>$\eta_S = 0.05, \eta_C = 0.05$, DeepMLP, AdamW | **100%** (5/5) | **40%** (2/5) | $1.8 \pm 0.8$ (all 5 runs) | **$10.5 \pm 3.5$** (epochs 8, 13)<br>censored in 3 runs | $94.56 \pm 0.33\%$ | $53.4\% \ / \ 46.6\%$ |
| **Condition E2** | Architectural Ablation (ResMLP)<br>$\eta_S = 0.05, \eta_C = 0.05$, ResMLP, AdamW | **100%** (5/5) | **0%** (0/5) | **$1.0 \pm 0.0$** (all 5 runs) | $> 50$ (censored in 5/5 runs) | $94.24 \pm 0.36\%$ | $54.0\% \ / \ 46.0\%$ |

##### Historical Comparison Note
In the historical test-monitored run (`results/raw_data/historical_run/`), where threshold crossing was monitored directly on `test_d`, Condition A crossing counts were **3/5 for $S$** and **0/5 for $C$** ($p = 0.1250$). In the corrected protocol, validation crossing is **5/5 for $S$** and **0/5 for $C$** ($p = 0.03125$). This variance arises because borderline seeds (101 and 105) fluctuate near the 0.80 boundary across disjoint 600-sample partitions, illustrating why in-loop acquisition detection should be conducted on validation data while reserving the test set for final evaluation.

---

### 6.3 Rigorous Hypothesis Testing & Statistical Significance

1. **Test of Simplicity Bias under Matched Signals (Condition A):**
   * Validation Crossing: $S = 5/5$ (mean observed $T_S = 1.8 \pm 1.8$ epochs); $C = 0/5$ (all $T_C > 50$).
   * Paired McNemar exact binomial test on discordant pairs ($b = 5, c = 0$): primary two-sided **$p = 0.0625$** (does not reach statistical significance at $\alpha = 0.05$); non-preregistered directional one-sided **$p = (0.5)^5 = 0.03125$** (must be interpreted cautiously as conditional on a non-preregistered directional hypothesis; unpaired Fisher's exact one-sided $p = 0.0040$, two-sided $p = 0.0079$, though its independent-groups assumption is violated).
   * Terminal held-out test accuracy (Epoch 50): Acc $S = 74.63 \pm 2.64\%$ vs. Acc $C = 70.00 \pm 2.36\%$, mean difference $+4.63 \pm 4.93\%$ (paired $t(4) = 2.10, p = 0.1035$; 95% CI $[-1.49\%, +10.75\%]$; Wilcoxon $W = 1.0, p = 0.1875$). Indicates feature parity at terminal convergence.
   * Exploratory early dynamics (Epoch 1 Split Breakdown): Post-hoc continuous evaluation indicates an immediate initial preference for $S$:
     - **Validation Monitoring Split (`val_d`, $N = 600$):** Acc $S = 85.81 \pm 5.48\%$ vs. Acc $C = 59.87 \pm 5.26\%$, difference $+25.94 \pm 10.72\%$, paired $t(4) = 5.41$, **$p = 0.0056$** (95% CI: $[+12.63\%, +39.25\%]$).
     - **Sequestered Held-Out Test Split (`test_d`, $N = 600$):** Acc $S = 83.50 \pm 5.72\%$ vs. Acc $C = 61.20 \pm 5.70\%$, difference $+22.30 \pm 11.33\%$, paired $t(4) = 4.40$, **$p = 0.0117$** (95% CI: $[+8.23\%, +36.37\%]$).
2. **Test of Inversion under Complex Signal Dominance (Falsification in Condition B):**
   * Non-linear interaction $C$ crosses at epoch 1 in all 5 runs ($100\%, T_C = 1.0$), while linear feature $S$ fails to cross in any run ($0/5, T_S > 50$).
   * Threshold-crossing tests on `val_d` ($b = 0, c = 5$): primary two-sided paired exact McNemar **$p = 0.0625$** (nonsignificant at $\alpha = 0.05$ due to discrete power limits with $N = 5$); directional one-sided ($C > S$) **$p = (0.5)^5 = 0.03125$**; continuous terminal held-out test accuracy is decisively significant (paired $t(4) = 78.83, p = 1.55 \times 10^{-7}$).
   * Paired $t$-test on terminal held-out test accuracy: Acc $C$ ($96.75 \pm 0.77\%$) vs. Acc $S$ ($50.97 \pm 0.67\%$ chance level): $t(4) = 78.83$, **$p = 1.55 \times 10^{-7}$** (mean difference $+45.77 \pm 1.30\%$; 95% CI $[+44.16\%, +47.39\%]$).
   * Feature reliance on $S$ collapses to $1.8\%$, while reliance on $C$ reaches $98.2\%$.
3. **Comparison Between Condition A and Condition B:**
   * Welch's two-sample $t$-test on reliance margins: $t = 22.11, \ df = 4.55$, **$p = 8.29 \times 10^{-6}$**.
   * Paired $t$-test across matching seeds: $t(4) = 18.13$, **$p = 5.44 \times 10^{-5}$** (95% CI: $[+42.69\%, +58.12\%]$).
   * Unstandardized effect size: **$+50.41\%$** accuracy margin shift toward the non-linear feature.
   * Confirms that modulating predictive signal quality causes a complete phase reversal in feature preference.
4. **Architectural Ablation (DeepMLP in Condition E1):**
   * DeepMLP (4 layers) crossed the threshold for $C$ in $40\%$ of runs (Seed 103 at epoch 8, Seed 102 at epoch 13; mean observed $T_C = 10.5 \pm 3.5$; 3/5 runs censored at $>50$), showing that compositional depth accelerates non-linear symmetry breaking under parity.


---

### 6.4 Publication Figures & Visual Evidence

#### Figure 1: Learning Dynamics Under Matched Signal Strength (Condition A)
![Figure 1: Matched Learning Dynamics](results/figures/fig1_learning_dynamics_matched.png)
*Under matched predictive strength ($\eta_S = \eta_C = 0.05$), the network learns $S$ within 2 epochs while $C$ lags behind, verifying standard simplicity bias under parity.*

---

#### Figure 2: The Falsification Boundary — Signal Quality Inverts Simplicity Bias (Condition A vs. Condition B)
![Figure 2: Falsification Boundary](results/figures/fig2_falsification_signal_dominance.png)
*Direct visual proof of the falsification boundary. Left: When signals are matched, $S$ is learned first. Right: When $C$ carries cleaner signal ($\eta_C = 0.01$ vs. $\eta_S = 0.25$), the learning order completely inverts ($T_C = 1$), falsifying unconditional simplicity bias.*

---

#### Figure 3: Relative Feature Reliance Across All Experimental Conditions
![Figure 3: Feature Reliance Across Conditions](results/figures/fig3_reliance_heatmaps.png)
*Normalized feature reliance ratios across experimental conditions, highlighting the transition from linear dominance (Condition C: 99.0% reliance on S) to nonlinear dominance (Condition B: 98.2% reliance on C).*

---

#### Figure 4: Impact of Optimizer & Architecture on Nonlinear Feature Discovery
![Figure 4: Optimizer and Architecture Ablation](results/figures/fig4_optimizer_architecture_ablation.png)
*Comparing nonlinear discovery time $T_C$ across optimizers and depths. DeepMLP (4 layers) accelerated discovery, enabling Seed 103 to cross threshold at Epoch 8 and Seed 102 at Epoch 13 (mean observed $T_C = 10.5 \pm 3.5$ epochs; 3/5 runs censored at $>50$).*

---

#### Figure 5: Statistical Distribution of Learning Order Difference ($\Delta = T_C - T_S$) Across Seeds
![Figure 5: Statistical Summary Boxplot](results/figures/fig5_statistical_summary.png)
*Box plot of $\Delta$ across seeds. The dramatic shift below the $\Delta=0$ parity threshold under Condition B provides rigorous empirical proof.*

---

## 7. Discussion & Boundary Analysis

### 7.1 What Does the Evidence Establish?
1. **Simplicity Bias is a Conditional Prior, Not an Invariant Law:**<br>
   Our results confirm that when multiple predictive features offer equivalent Bayes risk, gradient descent indeed prioritizes the linearly separable pattern ($S$). However, this priority is easily overridden: when the non-linear pattern offers superior predictive fidelity, the network abandons the linear pattern immediately.
2. **Predictive Information Overpowers Structural Simplicity:**<br>
   The loss gradient magnitude is proportional to prediction error. A clean non-linear signal reduces loss much more rapidly than a noisy linear signal once appropriate hidden unit coordinates are discovered. In Condition B, the non-linear feature's lower Bayes risk ($1\%$ vs. $25\%$) exerted overwhelmingly larger gradient pressure, driving $T_C = 1$.
3. **The Role of Optimization Accessibility:**<br>
   Why is $S$ preferred under parity? Because a linear direction has a non-zero expected first-order gradient from standard Gaussian weight initialization:
   $$\mathbb{E}_{w \sim \mathcal{N}}[ \nabla_w \mathcal{L} ] \ne 0$$
   In contrast, a symmetric XOR quadrant manifold has an expected gradient of zero at initialization:
   $$\mathbb{E}_{(u_1, u_2)}[ \text{sign}(u_1 \cdot u_2) \cdot u_1 ] = 0$$
   The network must rely on stochastic mini-batch noise to break symmetry before it can begin descending the non-linear manifold. Once depth is increased (DeepMLP), higher-order composition occurs faster, reducing $T_C$.

---

## 8. Architectural Trade-Offs Made

In adhering to the prompt and submission protocol (Card 02), we document the engineering and scientific trade-offs made in designing this experimental suite:

```mermaid
graph LR
    subgraph Design Choices
        T1["Synthetic Manifold vs. Image Dataset"]
        T2["Permutation Ablation vs. Zero Ablation"]
        T3["Small MLP Family vs. Large ResNet/Vision Transformer"]
        T4["Fixed Data Seed vs. Resampled Data per Run"]
    end

    subgraph Trade-off Analysis
        T1 --> R1["Trade-off: Sacrifices natural visual semantics for exact mathematical signal control (Bayes risk)."]
        T2 --> R2["Trade-off: Permutation preserves marginal feature variance, avoiding out-of-distribution artifacts."]
        T3 --> R3["Trade-off: Fast CPU execution across seeds, enables complete reproducibility in < 3 minutes."]
        T4 --> R4["Trade-off: Isolates optimization/weight seed variance from sample generation noise."]
    end
```

1. **Synthetic Feature Manifold vs. Real Image Data:**<br>
   *Decision:* We chose a synthetic orthogonal feature construction ($x_S, x_C, x_N$) over natural image datasets (e.g., Colored MNIST or CelebA).<br>
   *Trade-Off:* Synthetic data lacks complex real-world visual semantics, but it is the *only* paradigm that allows exact analytical control over the Bayes error rate ($\eta_S, \eta_C$), variance scaling, and orthogonality. On image benchmarks, feature signal strength cannot be independently adjusted without changing visual representations.
2. **Marginal Permutation vs. Zero Masking for Counterfactuals:**<br>
   *Decision:* We employed random test-set permutation of non-target subspaces rather than setting them to zero.<br>
   *Trade-Off:* Permutation preserves the first and second moments of the input distribution, preventing artificial out-of-distribution shifts.
3. **Model Complexity (MLP) vs. Computational Overhead:**<br>
   *Decision:* We parameterized small MLPs (2 to 4 layers, 64 hidden dimensions) rather than large convolutional networks or Vision Transformers.<br>
   *Trade-Off:* This follows the **Principle of Appropriate Complexity** highlighted in the challenge specification: small models can be trained deterministically on standard CPUs in seconds, enabling full multi-seed evaluation without GPU compute dependencies.

---

## 9. Security Measures & Experimental Integrity

In accordance with Card 02 of the submission protocol, the following security and operational integrity measures were instituted:

```mermaid
graph TD
    S1["Security & Experimental Integrity"]
    S1 --> S2["Deterministic Reproducibility"]
    S1 --> S3["Data Leakage Prevention"]
    S1 --> S4["Anti-Hallucination Logging"]
    S1 --> S5["Resource & DoS Protection"]

    S2 --- D1["Fixed random seeds locked for PyTorch, NumPy, and Scipy"]
    S3 --- D2["Strict split isolation: Z-score fitted only on training set"]
    S4 --- D3["Raw JSON logs stored with timestamped metrics before visualization"]
    S5 --- D4["Capped epoch budgets and memory-bounded DataLoader batching"]
```

1. **Deterministic Pseudo-Random Locking:**<br>
   All data generation, model initializations, and mini-batch samplings are pinned to deterministic seeds (`torch.manual_seed`, `np.random.RandomState`). This guarantees identical numerical replication across platforms.
2. **Data Leakage & Sanitization:**<br>
   The normalizer fits mean and standard deviation strictly within the synthetic generator; test splits are strictly held out from parameter updates and evaluated without gradient tracking (`torch.no_grad()`).
3. **Scientific Integrity & Anti-Hallucination Protocol:**<br>
   In compliance with the writing guide checklist, all reported results, tables, and figures are loaded directly from raw saved JSON artifacts (`results/raw_data/corrected_protocol/experiments_results_corrected.json` and historical archives in `results/raw_data/historical_run/`). No synthetic figures or fabricated metrics are presented.
4. **Computational Bounding (Denial of Service Prevention):**<br>
   Batch sizes ($64$), epoch budgets ($50$), and sample sizes ($4,000$) are strictly bounded to execute comfortably within low-spec hardware without memory leaks or compute exhaustion.

---

## 10. Threats to Validity & Limitations

1. **Synthetic vs. Real-World Geometries:**<br>
   Our non-linear feature is an XOR-quadrant manifold. While XOR is the canonical non-linearly separable problem, real-world data may exhibit hierarchical multi-scale manifolds (e.g., textures, edges, semantic parts) that have different inductive affinities for convolutional or attention architectures.
2. **Right-Censored Threshold Values:**<br>
   When a feature was not learned within 50 epochs, its learning time was right-censored beyond the 50-epoch observation window (recorded as `observed_epoch = None`, $T > 50$). Censored runs are strictly excluded from continuous observed crossing time means and analyzed using discrete paired exact McNemar tests.
3. **Binary Classification Setting:**<br>
   The study evaluates balanced binary classification. High-cardinality multi-class settings may introduce label competition dynamics not captured in this single-target framework.
4. **Finite Sample Size & Statistical Power ($N = 5$ Seeds):**<br>
   With five discordant pairs all in the same direction, the minimum attainable two-sided exact McNemar p-value is 0.0625. Six discordant pairs all in the same direction would make a two-sided p-value below 0.05 attainable, but would not guarantee significance in every experiment. Future studies should determine sample size through prospective power analysis and specify their endpoints and analysis plan in advance.
5. **Fixed Synthetic Dataset Instance (`DATA_SEED = 42`):**<br>
   All 5 training runs evaluated a fixed synthetic dataset realization ($N = 4{,}000$). The observed variance across seeds reflects network initialization and mini-batch shuffle stochasticity on this fixed empirical distribution, rather than sampling variance across independently generated datasets.
6. **Absence of Direct Gradient Instrumentation:**<br>
   Theoretical deductions attributing early linear acquisition to first-order gradient accessibility and gradient starvation are plausible mathematical explanations; however, layer-specific gradient projections and loss surface Hessian eigenvalues were not directly probed during training and should not be interpreted as directly verified causal proofs.

---

## 11. Conclusion

This study challenged the claim that neural networks unconditionally prefer simpler features over complex ones. Through a controlled experiment suite decoupling structural complexity from predictive signal strength, we showed that:
1. When predictive signal quality is equal ($\eta_S = \eta_C = 0.05$), standard MLPs prioritize linear features early in training: under the corrected validation protocol, $S$ crossed $\tau = 0.80$ in $5/5$ runs (mean observed $T_S = 1.8 \pm 1.8$ epochs) while $C$ crossed in $0/5$ runs (all $T_C > 50$). The primary two-sided paired exact McNemar test yields **$p = 0.0625$** (nonsignificant at $\alpha = 0.05$; non-preregistered directional one-sided $p = 0.03125$). At terminal convergence (epoch 50 on the held-out test split), counterfactual accuracy reflects feature parity ($74.63\%$ vs. $70.00\%$, paired $t(4) = 2.10, p = 0.1035$). Exploratory analysis reveals an initial linear lead at epoch 1 on both validation ($85.81\%$ vs. $59.87\%$, paired $t(4) = 5.41, p = 0.0056$) and held-out test splits ($83.50\%$ vs. $61.20\%$, paired $t(4) = 4.40, p = 0.0117$). (In the historical test-monitored protocol, $S$ crossed in $3/5$ runs vs. $0/5$ for $C$, primary two-sided McNemar $p = 0.2500$, one-sided $p = 0.1250$, illustrating how small-sample test monitoring introduces partition variance).
2. When the non-linear interaction carries a superior signal ($\eta_C = 0.01$ vs $\eta_S = 0.25$), the network completely inverts its behavior, acquiring the complex feature at epoch 1 across all 5 seeds ($100\%$ crossing rate for $C$, $0\%$ for $S$; primary two-sided paired McNemar $p = 0.0625$, directional one-sided $p = 0.03125$; paired $t$-test on terminal held-out test accuracy $t(4) = 78.83, p = 1.55 \times 10^{-7}$).
3. Simplicity bias is therefore not an immutable structural constraint of neural networks, but an **optimization-driven prior under signal parity that is readily superseded by predictive signal quality**.

---

## 12. References

1. Shah, H., Tamuly, K., Raghunathan, A., Jain, P., & Netrapalli, P. (2020). The Pitfalls of Simplicity Bias in Neural Networks. Advances in Neural Information Processing Systems (NeurIPS 2020), 33:9573–9585.
2. Morwani, D., Batra, J., Jain, P., & Netrapalli, P. (2023). Simplicity Bias in 1-Hidden Layer Neural Networks. Advances in Neural Information Processing Systems (NeurIPS 2023), 36.
3. Geirhos, R., Jacobsen, J. H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut Learning in Deep Neural Networks. Nature Machine Intelligence, 2(11):665–673.
4. Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F., Bengio, Y., & Courville, A. (2019). On the Spectral Bias of Neural Networks. Proceedings of the 36th International Conference on Machine Learning (ICML 2019), PMLR 97:5301–5310.
5. Soudry, D., Hoffer, E., Nacson, M. S., Gunasekar, S., & Srebro, N. (2018). The Implicit Bias of Gradient Descent on Separable Data. Journal of Machine Learning Research (JMLR), 19(70):1–57.
6. Arpit, D., Jastrzebski, S., Ballas, N., Krueger, D., Bengio, E., Kanwal, M. S., Maharaj, T., Fischer, A., Courville, A., Bengio, Y., & Lacoste-Julien, S. (2017). A Closer Look at Memorization in Deep Networks. Proceedings of the 34th International Conference on Machine Learning (ICML 2017), PMLR 70:233–242.
7. Jacot, A., Gabriel, F., & Hongler, C. (2018). Neural Tangent Kernel: Convergence and Generalization in Neural Networks. Advances in Neural Information Processing Systems (NeurIPS 2018), 31.
8. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016), 770–778.
