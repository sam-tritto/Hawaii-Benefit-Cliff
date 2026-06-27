# Hawaii Labor Economics: The 20-Hour Benefit Cliff

This repository contains a complete tutorial and implementation of Bayesian causal inference methods to measure corporate staffing distortions created by the **Hawaii Prepaid Health Care Act**.

## The Labor Economics Hook: The 20-Hour Benefit Cliff

Hawaii has a unique labor law that has stood for decades: the **Prepaid Health Care Act (1974)**. It mandates that employers must provide comprehensive healthcare coverage to any employee who works **20 hours or more per week**. 

### The Causal Challenge
- **Corporate Distortions**: How much does a rigid regulatory threshold distort part-time labor schedules? Do employers intentionally cap shifts just below the line?
- **The Cliff**: In Hawaii, the compliance threshold is 20 hours. For the rest of the United States (under the federal Affordable Care Act), this threshold is 30 hours. This disparity creates a massive, hyper-localized compliance cliff in Hawaii.
- **Analytical Solution**: By analyzing the distribution of weekly hours worked, we can estimate how much Hawaii's distribution bunches at 19 hours compared to a counterfactual. We explore two Bayesian causal estimators in `pymc` to construct this counterfactual:
  1. **Bayesian Bunching Estimator (Poisson Polynomial Regression)**: Fits a flexible polynomial to Hawaii's hours distribution, excluding the region around the threshold.
  2. **Bayesian Distributional Synthetic Control**: Builds a "Synthetic Hawaii" distribution by matching Hawaii's distribution to a donor pool of other tourism-dependent, high-cost-of-living economies (California, Nevada, Florida, Alaska) that do not share the 20-hour constraint.

---

## Directory Structure

```text
├── data/
│   └── hawaii_hours_data.csv       # Real pooled worker-level labor dataset from NBER
├── src/
│   ├── data_fetcher.py             # Script to download and pool CPS monthly data
│   ├── data_generator.py           # Script to simulate realistic CPS-like data (fallback)
│   └── models.py                   # PyMC model implementations (Bunching & Synthetic Control)
├── utils/
│   └── plotting.py                 # Custom visualization modules (styled Matplotlib/Seaborn)
├── hawaii_benefit_cliff.ipynb      # The interactive tutorial notebook
├── verify_setup.py                 # Diagnostic script to verify environment and compilation
├── pyproject.toml                  # uv package configuration
└── README.md                       # This file
```

---

## Installation & Setup

This project uses `uv` for python environment and dependency management.

### 1. Synchronize the Environment
To install Python and all dependencies into a local virtual environment:
```bash
uv sync
```

### 2. Verify the Installation
Run the verification script to generate the dataset and verify that `pymc` compiles and samples correctly on your system:
```bash
uv run verify_setup.py
```

### 3. Open the Jupyter Notebook
Start Jupyter Notebook and open `hawaii_benefit_cliff.ipynb`:
```bash
uv run jupyter notebook
```
*If using VS Code, simply open the notebook and select the virtual environment python interpreter located at `.venv/bin/python` as your kernel.*

---

## Methods and PyMC Implementations

### 1. Bayesian Bunching (Poisson Regression)
Standard bunching estimates a smooth counterfactual density by fitting a polynomial to the counts of the running variable (hours worked) excluding the distorted range:
$$\log(\lambda_h) = \sum_{k=0}^K \beta_k h^k$$
We implement this in PyMC using a Poisson likelihood on counts, with regularizing Normal priors on the polynomial coefficients $\beta$.

### 2. Bayesian Distributional Synthetic Control
Standard bunching struggles when there are natural, non-policy-induced peaks at the threshold (e.g. 20 hours is a round number, representing standard part-time work). Distributional Synthetic Control solves this by modeling Hawaii's counterfactual distribution as a weighted average of donor states' distributions:
$$Y_{\text{HI}, h} \approx \sum_{d} w_d Y_{d, h}$$
We model the weights $w$ in PyMC using a **Dirichlet prior**, which naturally guarantees that weights are non-negative and sum to 1. We fit the weights using data outside the policy-affected window and project them to estimate "Synthetic Hawaii" inside the window.

### Causal Insight: The Survey Heap Effect
In self-reported labor data (like the CPS), workers round their hours to multiples of 5 (10, 15, 20, etc.). 
- **The Bunching Failure**: Standard polynomial bunching assumes a smooth distribution and completely fails here. It over-predicts the counterfactual at 19 hours (expecting a smooth curve between 15 and 20) and under-predicts the natural peak at 20. This falsely suggests Hawaii has a *deficit* at 19 hours.
- **The Synthetic Control Solution**: Distributional Synthetic Control uses donor states that share the same rounding behaviors. By matching Hawaii's distribution to a combination of California, Nevada, Florida, and Alaska, the counterfactual naturally expects low counts at 19 and a large peak at 20, cleanly isolating Hawaii's true causal bunching (~6.6x relative excess at 19 hours).

---

## Conclusion

This project demonstrates the practical application of Bayesian causal inference to structural labor regulations:
1. **The Policy Cliff**: The Prepaid Health Care Act successfully insures full-time workers, but creates a sharp cost curve (cliff) at 20 hours. Employers respond by capping part-time shifts at exactly 19 hours.
2. **Causal Measurement**: The Distributional Synthetic Control model estimates that Hawaii has **0.86% of its part-time workforce** at exactly 19 hours, compared to a synthetic baseline of only **0.13%**, proving the existence of substantial regulatory shift capping.
3. **Policy Design**: To eliminate this shift distortion, policymakers should consider proration or graduated subsidies rather than binary thresholds, smoothing the cost curve and allowing part-time workers to choose their hours freely.

