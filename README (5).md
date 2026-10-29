# Calibrated and Explainable Random Forest Classification of Dengue

This repository contains a Google Colab / Jupyter research notebook supporting the manuscript **“Calibrated and Explainable Random Forest Classification of Dengue Using Routine Hematological Data.”** It implements a retrospective, dataset-specific evaluation of dengue outcome labels using routine hematological and demographic variables.

> **Research use only.** This work is not a validated dengue diagnostic test or a clinical decision-support system. The source dataset does not document a sufficiently verifiable diagnostic reference standard or provide patient/visit identifiers. Reported classification results should not be interpreted as independently verified clinical diagnostic accuracy.

## Repository contents

```text
.
├── README.md
└── Dengue_Reviewer_Revision_Step1 (3).ipynb
```

The notebook contains the main model-development workflow, additional analyses undertaken during peer review, and code to export tables and figures. The original dataset is **not included** in this repository.

## Dataset

- **Source:** [Predictive Clinical Dataset for Dengue Fever Using Vital Signs and Blood Parameters — Mendeley Data](https://doi.org/10.17632/xrsbyjs24t.1)
- **Companion data article:** [A benchmark dataset for analyzing hematological responses to dengue fever in Bangladesh (*Data in Brief*, 2024)](https://doi.org/10.1016/j.dib.2024.111030)
- **Expected input file:** `Dengue diseases dataset.csv` (or the source ZIP containing one CSV file).
- **Original rows:** 1,003; **rows with a recorded outcome label:** 989 (14 excluded for missing outcome labels).
- **Development/test partition:** 791/198 observations, stratified by recorded class with random seed 42.
- **Recorded class distribution:** 669 class `1` and 320 class `0` among the labelled records.

The notebook expects the following source columns, with their exact spelling:

```text
Age
Sex
Haemoglobin
WBC Count
Differential Count
RBC PANEL
Platelet Count
PDW
Final Output
```

The study interprets `Final Output = 1` as the dataset's dengue-positive label and `Final Output = 0` as its non-dengue label. **These interpretations are not proof of how the original labels were clinically assigned.** In addition, the meaning of the `Sex = Child` category and the binary `Differential Count` / `RBC PANEL` fields needs caution. The released data do not establish whether every row comes from a unique patient.

Before using or redistributing the dataset, consult its original repository page and applicable reuse terms.

## Requirements

The notebook was developed for **Python 3 in Google Colab** and relies on:

- NumPy, pandas, SciPy, Matplotlib
- scikit-learn
- XGBoost
- SHAP
- statsmodels

The first code cell attempts to install `xgboost`, `shap`, and `statsmodels` if they are missing. In another Python environment, dependencies can be installed with:

```bash
pip install numpy pandas scipy matplotlib scikit-learn xgboost shap statsmodels
```

Package versions are not pinned in the uploaded notebook. CPU execution is supported; some optimization and repeated-validation steps are computationally intensive.

## Running in Google Colab

1. Open [Google Colab](https://colab.research.google.com/) and load `Dengue_Reviewer_Revision_Step1 (3).ipynb` from this GitHub repository, or upload the notebook manually.
2. Download the original dataset using the Mendeley Data DOI above. Keep the CSV filename as **`Dengue diseases dataset.csv`**.
3. Place/upload the CSV into the Colab session's `/content/` directory. The dataset-loading cell also offers an interactive upload prompt when the file is not found.
4. Execute the setup, configuration, dataset-loading, audit, preprocessing, and reviewer-analysis cells in dependency order.
5. Proceed through the baseline model comparison, optimizer benchmark, calibration, held-out evaluation, bootstrap, SHAP and result-export cells.
6. Download the resulting `Mendeley_Dengue_Reviewer_Outputs/` directory and any final Figure 8 files saved in the runtime working directory, since Colab runtime storage is temporary. The optional final cell mounts Google Drive if persistence is needed.

### Important execution-order note

**The uploaded notebook is an analysis/revision notebook, not yet a verified clean “Runtime → Run all” release.** In its current cell order:

- The **Reviewer Step 8: Decision Curve Analysis** cell appears *before* the `derive_calibrated_threshold` helper and the **Development-only calibrated OOF threshold selection** cell that create `canonical_threshold_info`. Executing Step 8 at its current position raises a `RuntimeError`. **Skip it on the first pass**, or move/run it **after** the calibrated OOF threshold-selection cell. This exploratory DCA is not part of the reported final-manuscript figures.
- A **Reviewer Step 2: Feature ablation** cell occurs twice. The later copy is redundant and can be skipped after the earlier one has completed. The saved notebook contains a `NameError` from an earlier, out-of-order execution of this duplicate cell.
- Some saved outputs reflect separate interactive runs. A fresh, sequential end-to-end run has **not** been certified for this uploaded version. Check outputs against the manuscript before claiming full reproducibility.

The original source file is expected at `/content/Dengue diseases dataset.csv` for the smoothest Colab experience. The data are not downloaded automatically.

## Analysis workflow

The notebook includes:

1. **Dataset audit:** label completeness, missing values, recorded categorical coding, exact and near-similar records.
2. **Leakage-aware preprocessing:** training-fold imputation, scaling and one-hot encoding; locked stratified test partition.
3. **Baseline model comparison:** Logistic Regression, Random Forest and XGBoost using repeated stratified 5-fold cross-validation with 5 repeats.
4. **Prespecified utility:** `0.35 × sensitivity + 0.25 × F1 + 0.20 × ROC-AUC + 0.20 × balanced accuracy`, including weight-sensitivity analysis.
5. **Random Forest optimization:** canonical Red Panda Optimization (RPO) versus budget-matched Random Search, with **274 unique evaluated configurations per optimizer** and additional five-seed RPO stability analysis.
6. **Calibration and threshold selection:** development-only calibrated out-of-fold predictions, sigmoid calibration, fixed operating threshold, and held-out test evaluation.
7. **Model evaluation:** classification metrics, ROC/PR curves, Brier score, log loss and calibration diagnostics.
8. **Uncertainty:** 2,000-resample **conditional** bootstrap intervals and prevalence-sensitive PPV/NPV uncertainty using 50,000 posterior draws with Jeffreys priors.
9. **Reviewer robustness analyses:** feature ablation, 50-label-permutation test, record-similarity checks, similarity-grouped validation, median-versus-KNN imputation, and SHAP explanation stability.
10. **Explainability:** global SHAP feature importance and summary plot for the underlying fitted Random Forest, not the calibrated probability output.

The exploratory Decision Curve Analysis cell is present but, as explained above, requires reordering and is not an established clinical-utility result.

## Selected notebook results

The saved notebook reports the following results on the **198-observation locked test partition** for the canonical RPO Random Forest with threshold **0.52**:

| Metric | Reported value |
|---|---:|
| Accuracy | 0.9949 |
| Balanced accuracy | 0.9963 |
| Precision | 1.0000 |
| Sensitivity | 0.9925 |
| Specificity | 1.0000 |
| F1-score | 0.9963 |
| ROC-AUC | 1.0000 |
| PR-AUC | 1.0000 |
| Brier score | 0.002887 |
| Log loss | 0.021301 |

Recorded confusion matrix: **TN = 64, FP = 0, FN = 1, TP = 133**. These values describe performance against the *recorded dataset labels*, not externally verified dengue diagnoses.

The canonical optimization run in the saved notebook reports a best development utility of **0.998285** for RPO versus **0.997835** for Random Search. The manuscript does **not** establish that RPO has a statistically reliable superiority over conventional tuning: their held-out threshold-based confusion matrices were the same.

> **Reproduction caveat:** Some run-dependent values, notably optimizer elapsed times and conditional bootstrap interval endpoints, differ between the uploaded notebook's saved outputs and versions of the manuscript. Bootstrap intervals can change with resampling seeds. Do not claim an exact notebook–manuscript match without first reconciling these values through a clean, documented run.

## Output files

Most output tables and graphics are written to:

```text
Mendeley_Dengue_Reviewer_Outputs/
```

Representative files include:

```text
baseline_summary.csv
paired_baseline_tests_holm.csv
optimizer_development_summary.csv
rpo_multi_seed_stability.csv
canonical_threshold_analysis.csv
prespecified_independent_test_comparison.csv
calibration_diagnostics_canonical_rpo.csv
bootstrap_confidence_intervals_canonical_rpo.csv
global_shap_importance_canonical_rpo.csv
reviewer_step2_feature_ablation.csv
reviewer_step3_permutation.csv
reviewer_step5_grouped_cv.csv
reviewer_step6_shap_stability.csv
reviewer_step7_prevalence_uncertainty.csv
CANONICAL_MANUSCRIPT_RESULTS.json
CANONICAL_MANUSCRIPT_TEST_METRICS.csv
Figure_5_Canonical_RPO_Discrimination_Calibration.png
Figure_6_Threshold_Selection.png
Figure_7_Confusion_Matrix.png
Figure_9_Bootstrap_Confidence_Intervals.png
Figure_10_Global_SHAP_Summary.png
```

The later **Figure 8 with posterior credible intervals** is saved separately in the working directory as:

```text
Figure_8_PPV_NPV_Uncertainty.png
Figure_8_PPV_NPV_Uncertainty.pdf
Figure8_prevalence_uncertainty_results.csv
```

An earlier, **point-estimate-only** Figure 8 is also generated in the output directory; the uncertainty plot above corresponds to the revised manuscript analysis.

## Interpretation and limitations

- Data come from **one publicly released source**; no external multicenter validation was performed.
- The original diagnostic reference standard, exact label assignment procedure, and patient/visit uniqueness were **not independently verifiable**.
- No exact duplicate predictor profiles were observed in the analysed labelled cohort, but near-similar records were identified. Similarity-grouped validation cannot replace true patient-level validation.
- Predictor-to-label circularity or shortcut learning cannot be conclusively excluded.
- Bootstrap confidence intervals are **conditional on the final fitted model and test sample**; they are not full-pipeline uncertainty estimates.
- The PPV/NPV prevalence scenarios assume that sensitivity and specificity would remain constant across hypothetical populations.
- SHAP values explain model predictions and **do not establish causality or diagnostic validity**.
- This code must not be used to diagnose, triage, or treat patients without appropriate independent clinical validation and governance.

## Citations

If using the dataset, cite its original data article and repository:

1. Assaduzzaman M, Islam O, Nirob MAS, Mim MMH, Mahmud A. *A benchmark dataset for analyzing hematological responses to dengue fever in Bangladesh.* Data in Brief. 2024;57:111030. https://doi.org/10.1016/j.dib.2024.111030
2. Mendeley Data. *Predictive Clinical Dataset for Dengue Fever Using Vital Signs and Blood Parameters.* https://doi.org/10.17632/xrsbyjs24t.1

The corresponding dengue classification manuscript can be cited after its bibliographic details become publicly available.

## License

No software license is specified in the uploaded notebook. Add an explicit `LICENSE` file if you wish to grant others rights to reuse or redistribute the code. Dataset licensing is governed separately by the original Mendeley Data source.
