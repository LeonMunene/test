# Predicting 30-Day Hospital Readmission in Diabetic Patients

A machine learning group project using the UCI "Diabetes 130-US Hospitals (1999–2008)" dataset.

## 1. Real-world problem

Unplanned hospital readmissions within 30 days are a major driver of healthcare cost and a recognized marker of care quality. Predicting which patients are at high risk before discharge lets a hospital target follow-up calls, home health referrals, or medication reconciliation at the patients who need it most.

## 2. Dataset (secondary source)

- **Source:** UCI Machine Learning Repository — Diabetes 130-US Hospitals for Years 1999–2008 (Strack et al., 2014), provided as `diabetic_data.csv` (101,766 encounters, 50 raw columns) and `IDS_mapping.csv` (code lookup tables).
- **Cleaning steps applied:**
  1. Dropped encounters ending in death or hospice (2,423 rows) — these patients cannot be readmitted.
  2. Kept only the **first encounter per patient** (69,990 of 101,766 rows kept) to prevent the same patient appearing in both the training and test sets, which would leak information and inflate performance.
  3. Fixed a subtle parsing bug: pandas' default CSV reader silently treats the literal string `"None"` as missing, but in this dataset `A1Cresult="None"` and `max_glu_serum="None"` are a real category meaning "test not performed" — relabeled to `Not_tested` so it isn't lost.
  4. Dropped `weight` (96% missing) and `payer_code` (43% missing); grouped `medical_specialty` into top categories.
  5. Mapped 700+ raw ICD-9 diagnosis codes into 9 clinical categories (Circulatory, Diabetes, Respiratory, etc.).
  6. Grouped admission type/source and discharge disposition codes (via `IDS_mapping.csv`) into interpretable categories.
  7. Engineered `num_med_changes` and `num_meds_prescribed` from the 20+ individual diabetes medication columns.
- **Target:** binary — readmitted within **<30 days** (1) vs. not (0). This is the standard framing in the literature, since early readmission is the outcome most linked to care-transition quality. Base rate: **9.0%** positive class (imbalanced).

## 3. Models (hybrid/ensemble requirement)

Four models were trained on an identical preprocessing pipeline (standardization for numeric features, one-hot encoding for categorical features), split **64% train / 16% validation / 20% test** (stratified):

| Model | Type |
|---|---|
| Logistic Regression | Linear baseline, class-weight balanced |
| Random Forest | Bagging ensemble |
| XGBoost | Gradient boosting ensemble |
| **Stacking Ensemble** | **Hybrid model** — LR + RF + XGBoost base learners combined by a logistic-regression meta-learner (5-fold internal CV) |

Because only ~9% of patients are readmitted within 30 days, the default 0.5 classification threshold is inappropriate (it favors the majority class). For each model, the decision threshold was tuned on the **validation set** (maximizing F1) and then applied once to the untouched **test set** — this avoids leaking test data into model selection.

## 4. Performance evaluation (held-out test set, n=13,998)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.739 | 0.151 | 0.413 | 0.221 | 0.641 | 0.150 |
| Random Forest | 0.696 | 0.147 | 0.496 | 0.227 | 0.649 | 0.149 |
| XGBoost | 0.695 | 0.145 | 0.491 | 0.224 | 0.642 | 0.156 |
| **Stacking Ensemble (Hybrid)** | **0.796** | **0.172** | 0.333 | 0.227 | **0.650** | **0.158** |

**Interpretation:** the hybrid stacking ensemble achieves the best ROC-AUC, PR-AUC, precision, and accuracy of the four models, and ties for the best F1. All models land in the 0.64–0.65 ROC-AUC range — this is consistent with the published literature on this exact dataset; 30-day readmission is a genuinely hard prediction task because the available features capture only part of what drives readmission (social factors, care-transition quality, and post-discharge circumstances aren't recorded). The most predictive features (see feature importance chart) are **discharge disposition**, **number of lab procedures**, **prior inpatient visits**, **number of medications**, and **length of stay** — all clinically sensible.

See `figures/08_model_comparison.png`, `09_roc_curves.png`, `10_confusion_matrices.png`, and `12_precision_recall_curves.png`.

## 5. Deployment artifact

`readmission_risk_calculator.html` is a standalone, interactive risk calculator. It embeds the **real coefficients** of a simplified, interpretable logistic regression (retrained on 17 clinically actionable features, test ROC-AUC 0.640 — nearly identical to the full model, confirming most of the signal is captured by a small feature set). Enter a patient's admission details and it computes the readmission probability live in the browser, shows where the patient falls relative to the real patient population (median / 85th percentile markers), and explains the top contributing factors.

**Fairness note:** race and gender were deliberately excluded from this interactive tool — even though they were retained in the full benchmarking model above for completeness — to avoid presenting a demographic-based individual risk score. Worth mentioning explicitly in the presentation as a design decision.

## 6. Visualizations

- **Dataset (EDA):** target class balance, readmission rate by age, length-of-stay distribution, readmission rate by prior inpatient visits, diagnosis category breakdown, numeric feature correlation heatmap, demographic composition — `figures/01`–`07`.
- **Results:** model comparison bar chart, ROC curves, confusion matrices, feature importance (RF & XGBoost), precision-recall curves — `figures/08`–`12`.

## Folder contents

```
project_summary.md                    - this document
readmission_risk_calculator.html      - deployment artifact (open in any browser)
figures/                              - all 12 charts (EDA + results)
code/
  01_clean_features.py                - data cleaning & feature engineering
  02_eda_visuals.py                   - dataset visualizations
  03_modeling.py                      - trains all 4 models, evaluates on test set
  04_results_visuals.py               - results visualizations
  05_export_deploy_model.py           - trains simplified model, exports coefficients for the calculator
```

To reproduce: run the scripts in order 01 → 05 with `diabetic_data.csv` and `IDS_mapping.csv` in the same folder as the raw data path referenced at the top of `01_clean_features.py`.
