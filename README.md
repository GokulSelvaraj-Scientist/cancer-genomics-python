# Cancer Genomics Analysis in Python — Real Data

## Overview
An end-to-end cancer data science pipeline in Python using real public datasets. Covers exploratory data analysis, dimensionality reduction, machine learning classification, survival analysis, and biomarker discovery — demonstrating core Python data science skills applied to real clinical and genomic data.

## Datasets Used

### Wisconsin Breast Cancer Diagnostic Dataset (scikit-learn built-in)
- **Source:** UCI Machine Learning Repository via scikit-learn
- **Samples:** 569 real patient biopsies
- **Features:** 30 quantitative features from cell nucleus images
- **Target:** Binary diagnosis — Malignant vs Benign

### Rossi Recidivism Study (lifelines)
- **Source:** Rossi et al. (1980) — real longitudinal follow-up study
- **Samples:** 432 individuals followed for up to 52 weeks post-release
- **Purpose:** Demonstrates survival analysis methods on real longitudinal data

## Why This Matters
Python is the dominant language for data science and machine learning. This project demonstrates core Python skills directly applicable to data scientist roles:

- **scikit-learn pipelines** — the industry standard for ML in Python
- **Survival analysis** — lifelines is the standard Python tool for clinical research
- **Data wrangling** — pandas and numpy for real dataset manipulation
- **Visualisation** — matplotlib and seaborn for publication-quality figures

## Analysis Pipeline

### 1. Exploratory Data Analysis
- Patient diagnosis distribution, feature distributions by class
- Feature correlation heatmap, discriminating feature boxplots
- Feature space scatter plots

### 2. Dimensionality Reduction
- PCA scree plot — PC1 explains 44.3% variance
- PCA scatter — clear malignant/benign separation
- t-SNE visualisation of cluster structure

### 3. Machine Learning Classification
- Three models: Random Forest, Gradient Boosting, Logistic Regression
- 5-fold stratified cross-validation
- ROC curves for all models
- Best model: Logistic Regression (AUC = 0.995, Accuracy = 98.2%)
- Feature importance analysis

### 4. Survival Analysis
- Kaplan-Meier curves — overall cohort and by risk group
- Log-rank test: p = 0.0013 (significant difference by prior arrest history)
- Cox Proportional Hazards regression with hazard ratio forest plot

### 5. Biomarker Discovery
- Wilcoxon rank-sum test for top discriminating features
- Violin plots with significance labels (p < 1e-76)
- All top biomarkers highly significant: worst perimeter, worst area, worst concave points

## Results
| Model | CV AUC | Test AUC | Test Accuracy |
|---|---|---|---|
| Logistic Regression | 0.996 | 0.995 | 98.2% |
| Random Forest | 0.993 | 0.993 | 97.4% |
| Gradient Boosting | 0.989 | 0.989 | 96.5% |

## Outputs
| File | Description |
|---|---|
| `01_exploratory_analysis.png` | EDA dashboard — 6 panel figure |
| `02_dimensionality_reduction.png` | PCA scree plot, PCA scatter, t-SNE |
| `03_classification_results.png` | Model comparison, ROC curves, feature importance |
| `04_survival_analysis.png` | KM curves and Cox regression forest plot |
| `05_biomarker_analysis.png` | Top biomarker violin plots with p-values |
| `model_performance.csv` | Model performance summary |
| `feature_importance.csv` | Random Forest feature importances |
| `breast_cancer_data.csv` | Full Wisconsin dataset with diagnosis labels |

## How to Run

### Step 1 — Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn lifelines scipy
```

### Step 2 — Run analysis
```bash
python cancer_genomics_analysis.py
```

No data downloads needed — both datasets load automatically from scikit-learn and the web.

## Technical Stack
- **Data:** scikit-learn (Wisconsin BC), lifelines/GitHub (Rossi survival)
- **Machine learning:** scikit-learn (Random Forest, Gradient Boosting, Logistic Regression, PCA, t-SNE)
- **Survival analysis:** lifelines (KaplanMeierFitter, CoxPHFitter)
- **Visualisation:** matplotlib, seaborn
- **Statistics:** scipy (Mann-Whitney U, linear regression)

## Author
**Gokul Selvaraj, PhD**
GitHub: [GokulSelvaraj-Scientist](https://github.com/GokulSelvaraj-Scientist)
