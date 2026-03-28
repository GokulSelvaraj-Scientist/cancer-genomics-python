# Cancer Genomics Analysis in Python

## Overview
An end-to-end cancer genomics analysis pipeline in Python covering exploratory data analysis, dimensionality reduction, machine learning for cancer subtype classification, survival analysis, and drug response prediction. This project demonstrates core data science skills — NumPy, pandas, scikit-learn, matplotlib, seaborn, and lifelines — applied to a clinically relevant genomics dataset.

## Why This Matters
Python is the dominant language for data science and machine learning. In cancer genomics and drug development, Python is widely used for:

- **ML pipelines** — scikit-learn, XGBoost, and PyTorch power predictive models for patient stratification and drug response prediction
- **Genomics data processing** — large-scale expression and mutation datasets require efficient NumPy/pandas operations
- **Survival analysis** — the lifelines library is the standard Python tool for Kaplan-Meier and Cox regression in clinical research
- **Reproducible research** — Python pipelines integrate naturally with cloud workflows, Docker, and CI/CD systems
- **Visualization** — matplotlib and seaborn produce publication-quality figures for scientific communication

## Analysis Pipeline

### 1. Exploratory Data Analysis
- Sample distribution across cancer types
- Age at diagnosis by cancer type
- Tumor stage distribution
- Driver gene mutation frequencies
- Gene expression distributions
- Survival distributions by stage

### 2. Dimensionality Reduction and Clustering
- PCA variance explained analysis
- PCA scatter plot — cancer type separation
- t-SNE visualization of gene expression clusters

### 3. Machine Learning — Cancer Subtype Classification
- Feature selection: top 100 variable genes
- Three models: Random Forest, Gradient Boosting, Logistic Regression
- 5-fold stratified cross-validation
- Confusion matrix and classification report
- Feature importance analysis

### 4. Survival Analysis
- Kaplan-Meier curves by cancer type
- Kaplan-Meier curves by tumor stage
- Cox Proportional Hazards regression with hazard ratio plot

### 5. Mutation Landscape and Drug Response
- Driver gene co-mutation correlation heatmap
- Mutation burden by cancer type and stage
- EGFR mutation vs EGFR inhibitor response (Wilcoxon test)
- Mutation burden vs overall survival correlation

## Results

| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| Random Forest | ~0.92 ± 0.04 | ~0.93 |
| Gradient Boosting | ~0.90 ± 0.05 | ~0.90 |
| Logistic Regression | ~0.85 ± 0.05 | ~0.85 |

- EGFR-mutant tumors show significantly lower EGFR inhibitor IC50 (p < 0.001) — confirming predictive biomarker value
- t-SNE reveals clear separation of cancer subtypes in gene expression space
- Higher mutation burden is associated with reduced overall survival

## Outputs
| File | Description |
|---|---|
| `01_exploratory_analysis.png` | EDA dashboard — demographics, mutations, expression |
| `02_dimensionality_reduction.png` | PCA and t-SNE visualizations |
| `03_classification_results.png` | ML model comparison, confusion matrix, feature importance |
| `04_survival_analysis.png` | KM curves and Cox regression |
| `05_mutation_drug_analysis.png` | Mutation landscape and drug response |
| `model_performance.csv` | Model performance summary |
| `clinical_data.csv` | Simulated clinical dataset |

## How to Run

### Step 1 — Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn lifelines scipy
```

Or with conda:
```bash
conda install numpy pandas matplotlib seaborn scikit-learn scipy
pip install lifelines
```

### Step 2 — Run analysis
```bash
python cancer_genomics_analysis.py
```

## Requirements
- Python >= 3.8
- numpy, pandas, matplotlib, seaborn
- scikit-learn
- lifelines
- scipy

## Technical Stack
- **Data manipulation:** NumPy, pandas
- **Machine learning:** scikit-learn (Random Forest, Gradient Boosting, Logistic Regression, PCA, t-SNE)
- **Survival analysis:** lifelines (KaplanMeierFitter, CoxPHFitter)
- **Visualisation:** matplotlib, seaborn
- **Statistics:** scipy (Mann-Whitney U, linear regression)

## Author
**Gokul Selvaraj, PhD**
GitHub: [GokulSelvaraj-Scientist](https://github.com/GokulSelvaraj-Scientist)
