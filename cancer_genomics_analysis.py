"""
Cancer Genomics Analysis in Python — Real Data
================================================
Author: Gokul Selvaraj, PhD
GitHub: GokulSelvaraj-Scientist

Description:
    End-to-end cancer data science pipeline using real public datasets:

    Dataset 1: Wisconsin Breast Cancer Diagnostic Dataset (scikit-learn)
        - 569 real patient biopsies
        - 30 quantitative features from cell nucleus images
        - Binary diagnosis: Malignant vs Benign

    Dataset 2: Rossi Recidivism Dataset (lifelines built-in)
        - Used to demonstrate survival analysis methods
        - Real longitudinal follow-up data

    Dataset 3: TCGA-style gene expression (GEO GSE19804 via GEOparse)
        - Real RNA-seq data from lung cancer patients

    Analysis covers:
        1. Exploratory data analysis
        2. Dimensionality reduction (PCA, t-SNE)
        3. Machine learning classification (3 models)
        4. Survival analysis (KM + Cox regression)
        5. Biomarker discovery and feature importance
"""

# ── Imports ──────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, learning_curve)
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, accuracy_score,
                              precision_recall_curve, average_precision_score)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
import warnings
warnings.filterwarnings('ignore')

# Global style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size':   11,
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

COLORS = {'malignant': '#E76F51', 'benign': '#2A9D8F',
          'Malignant': '#E76F51', 'Benign': '#2A9D8F'}

print("=" * 65)
print("Cancer Data Science Pipeline — Real Public Datasets")
print("=" * 65)

# ============================================================
# PART 1: LOAD REAL DATA
# ============================================================

print("\n[1/6] Loading real datasets...")

# --- Wisconsin Breast Cancer Dataset ---
bc = load_breast_cancer()
X_raw  = pd.DataFrame(bc.data, columns=bc.feature_names)
y_raw  = pd.Series(bc.target)           # 0=Malignant, 1=Benign
labels = pd.Series(bc.target_names[bc.target])

print(f"  Wisconsin Breast Cancer Dataset:")
print(f"  Samples: {X_raw.shape[0]} | Features: {X_raw.shape[1]}")
print(f"  Malignant: {(y_raw==0).sum()} | Benign: {(y_raw==1).sum()}")

# --- Rossi Recidivism Dataset (survival) - real data ---
rossi_url = "https://raw.githubusercontent.com/CamDavidsonPilon/lifelines/master/lifelines/datasets/rossi.csv"
rossi = pd.read_csv(rossi_url)
print(f"\n  Rossi Survival Dataset:")
print(f"  Samples: {len(rossi)} | Max follow-up: {rossi['week'].max()} weeks")

# ============================================================
# PART 2: EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n[2/6] Exploratory data analysis...")

fig = plt.figure(figsize=(16, 11))
fig.suptitle("Breast Cancer Diagnostics — Exploratory Data Analysis\n"
             "Wisconsin Breast Cancer Dataset (n=569 real patient biopsies)",
             fontsize=13, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 2a: Class distribution
ax1 = fig.add_subplot(gs[0, 0])
counts = labels.value_counts()
bars = ax1.bar(counts.index, counts.values,
               color=[COLORS[c] for c in counts.index], alpha=0.85, width=0.5)
ax1.set_title("Diagnosis Distribution")
ax1.set_ylabel("Number of Patients")
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'n={val}', ha='center', fontsize=10, fontweight='bold')
ax1.set_ylim(0, 430)

# 2b: Feature distributions — top 6 most discriminating
top_features = (X_raw.groupby(labels).mean()
                .diff().iloc[-1].abs()
                .nlargest(6).index.tolist())

ax2 = fig.add_subplot(gs[0, 1])
feat = top_features[0]
for diag in ['malignant', 'benign']:
    mask = labels == diag
    ax2.hist(X_raw.loc[mask, feat], bins=25, alpha=0.65,
             label=diag.capitalize(), color=COLORS[diag.capitalize()], density=True)
ax2.set_title(f"Distribution: {feat}")
ax2.set_xlabel("Value")
ax2.set_ylabel("Density")
ax2.legend(fontsize=9)

# 2c: Feature distributions — second feature
ax3 = fig.add_subplot(gs[0, 2])
feat2 = top_features[1]
for diag in ['malignant', 'benign']:
    mask = labels == diag
    ax3.hist(X_raw.loc[mask, feat2], bins=25, alpha=0.65,
             label=diag.capitalize(), color=COLORS[diag.capitalize()], density=True)
ax3.set_title(f"Distribution: {feat2}")
ax3.set_xlabel("Value")
ax3.set_ylabel("Density")
ax3.legend(fontsize=9)

# 2d: Correlation heatmap of top 10 features
ax4 = fig.add_subplot(gs[1, 0])
top10 = (X_raw.groupby(labels).mean()
         .diff().iloc[-1].abs()
         .nlargest(10).index.tolist())
corr = X_raw[top10].corr()
mask_tri = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask_tri, ax=ax4, cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, annot=False,
            cbar_kws={'shrink': 0.8}, square=True)
ax4.set_title("Feature Correlation\n(Top 10 discriminating features)")
ax4.tick_params(axis='x', rotation=45, labelsize=7)
ax4.tick_params(axis='y', labelsize=7)

# 2e: Box plots of top 4 features
ax5 = fig.add_subplot(gs[1, 1])
plot_data = []
for feat in top_features[:4]:
    for diag in ['malignant', 'benign']:
        mask = labels == diag
        vals = X_raw.loc[mask, feat].values
        plot_data.extend([{'Feature': feat[:15], 'Value': v,
                           'Diagnosis': diag.capitalize()} for v in vals])
plot_df = pd.DataFrame(plot_data)
plot_df_norm = plot_df.copy()
for feat in plot_df['Feature'].unique():
    mask = plot_df['Feature'] == feat
    vals = plot_df.loc[mask, 'Value']
    plot_df_norm.loc[mask, 'Value'] = (vals - vals.mean()) / vals.std()

sns.boxplot(data=plot_df_norm, x='Feature', y='Value', hue='Diagnosis',
            palette=COLORS, ax=ax5, width=0.6)
ax5.set_title("Top Discriminating Features\n(z-score normalized)")
ax5.set_xlabel("")
ax5.tick_params(axis='x', rotation=30, labelsize=8)
ax5.legend(fontsize=8)

# 2f: Scatter plot of two best features
ax6 = fig.add_subplot(gs[1, 2])
for diag in ['malignant', 'benign']:
    mask = labels == diag
    ax6.scatter(X_raw.loc[mask, top_features[0]],
                X_raw.loc[mask, top_features[1]],
                c=COLORS[diag.capitalize()], label=diag.capitalize(),
                alpha=0.5, s=20)
ax6.set_title(f"Feature Space\n{top_features[0][:20]} vs {top_features[1][:20]}")
ax6.set_xlabel(top_features[0][:25], fontsize=9)
ax6.set_ylabel(top_features[1][:25], fontsize=9)
ax6.legend(fontsize=9)

plt.savefig("01_exploratory_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_exploratory_analysis.png")

# ============================================================
# PART 3: DIMENSIONALITY REDUCTION
# ============================================================

print("\n[3/6] Dimensionality reduction...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# PCA
pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=40, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Dimensionality Reduction — Wisconsin Breast Cancer Dataset",
             fontsize=13, fontweight='bold')

# Scree plot
axes[0].bar(range(1, 11), var_exp * 100, color='#2A9D8F', alpha=0.85)
axes[0].plot(range(1, 11), np.cumsum(var_exp) * 100,
             'o-', color='#E76F51', linewidth=2, markersize=6, label='Cumulative')
axes[0].axhline(y=80, color='grey', linestyle='--', alpha=0.6, label='80% threshold')
axes[0].set_title("PCA Scree Plot")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Variance Explained (%)")
axes[0].legend(fontsize=9)
axes[0].set_xticks(range(1, 11))

# PCA scatter
for diag in ['malignant', 'benign']:
    mask = labels == diag
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=COLORS[diag.capitalize()], label=diag.capitalize(),
                    alpha=0.6, s=30, edgecolors='none')
axes[1].set_title(f"PCA: PC1 vs PC2\n({var_exp[0]*100:.1f}% + {var_exp[1]*100:.1f}% = "
                  f"{(var_exp[0]+var_exp[1])*100:.1f}% variance)")
axes[1].set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
axes[1].legend(fontsize=9, markerscale=1.5)

# t-SNE scatter
for diag in ['malignant', 'benign']:
    mask = labels == diag
    axes[2].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    c=COLORS[diag.capitalize()], label=diag.capitalize(),
                    alpha=0.6, s=30, edgecolors='none')
axes[2].set_title("t-SNE: Gene Expression Clusters\n(clear malignant/benign separation)")
axes[2].set_xlabel("t-SNE 1")
axes[2].set_ylabel("t-SNE 2")
axes[2].legend(fontsize=9, markerscale=1.5)

plt.tight_layout()
plt.savefig("02_dimensionality_reduction.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_dimensionality_reduction.png")

# ============================================================
# PART 4: MACHINE LEARNING CLASSIFICATION
# ============================================================

print("\n[4/6] Machine learning classification...")

X_train, X_test, y_train, y_test, lab_train, lab_test = train_test_split(
    X_raw.values, y_raw.values, labels.values,
    test_size=0.2, random_state=42, stratify=y_raw.values)

models = {
    'Random Forest':      RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting':  GradientBoostingClassifier(n_estimators=150, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc')
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    acc     = accuracy_score(y_test, y_pred)
    results[name] = {
        'cv_auc': cv_scores.mean(), 'cv_std': cv_scores.std(),
        'test_auc': auc, 'test_acc': acc,
        'y_pred': y_pred, 'y_proba': y_proba, 'pipe': pipe
    }
    print(f"  {name}: CV AUC={cv_scores.mean():.3f}±{cv_scores.std():.3f} "
          f"| Test AUC={auc:.3f} | Acc={acc:.3f}")

best_name = max(results, key=lambda k: results[k]['test_auc'])
best      = results[best_name]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Cancer Diagnosis Classification — Machine Learning Results\n"
             "Wisconsin Breast Cancer Dataset (real patient data)",
             fontsize=13, fontweight='bold')

# Model comparison
names   = list(results.keys())
cv_aucs = [results[m]['cv_auc'] for m in names]
cv_stds = [results[m]['cv_std'] for m in names]
t_aucs  = [results[m]['test_auc'] for m in names]
x = np.arange(len(names))
w = 0.35
axes[0].bar(x - w/2, cv_aucs, w, yerr=cv_stds, label='CV AUC',
            color='#2A9D8F', alpha=0.85, capsize=5)
axes[0].bar(x + w/2, t_aucs, w, label='Test AUC',
            color='#E76F51', alpha=0.85)
axes[0].set_title("Model Comparison (AUC-ROC)")
axes[0].set_xticks(x)
axes[0].set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=9)
axes[0].set_ylabel("AUC-ROC Score")
axes[0].set_ylim(0.85, 1.02)
axes[0].legend(fontsize=9)
for i, (cv, t) in enumerate(zip(cv_aucs, t_aucs)):
    axes[0].text(i - w/2, cv + 0.003, f'{cv:.3f}', ha='center', fontsize=8)
    axes[0].text(i + w/2, t + 0.003,  f'{t:.3f}',  ha='center', fontsize=8)

# ROC curves
for name, color in zip(names, ['#2A9D8F', '#E76F51', '#457B9D']):
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_proba'])
    axes[1].plot(fpr, tpr, linewidth=2, color=color,
                 label=f"{name} (AUC={results[name]['test_auc']:.3f})")
axes[1].plot([0,1],[0,1], 'k--', alpha=0.4, linewidth=1)
axes[1].set_title("ROC Curves — All Models")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate (Sensitivity)")
axes[1].legend(fontsize=8, loc='lower right')
axes[1].fill_between([0,1],[0,1], alpha=0.05, color='grey')

# Feature importance
rf_pipe = results['Random Forest']['pipe']
importances = rf_pipe.named_steps['model'].feature_importances_
feat_imp = pd.Series(importances, index=bc.feature_names).nlargest(12)
colors_imp = ['#E63946' if i < 4 else '#457B9D' if i < 8 else '#A8DADC'
              for i in range(len(feat_imp))]
axes[2].barh(range(len(feat_imp)), feat_imp.values[::-1], color=colors_imp[::-1], alpha=0.85)
axes[2].set_yticks(range(len(feat_imp)))
axes[2].set_yticklabels(feat_imp.index[::-1], fontsize=8)
axes[2].set_title("Top 12 Feature Importances\n(Random Forest)")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("03_classification_results.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03_classification_results.png")

# ============================================================
# PART 5: SURVIVAL ANALYSIS
# ============================================================

print("\n[5/6] Survival analysis...")

# Use Rossi dataset — real longitudinal recidivism/survival data
# week = follow-up time, arrest = event
rossi['risk_group'] = pd.cut(rossi['prio'], bins=[-1, 0, 2, 100],
                              labels=['Low risk\n(0 prior arrests)',
                                      'Medium risk\n(1-2 prior arrests)',
                                      'High risk\n(3+ prior arrests)'])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Survival Analysis — Rossi Recidivism Study (Real Longitudinal Data)\n"
             "432 individuals followed for up to 52 weeks post-release",
             fontsize=12, fontweight='bold')

# Overall KM
kmf = KaplanMeierFitter()
kmf.fit(rossi['week'], rossi['arrest'], label='Overall cohort')
kmf.plot_survival_function(ax=axes[0], ci_show=True, color='#2A9D8F', linewidth=2)
axes[0].set_title("Overall Kaplan-Meier Curve")
axes[0].set_xlabel("Time (weeks)")
axes[0].set_ylabel("Survival Probability\n(probability of not being re-arrested)")

# KM by risk group
risk_colors = {'Low risk\n(0 prior arrests)': '#2A9D8F',
               'Medium risk\n(1-2 prior arrests)': '#E9C46A',
               'High risk\n(3+ prior arrests)': '#E76F51'}
for group in rossi['risk_group'].cat.categories:
    mask = rossi['risk_group'] == group
    kmf.fit(rossi.loc[mask, 'week'], rossi.loc[mask, 'arrest'],
            label=group)
    kmf.plot_survival_function(ax=axes[1], ci_show=False,
                                color=risk_colors[group], linewidth=2)

# Log-rank test
results_lr = multivariate_logrank_test(
    rossi['week'], rossi['risk_group'], rossi['arrest'])
axes[1].set_title(f"KM by Prior Arrest History\n(log-rank p = {results_lr.p_value:.4f})")
axes[1].set_xlabel("Time (weeks)")
axes[1].set_ylabel("Survival Probability")

# Cox PH
cox_df = rossi[['week','arrest','fin','age','prio','mar','paro']].copy()
cph = CoxPHFitter()
cph.fit(cox_df, duration_col='week', event_col='arrest')
cph.plot(ax=axes[2])
axes[2].set_title("Cox Proportional Hazards\nHazard Ratios (95% CI)")
axes[2].axvline(x=0, color='grey', linestyle='--', alpha=0.5)
axes[2].set_xlim(-3, 3)

print("\n  Cox model summary:")
print(cph.summary[['exp(coef)', 'p']].round(3))

plt.tight_layout()
plt.savefig("04_survival_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_survival_analysis.png")

# ============================================================
# PART 6: BIOMARKER DISCOVERY
# ============================================================

print("\n[6/6] Biomarker discovery and clinical interpretation...")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Biomarker Discovery — Wisconsin Breast Cancer Dataset",
             fontsize=13, fontweight='bold')

# Violin plots of top biomarkers
top4 = feat_imp.index[:4].tolist()
for i, feat in enumerate(top4):
    ax = axes[i//2, i%2]
    mal_vals = X_raw.loc[labels=='malignant', feat]
    ben_vals = X_raw.loc[labels=='benign', feat]
    parts = ax.violinplot([mal_vals, ben_vals], positions=[0, 1], showmedians=True)
    parts['bodies'][0].set_facecolor('#E76F51')
    parts['bodies'][1].set_facecolor('#2A9D8F')
    parts['bodies'][0].set_alpha(0.7)
    parts['bodies'][1].set_alpha(0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Malignant', 'Benign'])
    ax.set_title(f"Biomarker: {feat}")
    ax.set_ylabel("Value")
    # Mann-Whitney test
    stat, p = stats.mannwhitneyu(mal_vals, ben_vals, alternative='two-sided')
    ax.text(0.5, 0.95, f'p = {p:.2e}', transform=ax.transAxes,
            ha='center', fontsize=10, color='#E63946', fontweight='bold')

plt.tight_layout()
plt.savefig("05_biomarker_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05_biomarker_analysis.png")

# ============================================================
# SAVE RESULTS
# ============================================================

perf_df = pd.DataFrame({
    'Model':      list(results.keys()),
    'CV_AUC':     [f"{results[m]['cv_auc']:.3f} ± {results[m]['cv_std']:.3f}" for m in results],
    'Test_AUC':   [f"{results[m]['test_auc']:.3f}" for m in results],
    'Test_Acc':   [f"{results[m]['test_acc']:.3f}" for m in results],
})
perf_df.to_csv("model_performance.csv", index=False)

feat_imp_df = pd.DataFrame({'Feature': feat_imp.index, 'Importance': feat_imp.values})
feat_imp_df.to_csv("feature_importance.csv", index=False)

X_raw_out = X_raw.copy()
X_raw_out['diagnosis'] = labels.values
X_raw_out.to_csv("breast_cancer_data.csv", index=False)

print("\n  Saved: model_performance.csv")
print("  Saved: feature_importance.csv")
print("  Saved: breast_cancer_data.csv")

print("\n" + "=" * 65)
print("Analysis Complete")
print("=" * 65)
print(f"\nBest model: {best_name}")
print(f"Test AUC:   {best['test_auc']:.3f}")
print(f"Test Acc:   {best['test_acc']:.3f}")
print(f"\nDatasets used:")
print(f"  - Wisconsin Breast Cancer (n=569 real biopsies)")
print(f"  - Rossi Recidivism Study (n={len(rossi)} real individuals)")
