"""
Cancer Genomics Analysis in Python
===================================
Author: Gokul Selvaraj, PhD
GitHub: GokulSelvaraj-Scientist

Description:
    End-to-end cancer genomics analysis pipeline in Python covering:
    1. Exploratory data analysis of cancer genomics data
    2. Mutation landscape analysis
    3. Gene expression clustering and dimensionality reduction
    4. Machine learning for cancer subtype classification
    5. Survival analysis and visualisation
    6. Drug response prediction

Dataset:
    Simulated data based on TCGA/CCLE distributions
    - 200 cancer samples across 5 cancer types
    - 500 gene expression features
    - Somatic mutation profiles
    - Clinical outcomes
"""

# ── Imports ──────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, accuracy_score)
from sklearn.pipeline import Pipeline
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings('ignore')

# Set global style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

PALETTE = {
    'LUAD': '#2A9D8F', 'BRCA': '#E76F51',
    'COAD': '#457B9D', 'GBM': '#E9C46A', 'PRAD': '#A8DADC'
}

print("=" * 60)
print("Cancer Genomics Analysis Pipeline")
print("=" * 60)

# ============================================================
# PART 1: DATA SIMULATION
# ============================================================

np.random.seed(42)

N_SAMPLES   = 200
N_GENES     = 500
CANCER_TYPES = ['LUAD', 'BRCA', 'COAD', 'GBM', 'PRAD']
N_PER_TYPE  = N_SAMPLES // len(CANCER_TYPES)

print("\n[1/6] Simulating cancer genomics dataset...")

# Simulate gene expression with cancer-type-specific signatures
def simulate_expression(n, cancer_type, n_genes=N_GENES):
    base = np.random.poisson(lam=10, size=(n, n_genes)).astype(float)
    base += np.random.uniform(0.1, 1.0, size=(n, n_genes))
    # Add cancer-type-specific signal to first 50 genes
    offsets = {'LUAD': 3.0, 'BRCA': 2.0, 'COAD': -1.5, 'GBM': 4.0, 'PRAD': -2.0}
    base[:, :50] += offsets.get(cancer_type, 0)
    base = np.clip(base, 0.01, None)
    return np.log2(base + 1)

expr_blocks = [simulate_expression(N_PER_TYPE, ct) for ct in CANCER_TYPES]
expr_matrix = np.vstack(expr_blocks)
cancer_labels = np.repeat(CANCER_TYPES, N_PER_TYPE)
sample_ids = [f"S{i:03d}" for i in range(N_SAMPLES)]

# Simulate clinical data
ages = np.concatenate([
    np.random.normal(mu, 8, N_PER_TYPE)
    for mu in [62, 52, 68, 55, 67]
]).astype(int).clip(30, 85)

stages = np.random.choice(['I', 'II', 'III', 'IV'], N_SAMPLES,
                           p=[0.25, 0.30, 0.25, 0.20])

# Simulate survival with stage and cancer-type effects
base_survival = {'I': 2000, 'II': 1500, 'III': 900, 'IV': 400}
os_days = np.array([
    max(30, int(np.random.exponential(base_survival[s] * (1.2 if ct in ['LUAD','BRCA'] else 0.8))))
    for s, ct in zip(stages, cancer_labels)
])
vital_status = (os_days < 1500).astype(int)

# Simulate key mutations
tp53_mut  = np.random.binomial(1, 0.45, N_SAMPLES)
kras_mut  = np.where(cancer_labels == 'LUAD',
                     np.random.binomial(1, 0.35, N_SAMPLES),
                     np.random.binomial(1, 0.10, N_SAMPLES))
egfr_mut  = np.where(cancer_labels == 'LUAD',
                     np.random.binomial(1, 0.25, N_SAMPLES),
                     np.random.binomial(1, 0.05, N_SAMPLES))
brca1_mut = np.where(cancer_labels == 'BRCA',
                     np.random.binomial(1, 0.20, N_SAMPLES),
                     np.random.binomial(1, 0.03, N_SAMPLES))

# Build dataframes
gene_names = [f"GENE_{i:04d}" for i in range(N_GENES)]
expr_df = pd.DataFrame(expr_matrix, index=sample_ids, columns=gene_names)

clinical_df = pd.DataFrame({
    'sample_id':    sample_ids,
    'cancer_type':  cancer_labels,
    'age':          ages,
    'stage':        stages,
    'os_days':      os_days,
    'vital_status': vital_status,
    'TP53_mut':     tp53_mut,
    'KRAS_mut':     kras_mut,
    'EGFR_mut':     egfr_mut,
    'BRCA1_mut':    brca1_mut,
    'mutation_burden': tp53_mut + kras_mut + egfr_mut + brca1_mut
})

print(f"  Expression matrix: {expr_df.shape[0]} samples × {expr_df.shape[1]} genes")
print(f"  Cancer types: {dict(pd.Series(cancer_labels).value_counts())}")
print(f"  Vital status: {vital_status.sum()} events / {N_SAMPLES} total")

# ============================================================
# PART 2: EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n[2/6] Exploratory data analysis...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Cancer Genomics — Exploratory Data Analysis", fontsize=14, fontweight='bold', y=1.01)

# 2a: Sample distribution
ct_counts = pd.Series(cancer_labels).value_counts()
colors = [PALETTE[ct] for ct in ct_counts.index]
axes[0,0].bar(ct_counts.index, ct_counts.values, color=colors, alpha=0.85, edgecolor='white')
axes[0,0].set_title("Samples per Cancer Type")
axes[0,0].set_xlabel("Cancer Type")
axes[0,0].set_ylabel("Number of Samples")

# 2b: Age distribution by cancer type
ct_order = CANCER_TYPES
age_data = [ages[cancer_labels == ct] for ct in ct_order]
bp = axes[0,1].boxplot(age_data, patch_artist=True, labels=ct_order)
for patch, ct in zip(bp['boxes'], ct_order):
    patch.set_facecolor(PALETTE[ct])
    patch.set_alpha(0.8)
axes[0,1].set_title("Age at Diagnosis by Cancer Type")
axes[0,1].set_xlabel("Cancer Type")
axes[0,1].set_ylabel("Age")

# 2c: Stage distribution
stage_ct = pd.crosstab(cancer_labels, stages)
stage_ct.plot(kind='bar', ax=axes[0,2], colormap='Blues', alpha=0.85, edgecolor='white')
axes[0,2].set_title("Tumor Stage Distribution")
axes[0,2].set_xlabel("Cancer Type")
axes[0,2].set_ylabel("Count")
axes[0,2].tick_params(axis='x', rotation=0)
axes[0,2].legend(title='Stage', loc='upper right', fontsize=8)

# 2d: Mutation frequency
mut_cols = ['TP53_mut', 'KRAS_mut', 'EGFR_mut', 'BRCA1_mut']
mut_freq = clinical_df[mut_cols].mean() * 100
mut_colors = ['#E63946' if f > 20 else '#457B9D' for f in mut_freq]
axes[1,0].barh(mut_cols, mut_freq, color=mut_colors, alpha=0.85)
axes[1,0].set_title("Driver Gene Mutation Frequencies")
axes[1,0].set_xlabel("% Samples Mutated")
for i, v in enumerate(mut_freq):
    axes[1,0].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)

# 2e: Expression distribution
sample_means = expr_matrix.mean(axis=1)
for ct in CANCER_TYPES:
    mask = cancer_labels == ct
    axes[1,1].hist(sample_means[mask], bins=20, alpha=0.6,
                   label=ct, color=PALETTE[ct], density=True)
axes[1,1].set_title("Mean Gene Expression by Cancer Type")
axes[1,1].set_xlabel("Mean Log2 Expression")
axes[1,1].set_ylabel("Density")
axes[1,1].legend(fontsize=8)

# 2f: Survival by stage
for stage, color in zip(['I','II','III','IV'], ['#2A9D8F','#A8DADC','#E9C46A','#E76F51']):
    mask = stages == stage
    axes[1,2].hist(os_days[mask], bins=15, alpha=0.6,
                   label=f'Stage {stage}', color=color, density=True)
axes[1,2].set_title("Survival Distribution by Stage")
axes[1,2].set_xlabel("Overall Survival (days)")
axes[1,2].set_ylabel("Density")
axes[1,2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("01_exploratory_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_exploratory_analysis.png")

# ============================================================
# PART 3: DIMENSIONALITY REDUCTION & CLUSTERING
# ============================================================

print("\n[3/6] Dimensionality reduction and clustering...")

# Standardize
scaler = StandardScaler()
expr_scaled = scaler.fit_transform(expr_matrix)

# PCA
pca = PCA(n_components=50, random_state=42)
expr_pca = pca.fit_transform(expr_scaled)
var_exp = pca.explained_variance_ratio_

# t-SNE on top 20 PCs
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
expr_tsne = tsne.fit_transform(expr_pca[:, :20])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Dimensionality Reduction: Gene Expression Landscape",
             fontsize=14, fontweight='bold')

# PCA variance explained
axes[0].plot(range(1, 21), np.cumsum(var_exp[:20]) * 100,
             'o-', color='#2A9D8F', linewidth=2, markersize=5)
axes[0].axhline(y=80, color='grey', linestyle='--', alpha=0.6)
axes[0].set_title("PCA: Cumulative Variance Explained")
axes[0].set_xlabel("Number of Principal Components")
axes[0].set_ylabel("Cumulative Variance (%)")
axes[0].fill_between(range(1, 21), np.cumsum(var_exp[:20]) * 100, alpha=0.1, color='#2A9D8F')

# PCA scatter
for ct in CANCER_TYPES:
    mask = cancer_labels == ct
    axes[1].scatter(expr_pca[mask, 0], expr_pca[mask, 1],
                    c=PALETTE[ct], label=ct, alpha=0.7, s=30)
axes[1].set_title(f"PCA — PC1 vs PC2\n({var_exp[0]*100:.1f}% + {var_exp[1]*100:.1f}% variance)")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].legend(fontsize=8, markerscale=1.5)

# t-SNE scatter
for ct in CANCER_TYPES:
    mask = cancer_labels == ct
    axes[2].scatter(expr_tsne[mask, 0], expr_tsne[mask, 1],
                    c=PALETTE[ct], label=ct, alpha=0.7, s=30)
axes[2].set_title("t-SNE — Gene Expression Clusters")
axes[2].set_xlabel("t-SNE 1")
axes[2].set_ylabel("t-SNE 2")
axes[2].legend(fontsize=8, markerscale=1.5)

plt.tight_layout()
plt.savefig("02_dimensionality_reduction.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_dimensionality_reduction.png")

# ============================================================
# PART 4: MACHINE LEARNING - CANCER SUBTYPE CLASSIFICATION
# ============================================================

print("\n[4/6] Machine learning — cancer subtype classification...")

# Use top 100 variable genes
gene_vars = expr_matrix.var(axis=0)
top100_idx = np.argsort(gene_vars)[-100:]
X = expr_matrix[:, top100_idx]
le = LabelEncoder()
y = le.fit_transform(cancer_labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=0.1)
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')
    pipe.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, pipe.predict(X_test))
    results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std':  cv_scores.std(),
        'test_acc': test_acc,
        'model': pipe
    }
    print(f"  {name}: CV={cv_scores.mean():.3f}±{cv_scores.std():.3f} | Test={test_acc:.3f}")

best_name = max(results, key=lambda k: results[k]['test_acc'])
best_pipe = results[best_name]['model']
y_pred    = best_pipe.predict(X_test)

# Plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Cancer Subtype Classification — Machine Learning Results",
             fontsize=14, fontweight='bold')

# Model comparison
model_names  = list(results.keys())
cv_means     = [results[m]['cv_mean'] for m in model_names]
cv_stds      = [results[m]['cv_std']  for m in model_names]
test_accs    = [results[m]['test_acc'] for m in model_names]
x = np.arange(len(model_names))
w = 0.35
axes[0].bar(x - w/2, cv_means, w, yerr=cv_stds, label='CV Accuracy',
            color='#2A9D8F', alpha=0.85, capsize=4)
axes[0].bar(x + w/2, test_accs, w, label='Test Accuracy',
            color='#E76F51', alpha=0.85)
axes[0].set_title("Model Comparison")
axes[0].set_xticks(x)
axes[0].set_xticklabels([m.replace(' ', '\n') for m in model_names], fontsize=9)
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0, 1.1)
axes[0].legend(fontsize=9)
axes[0].axhline(y=1/5, color='grey', linestyle='--', alpha=0.5, label='Chance')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[1],
            xticklabels=le.classes_, yticklabels=le.classes_,
            cmap='Blues', cbar=False)
axes[1].set_title(f"Confusion Matrix\n{best_name} (Test Acc = {results[best_name]['test_acc']:.1%})")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
axes[1].tick_params(axis='x', rotation=45)

# Feature importance (Random Forest)
rf_pipe = results['Random Forest']['model']
importances = rf_pipe.named_steps['model'].feature_importances_
top10_idx_local = np.argsort(importances)[-10:]
top10_names  = [gene_names[top100_idx[i]] for i in top10_idx_local]
top10_values = importances[top10_idx_local]
axes[2].barh(range(10), top10_values[np.argsort(top10_values)],
             color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, 10)), alpha=0.85)
axes[2].set_yticks(range(10))
axes[2].set_yticklabels([top10_names[i] for i in np.argsort(top10_values)], fontsize=9)
axes[2].set_title("Top 10 Feature Importances\n(Random Forest)")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("03_classification_results.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03_classification_results.png")

# ============================================================
# PART 5: SURVIVAL ANALYSIS
# ============================================================

print("\n[5/6] Survival analysis...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Survival Analysis — Cancer Genomics Cohort",
             fontsize=14, fontweight='bold')

# KM by cancer type
kmf = KaplanMeierFitter()
for ct in CANCER_TYPES:
    mask = cancer_labels == ct
    kmf.fit(os_days[mask], vital_status[mask], label=ct)
    kmf.plot_survival_function(ax=axes[0], ci_show=False,
                                color=PALETTE[ct], linewidth=2)
axes[0].set_title("Kaplan-Meier: Overall Survival\nby Cancer Type")
axes[0].set_xlabel("Time (days)")
axes[0].set_ylabel("Survival Probability")
axes[0].legend(fontsize=8)

# KM by stage
stage_colors = {'I': '#2A9D8F', 'II': '#A8DADC', 'III': '#E9C46A', 'IV': '#E76F51'}
for stage in ['I', 'II', 'III', 'IV']:
    mask = stages == stage
    kmf.fit(os_days[mask], vital_status[mask], label=f'Stage {stage}')
    kmf.plot_survival_function(ax=axes[1], ci_show=False,
                                color=stage_colors[stage], linewidth=2)
axes[1].set_title("Kaplan-Meier: Overall Survival\nby Tumor Stage")
axes[1].set_xlabel("Time (days)")
axes[1].set_ylabel("Survival Probability")
axes[1].legend(fontsize=8)

# Cox PH regression
cox_df = pd.DataFrame({
    'os_days':      os_days,
    'vital_status': vital_status,
    'age':          ages,
    'stage_num':    pd.Categorical(stages).codes,
    'TP53_mut':     tp53_mut,
    'EGFR_mut':     egfr_mut,
    'mut_burden':   clinical_df['mutation_burden'].values
})
cph = CoxPHFitter()
cph.fit(cox_df, duration_col='os_days', event_col='vital_status')
cph.plot(ax=axes[2])
axes[2].set_title("Cox Proportional Hazards\nHazard Ratios (95% CI)")
axes[2].axvline(x=0, color='grey', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("04_survival_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_survival_analysis.png")

# ============================================================
# PART 6: MUTATION LANDSCAPE & DRUG RESPONSE
# ============================================================

print("\n[6/6] Mutation landscape and drug response analysis...")

# Simulate drug response
egfr_inhibitor_response = np.where(
    egfr_mut == 1,
    np.random.normal(-2.0, 0.5, N_SAMPLES),
    np.random.normal(1.5, 0.8, N_SAMPLES)
)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Mutation Landscape and Drug Response Analysis",
             fontsize=14, fontweight='bold')

# Mutation co-occurrence heatmap
mut_matrix = clinical_df[['TP53_mut','KRAS_mut','EGFR_mut','BRCA1_mut']].values
mut_corr = pd.DataFrame(mut_matrix,
           columns=['TP53','KRAS','EGFR','BRCA1']).corr()
sns.heatmap(mut_corr, annot=True, fmt='.2f', ax=axes[0,0],
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={'shrink': 0.8})
axes[0,0].set_title("Driver Gene Co-mutation Correlation")

# Mutation burden by cancer type and stage
mut_data = []
for ct in CANCER_TYPES:
    for s in ['I','II','III','IV']:
        mask = (cancer_labels == ct) & (stages == s)
        if mask.sum() > 0:
            mut_data.append({
                'cancer_type': ct, 'stage': s,
                'mean_burden': clinical_df.loc[mask, 'mutation_burden'].mean()
            })
mut_burden_df = pd.DataFrame(mut_data)
pivot = mut_burden_df.pivot(index='cancer_type', columns='stage', values='mean_burden')
sns.heatmap(pivot, annot=True, fmt='.1f', ax=axes[0,1],
            cmap='YlOrRd', cbar_kws={'shrink': 0.8})
axes[0,1].set_title("Mean Mutation Burden\nby Cancer Type and Stage")

# EGFR mutation vs drug response
egfr_wt   = egfr_inhibitor_response[egfr_mut == 0]
egfr_mt   = egfr_inhibitor_response[egfr_mut == 1]
axes[1,0].violinplot([egfr_wt, egfr_mt], positions=[0, 1], showmedians=True)
axes[1,0].set_xticks([0, 1])
axes[1,0].set_xticklabels(['EGFR Wild-type', 'EGFR Mutant'])
axes[1,0].set_title("EGFR Inhibitor Response\nby EGFR Mutation Status")
axes[1,0].set_ylabel("Log IC50")
axes[1,0].axhline(y=0, color='grey', linestyle='--', alpha=0.5)
t_stat, p_val = stats.mannwhitneyu(egfr_wt, egfr_mt, alternative='two-sided')
axes[1,0].text(0.5, 0.92, f'p = {p_val:.2e}', transform=axes[1,0].transAxes,
               ha='center', fontsize=10, color='#E63946', fontweight='bold')

# Mutation burden vs survival
axes[1,1].scatter(clinical_df['mutation_burden'], os_days,
                  c=[list(PALETTE.values())[i] for i in le.transform(cancer_labels)],
                  alpha=0.5, s=25)
m, b, r, p, _ = stats.linregress(clinical_df['mutation_burden'], os_days)
x_line = np.linspace(0, 4, 100)
axes[1,1].plot(x_line, m * x_line + b, 'k--', linewidth=1.5, alpha=0.7)
axes[1,1].set_title(f"Mutation Burden vs Overall Survival\n(r = {r:.2f}, p = {p:.3f})")
axes[1,1].set_xlabel("Total Driver Mutation Burden")
axes[1,1].set_ylabel("Overall Survival (days)")

plt.tight_layout()
plt.savefig("05_mutation_drug_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05_mutation_drug_analysis.png")

# ============================================================
# SAVE SUMMARY RESULTS
# ============================================================

# Model performance summary
perf_df = pd.DataFrame({
    'Model':     list(results.keys()),
    'CV_Accuracy': [f"{results[m]['cv_mean']:.3f} ± {results[m]['cv_std']:.3f}" for m in results],
    'Test_Accuracy': [f"{results[m]['test_acc']:.3f}" for m in results]
})
perf_df.to_csv("model_performance.csv", index=False)

# Clinical summary
clinical_df.to_csv("clinical_data.csv", index=False)

print("\n  Saved: model_performance.csv")
print("  Saved: clinical_data.csv")

print("\n" + "=" * 60)
print("Analysis Complete")
print("=" * 60)
print(f"\nBest model: {best_name}")
print(f"Test accuracy: {results[best_name]['test_acc']:.1%}")
print(f"\nOutputs:")
for f in ["01_exploratory_analysis.png", "02_dimensionality_reduction.png",
          "03_classification_results.png", "04_survival_analysis.png",
          "05_mutation_drug_analysis.png", "model_performance.csv", "clinical_data.csv"]:
    print(f"  - {f}")
