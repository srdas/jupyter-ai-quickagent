"""
Research Question 1 Analysis:
Which combination of behavioral and lifestyle factors best predict depression risk
in teenagers, and what are the relative effect sizes?
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATA_PATH = "/Users/sanjivda/Desktop/mentalhealth/Teen_Mental_Health_Dataset.csv"
OUT_DIR   = "/Users/sanjivda/Desktop/mentalhealth/output"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print(f"\nDepression label distribution:\n{df['depression_label'].value_counts()}")
print(f"Missing values:\n{df.isnull().sum()}")

# ─────────────────────────────────────────────
# 2. SUMMARY STATISTICS (raw numeric cols)
# ─────────────────────────────────────────────
numeric_cols_raw = df.select_dtypes(include=[np.number]).columns.tolist()
summary = df[numeric_cols_raw].agg(['mean', 'std', 'min', 'max']).T
summary.columns = ['mean', 'std', 'min', 'max']
summary.index.name = 'feature'
summary_path = os.path.join(OUT_DIR, "rq1_summary_stats.csv")
summary.to_csv(summary_path)
print(f"\n[✓] Summary stats saved → {summary_path}")
print(summary.to_string())

# ─────────────────────────────────────────────
# 3. ENCODE CATEGORICALS
# ─────────────────────────────────────────────
df_enc = df.copy()

# gender: male=0, female=1
df_enc['gender'] = df_enc['gender'].str.strip().str.lower().map({'male': 0, 'female': 1})

# platform_usage: Instagram=0, TikTok=1, Both=2
df_enc['platform_usage'] = df_enc['platform_usage'].str.strip().map(
    {'Instagram': 0, 'TikTok': 1, 'Both': 2}
)

# social_interaction_level: low=0, medium=1, high=2
df_enc['social_interaction_level'] = df_enc['social_interaction_level'].str.strip().str.lower().map(
    {'low': 0, 'medium': 1, 'high': 2}
)

print(f"\nEncoding check – null counts after encoding:")
print(df_enc[['gender','platform_usage','social_interaction_level']].isnull().sum())

# ─────────────────────────────────────────────
# 4. PREPARE FEATURES & TARGET
# ─────────────────────────────────────────────
feature_cols = [c for c in df_enc.columns if c != 'depression_label']
X = df_enc[feature_cols].dropna()
y = df_enc.loc[X.index, 'depression_label']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape:         {y.shape}")

# Scale features for logistic regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 5. LOGISTIC REGRESSION
# ─────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
model.fit(X_train, y_train)

y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc      = roc_auc_score(y_test, y_pred_proba)

# McFadden's pseudo-R²
# R² = 1 - (log-likelihood of model / log-likelihood of null model)
ll_model = -log_loss(y_test, model.predict_proba(X_test), normalize=False)
# Null model: always predict the base rate
p_null   = y_train.mean()
ll_null  = len(y_test) * (p_null * np.log(p_null + 1e-15) + (1 - p_null) * np.log(1 - p_null + 1e-15))
mcfadden_r2 = 1 - (ll_model / ll_null)

# Build coefficient table
coef_df = pd.DataFrame({
    'feature':    feature_cols,
    'coefficient': model.coef_[0],
    'odds_ratio':  np.exp(model.coef_[0])
}).sort_values('coefficient', key=abs, ascending=False).reset_index(drop=True)

coef_path = os.path.join(OUT_DIR, "rq1_logreg_coefs.csv")
coef_df.to_csv(coef_path, index=False)
print(f"\n[✓] Logistic regression coefficients saved → {coef_path}")

# ─────────────────────────────────────────────
# 6. POINT-BISERIAL CORRELATIONS
# ─────────────────────────────────────────────
all_numeric = df_enc.select_dtypes(include=[np.number]).columns.tolist()
predictors  = [c for c in all_numeric if c != 'depression_label']

corr_rows = []
for col in predictors:
    valid = df_enc[[col, 'depression_label']].dropna()
    r, p = stats.pointbiserialr(valid[col], valid['depression_label'])
    corr_rows.append({'feature': col, 'point_biserial_r': round(r, 4), 'p_value': round(p, 6)})

corr_df = pd.DataFrame(corr_rows).sort_values('point_biserial_r', key=abs, ascending=False).reset_index(drop=True)
corr_path = os.path.join(OUT_DIR, "rq1_correlations.csv")
corr_df.to_csv(corr_path, index=False)
print(f"[✓] Point-biserial correlations saved → {corr_path}")

# ─────────────────────────────────────────────
# 7. FIGURE 1 – Feature Importance Bar Chart
# ─────────────────────────────────────────────
fig1_df = coef_df.copy()
fig1_df['abs_coef'] = fig1_df['coefficient'].abs()
fig1_df = fig1_df.sort_values('abs_coef', ascending=True)   # ascending so top bar is at top

# Colour: positive coef = red (risk), negative = blue (protective)
bar_colors = ['#d73027' if c > 0 else '#4575b4' for c in fig1_df['coefficient']]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(fig1_df['feature'], fig1_df['abs_coef'], color=bar_colors, edgecolor='white', height=0.65)

# Value labels
for bar, val in zip(bars, fig1_df['abs_coef']):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', ha='left', fontsize=9, color='#333333')

ax.set_xlabel('Absolute Logistic Regression Coefficient (Standardised)', fontsize=11)
ax.set_ylabel('Feature', fontsize=11)
ax.set_title('Figure 1: Logistic Regression Feature Importance for Depression Prediction',
             fontsize=12, fontweight='bold', pad=14)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#d73027', label='Positive (risk-increasing)'),
                   Patch(facecolor='#4575b4', label='Negative (protective)')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

ax.spines[['top', 'right']].set_visible(False)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.tight_layout()

fig1_png = os.path.join(OUT_DIR, "fig1_feature_importance.png")
fig1_pdf = os.path.join(OUT_DIR, "fig1_feature_importance.pdf")
fig.savefig(fig1_png, dpi=150, bbox_inches='tight')
fig.savefig(fig1_pdf, bbox_inches='tight')
plt.close(fig)
print(f"[✓] Figure 1 saved → {fig1_png}")
print(f"[✓] Figure 1 saved → {fig1_pdf}")

# ─────────────────────────────────────────────
# 8. FIGURE 2 – Correlation Heatmap
# ─────────────────────────────────────────────
numeric_df = df_enc[all_numeric].copy()
# Rename columns for readability in heatmap
rename_map = {
    'daily_social_media_hours': 'SM Hours',
    'platform_usage':           'Platform',
    'sleep_hours':              'Sleep Hrs',
    'screen_time_before_sleep': 'Screen@Sleep',
    'academic_performance':     'Academic',
    'physical_activity':        'Physical Act.',
    'social_interaction_level': 'Social Interact.',
    'stress_level':             'Stress',
    'anxiety_level':            'Anxiety',
    'addiction_level':          'Addiction',
    'depression_label':         'Depression',
    'age':                      'Age',
    'gender':                   'Gender',
}
numeric_df_plot = numeric_df.rename(columns=rename_map)

corr_matrix = numeric_df_plot.corr()

fig2, ax2 = plt.subplots(figsize=(13, 10))
mask = np.zeros_like(corr_matrix, dtype=bool)   # show full matrix (no mask)

hm = sns.heatmap(
    corr_matrix,
    ax=ax2,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.5,
    linecolor='#cccccc',
    annot_kws={'size': 8},
    square=True,
    cbar_kws={'shrink': 0.8, 'label': 'Pearson r'}
)

ax2.set_title('Figure 2: Correlation Matrix of Mental Health Indicators',
              fontsize=13, fontweight='bold', pad=16)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=9)
plt.tight_layout()

fig2_png = os.path.join(OUT_DIR, "fig2_correlation_heatmap.png")
fig2_pdf = os.path.join(OUT_DIR, "fig2_correlation_heatmap.pdf")
fig2.savefig(fig2_png, dpi=150, bbox_inches='tight')
fig2.savefig(fig2_pdf, bbox_inches='tight')
plt.close(fig2)
print(f"[✓] Figure 2 saved → {fig2_png}")
print(f"[✓] Figure 2 saved → {fig2_pdf}")

# ─────────────────────────────────────────────
# 9. PRINT KEY STATISTICS
# ─────────────────────────────────────────────
print("\n" + "="*65)
print("  RQ1 KEY STATISTICS SUMMARY")
print("="*65)
print(f"\n  MODEL PERFORMANCE")
print(f"  {'Accuracy':<30} {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  {'AUC-ROC':<30} {auc:.4f}")
print(f"  {'McFadden Pseudo-R²':<30} {mcfadden_r2:.4f}")

print(f"\n  TOP 5 PREDICTORS (by |coefficient|)")
print(f"  {'Rank':<5} {'Feature':<30} {'Coefficient':>12} {'Odds Ratio':>12}")
print(f"  {'-'*60}")
for i, row in coef_df.head(5).iterrows():
    direction = '▲ risk' if row['coefficient'] > 0 else '▼ protective'
    print(f"  {i+1:<5} {row['feature']:<30} {row['coefficient']:>12.4f} {row['odds_ratio']:>12.4f}  [{direction}]")

print(f"\n  TOP 5 POINT-BISERIAL CORRELATIONS WITH depression_label")
print(f"  {'Rank':<5} {'Feature':<30} {'r':>10} {'p-value':>12}")
print(f"  {'-'*60}")
for i, row in corr_df.head(5).iterrows():
    sig = '***' if row['p_value'] < 0.001 else ('**' if row['p_value'] < 0.01 else ('*' if row['p_value'] < 0.05 else ''))
    print(f"  {i+1:<5} {row['feature']:<30} {row['point_biserial_r']:>10.4f} {row['p_value']:>12.6f}  {sig}")

print(f"\n  OUTPUT FILES")
for path in [summary_path, coef_path, corr_path, fig1_png, fig1_pdf, fig2_png, fig2_pdf]:
    print(f"  → {path}")

print("\n" + "="*65)

# ─────────────────────────────────────────────
# RETURN DICT for programmatic access
# ─────────────────────────────────────────────
results = {
    "accuracy":     round(accuracy,    4),
    "auc":          round(auc,         4),
    "mcfadden_r2":  round(mcfadden_r2, 4),
    "top5_coefs":   coef_df.head(5).to_dict(orient='records'),
    "top5_corrs":   corr_df.head(5).to_dict(orient='records'),
    "output_files": [summary_path, coef_path, corr_path, fig1_png, fig1_pdf, fig2_png, fig2_pdf],
    "coef_df":      coef_df,
    "corr_df":      corr_df,
}
print("\nScript completed successfully.")
