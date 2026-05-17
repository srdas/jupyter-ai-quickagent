"""Full analysis pipeline for customer churn dataset.
Verifies findings before notebook/paper generation.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

OUT = Path('/Users/sanjivda/Desktop/customer_churn/output')
OUT.mkdir(exist_ok=True)

SRC = '/Users/sanjivda/Desktop/customer_churn/customer churn.csv'
df_raw = pd.read_csv(SRC)
print(f"Raw shape: {df_raw.shape}")

# ---------- CLEANING ----------
df = df_raw.copy()
# Normalize categorical case
df['Region'] = df['Region'].str.title()
df['Subscription_Plan'] = df['Subscription_Plan'].str.title()

# Flag invalid numeric values (sentinels / impossible)
df.loc[(df['Age'] < 0) | (df['Age'] > 100), 'Age'] = np.nan
df.loc[(df['Monthly_Spend'] < 0) | (df['Monthly_Spend'] > 10000), 'Monthly_Spend'] = np.nan

# Impute
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Monthly_Spend'] = df['Monthly_Spend'].fillna(df['Monthly_Spend'].median())
df['Payment_Method'] = df['Payment_Method'].fillna('Unknown')

print(f"Churn rate: {df['Churn'].mean():.3f}")
print(df[['Age','Monthly_Spend']].describe())

# ---------- STATISTICAL TESTS ----------
results = {}
# Numeric: t-test churners vs non-churners
for col in ['Age','Days_Since_Last_Login','Customer_Service_Calls','Monthly_Spend']:
    g1 = df.loc[df['Churn']==1, col]
    g0 = df.loc[df['Churn']==0, col]
    t,p = stats.ttest_ind(g1, g0, equal_var=False)
    results[col] = {'mean_churn': g1.mean(), 'mean_retain': g0.mean(), 't': t, 'p': p}
    print(f"{col}: churn={g1.mean():.2f} retain={g0.mean():.2f} t={t:.3f} p={p:.4g}")

# Categorical: chi-square
print("\nChi-square tests:")
for col in ['Gender','Region','Subscription_Plan','Payment_Method']:
    ct = pd.crosstab(df[col], df['Churn'])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    print(f"{col}: chi2={chi2:.2f} dof={dof} p={p:.4g}")

# ---------- MODEL ----------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

cat_cols = ['Gender','Region','Subscription_Plan','Payment_Method']
num_cols = ['Age','Days_Since_Last_Login','Customer_Service_Calls','Monthly_Spend']
X = pd.get_dummies(df[cat_cols + num_cols], columns=cat_cols, drop_first=True)
y = df['Churn']

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X_tr, y_tr)
p_rf = rf.predict_proba(X_te)[:,1]
print(f"\nRF AUC: {roc_auc_score(y_te, p_rf):.4f}")

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_tr_s, y_tr)
p_lr = lr.predict_proba(X_te_s)[:,1]
print(f"LR AUC: {roc_auc_score(y_te, p_lr):.4f}")

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 RF importances:")
print(importances.head(10))

lr_coefs = pd.Series(lr.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
print("\nTop 10 LR |coef|:")
print(lr_coefs.head(10))

# ---------- VISUALIZATIONS ----------
plt.rcParams.update({'figure.dpi':110, 'savefig.dpi':150, 'font.size':10})

def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches='tight')
    fig.savefig(OUT / f"{name}.pdf", bbox_inches='tight')
    plt.close(fig)

# 1. Data quality: before/after
fig, axes = plt.subplots(1,2, figsize=(11,4))
axes[0].hist(df_raw['Monthly_Spend'].dropna(), bins=60, color='crimson', alpha=0.8)
axes[0].set_title('Monthly Spend — Raw (note sentinel at 99999.99)')
axes[0].set_xlabel('Monthly Spend'); axes[0].set_ylabel('Count')
axes[1].hist(df['Monthly_Spend'], bins=60, color='steelblue', alpha=0.8)
axes[1].set_title('Monthly Spend — Cleaned')
axes[1].set_xlabel('Monthly Spend')
fig.suptitle('Figure 1: Impact of cleaning on Monthly Spend distribution')
save(fig, 'fig1_data_quality')

# 2. Churn rate by categorical
fig, axes = plt.subplots(2,2, figsize=(11,8))
for ax, col in zip(axes.flat, cat_cols):
    g = df.groupby(col)['Churn'].agg(['mean','count']).sort_values('mean')
    ax.barh(g.index, g['mean'], color='teal')
    ax.axvline(df['Churn'].mean(), ls='--', color='red', label=f'overall={df["Churn"].mean():.2f}')
    ax.set_xlabel('Churn rate'); ax.set_title(col); ax.legend(fontsize=8)
fig.suptitle('Figure 2: Churn rate by categorical attribute')
fig.tight_layout()
save(fig, 'fig2_churn_by_category')

# 3. Numeric distributions split by churn
fig, axes = plt.subplots(2,2, figsize=(11,8))
for ax, col in zip(axes.flat, num_cols):
    ax.hist(df.loc[df['Churn']==0, col], bins=30, alpha=0.6, label='Retained', color='steelblue', density=True)
    ax.hist(df.loc[df['Churn']==1, col], bins=30, alpha=0.6, label='Churned', color='crimson', density=True)
    ax.set_title(col); ax.legend(fontsize=8)
fig.suptitle('Figure 3: Numeric feature distributions, by churn status')
fig.tight_layout()
save(fig, 'fig3_numeric_by_churn')

# 4. Behavioral signature: 2D heatmap of churn rate
bins_d = pd.cut(df['Days_Since_Last_Login'], bins=10)
bins_c = pd.cut(df['Customer_Service_Calls'], bins=min(10, df['Customer_Service_Calls'].nunique()))
pivot = df.groupby([bins_c, bins_d], observed=True)['Churn'].mean().unstack()
fig, ax = plt.subplots(figsize=(9,5))
im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([str(c) for c in pivot.columns], rotation=40, ha='right', fontsize=7)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([str(c) for c in pivot.index], fontsize=8)
ax.set_xlabel('Days Since Last Login (binned)')
ax.set_ylabel('Customer Service Calls (binned)')
ax.set_title('Figure 4: Churn rate heatmap — disengagement × support friction')
fig.colorbar(im, ax=ax, label='Churn rate')
save(fig, 'fig4_behavior_heatmap')

# 5. Feature importance
fig, axes = plt.subplots(1,2, figsize=(12,5))
top = importances.head(12)[::-1]
axes[0].barh(top.index, top.values, color='darkgreen')
axes[0].set_title('Random Forest feature importance (top 12)')
axes[0].set_xlabel('Importance')
top_lr = lr_coefs.head(12)[::-1]
colors = ['crimson' if v>0 else 'steelblue' for v in top_lr.values]
axes[1].barh(top_lr.index, top_lr.values, color=colors)
axes[1].axvline(0, color='black', lw=0.6)
axes[1].set_title('Logistic regression coefficients (top 12 by |coef|)')
axes[1].set_xlabel('Standardized coefficient')
fig.suptitle('Figure 5: Feature contributions to churn prediction')
fig.tight_layout()
save(fig, 'fig5_feature_importance')

# 6. ROC curves
from sklearn.metrics import roc_curve
fig, ax = plt.subplots(figsize=(6,5))
for name, p in [('Random Forest', p_rf), ('Logistic Regression', p_lr)]:
    fpr, tpr, _ = roc_curve(y_te, p)
    auc = roc_auc_score(y_te, p)
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
ax.plot([0,1],[0,1],'k--',lw=0.7)
ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
ax.set_title('Figure 6: ROC curves')
ax.legend()
save(fig, 'fig6_roc')

# Save summary metrics
import json
summary = {
    'n_rows': int(len(df)),
    'churn_rate': float(df['Churn'].mean()),
    'ttest': {k: {kk: float(vv) for kk,vv in v.items()} for k,v in results.items()},
    'rf_auc': float(roc_auc_score(y_te, p_rf)),
    'lr_auc': float(roc_auc_score(y_te, p_lr)),
    'top_rf_features': importances.head(10).to_dict(),
    'top_lr_features': lr_coefs.head(10).to_dict(),
}
with open(OUT/'summary.json','w') as f:
    json.dump(summary, f, indent=2, default=str)

print("\nDone. Files written to", OUT)
