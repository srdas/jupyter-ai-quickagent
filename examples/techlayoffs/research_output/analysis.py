import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from math import erf, sqrt

OUT = Path('/Users/sanjivda/Downloads/techlayoffs/research_output')
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
DATA = '/Users/sanjivda/Downloads/techlayoffs/tech_layoffs_hiring_trends_elite_v2.csv'

plt.rcParams.update({'figure.dpi': 110, 'savefig.bbox': 'tight'})

df = pd.read_csv(DATA)
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)
df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str), format='%Y-%b')

results = {}

# ---------- 1. Correlation heatmap ----------
num = df.select_dtypes(include=[np.number])
corr = num.corr()
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=70, ha='right', fontsize=8)
ax.set_yticklabels(corr.columns, fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i,j]:.2f}", ha='center', va='center',
                color='white' if abs(corr.values[i,j]) > 0.5 else 'black', fontsize=6)
plt.colorbar(im, ax=ax, fraction=0.04)
ax.set_title('Correlation matrix of numeric features (n=12,000)')
plt.savefig(FIG / 'fig1_correlation.png')
plt.savefig(FIG / 'fig1_correlation.pdf')
plt.close()
results['top_correlations'] = (
    corr.where(~np.eye(len(corr), dtype=bool))
        .stack().abs().sort_values(ascending=False).head(12).round(3).to_dict()
)

# ---------- 2. Layoff % by reason ----------
fig, ax = plt.subplots(figsize=(9, 5))
order = df.groupby('reason_for_layoffs')['layoff_percentage'].median().sort_values(ascending=False).index
data = [df.loc[df['reason_for_layoffs'] == r, 'layoff_percentage'].values for r in order]
bp = ax.boxplot(data, labels=order, showfliers=False, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#88aacc')
ax.set_ylabel('Layoff percentage (%)')
ax.set_xlabel('Stated reason')
ax.set_title('Layoff intensity by stated reason')
plt.xticks(rotation=15)
plt.savefig(FIG / 'fig2_reason_box.png')
plt.savefig(FIG / 'fig2_reason_box.pdf')
plt.close()
results['layoff_pct_by_reason'] = (
    df.groupby('reason_for_layoffs')['layoff_percentage']
      .agg(['mean','median','std','count']).round(2).to_dict()
)

# ---------- 3. AI risk vs layoff %, colored by reason ----------
fig, ax = plt.subplots(figsize=(9, 6))
reasons = df['reason_for_layoffs'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(reasons)))
for r, c in zip(reasons, colors):
    sub_idx = df[df['reason_for_layoffs'] == r].sample(min(400, (df['reason_for_layoffs']==r).sum()), random_state=1)
    ax.scatter(sub_idx['ai_replacement_risk'], sub_idx['layoff_percentage'], s=10, alpha=0.5, color=c, label=r)
ax.set_xlabel('AI replacement risk (1-10)')
ax.set_ylabel('Layoff percentage (%)')
ax.set_title('AI replacement risk vs layoff percentage')
ax.legend(fontsize=8, loc='upper left')
plt.savefig(FIG / 'fig3_ai_risk_scatter.png')
plt.savefig(FIG / 'fig3_ai_risk_scatter.pdf')
plt.close()

# ---------- 4. Industry x country heatmap of mean layoff % ----------
piv = df.pivot_table(index='industry', columns='country', values='layoff_percentage', aggfunc='mean')
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(piv.values, cmap='YlOrRd')
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=30)
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        ax.text(j, i, f"{piv.values[i,j]:.1f}", ha='center', va='center', fontsize=8)
plt.colorbar(im, ax=ax, label='Mean layoff %')
ax.set_title('Mean layoff % by industry and country')
plt.savefig(FIG / 'fig4_industry_country.png')
plt.savefig(FIG / 'fig4_industry_country.pdf')
plt.close()
results['industry_country_mean'] = piv.round(2).to_dict()

# ---------- 5. Hiring trend vs layoff % ----------
hire_order = ['Aggressive Hiring', 'Moderate Hiring', 'Hiring Freeze', 'Downsizing']
fig, ax = plt.subplots(figsize=(8, 5))
data2 = [df.loc[df['hiring_trend'] == h, 'layoff_percentage'].values for h in hire_order]
bp = ax.boxplot(data2, labels=hire_order, showfliers=False, patch_artist=True)
for patch, col in zip(bp['boxes'], ['#2ca02c','#98df8a','#ffbb78','#d62728']):
    patch.set_facecolor(col)
ax.set_ylabel('Layoff percentage (%)')
ax.set_title('Layoffs co-occurring with each hiring posture')
plt.savefig(FIG / 'fig5_hiring_paradox.png')
plt.savefig(FIG / 'fig5_hiring_paradox.pdf')
plt.close()
paradox = df.groupby('hiring_trend').agg(
    n=('layoff_percentage','count'),
    mean_layoff_pct=('layoff_percentage','mean'),
    median_open_roles=('open_roles','median'),
    mean_sentiment=('employee_sentiment','mean'),
    mean_job_security=('job_security_score','mean'),
).round(2)
results['hiring_paradox'] = paradox.to_dict()

# ---------- 6. Top hiring roles ----------
role_share = (df['top_hiring_role'].value_counts(normalize=True) * 100).round(2)
fig, ax = plt.subplots(figsize=(8, 5))
role_share.sort_values().plot(kind='barh', ax=ax, color='#4477aa')
ax.set_xlabel('Share of records (%)')
ax.set_title('Most-hired roles during layoff records')
plt.savefig(FIG / 'fig6_top_roles.png')
plt.savefig(FIG / 'fig6_top_roles.pdf')
plt.close()
results['top_role_share'] = role_share.to_dict()

# ---------- 7. Time series ----------
ts = df.groupby('date')['layoff_percentage'].mean().sort_index()
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(ts.index, ts.values, marker='o', color='#cc4444')
ax.set_ylabel('Mean layoff % across firms')
ax.set_title('Monthly mean layoff percentage (2024-2026)')
ax.grid(alpha=0.3)
plt.savefig(FIG / 'fig7_timeseries.png')
plt.savefig(FIG / 'fig7_timeseries.pdf')
plt.close()
results['ts_summary'] = {
    'min_month': str(ts.idxmin().date()), 'min_val': float(ts.min().round(3)),
    'max_month': str(ts.idxmax().date()), 'max_val': float(ts.max().round(3)),
    'overall_mean': float(ts.mean().round(3)),
    'std_across_months': float(ts.std().round(3)),
}

# ---------- 8. OLS regression ----------
y = df['layoff_percentage'].values
predictors_num = ['ai_replacement_risk', 'ai_adoption_level', 'ai_automation_impact',
                  'revenue_growth_percent', 'salary_budget_change', 'stock_growth_percent',
                  'remote_jobs_percentage']
X_num = df[predictors_num].values
dummies = pd.get_dummies(df[['industry','country','company_size','market_condition','reason_for_layoffs']], drop_first=True)
X_full = np.hstack([np.ones((len(df),1)), X_num, dummies.values.astype(float)])
names = ['Intercept'] + predictors_num + list(dummies.columns)
beta, *_ = np.linalg.lstsq(X_full, y, rcond=None)
yhat = X_full @ beta
resid = y - yhat
ss_res = (resid**2).sum(); ss_tot = ((y - y.mean())**2).sum()
r2 = 1 - ss_res/ss_tot
n, k = X_full.shape
sigma2 = ss_res / (n - k)
cov = sigma2 * np.linalg.pinv(X_full.T @ X_full)
se = np.sqrt(np.diag(cov))
tvals = beta / se
def pval(z):
    return 2 * (1 - 0.5*(1+erf(abs(z)/sqrt(2))))
pvals = [pval(t) for t in tvals]
reg = pd.DataFrame({'coef': beta, 'se': se, 't': tvals, 'p': pvals}, index=names).round(4)
reg.to_csv(OUT / 'regression_layoff_pct.csv')
results['regression'] = {'r2': float(round(r2, 4)), 'n': int(n), 'k': int(k)}
print('R^2 =', round(r2, 4))
print(reg.to_string())

def _stringify_keys(obj):
    if isinstance(obj, dict):
        return {str(k): _stringify_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_keys(v) for v in obj]
    return obj

with open(OUT / 'results.json', 'w') as f:
    json.dump(_stringify_keys(results), f, indent=2, default=str)

print('\nDone. Figures in', FIG)
