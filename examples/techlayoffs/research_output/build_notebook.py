import nbformat as nbf
from pathlib import Path

OUT = Path('/Users/sanjivda/Downloads/techlayoffs/research_output')
nb = nbf.v4.new_notebook()
cells = []

INTRO_MD = """# Tech Layoffs & AI Hiring Trends (2024-2026): An Empirical Study

**Dataset:** `tech_layoffs_hiring_trends_elite_v2.csv` (12,000 records, 23 columns)

**Research questions**

1. Are firms with higher AI adoption / AI replacement risk associated with larger layoff percentages, after controlling for revenue, market condition, geography, and company size?
2. Do companies show a *layoff-hiring paradox*: laying off staff while simultaneously hiring (especially ML / data roles)?
3. Which industry x country combinations exhibit the most severe layoff intensity?

This notebook reproduces every figure and statistic in the accompanying paper. All figures are saved as both PNG (inline) and PDF (for LaTeX inclusion) under `figures/`.
"""
cells.append(nbf.v4.new_markdown_cell(INTRO_MD))

SETUP = '''%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import erf, sqrt

OUT = Path("/Users/sanjivda/Downloads/techlayoffs/research_output")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DATA = "/Users/sanjivda/Downloads/techlayoffs/tech_layoffs_hiring_trends_elite_v2.csv"

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})

df = pd.read_csv(DATA)
month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
df["month"] = pd.Categorical(df["month"], categories=month_order, ordered=True)
df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str), format="%Y-%b")
print("Shape:", df.shape)
df.head()
'''
cells.append(nbf.v4.new_markdown_cell("## 1. Setup and data load"))
cells.append(nbf.v4.new_code_cell(SETUP))

CORR = '''num = df.select_dtypes(include=[np.number])
corr = num.corr()
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=70, ha="right", fontsize=8)
ax.set_yticklabels(corr.columns, fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                color="white" if abs(corr.values[i,j]) > 0.5 else "black", fontsize=6)
plt.colorbar(im, ax=ax, fraction=0.04)
ax.set_title("Correlation matrix of numeric features (n=12,000)")
plt.savefig(FIG / "fig1_correlation.png"); plt.savefig(FIG / "fig1_correlation.pdf")
plt.show()
top = corr.where(~np.eye(len(corr), dtype=bool)).stack().abs().sort_values(ascending=False).head(10)
print(top.round(3))
'''
cells.append(nbf.v4.new_markdown_cell("## 2. Correlation structure\\n\\nFirst we examine the correlation among numeric features. The strongest associations are between `layoff_percentage` and `job_security_score` (negative), and among the AI-related variables."))
cells.append(nbf.v4.new_code_cell(CORR))

REASON = '''order = df.groupby("reason_for_layoffs")["layoff_percentage"].median().sort_values(ascending=False).index
data = [df.loc[df["reason_for_layoffs"] == r, "layoff_percentage"].values for r in order]
fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot(data, tick_labels=order, showfliers=False, patch_artist=True)
for patch in bp["boxes"]: patch.set_facecolor("#88aacc")
ax.set_ylabel("Layoff percentage (%)"); ax.set_xlabel("Stated reason")
ax.set_title("Layoff intensity by stated reason"); plt.xticks(rotation=15)
plt.savefig(FIG / "fig2_reason_box.png"); plt.savefig(FIG / "fig2_reason_box.pdf")
plt.show()
df.groupby("reason_for_layoffs")["layoff_percentage"].agg(["mean","median","std","count"]).round(2)
'''
cells.append(nbf.v4.new_markdown_cell("## 3. Layoff intensity by stated reason\\n\\nThe five stated reasons (AI Automation, Cost Cutting, Market Slowdown, Overhiring Correction, Restructuring) have nearly indistinguishable distributions of `layoff_percentage`. The 'AI Automation' label, on its own, is **not** associated with deeper cuts than the others."))
cells.append(nbf.v4.new_code_cell(REASON))

AISCAT = '''fig, ax = plt.subplots(figsize=(9, 6))
reasons = df["reason_for_layoffs"].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(reasons)))
for r, c in zip(reasons, colors):
    sub = df[df["reason_for_layoffs"] == r].sample(min(400, (df["reason_for_layoffs"] == r).sum()), random_state=1)
    ax.scatter(sub["ai_replacement_risk"], sub["layoff_percentage"], s=10, alpha=0.5, color=c, label=r)
ax.set_xlabel("AI replacement risk (1-10)"); ax.set_ylabel("Layoff percentage (%)")
ax.set_title("AI replacement risk vs layoff percentage")
ax.legend(fontsize=8, loc="upper left")
plt.savefig(FIG / "fig3_ai_risk_scatter.png"); plt.savefig(FIG / "fig3_ai_risk_scatter.pdf")
plt.show()
print("Pearson r:", df[["ai_replacement_risk","layoff_percentage"]].corr().iloc[0,1].round(3))
'''
cells.append(nbf.v4.new_markdown_cell("## 4. AI replacement risk vs layoff percentage\\n\\nUnlike the *stated reason*, the *measured* `ai_replacement_risk` score correlates strongly with layoff percentage (r ~ 0.47)."))
cells.append(nbf.v4.new_code_cell(AISCAT))

INDCO = '''piv = df.pivot_table(index="industry", columns="country", values="layoff_percentage", aggfunc="mean")
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(piv.values, cmap="YlOrRd")
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=30)
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        ax.text(j, i, f"{piv.values[i,j]:.1f}", ha="center", va="center", fontsize=8)
plt.colorbar(im, ax=ax, label="Mean layoff %")
ax.set_title("Mean layoff % by industry and country")
plt.savefig(FIG / "fig4_industry_country.png"); plt.savefig(FIG / "fig4_industry_country.pdf")
plt.show()
piv.round(2)
'''
cells.append(nbf.v4.new_markdown_cell("## 5. Industry x country heterogeneity\\n\\nThe Cloud and Social Media sectors in the UK show the highest mean layoff percentages; FinTech tends to be milder."))
cells.append(nbf.v4.new_code_cell(INDCO))

PARADOX = '''hire_order = ["Aggressive Hiring", "Moderate Hiring", "Hiring Freeze", "Downsizing"]
data2 = [df.loc[df["hiring_trend"] == h, "layoff_percentage"].values for h in hire_order]
fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot(data2, tick_labels=hire_order, showfliers=False, patch_artist=True)
for patch, col in zip(bp["boxes"], ["#2ca02c","#98df8a","#ffbb78","#d62728"]):
    patch.set_facecolor(col)
ax.set_ylabel("Layoff percentage (%)")
ax.set_title("Layoffs co-occurring with each hiring posture")
plt.savefig(FIG / "fig5_hiring_paradox.png"); plt.savefig(FIG / "fig5_hiring_paradox.pdf")
plt.show()
df.groupby("hiring_trend").agg(
    n=("layoff_percentage","count"),
    mean_layoff_pct=("layoff_percentage","mean"),
    median_open_roles=("open_roles","median"),
    mean_sentiment=("employee_sentiment","mean"),
    mean_job_security=("job_security_score","mean"),
).round(2).loc[hire_order]
'''
cells.append(nbf.v4.new_markdown_cell("## 6. The layoff-hiring paradox\\n\\nFirms in 'Aggressive Hiring' still post a mean ~5.2% layoff rate; firms in 'Hiring Freeze' average 13%, but maintain a median of ~1,650 open roles. Layoffs and hiring are not mutually exclusive."))
cells.append(nbf.v4.new_code_cell(PARADOX))

ROLES = '''role_share = (df["top_hiring_role"].value_counts(normalize=True) * 100).round(2)
fig, ax = plt.subplots(figsize=(8, 5))
role_share.sort_values().plot(kind="barh", ax=ax, color="#4477aa")
ax.set_xlabel("Share of records (%)")
ax.set_title("Most-hired roles during layoff records")
plt.savefig(FIG / "fig6_top_roles.png"); plt.savefig(FIG / "fig6_top_roles.pdf")
plt.show()
role_share
'''
cells.append(nbf.v4.new_markdown_cell("## 7. Which roles are still being hired?\\n\\nML Engineer and Data Scientist together account for ~36% of 'top hiring role' assignments - reinforcing the paradox: AI-related talent is in demand even at firms cutting headcount."))
cells.append(nbf.v4.new_code_cell(ROLES))

TS = '''ts = df.groupby("date")["layoff_percentage"].mean().sort_index()
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(ts.index, ts.values, marker="o", color="#cc4444")
ax.set_ylabel("Mean layoff % across firms")
ax.set_title("Monthly mean layoff percentage (2024-2026)")
ax.grid(alpha=0.3)
plt.savefig(FIG / "fig7_timeseries.png"); plt.savefig(FIG / "fig7_timeseries.pdf")
plt.show()
print("Range:", ts.min().round(2), "to", ts.max().round(2), "| std across months:", ts.std().round(3))
'''
cells.append(nbf.v4.new_markdown_cell("## 8. Temporal trend\\n\\nMonthly mean layoff percentage fluctuates within ~12-14% with no clear linear trend across 2024-2026."))
cells.append(nbf.v4.new_code_cell(TS))

REG = '''y = df["layoff_percentage"].values
predictors_num = ["ai_replacement_risk","ai_adoption_level","ai_automation_impact",
                  "revenue_growth_percent","salary_budget_change","stock_growth_percent",
                  "remote_jobs_percentage"]
X_num = df[predictors_num].values
dummies = pd.get_dummies(df[["industry","country","company_size","market_condition","reason_for_layoffs"]], drop_first=True)
X_full = np.hstack([np.ones((len(df),1)), X_num, dummies.values.astype(float)])
names = ["Intercept"] + predictors_num + list(dummies.columns)
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
def pval(z): return 2 * (1 - 0.5*(1+erf(abs(z)/sqrt(2))))
pvals = [pval(t) for t in tvals]
reg = pd.DataFrame({"coef": beta, "se": se, "t": tvals, "p": pvals}, index=names).round(4)
print("R^2 =", round(r2, 4), " n =", n, " k =", k)
reg
'''
cells.append(nbf.v4.new_markdown_cell("## 9. OLS regression: drivers of layoff percentage\\n\\nWe regress `layoff_percentage` on AI-related scores, financial signals, and one-hot fixed effects for industry, country, company size, market condition, and stated reason. The model achieves R^2 ~ 0.72."))
cells.append(nbf.v4.new_code_cell(REG))

CONCL = """## 10. Findings summary

- **AI exposure matters, but not the AI label.** `ai_replacement_risk` is one of the largest standardised drivers of `layoff_percentage` (coef ~ +1.34, p < 1e-50). The stated reason 'AI Automation' itself is statistically indistinguishable from 'Cost Cutting' or 'Restructuring'. Firms experiencing automation risk lay off more *regardless of how they label it*.
- **Macro conditions dominate.** `market_condition = Recession` adds ~+14 percentage points to layoff rate vs. a Bull-Market baseline - the single largest effect in the model.
- **Revenue, not stock, is what is being defended.** `revenue_growth_percent` and `salary_budget_change` (correlation 0.82) move together and predict lower layoffs and higher sentiment. `stock_growth_percent` is essentially uncorrelated with layoffs (coef p = 0.69).
- **The hiring paradox is real.** Firms classified as 'Aggressive Hiring' still average 5.2% layoff rates; firms in 'Hiring Freeze' still post ~1,650 open roles (median). ML Engineer + Data Scientist account for ~36% of top hires - AI talent is being acquired at the same time other roles are eliminated.
- **Geography matters at the margin.** Country fixed effects are individually small (~0.1-0.2 pp) and largely insignificant once industry and macro controls are present. Industry effects are significant for Cloud and E-Commerce (~+0.5 pp vs AI baseline).
- **No clear time trend.** Monthly mean layoff % oscillates within 11.7-14.0 across 36 months with no monotonic direction.
"""
cells.append(nbf.v4.new_markdown_cell(CONCL))

nb["cells"] = cells
with open(OUT / "research_notebook.ipynb", "w") as f:
    nbf.write(nb, f)
print("Wrote", OUT / "research_notebook.ipynb", "with", len(cells), "cells")
