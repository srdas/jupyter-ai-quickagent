"""Deep statistical analysis for the student_performance dataset."""
import pandas as pd
import numpy as np
from scipy import stats
import json

CSV = "/Users/sanjivda/Desktop/student/student_performance.csv"
df = pd.read_csv(CSV)

# Encode binary yes/no
df["passed_bin"] = (df["passed"] == "Yes").astype(int)
df["internet_bin"] = (df["internet_access"] == "Yes").astype(int)
df["extracur_bin"] = (df["extracurricular"] == "Yes").astype(int)
df["female_bin"] = (df["gender"] == "Female").astype(int)

# Parent-education ordinal: None<HS<Bachelor<Master<PhD. Keep NaN for missing.
edu_map = {"None": 0, "High School": 1, "Bachelor": 2, "Master": 3, "PhD": 4}
df["parent_edu_ord"] = df["parent_education"].map(edu_map)

print("=" * 60)
print("PEARSON CORRELATIONS WITH final_score")
print("=" * 60)
num_cols = ["study_hours_per_week", "attendance_rate", "previous_score",
            "age", "internet_bin", "extracur_bin", "female_bin", "parent_edu_ord"]
for c in num_cols:
    sub = df[[c, "final_score"]].dropna()
    r, p = stats.pearsonr(sub[c], sub["final_score"])
    print(f"  {c:25s}  r={r:+.3f}  p={p:.2e}  n={len(sub)}")

print("\n" + "=" * 60)
print("OLS REGRESSION: final_score ~ predictors")
print("=" * 60)
# Manual OLS via numpy (avoids statsmodels dep). Impute parent_edu with median, add missing flag.
X_df = df[["study_hours_per_week", "attendance_rate", "previous_score", "age",
           "internet_bin", "extracur_bin", "female_bin"]].copy()
edu_missing = df["parent_edu_ord"].isna().astype(int)
edu_filled = df["parent_edu_ord"].fillna(df["parent_edu_ord"].median())
X_df["parent_edu_ord"] = edu_filled
X_df["parent_edu_missing"] = edu_missing

y = df["final_score"].values
X = X_df.values.astype(float)
Xc = np.column_stack([np.ones(len(X)), X])
names = ["Intercept"] + list(X_df.columns)

# beta = (X'X)^-1 X'y
beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
yhat = Xc @ beta
resid = y - yhat
n, k = Xc.shape
dof = n - k
sigma2 = (resid @ resid) / dof
cov = sigma2 * np.linalg.inv(Xc.T @ Xc)
se = np.sqrt(np.diag(cov))
tvals = beta / se
pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), dof))

ss_res = (resid ** 2).sum()
ss_tot = ((y - y.mean()) ** 2).sum()
r2 = 1 - ss_res / ss_tot
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)

print(f"  n={n}  R^2={r2:.4f}  Adj R^2={adj_r2:.4f}  RMSE={np.sqrt(sigma2):.3f}")
print(f"  {'Variable':25s}  {'Coef':>9s}  {'SE':>7s}  {'t':>7s}  {'p':>9s}")
for name, b, s, t, p in zip(names, beta, se, tvals, pvals):
    print(f"  {name:25s}  {b:+9.4f}  {s:7.3f}  {t:+7.2f}  {p:9.2e}")

# Standardized coefficients (betas) so we can rank importance on comparable scale
print("\n  Standardized (beta) coefficients:")
Xs = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
ys = (y - y.mean()) / y.std(ddof=1)
Xsc = np.column_stack([np.ones(len(Xs)), Xs])
beta_s, *_ = np.linalg.lstsq(Xsc, ys, rcond=None)
for name, b in zip(names[1:], beta_s[1:]):
    print(f"    {name:25s}  beta={b:+.3f}")

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION (IRLS): passed ~ predictors")
print("=" * 60)
# Simple IRLS
y_bin = df["passed_bin"].values.astype(float)
b = np.zeros(Xc.shape[1])
for _ in range(50):
    eta = Xc @ b
    p = 1 / (1 + np.exp(-eta))
    W = p * (1 - p)
    grad = Xc.T @ (y_bin - p)
    H = -(Xc.T * W) @ Xc
    try:
        step = np.linalg.solve(H, -grad)
    except np.linalg.LinAlgError:
        break
    b_new = b + step
    if np.max(np.abs(b_new - b)) < 1e-8:
        b = b_new
        break
    b = b_new

eta = Xc @ b
p = 1 / (1 + np.exp(-eta))
W = p * (1 - p)
cov_b = np.linalg.inv((Xc.T * W) @ Xc)
se_b = np.sqrt(np.diag(cov_b))
z = b / se_b
pv = 2 * (1 - stats.norm.cdf(np.abs(z)))
pred = (p >= 0.5).astype(int)
acc = (pred == y_bin).mean()

print(f"  n={len(y_bin)}  Accuracy={acc:.3f}  Base rate (passed)={y_bin.mean():.3f}")
print(f"  {'Variable':25s}  {'Coef':>9s}  {'OR':>7s}  {'SE':>7s}  {'z':>7s}  {'p':>9s}")
for name, bi, si, zi, pi in zip(names, b, se_b, z, pv):
    orv = np.exp(bi) if name != "Intercept" else float("nan")
    or_str = f"{orv:7.3f}" if name != "Intercept" else "   -   "
    print(f"  {name:25s}  {bi:+9.4f}  {or_str}  {si:7.3f}  {zi:+7.2f}  {pi:9.2e}")

print("\n" + "=" * 60)
print("GROUP COMPARISONS (final_score means + Welch t / ANOVA)")
print("=" * 60)

def ttest(col):
    groups = df.groupby(col)["final_score"]
    vals = [g.values for _, g in groups]
    names_g = [str(n) for n, _ in groups]
    means = [v.mean() for v in vals]
    t, pv = stats.ttest_ind(vals[0], vals[1], equal_var=False)
    print(f"  {col}: {names_g[0]} mean={means[0]:.2f} (n={len(vals[0])})  "
          f"{names_g[1]} mean={means[1]:.2f} (n={len(vals[1])})  "
          f"t={t:+.2f}  p={pv:.3e}")

ttest("gender")
ttest("internet_access")
ttest("extracurricular")
ttest("passed")

# One-way ANOVA: parent_education
sub = df.dropna(subset=["parent_education"])
groups = [g["final_score"].values for _, g in sub.groupby("parent_education")]
F, pv = stats.f_oneway(*groups)
print(f"  parent_education ANOVA:  F={F:.3f}  p={pv:.3e}")
for name, g in sub.groupby("parent_education"):
    print(f"    {name:12s}  mean={g['final_score'].mean():.2f}  n={len(g)}")

print("\nDONE")
