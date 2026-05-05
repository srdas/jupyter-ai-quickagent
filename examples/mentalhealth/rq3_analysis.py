"""
Research Question 3:
Does sleep duration mediate the relationship between screen time before sleep
and mental health outcomes (stress, anxiety, depression),
controlling for age, gender, and physical activity?
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.linear_model import LinearRegression

# ─────────────────────────────────────────────────────────────
# 1. Setup
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR = "/Users/sanjivda/Desktop/mentalhealth/output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("/Users/sanjivda/Desktop/mentalhealth/Teen_Mental_Health_Dataset.csv")
print(f"Dataset shape: {df.shape}")
print(df.dtypes)
print(df.head(3))

# ─────────────────────────────────────────────────────────────
# 2. Encode categorical variables
# ─────────────────────────────────────────────────────────────
df["gender"] = df["gender"].str.strip().str.lower().map({"male": 0, "female": 1})
df["social_interaction_level"] = (
    df["social_interaction_level"].str.strip().str.lower()
    .map({"low": 0, "medium": 1, "high": 2})
)

print("\nAfter encoding:")
print(df[["gender", "social_interaction_level"]].value_counts())

# ─────────────────────────────────────────────────────────────
# 3. Helper: OLS with sklearn + std-error via residuals
# ─────────────────────────────────────────────────────────────
def ols_with_se(X_df, y_series):
    """
    Fit OLS, return coefficients dict with std-errors via (X'X)^-1 * s^2.
    Returns: (model, coef_dict)  coef_dict keys = feature names + 'intercept'
    """
    X = X_df.values
    y = y_series.values
    n, p = X.shape

    # Add intercept column
    X_aug = np.column_stack([np.ones(n), X])
    # OLS closed form
    try:
        XtX_inv = np.linalg.inv(X_aug.T @ X_aug)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X_aug.T @ X_aug)

    beta = XtX_inv @ X_aug.T @ y
    y_hat = X_aug @ beta
    residuals = y - y_hat
    s2 = residuals @ residuals / max(n - p - 1, 1)
    var_beta = s2 * XtX_inv
    se_beta = np.sqrt(np.diag(var_beta))

    names = ["intercept"] + list(X_df.columns)
    coef_dict = {name: {"coef": beta[i], "se": se_beta[i]}
                 for i, name in enumerate(names)}
    return coef_dict

# ─────────────────────────────────────────────────────────────
# 4. Baron & Kenny Mediation Analysis
# ─────────────────────────────────────────────────────────────
# X  = screen_time_before_sleep
# M  = sleep_hours
# Y  = stress_level / anxiety_level
# Controls = age, gender, physical_activity

controls = ["age", "gender", "physical_activity"]
X_col = "screen_time_before_sleep"
M_col = "sleep_hours"
outcomes = ["stress_level", "anxiety_level"]

mediation_rows = []

for Y_col in outcomes:
    label = Y_col.replace("_level", "").capitalize()

    # ── Step 1: Total effect  Y ~ X + controls  (path c)
    step1_features = [X_col] + controls
    coef1 = ols_with_se(df[step1_features], df[Y_col])
    c      = coef1[X_col]["coef"]
    c_se   = coef1[X_col]["se"]

    # ── Step 2: X → M  (path a)
    step2_features = [X_col] + controls
    coef2 = ols_with_se(df[step2_features], df[M_col])
    a      = coef2[X_col]["coef"]
    a_se   = coef2[X_col]["se"]

    # ── Step 3: Direct effect  Y ~ X + M + controls  (paths c' and b)
    step3_features = [X_col, M_col] + controls
    coef3  = ols_with_se(df[step3_features], df[Y_col])
    c_prime      = coef3[X_col]["coef"]
    c_prime_se   = coef3[X_col]["se"]
    b            = coef3[M_col]["coef"]
    b_se         = coef3[M_col]["se"]

    # ── Indirect effect = a * b  (Sobel-style SE)
    indirect      = a * b
    indirect_se   = np.sqrt((b**2) * (a_se**2) + (a**2) * (b_se**2))
    sobel_z       = indirect / indirect_se if indirect_se > 0 else np.nan

    print(f"\n{'='*55}")
    print(f"Mediation for Y = {Y_col}")
    print(f"  Path c  (total effect X→Y):        {c:.4f}  SE={c_se:.4f}")
    print(f"  Path a  (X→M):                     {a:.4f}  SE={a_se:.4f}")
    print(f"  Path b  (M→Y | X):                 {b:.4f}  SE={b_se:.4f}")
    print(f"  Path c' (direct effect X→Y | M):   {c_prime:.4f}  SE={c_prime_se:.4f}")
    print(f"  Indirect effect (a*b):             {indirect:.4f}  SE={indirect_se:.4f}")
    print(f"  Sobel Z:                           {sobel_z:.4f}")

    # Mediation conclusion (Baron & Kenny criteria)
    # Full mediation : c sig, a sig, b sig, c' non-sig
    # Partial mediation: c sig, a sig, b sig, c' still sig but reduced
    # No mediation: a or b non-sig
    alpha = 0.05
    z_crit = 1.96

    a_sig = abs(a / a_se) > z_crit
    b_sig = abs(b / b_se) > z_crit
    c_sig = abs(c / c_se) > z_crit
    cp_sig = abs(c_prime / c_prime_se) > z_crit

    if not a_sig or not b_sig:
        conclusion = "No mediation (path a or b non-significant)"
    elif not c_sig:
        conclusion = "No mediation (total effect c non-significant)"
    elif not cp_sig:
        conclusion = "Full mediation (c' non-significant after controlling M)"
    else:
        reduction = abs(c - c_prime) / abs(c) * 100
        conclusion = f"Partial mediation (c reduced by {reduction:.1f}% after adding M)"

    print(f"  Conclusion: {conclusion}")

    # Build rows for CSV
    def row(outcome, path, coef, se, interp):
        return {"outcome": outcome, "path": path,
                "coefficient": round(coef, 6),
                "std_error": round(se, 6),
                "interpretation": interp}

    mediation_rows += [
        row(Y_col, "c  (total effect X→Y)",      c,        c_se,        "Total effect of screen_time on " + Y_col),
        row(Y_col, "a  (X→M)",                    a,        a_se,        "Effect of screen_time on sleep_hours"),
        row(Y_col, "b  (M→Y | X)",                b,        b_se,        "Effect of sleep_hours on " + Y_col + " controlling X"),
        row(Y_col, "c' (direct X→Y | M)",         c_prime,  c_prime_se,  "Direct effect of screen_time on " + Y_col + " after controlling sleep"),
        row(Y_col, "indirect (a*b)",              indirect, indirect_se, f"Indirect effect via sleep_hours; Sobel Z={sobel_z:.3f}; {conclusion}"),
    ]

med_df = pd.DataFrame(mediation_rows)
med_df.to_csv(OUTPUT_DIR + "rq3_mediation.csv", index=False)
print(f"\nSaved: {OUTPUT_DIR}rq3_mediation.csv")

# ─────────────────────────────────────────────────────────────
# 5. Sleep bins
# ─────────────────────────────────────────────────────────────
def sleep_bin(h):
    if h < 6:
        return "Short (<6h)"
    elif h <= 8:
        return "Adequate (6-8h)"
    else:
        return "Long (>8h)"

df["sleep_bin"] = df["sleep_hours"].apply(sleep_bin)
print("\nSleep bin counts:")
print(df["sleep_bin"].value_counts())

bin_stats = (
    df.groupby("sleep_bin")[["stress_level", "anxiety_level", "depression_label"]]
    .agg(["mean", "std", "count"])
    .round(4)
)
bin_stats.columns = ["_".join(c) for c in bin_stats.columns]
bin_stats = bin_stats.reset_index()
bin_stats.to_csv(OUTPUT_DIR + "rq3_sleep_bin_stats.csv", index=False)
print(f"\nSaved: {OUTPUT_DIR}rq3_sleep_bin_stats.csv")
print(bin_stats)

# ─────────────────────────────────────────────────────────────
# 6. Pearson Correlations
# ─────────────────────────────────────────────────────────────
corr_pairs = [
    ("screen_time_before_sleep", "sleep_hours",    "Screen Time vs Sleep Hours"),
    ("sleep_hours",              "stress_level",   "Sleep Hours vs Stress Level"),
    ("sleep_hours",              "anxiety_level",  "Sleep Hours vs Anxiety Level"),
    ("sleep_hours",              "depression_label","Sleep Hours vs Depression Label"),
]

corr_rows = []
print("\nPearson Correlations:")
for x_var, y_var, label in corr_pairs:
    r, p = stats.pearsonr(df[x_var], df[y_var])
    sig = "Yes" if p < 0.05 else "No"
    print(f"  {label:45s}  r={r:+.4f}  p={p:.4e}  sig={sig}")
    corr_rows.append({"pair": label, "var1": x_var, "var2": y_var,
                      "r": round(r, 6), "p_value": round(p, 8),
                      "significant_alpha0.05": sig})

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(OUTPUT_DIR + "rq3_sleep_correlations.csv", index=False)
print(f"\nSaved: {OUTPUT_DIR}rq3_sleep_correlations.csv")

# ─────────────────────────────────────────────────────────────
# 7. Figure 6 — 3-panel scatter + regression lines
# ─────────────────────────────────────────────────────────────
def add_regression_line(ax, x, y, color="crimson"):
    m, b_int, r_val, p_val, _ = stats.linregress(x, y)
    xfit = np.linspace(x.min(), x.max(), 200)
    ax.plot(xfit, m * xfit + b_int, color=color, linewidth=2,
            label=f"r={r_val:+.3f}, p={p_val:.3e}")
    return r_val, p_val

fig6, axes = plt.subplots(1, 3, figsize=(16, 5))
scatter_kw = dict(alpha=0.35, s=25, edgecolors="none")

# Panel A
ax = axes[0]
ax.scatter(df["screen_time_before_sleep"], df["sleep_hours"],
           color="#4e79a7", **scatter_kw)
r_a, p_a = add_regression_line(ax,
                                df["screen_time_before_sleep"],
                                df["sleep_hours"])
ax.set_xlabel("Screen Time Before Sleep (hrs)", fontsize=12)
ax.set_ylabel("Sleep Hours", fontsize=12)
ax.set_title("Panel A: Screen Time → Sleep Duration", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)

# Panel B
ax = axes[1]
ax.scatter(df["sleep_hours"], df["stress_level"],
           color="#f28e2b", **scatter_kw)
r_b, p_b = add_regression_line(ax, df["sleep_hours"], df["stress_level"])
ax.set_xlabel("Sleep Hours", fontsize=12)
ax.set_ylabel("Stress Level", fontsize=12)
ax.set_title("Panel B: Sleep Duration → Stress", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)

# Panel C
ax = axes[2]
ax.scatter(df["sleep_hours"], df["anxiety_level"],
           color="#59a14f", **scatter_kw)
r_c, p_c = add_regression_line(ax, df["sleep_hours"], df["anxiety_level"])
ax.set_xlabel("Sleep Hours", fontsize=12)
ax.set_ylabel("Anxiety Level", fontsize=12)
ax.set_title("Panel C: Sleep Duration → Anxiety", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)

fig6.suptitle(
    "Figure 6: Mediation Pathway — Screen Time → Sleep → Mental Health",
    fontsize=14, fontweight="bold", y=1.02
)
plt.tight_layout()
fig6.savefig(OUTPUT_DIR + "fig6_mediation_scatter.png", dpi=150, bbox_inches="tight")
fig6.savefig(OUTPUT_DIR + "fig6_mediation_scatter.pdf", bbox_inches="tight")
plt.close(fig6)
print(f"Saved: {OUTPUT_DIR}fig6_mediation_scatter.png / .pdf")

# ─────────────────────────────────────────────────────────────
# 8. Figure 7 — Grouped bar chart by sleep bin
# ─────────────────────────────────────────────────────────────
bin_order = ["Short (<6h)", "Adequate (6-8h)", "Long (>8h)"]
mean_stress  = [df[df["sleep_bin"] == b]["stress_level"].mean()  for b in bin_order]
mean_anxiety = [df[df["sleep_bin"] == b]["anxiety_level"].mean() for b in bin_order]

x_pos = np.arange(len(bin_order))
bar_width = 0.35

fig7, ax7 = plt.subplots(figsize=(9, 6))
bars1 = ax7.bar(x_pos - bar_width/2, mean_stress,  bar_width,
                label="Stress Level",  color="#e15759", edgecolor="white", linewidth=0.8)
bars2 = ax7.bar(x_pos + bar_width/2, mean_anxiety, bar_width,
                label="Anxiety Level", color="#76b7b2", edgecolor="white", linewidth=0.8)

# Value labels
for bar in bars1:
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10)
for bar in bars2:
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10)

ax7.set_xticks(x_pos)
ax7.set_xticklabels(bin_order, fontsize=12)
ax7.set_xlabel("Sleep Duration Category", fontsize=13)
ax7.set_ylabel("Mean Score", fontsize=13)
ax7.set_title("Figure 7: Mental Health Outcomes by Sleep Duration Category",
              fontsize=13, fontweight="bold")
ax7.legend(fontsize=12)
ax7.set_ylim(0, max(max(mean_stress), max(mean_anxiety)) * 1.2)
plt.tight_layout()
fig7.savefig(OUTPUT_DIR + "fig7_sleep_bins.png", dpi=150, bbox_inches="tight")
fig7.savefig(OUTPUT_DIR + "fig7_sleep_bins.pdf", bbox_inches="tight")
plt.close(fig7)
print(f"Saved: {OUTPUT_DIR}fig7_sleep_bins.png / .pdf")

# ─────────────────────────────────────────────────────────────
# 9. Multiple regression for stress_level
# ─────────────────────────────────────────────────────────────
mr_features = ["screen_time_before_sleep", "sleep_hours", "age",
               "gender", "physical_activity", "daily_social_media_hours"]
mr_coefs = ols_with_se(df[mr_features], df["stress_level"])

mr_rows = []
for name, vals in mr_coefs.items():
    t_stat = vals["coef"] / vals["se"] if vals["se"] > 0 else np.nan
    # approx p-value (large-sample)
    p_approx = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    sig = "Yes" if p_approx < 0.05 else "No"
    mr_rows.append({
        "predictor": name,
        "coefficient": round(vals["coef"], 6),
        "std_error": round(vals["se"], 6),
        "t_statistic": round(t_stat, 4),
        "p_value_approx": round(p_approx, 8),
        "significant_alpha0.05": sig
    })
    print(f"  {name:35s}  β={vals['coef']:+.4f}  SE={vals['se']:.4f}  t={t_stat:.3f}  p={p_approx:.4e}  sig={sig}")

mr_df = pd.DataFrame(mr_rows)
mr_df.to_csv(OUTPUT_DIR + "rq3_multireg.csv", index=False)
print(f"\nSaved: {OUTPUT_DIR}rq3_multireg.csv")

# ─────────────────────────────────────────────────────────────
# 10. Print final summary JSON-style
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("FINAL SUMMARY")
print("="*65)

print("\n[MEDIATION RESULTS]")
for row in mediation_rows:
    print(f"  {row['outcome']:15s} | {row['path']:30s} | β={row['coefficient']:+.4f} | SE={row['std_error']:.4f}")
    if "indirect" in row["path"]:
        print(f"    └─ {row['interpretation']}")

print("\n[SLEEP BIN STATS]")
print(bin_stats.to_string(index=False))

print("\n[CORRELATIONS]")
for _, r in corr_df.iterrows():
    print(f"  {r['pair']:45s} r={r['r']:+.4f}  p={r['p_value']:.4e}  sig={r['significant_alpha0.05']}")

print("\n[MULTIPLE REGRESSION — stress_level]")
for r in mr_rows:
    print(f"  {r['predictor']:35s} β={r['coefficient']:+.6f}  p={r['p_value_approx']:.4e}  sig={r['significant_alpha0.05']}")

print("\n[OUTPUT FILES]")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {OUTPUT_DIR}{f}")
