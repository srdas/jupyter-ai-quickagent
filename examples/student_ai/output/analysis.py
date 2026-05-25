"""
Empirical analysis of GenAI usage and student academic outcomes.
Generates statistics, models, and figures for RQ1, RQ2, RQ3.
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
warnings.filterwarnings("ignore")

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(OUT), "ai_student_impact_dataset.csv")
FIGS = OUT  # figures saved alongside

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150,
    "axes.titlesize": 12, "axes.labelsize": 10,
    "font.size": 10, "figure.autolayout": True,
})

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)

# ---------- Load ----------
df = pd.read_csv(DATA)
df["GPA_Delta"] = df["Post_Semester_GPA"] - df["Pre_Semester_GPA"]
results = {"n": int(len(df)), "columns": list(df.columns)}

# Categorical encodings used across models
CAT = ["Major_Category", "Year_of_Study", "Primary_Use_Case",
       "Institutional_Policy", "Paid_Subscription"]
for c in CAT:
    df[c] = df[c].astype("category")


# =================================================================
# RQ1: Does heavier weekly GenAI use harm GPA growth and skill retention,
# controlling for major, year, prompt skill, and traditional study hours?
# =================================================================
print("\n=== RQ1: GenAI hours and outcomes ===")

rq1_models = {}

# Model A: GPA_Delta ~ GenAI hours + controls
m1 = smf.ols(
    "GPA_Delta ~ Weekly_GenAI_Hours + Pre_Semester_GPA + Traditional_Study_Hours "
    "+ Prompt_Engineering_Skill + Tool_Diversity + Perceived_AI_Dependency "
    "+ C(Major_Category) + C(Year_of_Study) + C(Primary_Use_Case)",
    data=df,
).fit()
rq1_models["GPA_Delta"] = m1
print(m1.summary().tables[1])

# Model B: Skill retention
m2 = smf.ols(
    "Skill_Retention_Score ~ Weekly_GenAI_Hours + Pre_Semester_GPA + Traditional_Study_Hours "
    "+ Prompt_Engineering_Skill + Tool_Diversity + Perceived_AI_Dependency "
    "+ C(Major_Category) + C(Year_of_Study) + C(Primary_Use_Case)",
    data=df,
).fit()
rq1_models["Skill_Retention"] = m2
print(m2.summary().tables[1])

# --- Figure 1: GPA delta vs GenAI hours, binned with 95% CI ---
bins = pd.cut(df["Weekly_GenAI_Hours"], bins=12)
agg = df.groupby(bins, observed=True).agg(
    x=("Weekly_GenAI_Hours", "mean"),
    y=("GPA_Delta", "mean"),
    se=("GPA_Delta", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
    n=("GPA_Delta", "size"),
).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.errorbar(agg["x"], agg["y"], yerr=1.96 * agg["se"], fmt="o-",
            color="#1f4e79", ecolor="#9ab7d3", capsize=3)
ax.axhline(0, color="grey", lw=0.8, ls="--")
ax.set_xlabel("Weekly GenAI usage (hours)")
ax.set_ylabel("Mean GPA change (Post - Pre)")
ax.set_title("Figure 1. GPA change vs. weekly GenAI use (binned, 95% CI)")
save(fig, "fig1_gpa_delta_vs_genai")

# --- Figure 2: Skill retention vs GenAI hours, by tool diversity ---
fig, ax = plt.subplots(figsize=(7, 4.2))
df["TD_bin"] = pd.cut(df["Tool_Diversity"], bins=[-0.1, 1, 3, 6, 10],
                      labels=["1", "2-3", "4-6", "7+"])
for label, sub in df.groupby("TD_bin", observed=True):
    b = pd.cut(sub["Weekly_GenAI_Hours"], bins=10)
    g = sub.groupby(b, observed=True).agg(
        x=("Weekly_GenAI_Hours", "mean"),
        y=("Skill_Retention_Score", "mean"),
    ).reset_index(drop=True)
    ax.plot(g["x"], g["y"], "o-", label=f"Tools used: {label}")
ax.set_xlabel("Weekly GenAI usage (hours)")
ax.set_ylabel("Mean skill retention score")
ax.set_title("Figure 2. Skill retention vs. GenAI hours, by tool diversity")
ax.legend(title="", fontsize=8)
save(fig, "fig2_retention_vs_genai")

rq1_summary = {
    "GPA_Delta_R2": float(m1.rsquared),
    "GPA_Delta_GenAI_coef": float(m1.params["Weekly_GenAI_Hours"]),
    "GPA_Delta_GenAI_p": float(m1.pvalues["Weekly_GenAI_Hours"]),
    "Skill_R2": float(m2.rsquared),
    "Skill_GenAI_coef": float(m2.params["Weekly_GenAI_Hours"]),
    "Skill_GenAI_p": float(m2.pvalues["Weekly_GenAI_Hours"]),
}
results["RQ1"] = rq1_summary
print(json.dumps(rq1_summary, indent=2))


# =================================================================
# RQ2: Does institutional policy moderate the GenAI -> outcome link?
# =================================================================
print("\n=== RQ2: Policy moderation ===")

m3 = smf.ols(
    "GPA_Delta ~ Weekly_GenAI_Hours * C(Institutional_Policy) "
    "+ Pre_Semester_GPA + Traditional_Study_Hours + Prompt_Engineering_Skill "
    "+ Tool_Diversity + C(Major_Category) + C(Year_of_Study)",
    data=df,
).fit()
print(m3.summary().tables[1])

m4 = smf.ols(
    "Skill_Retention_Score ~ Weekly_GenAI_Hours * C(Institutional_Policy) "
    "+ Pre_Semester_GPA + Traditional_Study_Hours + Prompt_Engineering_Skill "
    "+ Tool_Diversity + C(Major_Category) + C(Year_of_Study)",
    data=df,
).fit()
print(m4.summary().tables[1])

# Per-policy slope summary
slopes = []
for pol, sub in df.groupby("Institutional_Policy", observed=True):
    mm = smf.ols("GPA_Delta ~ Weekly_GenAI_Hours + Pre_Semester_GPA + Traditional_Study_Hours "
                 "+ Prompt_Engineering_Skill + Tool_Diversity", data=sub).fit()
    sm2 = smf.ols("Skill_Retention_Score ~ Weekly_GenAI_Hours + Pre_Semester_GPA + Traditional_Study_Hours "
                  "+ Prompt_Engineering_Skill + Tool_Diversity", data=sub).fit()
    slopes.append({
        "policy": pol, "n": int(len(sub)),
        "gpa_slope": float(mm.params["Weekly_GenAI_Hours"]),
        "gpa_p": float(mm.pvalues["Weekly_GenAI_Hours"]),
        "skill_slope": float(sm2.params["Weekly_GenAI_Hours"]),
        "skill_p": float(sm2.pvalues["Weekly_GenAI_Hours"]),
    })
slopes_df = pd.DataFrame(slopes)
print(slopes_df)

# --- Figure 3: GPA-delta vs hours by policy ---
fig, ax = plt.subplots(figsize=(7.5, 4.5))
colors = {"Strict_Ban": "#c0392b", "Allowed_With_Citation": "#2c3e50",
          "Actively_Encouraged": "#27ae60"}
for pol, sub in df.groupby("Institutional_Policy", observed=True):
    b = pd.cut(sub["Weekly_GenAI_Hours"], bins=10)
    g = sub.groupby(b, observed=True).agg(
        x=("Weekly_GenAI_Hours", "mean"),
        y=("GPA_Delta", "mean"),
        se=("GPA_Delta", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
    ).reset_index(drop=True)
    ax.errorbar(g["x"], g["y"], yerr=1.96 * g["se"], fmt="o-",
                color=colors.get(pol, "k"), label=pol, capsize=2)
ax.axhline(0, color="grey", lw=0.8, ls="--")
ax.set_xlabel("Weekly GenAI usage (hours)")
ax.set_ylabel("Mean GPA change (Post - Pre)")
ax.set_title("Figure 3. GPA change vs. GenAI hours by institutional policy")
ax.legend(fontsize=8)
save(fig, "fig3_policy_moderation_gpa")

# --- Figure 4: Skill retention by policy ---
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for pol, sub in df.groupby("Institutional_Policy", observed=True):
    b = pd.cut(sub["Weekly_GenAI_Hours"], bins=10)
    g = sub.groupby(b, observed=True).agg(
        x=("Weekly_GenAI_Hours", "mean"),
        y=("Skill_Retention_Score", "mean"),
    ).reset_index(drop=True)
    ax.plot(g["x"], g["y"], "o-", color=colors.get(pol, "k"), label=pol)
ax.set_xlabel("Weekly GenAI usage (hours)")
ax.set_ylabel("Mean skill retention score")
ax.set_title("Figure 4. Skill retention vs. GenAI hours by institutional policy")
ax.legend(fontsize=8)
save(fig, "fig4_policy_moderation_skill")

results["RQ2"] = {
    "interaction_p_GPA": {k: float(v) for k, v in m3.pvalues.items()
                          if "Weekly_GenAI_Hours:C(Institutional_Policy)" in k},
    "interaction_p_Skill": {k: float(v) for k, v in m4.pvalues.items()
                            if "Weekly_GenAI_Hours:C(Institutional_Policy)" in k},
    "slopes_by_policy": slopes,
}


# =================================================================
# RQ3: What predicts high burnout risk and exam anxiety?
# =================================================================
print("\n=== RQ3: Burnout & Anxiety predictors ===")

# Targets: burnout categorical (Low/Medium/High); anxiety numeric 1-10.
burnout_order = ["Low", "Medium", "High"]
df["Burnout_Risk_Level"] = pd.Categorical(df["Burnout_Risk_Level"],
                                          categories=burnout_order, ordered=True)

df["High_Burnout"] = (df["Burnout_Risk_Level"] == "High").astype(int)
# Define "high anxiety" as upper third of the 1-10 scale (>=7)
df["High_Anxiety"] = (df["Anxiety_Level_During_Exams"] >= 7).astype(int)

# --- Logistic regression: high burnout ---
logit_burn = smf.logit(
    "High_Burnout ~ Weekly_GenAI_Hours + Perceived_AI_Dependency "
    "+ Traditional_Study_Hours + Prompt_Engineering_Skill + Tool_Diversity "
    "+ Pre_Semester_GPA + C(Major_Category) + C(Year_of_Study) "
    "+ C(Primary_Use_Case) + C(Institutional_Policy)",
    data=df,
).fit(disp=False)
print(logit_burn.summary().tables[1])

logit_anx = smf.logit(
    "High_Anxiety ~ Weekly_GenAI_Hours + Perceived_AI_Dependency "
    "+ Traditional_Study_Hours + Prompt_Engineering_Skill + Tool_Diversity "
    "+ Pre_Semester_GPA + C(Major_Category) + C(Year_of_Study) "
    "+ C(Primary_Use_Case) + C(Institutional_Policy)",
    data=df,
).fit(disp=False)
print(logit_anx.summary().tables[1])

# Odds ratios (key continuous predictors only)
key = ["Weekly_GenAI_Hours", "Perceived_AI_Dependency",
       "Traditional_Study_Hours", "Tool_Diversity"]
or_burn = pd.DataFrame({
    "OR": np.exp(logit_burn.params[key]),
    "p": logit_burn.pvalues[key],
})
or_anx = pd.DataFrame({
    "OR": np.exp(logit_anx.params[key]),
    "p": logit_anx.pvalues[key],
})
print("Burnout ORs:\n", or_burn)
print("Anxiety ORs:\n", or_anx)

# --- Random Forest feature importance for both targets ---
feature_cols = ["Weekly_GenAI_Hours", "Traditional_Study_Hours",
                "Tool_Diversity", "Perceived_AI_Dependency", "Pre_Semester_GPA"]
X_num = df[feature_cols].values
X_cat = pd.get_dummies(df[["Major_Category", "Year_of_Study",
                           "Primary_Use_Case", "Institutional_Policy",
                           "Paid_Subscription", "Prompt_Engineering_Skill"]],
                       drop_first=True)
X = np.hstack([X_num, X_cat.values])
feat_names = feature_cols + list(X_cat.columns)

importances = {}
aucs = {}
for target_name, y in [("Burnout", df["High_Burnout"].values),
                       ("Anxiety", df["High_Anxiety"].values)]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1,
                                class_weight="balanced_subsample")
    rf.fit(Xtr, ytr)
    p = rf.predict_proba(Xte)[:, 1]
    aucs[target_name] = float(roc_auc_score(yte, p))
    imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)
    importances[target_name] = imp
    print(f"{target_name} AUC = {aucs[target_name]:.3f}")
    print(imp.head(10))

# --- Figure 5: Top-10 RF importances side by side ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, (name, imp) in zip(axes, importances.items()):
    top = imp.head(10).iloc[::-1]
    ax.barh(top.index, top.values, color="#1f4e79")
    ax.set_title(f"{name}: top predictors (RF importance, AUC={aucs[name]:.2f})")
    ax.set_xlabel("Importance")
plt.suptitle("Figure 5. Random-forest feature importances for high burnout and high anxiety",
             y=1.02)
save(fig, "fig5_feature_importance")

# --- Figure 6: Burnout/Anxiety prevalence vs GenAI hours ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
b = pd.cut(df["Weekly_GenAI_Hours"], bins=10)
agg = df.groupby(b, observed=True).agg(
    x=("Weekly_GenAI_Hours", "mean"),
    burn=("High_Burnout", "mean"),
    anx=("High_Anxiety", "mean"),
    n=("High_Burnout", "size"),
).reset_index(drop=True)
axes[0].plot(agg["x"], agg["burn"]*100, "o-", color="#c0392b")
axes[0].set_title("High burnout prevalence vs. GenAI hours")
axes[0].set_xlabel("Weekly GenAI hours"); axes[0].set_ylabel("% with high burnout")
axes[1].plot(agg["x"], agg["anx"]*100, "o-", color="#8e44ad")
axes[1].set_title("High/severe anxiety prevalence vs. GenAI hours")
axes[1].set_xlabel("Weekly GenAI hours"); axes[1].set_ylabel("% with high/severe anxiety")
plt.suptitle("Figure 6. Wellbeing prevalence by weekly GenAI usage", y=1.02)
save(fig, "fig6_wellbeing_prevalence")

results["RQ3"] = {
    "burnout_AUC": aucs["Burnout"],
    "anxiety_AUC": aucs["Anxiety"],
    "burnout_top_features": importances["Burnout"].head(10).to_dict(),
    "anxiety_top_features": importances["Anxiety"].head(10).to_dict(),
    "logit_burnout_OR": or_burn.to_dict(),
    "logit_anxiety_OR": or_anx.to_dict(),
}

# ---------- Persist results ----------
with open(os.path.join(OUT, "results.json"), "w") as fh:
    json.dump(results, fh, indent=2, default=str)
print("\nSaved results.json and figures.")
