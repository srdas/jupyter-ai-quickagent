"""
Online vs. Offline Learning — Empirical Analysis
=================================================
Dataset : /Users/sanjivda/Desktop/online_learning/online_vs_offline_learning_dataset.csv
Output  : /Users/sanjivda/Desktop/online_learning/output/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for any environment)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = "/Users/sanjivda/Desktop/online_learning/online_vs_offline_learning_dataset.csv"
OUTPUT_DIR  = "/Users/sanjivda/Desktop/online_learning/output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        plt.style.use("ggplot")

PALETTE   = {"Online": "#2196F3", "Offline": "#FF9800"}   # blue / orange
FONT_TITLE = dict(fontsize=14, fontweight="bold")
FONT_AXIS  = dict(fontsize=11)

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(f"Shape : {df.shape}")
print(f"Columns : {list(df.columns)}")
print(f"\nLearning_Mode counts:\n{df['Learning_Mode'].value_counts().to_string()}")
print(f"\nSubject counts:\n{df['Subject'].value_counts().to_string()}")

online  = df[df["Learning_Mode"] == "Online"]
offline = df[df["Learning_Mode"] == "Offline"]
NUMERIC = ["Exam_Score", "Retention_Score", "Focus_Level", "Study_Hours"]

# ═══════════════════════════════════════════════════════════════════════════════
#  STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Mean ± Std by Learning Mode ────────────────────────────────────────────
print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS — Mean ± Std by Learning Mode")
print("=" * 70)
summary = df.groupby("Learning_Mode")[NUMERIC].agg(["mean", "std"])
for mode in ["Online", "Offline"]:
    print(f"\n{mode}:")
    for col in NUMERIC:
        m = summary.loc[mode, (col, "mean")]
        s = summary.loc[mode, (col, "std")]
        print(f"  {col:<18}: {m:6.2f} ± {s:5.2f}")

# ── 2. Hypothesis Tests (Mann-Whitney U + independent t-test) ─────────────────
print("\n" + "=" * 70)
print("HYPOTHESIS TESTS — Online vs. Offline")
print("=" * 70)
print(f"{'Metric':<18}  {'t-stat':>8}  {'t p-val':>8}  {'U-stat':>10}  {'MWU p-val':>10}  Sig?")
print("-" * 70)
for col in NUMERIC:
    a, b = online[col].dropna(), offline[col].dropna()
    t_stat, t_p   = stats.ttest_ind(a, b)
    u_stat, u_p   = stats.mannwhitneyu(a, b, alternative="two-sided")
    sig = "***" if t_p < 0.001 else ("**" if t_p < 0.01 else ("*" if t_p < 0.05 else "ns"))
    print(f"{col:<18}  {t_stat:>8.3f}  {t_p:>8.4f}  {u_stat:>10.0f}  {u_p:>10.4f}  {sig}")

# ── 3. Pearson correlations ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PEARSON CORRELATIONS with Exam_Score (by Learning Mode)")
print("=" * 70)
for mode, grp in [("Online", online), ("Offline", offline)]:
    print(f"\n{mode}:")
    for col in ["Retention_Score", "Focus_Level", "Study_Hours"]:
        r, p = stats.pearsonr(grp[col].dropna(), grp["Exam_Score"].dropna())
        sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  {col:<18} vs Exam_Score : r = {r:+.4f},  p = {p:.4f}  {sig}")

# ── 4. Mean Exam_Score pivot: Subject × Learning_Mode ─────────────────────────
print("\n" + "=" * 70)
print("MEAN EXAM_SCORE — Subject × Learning_Mode (Pivot Table)")
print("=" * 70)
pivot = df.pivot_table(values="Exam_Score",
                       index="Subject",
                       columns="Learning_Mode",
                       aggfunc="mean")
pivot["Diff (Online−Offline)"] = pivot["Online"] - pivot["Offline"]
print(pivot.round(2).to_string())


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER — save figure
# ═══════════════════════════════════════════════════════════════════════════════
def save_fig(fig, name):
    for ext in ("png", "pdf"):
        path = os.path.join(OUTPUT_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  ✓  Saved {name}.png  /  {name}.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 1 — Side-by-side box plots
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING FIGURES")
print("=" * 70)

metrics    = ["Exam_Score", "Retention_Score", "Focus_Level"]
ylabels    = ["Exam Score",  "Retention Score",  "Focus Level"]
fig1, axes = plt.subplots(1, 3, figsize=(15, 6))
fig1.suptitle("Online vs. Offline Learning: Score Distributions", **FONT_TITLE, y=1.02)

for ax, metric, ylabel in zip(axes, metrics, ylabels):
    data_on  = online[metric].dropna()
    data_off = offline[metric].dropna()

    bp = ax.boxplot(
        [data_on, data_off],
        labels=["Online", "Offline"],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )
    bp["boxes"][0].set_facecolor(PALETTE["Online"])
    bp["boxes"][1].set_facecolor(PALETTE["Offline"])
    bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_alpha(0.75)

    ax.set_title(metric.replace("_", " "), fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, **FONT_AXIS)
    ax.set_xlabel("Learning Mode", **FONT_AXIS)

    # annotate medians
    for i, data in enumerate([data_on, data_off], 1):
        med = np.median(data)
        ax.text(i, med + 0.5, f"{med:.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="black")

    # significance annotation
    _, p = stats.ttest_ind(data_on, data_off)
    sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    y_top = max(data_on.max(), data_off.max()) + 2
    ax.annotate("", xy=(2, y_top), xytext=(1, y_top),
                arrowprops=dict(arrowstyle="-", lw=1.5))
    ax.text(1.5, y_top + 0.5, sig, ha="center", fontsize=11, fontweight="bold")

patches = [mpatches.Patch(color=PALETTE[m], alpha=0.75, label=m) for m in ["Online", "Offline"]]
fig1.legend(handles=patches, loc="upper right", fontsize=10, title="Mode")
fig1.tight_layout()
save_fig(fig1, "fig1_score_distributions")
plt.close(fig1)

# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 2 — Grouped bar chart: mean Exam_Score by Subject × Mode
# ═══════════════════════════════════════════════════════════════════════════════
agg = df.groupby(["Subject", "Learning_Mode"])["Exam_Score"].agg(["mean", "sem"]).reset_index()
subjects = sorted(df["Subject"].unique())
x        = np.arange(len(subjects))
width    = 0.35

fig2, ax2 = plt.subplots(figsize=(11, 6))
for i, mode in enumerate(["Online", "Offline"]):
    sub = agg[agg["Learning_Mode"] == mode].set_index("Subject").reindex(subjects)
    ax2.bar(x + i * width, sub["mean"], width,
            yerr=sub["sem"], capsize=4,
            color=PALETTE[mode], alpha=0.85, label=mode,
            error_kw=dict(elinewidth=1.5, ecolor="black"))
    # value labels
    for j, (m, s) in enumerate(zip(sub["mean"], sub["sem"])):
        ax2.text(j + i * width, m + s + 0.8, f"{m:.1f}",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax2.set_xticks(x + width / 2)
ax2.set_xticklabels(subjects, fontsize=11)
ax2.set_xlabel("Subject", **FONT_AXIS)
ax2.set_ylabel("Mean Exam Score", **FONT_AXIS)
ax2.set_title("Mean Exam Score by Subject and Learning Mode\n(error bars = SEM)",
              **FONT_TITLE)
ax2.legend(title="Learning Mode", fontsize=10)
ax2.set_ylim(0, agg["mean"].max() + 10)
fig2.tight_layout()
save_fig(fig2, "fig2_exam_by_subject")
plt.close(fig2)

# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 3 — Scatter: Retention_Score vs Exam_Score with regression lines
# ═══════════════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(9, 6))
for mode, grp in [("Online", online), ("Offline", offline)]:
    x_vals = grp["Retention_Score"].values
    y_vals = grp["Exam_Score"].values
    ax3.scatter(x_vals, y_vals, c=PALETTE[mode], alpha=0.35,
                s=35, label=mode, edgecolors="none")
    # regression line
    slope, intercept, r, p, _ = stats.linregress(x_vals, y_vals)
    xr = np.linspace(x_vals.min(), x_vals.max(), 200)
    ax3.plot(xr, intercept + slope * xr, color=PALETTE[mode],
             linewidth=2.2, label=f"{mode} fit  (r={r:+.3f}, p={p:.3f})")

ax3.set_xlabel("Retention Score", **FONT_AXIS)
ax3.set_ylabel("Exam Score", **FONT_AXIS)
ax3.set_title("Retention Score vs. Exam Score by Learning Mode", **FONT_TITLE)
ax3.legend(fontsize=10, title="Learning Mode / Fit")
fig3.tight_layout()
save_fig(fig3, "fig3_retention_vs_exam")
plt.close(fig3)

# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 4 — Scatter: Focus_Level vs Exam_Score with regression lines
# ═══════════════════════════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(9, 6))
for mode, grp in [("Online", online), ("Offline", offline)]:
    x_vals = grp["Focus_Level"].values
    y_vals = grp["Exam_Score"].values
    ax4.scatter(x_vals, y_vals, c=PALETTE[mode], alpha=0.35,
                s=35, label=mode, edgecolors="none")
    slope, intercept, r, p, _ = stats.linregress(x_vals, y_vals)
    xr = np.linspace(x_vals.min(), x_vals.max(), 200)
    ax4.plot(xr, intercept + slope * xr, color=PALETTE[mode],
             linewidth=2.2, label=f"{mode} fit  (r={r:+.3f}, p={p:.3f})")

ax4.set_xlabel("Focus Level", **FONT_AXIS)
ax4.set_ylabel("Exam Score", **FONT_AXIS)
ax4.set_title("Focus Level vs. Exam Score by Learning Mode", **FONT_TITLE)
ax4.legend(fontsize=10, title="Learning Mode / Fit")
fig4.tight_layout()
save_fig(fig4, "fig4_focus_vs_exam")
plt.close(fig4)

# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 5 — Overlapping histograms + KDE for Study_Hours
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from scipy.stats import gaussian_kde
    HAS_KDE = True
except ImportError:
    HAS_KDE = False

fig5, ax5 = plt.subplots(figsize=(9, 5))
bins = np.linspace(df["Study_Hours"].min() - 0.5,
                   df["Study_Hours"].max() + 0.5, 30)

for mode, grp in [("Online", online), ("Offline", offline)]:
    vals = grp["Study_Hours"].dropna().values
    ax5.hist(vals, bins=bins, color=PALETTE[mode],
             alpha=0.45, label=f"{mode} (histogram)", density=True)
    if HAS_KDE and len(vals) > 1:
        kde = gaussian_kde(vals, bw_method=0.4)
        xr  = np.linspace(vals.min() - 0.3, vals.max() + 0.3, 300)
        ax5.plot(xr, kde(xr), color=PALETTE[mode], linewidth=2.5,
                 label=f"{mode} KDE")

ax5.set_xlabel("Study Hours", **FONT_AXIS)
ax5.set_ylabel("Density", **FONT_AXIS)
ax5.set_title("Distribution of Study Hours by Learning Mode", **FONT_TITLE)
ax5.legend(fontsize=10)
fig5.tight_layout()
save_fig(fig5, "fig5_study_hours_distribution")
plt.close(fig5)

# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 6 — Heatmap: mean Exam_Score by Subject × Learning_Mode
# ═══════════════════════════════════════════════════════════════════════════════
heat_data = df.pivot_table(values="Exam_Score",
                           index="Subject",
                           columns="Learning_Mode",
                           aggfunc="mean")

fig6, ax6 = plt.subplots(figsize=(7, 5))
im = ax6.imshow(heat_data.values, cmap="YlOrRd", aspect="auto",
                vmin=heat_data.values.min() - 2,
                vmax=heat_data.values.max() + 2)

cbar = fig6.colorbar(im, ax=ax6, shrink=0.85)
cbar.set_label("Mean Exam Score", fontsize=11)

ax6.set_xticks(range(len(heat_data.columns)))
ax6.set_xticklabels(heat_data.columns, fontsize=11)
ax6.set_yticks(range(len(heat_data.index)))
ax6.set_yticklabels(heat_data.index, fontsize=11)
ax6.set_title("Mean Exam Score: Subject × Learning Mode", **FONT_TITLE)
ax6.set_xlabel("Learning Mode", **FONT_AXIS)
ax6.set_ylabel("Subject", **FONT_AXIS)

# annotate cells
for i in range(heat_data.shape[0]):
    for j in range(heat_data.shape[1]):
        val  = heat_data.values[i, j]
        vmin = heat_data.values.min()
        vmax = heat_data.values.max()
        txt_color = "white" if val > (vmin + vmax) / 2 + 2 else "black"
        ax6.text(j, i, f"{val:.1f}", ha="center", va="center",
                 fontsize=12, fontweight="bold", color=txt_color)

fig6.tight_layout()
save_fig(fig6, "fig6_subject_heatmap")
plt.close(fig6)

# ── Final confirmation ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ALL FIGURES SAVED")
print("=" * 70)
saved = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png") or f.endswith(".pdf")]
for f in sorted(saved):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f:<45}  ({size/1024:6.1f} KB)")

print("\nAnalysis complete.")
