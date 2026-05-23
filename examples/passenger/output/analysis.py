import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

CSV = '/Users/sanjivda/Downloads/passenger/passenger_survey_balanced.csv'
OUT = '/Users/sanjivda/Downloads/passenger/output'
os.makedirs(OUT, exist_ok=True)

print('Loading...')
df = pd.read_csv(CSV)
print('Shape:', df.shape)

applicable_cols = [c for c in df.columns if c.endswith('_is_applicable')]
paired_ratings  = [c.replace('_is_applicable','') for c in applicable_cols if c.replace('_is_applicable','') in df.columns]
extra_ratings   = ['immigration_control','location_and_movement','overall_airport_cleanliness']
RATINGS = paired_ratings + [c for c in extra_ratings if c in df.columns]
print('Rating columns:', len(RATINGS))

DEMOG = ['nationality','gender','age_group','education','household_income',
         'traveling_alone','number_of_companions','trip_purpose',
         'trips_last_12_months','used_airport_before_last_12_months']
CONTEXT = ['process','flight_type','connection','ticket_purchased_by','checkin_method']

results = {}

# ====================== RQ1: Driver analysis ======================
print('\n=== RQ1: Driver analysis ===')

X_full = df[RATINGS].copy()
y = df['liked'].astype(int).values

# Coverage: fraction of respondents who actually rated each item
coverage = X_full.notna().mean().sort_values(ascending=False)

# Pearson correlation with `liked` (only over rows where the rating is present)
corrs = {}
for c in RATINGS:
    s = X_full[c]
    mask = s.notna()
    if mask.sum() < 200:
        continue
    corrs[c] = np.corrcoef(s[mask], y[mask])[0,1]
corr_s = pd.Series(corrs).sort_values(ascending=False)

# Impute medians, then fit logistic regression and random forest on full feature set
imp = SimpleImputer(strategy='median')
X_imp = imp.fit_transform(X_full)
scaler = StandardScaler()
Xs = scaler.fit_transform(X_imp)

X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.25, random_state=42, stratify=y)
logit = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
logit.fit(X_tr, y_tr)
auc_logit = roc_auc_score(y_te, logit.predict_proba(X_te)[:,1])
print(f'Logit AUC: {auc_logit:.4f}')

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_imp, y, test_size=0.25, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
rf.fit(X_tr2, y_tr2)
auc_rf = roc_auc_score(y_te2, rf.predict_proba(X_te2)[:,1])
print(f'RF AUC:    {auc_rf:.4f}')

logit_coef = pd.Series(logit.coef_[0], index=RATINGS).sort_values(ascending=False)
rf_imp     = pd.Series(rf.feature_importances_, index=RATINGS).sort_values(ascending=False)

results['rq1'] = {
    'auc_logit': float(auc_logit),
    'auc_rf':    float(auc_rf),
    'top10_corr':    corr_s.head(10).round(3).to_dict(),
    'top10_logit':   logit_coef.head(10).round(3).to_dict(),
    'top10_rf':      rf_imp.head(10).round(4).to_dict(),
    'coverage_top':  coverage.head(15).round(3).to_dict(),
    'coverage_bot':  coverage.tail(15).round(3).to_dict(),
}

# Save tables
corr_s.to_csv(f'{OUT}/rq1_corr.csv', header=['pearson_r'])
logit_coef.to_csv(f'{OUT}/rq1_logit_coef.csv', header=['std_coef'])
rf_imp.to_csv(f'{OUT}/rq1_rf_importance.csv', header=['importance'])
coverage.to_csv(f'{OUT}/rq1_coverage.csv', header=['frac_rated'])

# ---- Plot 1: top-15 RF feature importances ----
fig, ax = plt.subplots(figsize=(9,7))
top15 = rf_imp.head(15)[::-1]
ax.barh(top15.index, top15.values, color='#2b6cb0')
ax.set_xlabel('Random Forest feature importance')
ax.set_title('RQ1 — Top 15 drivers of "liked" (Random Forest)')
plt.tight_layout()
for ext in ('png','pdf'):
    plt.savefig(f'{OUT}/fig_rq1_rf_top15.{ext}', dpi=150)
plt.close()

# ---- Plot 2: top-15 Pearson correlations ----
fig, ax = plt.subplots(figsize=(9,7))
top15c = corr_s.head(15)[::-1]
ax.barh(top15c.index, top15c.values, color='#2f855a')
ax.set_xlabel('Pearson correlation with `liked`')
ax.set_title('RQ1 — Top 15 ratings correlated with "liked"')
plt.tight_layout()
for ext in ('png','pdf'):
    plt.savefig(f'{OUT}/fig_rq1_corr_top15.{ext}', dpi=150)
plt.close()
print('RQ1 figures saved.')

# ====================== RQ2: Segment-specific drivers ======================
print('\n=== RQ2: Segmented drivers ===')

def fit_segment(mask, label):
    Xseg = X_full.loc[mask].copy()
    yseg = df.loc[mask,'liked'].astype(int).values
    # Fill all-NaN cols with global median to preserve column count
    global_med = X_full.median(numeric_only=True)
    for c in Xseg.columns:
        if Xseg[c].isna().all():
            Xseg[c] = global_med[c] if not pd.isna(global_med[c]) else 3.0
    Ximp = SimpleImputer(strategy='median').fit_transform(Xseg)
    Xs   = StandardScaler().fit_transform(Ximp)
    Xtr,Xte,ytr,yte = train_test_split(Xs,yseg,test_size=0.25,random_state=42,stratify=yseg)
    lg = LogisticRegression(max_iter=2000).fit(Xtr,ytr)
    auc_lg = roc_auc_score(yte, lg.predict_proba(Xte)[:,1])
    rf2 = RandomForestClassifier(n_estimators=300,n_jobs=-1,random_state=42).fit(Ximp, yseg)
    imp = pd.Series(rf2.feature_importances_, index=RATINGS).sort_values(ascending=False)
    coef = pd.Series(lg.coef_[0], index=RATINGS).sort_values(ascending=False)
    print(f'  {label}: n={mask.sum()}, like_rate={yseg.mean():.3f}, AUC_logit={auc_lg:.3f}')
    return {'n':int(mask.sum()),'like_rate':float(yseg.mean()),
            'auc_logit':float(auc_lg),'rf_imp':imp,'logit_coef':coef}

segments = {
    'Boarding':       df['process']=='Embarque',
    'Disembarkation': df['process']=='Desembarque',
    'Domestic':       df['flight_type']=='Doméstico',
    'International':  df['flight_type']=='Internacional',
}
# fall back if Portuguese labels aren't matched
proc_vals = df['process'].value_counts()
ft_vals   = df['flight_type'].value_counts()
print('process values:', proc_vals.to_dict())
print('flight_type values:', ft_vals.to_dict())

# Re-derive segments using actual labels (top 2 of each)
proc_labels = list(proc_vals.index)
ft_labels   = list(ft_vals.index)
segments = {
    f'process={proc_labels[0]}': df['process']==proc_labels[0],
    f'process={proc_labels[1]}': df['process']==proc_labels[1],
    f'flight_type={ft_labels[0]}': df['flight_type']==ft_labels[0],
    f'flight_type={ft_labels[1]}': df['flight_type']==ft_labels[1],
}

seg_results = {}
for label,mask in segments.items():
    seg_results[label] = fit_segment(mask,label)

# Build comparison table for top-N from full model
TOPN = 15
top_features = rf_imp.head(TOPN).index.tolist()
comp = pd.DataFrame({lab: seg_results[lab]['rf_imp'].reindex(top_features) for lab in seg_results})
comp.to_csv(f'{OUT}/rq2_importance_comparison.csv')

# Plot grouped bar of top-10 drivers across the 4 segments
fig, ax = plt.subplots(figsize=(11,7))
top10f = rf_imp.head(10).index.tolist()
labels = list(seg_results.keys())
x = np.arange(len(top10f)); w = 0.2
colors = ['#2b6cb0','#9f7aea','#dd6b20','#38a169']
for i,lab in enumerate(labels):
    vals = seg_results[lab]['rf_imp'].reindex(top10f).values
    ax.bar(x + (i-1.5)*w, vals, w, label=lab, color=colors[i])
ax.set_xticks(x); ax.set_xticklabels(top10f, rotation=40, ha='right')
ax.set_ylabel('RF feature importance')
ax.set_title('RQ2 — Top-10 drivers of "liked" across segments')
ax.legend(fontsize=8)
plt.tight_layout()
for ext in ('png','pdf'):
    plt.savefig(f'{OUT}/fig_rq2_segments.{ext}', dpi=150)
plt.close()

# Also: each segment's own top-10 (set difference)
own_top = {lab: seg_results[lab]['rf_imp'].head(10).index.tolist() for lab in seg_results}
results['rq2'] = {
    'segments': {lab:{'n':seg_results[lab]['n'],
                      'like_rate':seg_results[lab]['like_rate'],
                      'auc_logit':seg_results[lab]['auc_logit'],
                      'top10':seg_results[lab]['rf_imp'].head(10).round(4).to_dict()}
                  for lab in seg_results},
    'shared_top': top10f,
    'own_top': own_top,
}
print('RQ2 figures saved.')

# ====================== RQ3: Demographic equity ======================
print('\n=== RQ3: Demographic equity ===')

# Raw like-rates per demographic level
raw_rates = {}
for col in DEMOG:
    if col in df.columns:
        gb = df.groupby(col)['liked'].agg(['mean','count'])
        gb.columns = ['like_rate','n']
        gb = gb.sort_values('like_rate', ascending=False)
        raw_rates[col] = gb
        gb.to_csv(f'{OUT}/rq3_raw_{col}.csv')

# Pretty-print a few
for col,gb in raw_rates.items():
    print(f'\n  {col} (range {gb["like_rate"].min():.3f}..{gb["like_rate"].max():.3f}):')
    print(gb.round(3).to_string())

# --- Adjusted analysis: control for ratings via residualisation ---
# Build full feature matrix: all RATINGS (median-imputed) -> predict liked.
# Then test whether demographics explain residual variance beyond ratings.
imp_full = SimpleImputer(strategy='median').fit_transform(X_full)
Xs_full  = StandardScaler().fit_transform(imp_full)
base = LogisticRegression(max_iter=2000).fit(Xs_full, y)
p_hat = base.predict_proba(Xs_full)[:,1]
resid = y - p_hat   # +ve => liked more than ratings would predict

resid_by_demo = {}
for col in DEMOG:
    if col in df.columns:
        gb = df.assign(resid=resid).groupby(col)['resid'].agg(['mean','count'])
        gb.columns = ['mean_resid','n']
        gb = gb.sort_values('mean_resid', ascending=False)
        resid_by_demo[col] = gb
        gb.to_csv(f'{OUT}/rq3_resid_{col}.csv')

# Plot raw vs residual for the 4 most informative demographics
focus = ['age_group','household_income','education','trip_purpose']
focus = [c for c in focus if c in raw_rates]
fig, axes = plt.subplots(len(focus),2, figsize=(13, 3.4*len(focus)))
for i,col in enumerate(focus):
    raw = raw_rates[col]; res = resid_by_demo[col].reindex(raw.index)
    ax1,ax2 = axes[i]
    ax1.barh(raw.index, raw['like_rate'], color='#3182ce')
    ax1.axvline(0.5, color='k', lw=0.7, ls='--')
    ax1.set_title(f'Raw like-rate by {col}')
    ax1.set_xlim(0.3,0.7)
    ax2.barh(res.index, res['mean_resid'], color='#dd6b20')
    ax2.axvline(0, color='k', lw=0.7, ls='--')
    ax2.set_title(f'Residual (after rating-controls) by {col}')
plt.tight_layout()
for ext in ('png','pdf'):
    plt.savefig(f'{OUT}/fig_rq3_demographics.{ext}', dpi=150)
plt.close()

results['rq3'] = {
    'raw': {col: raw_rates[col].round(4).to_dict() for col in raw_rates},
    'residual': {col: resid_by_demo[col].round(4).to_dict() for col in resid_by_demo},
}

# --- Save full results JSON ---
def _to_jsonable(o):
    if isinstance(o,dict): return {str(k):_to_jsonable(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [_to_jsonable(x) for x in o]
    if isinstance(o,(np.floating,float)): return float(o)
    if isinstance(o,(np.integer,int)): return int(o)
    return o
with open(f'{OUT}/results.json','w') as fp:
    json.dump(_to_jsonable(results), fp, indent=2, default=str)

print('\nAll analysis complete.')
print('Outputs in:', OUT)
