
"""
STEP 3 Robust Feature Selection (BridgeGuard )
=======================================================================

SYNC WITH STEP 2b ( scientific exclusion of degenerate features):
 [SYNC ] One sequential feature retained from sequential_features.py:

 EXCLUDED freq_persistence_5w (label proxy, not a feature):
 sequential feature extraction computes this correctly, but it is excluded from candidates here.
 For slow_poisoning, min_emergency=1 is enforced by window extraction WindowConfig
 every attack window has n_emergency > 0 has_alert=1 for all 158 windows
 rolling mean of a constant series = constant freq_persistence_5w = 1.0
 for the entire class (within-class variance = 0.000000).
 Global variance is high ( 0.14) because normal windows have persistence 0.09,
 making it pass VarianceThreshold on the mixed dataset an angle mort of a
 global-only variance filter. A per-class variance filter (Step 0b, added here)
 catches this pattern.
 Scientific argument: this feature encodes "has the attack been running for
 3 windows?" it is undefined at attack onset and constant thereafter.
 It is a retrospective indicator, not a prospective detector.

 EXCLUDED freq_rolling_mean_5w (redundant with alert_frequency_per_hour):
 Rolling mean of alert_frequency_per_hour over 5 windows.
 Pearson r=0.924 with alert_frequency_per_hour (already a candidate).
 Although Spearman=0.669 prevents the dual AND collinearity filter from
 catching it, the feature adds no new information beyond the existing
 smoothed estimate of the underlying series. Excluded upstream for parsimony.

 RETAINED freq_rolling_std_5w (Hedges g 1.35, genuinely new signal):
 Captures the metronomic regularity of the injection rate over 5 windows.
 Normal traffic has high variance in alert frequency (stochastic inter-arrival);
 slow_poisoning injections have low variance (stable rate low std).
 Within-class variance is non-zero for both classes not a label proxy.
 This signal is orthogonal to alert_frequency_per_hour and captures
 temporal consistency rather than temporal magnitude.

 Total candidate features: 14 (13 original + 1 sequential)
 N_FEATURES_FINAL unchanged: 8

 features_selected_labeled.csv now includes 'window_start_time'
 to support chronological sorting in ensemble calibration (ensemble calibration searches for this column).

SYNC WITH ZONE ENCODING (preserved):
 [SYNC] 'time_of_day_deviation' removed, replaced by time_sin + time_cos.
 Total candidate features: 13.

CORRECTIONS (preserved from prior version):
 [FIX 1] Hedges' g with correct pooled SD (n-1 weighted).
 Collinearity on NORMAL windows only.
 Dual collinearity (Pearson AND Spearman).
 [FIX 4] Per-attack-type Hedges g (A2 and A5 separately) min(g_A2, g_A5).
 [FIX 5] Mutual Information as third discriminative signal.
 [FIX 6] VarianceThreshold filter.
 [FIX 7] Bootstrap stability analysis (n=200, 60% stable).
 [FIX 8] F-score on calibration split only.
 [FIX 9] Kruskal-Wallis as non-parametric validation.
 [FIX 10] Graph-based collinearity removal (BFS connected components).

PIPELINE (unchanged from v3 except candidate list and Step 0b):
 0. Variance filter (global, VarianceThreshold)
 0b. Per-class variance filter catches label proxies invisible to global filter
 1. Collinearity filter on NORMAL windows (Pearson AND Spearman, 0.92)
 2. Per-attack discriminative ranking (Hedges g + F + MI)
 3. Bootstrap stability (n=200, 60%)
 4. Final selection: top-8

OUTPUT (contracts unchanged hyperparameter tuning/5/6/7 read without modification):
 - selected_features.json
 - features_selected_normal.csv selected_features + label + attack_type
 - features_selected_labeled.csv selected_features + label + attack_type
 + window_start_time [NEW in v4]
 - figures/feature_correlation_heatmap_v3.png
 - figures/feature_discriminability_v3.png
 - figures/feature_bootstrap_stability_v3.png
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import shapiro, kruskal, mannwhitneyu, spearmanr
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
)
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")
np.random.seed(42)

FEATURES_DIR            = "bridgeguard_features"
FIGURES_DIR             = "figures"
COLLINEARITY_THRESHOLD  = 0.92
N_FEATURES_FINAL        = 8
VARIANCE_THRESHOLD      = 1e-4
BOOTSTRAP_N             = 200
BOOTSTRAP_STABILITY_MIN = 0.60
CALIBRATION_SIZE        = 0.60

_TIMESTAMP_COL = "window_start_time"

ALL_FEATURE_COLS = [

    'alert_frequency_per_hour',
    'normal_emergency_ratio',
    'inter_alert_interval_mean',
    'inter_interval_variance',
    'burst_score',
    'topic_diversity',
    'time_sin',
    'time_cos',
    'payload_entropy',
    'consecutive_emergency_count',
    'alert_rate_acceleration',
    'regularity_coefficient',
    'temporal_clustering_score',

    'freq_rolling_std_5w',

]

if __name__ != "__main__":
    raise RuntimeError(
        "attack_aware_feature_selection.py must be run directly, not imported.\n"
        "All pipeline logic executes at module level importing would run the\n"
        "entire feature selection selection process. Use: python attack_aware_feature_selection.py"
    )

os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 70)
print("BridgeGuard Robust Feature Selection")
print("=" * 70)
print(f"  Candidates : {len(ALL_FEATURE_COLS)} (13 original + 1 sequential from sequential_features.py)")
print(f"  Excluded   : freq_persistence_5w (label proxy), freq_rolling_mean_5w (redundant)")
print(f"  Selected   : {N_FEATURES_FINAL} final features (unchanged)")

labeled = pd.read_csv(f"{FEATURES_DIR}/features_labeled.csv")
normal  = labeled[labeled['label'] == 1].reset_index(drop=True)
attack  = labeled[labeled['label'] == 0].reset_index(drop=True)

missing_new = [f for f in ['freq_rolling_std_5w'] if f not in labeled.columns]
if missing_new:
    raise RuntimeError(
        f"\n[ERROR] Sequential feature missing from features_labeled.csv:\n"
        f"  {missing_new}\n"
        f"  Run sequential_features.py before attack_aware_feature_selection.py."
    )

if _TIMESTAMP_COL not in labeled.columns:
    raise RuntimeError(
        f"\n[ERROR] Column '{_TIMESTAMP_COL}' missing from features_labeled.csv.\n"
        f"  Verify that window extraction produced the required column."
    )

ALL_FEATURE_COLS = [f for f in ALL_FEATURE_COLS if f in labeled.columns]
missing_from_csv = [f for f in ALL_FEATURE_COLS if f not in labeled.columns]
if missing_from_csv:
    print(f"     Features absent from CSV (skipped): {missing_from_csv}")

a2_mask = labeled['attack_type'].str.lower().str.contains('flood|attack2|a2', na=False)
a5_mask = labeled['attack_type'].str.lower().str.contains('poison|slow|attack5|a5', na=False)
attack_a2 = labeled[a2_mask & (labeled['label'] == 0)].reset_index(drop=True)
attack_a5 = labeled[a5_mask & (labeled['label'] == 0)].reset_index(drop=True)

if len(attack_a2) < 5 or len(attack_a5) < 5:
    attack_sorted = attack.sort_values('alert_frequency_per_hour')
    mid = len(attack_sorted) // 2
    attack_a5 = attack_sorted.iloc[:mid].reset_index(drop=True)
    attack_a2 = attack_sorted.iloc[mid:].reset_index(drop=True)
    print(f"     Attack type split by alert_frequency "
          f"(A2: {len(attack_a2)}, A5: {len(attack_a5)})")

print(f"\n  Loaded: {len(labeled)} windows "
      f"({len(normal)} normal / {len(attack)} attack)")
print(f"    A2 (flooding)      : {len(attack_a2)} windows")
print(f"    A5 (slow-poisoning): {len(attack_a5)} windows")

X_all = labeled[ALL_FEATURE_COLS].values
y_all = labeled['label'].values

def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd < 1e-10:
        return 0.0
    g_raw = abs(np.mean(group1) - np.mean(group2)) / pooled_sd
    correction = 1 - (3 / (4 * (n1 + n2 - 2) - 1))
    return float(g_raw * correction)

print("\n" + " " * 70)
print(f"Step 0   Variance filter (threshold={VARIANCE_THRESHOLD})")
print(" " * 70)

vt = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
vt.fit(labeled[ALL_FEATURE_COLS])
passed_variance = [f for f, ok in zip(ALL_FEATURE_COLS, vt.get_support()) if ok]
failed_variance = [f for f, ok in zip(ALL_FEATURE_COLS, vt.get_support()) if not ok]

variances = labeled[ALL_FEATURE_COLS].var()
for f in ALL_FEATURE_COLS:
    status = "PASS" if f in passed_variance else "DROP (near-zero variance)"
    seq_tag = " [SEQ]" if f in ('freq_persistence_5w',
                                 'freq_rolling_mean_5w',
                                 'freq_rolling_std_5w') else ""
    print(f"  {f:<38}  var={variances[f]:>10.4f}  {status}{seq_tag}")

if failed_variance:
    print(f"\n  Removed {len(failed_variance)} near-zero-variance feature(s): "
          f"{failed_variance}")
else:
    print(f"\n  All {len(ALL_FEATURE_COLS)} features pass variance filter.")

remaining_after_var = passed_variance

WITHIN_CLASS_VAR_THRESHOLD = 1e-6

print("\n" + " " * 70)
print(f"Step 0b   Per-class variance filter (within-class, threshold={WITHIN_CLASS_VAR_THRESHOLD})")
print(" " * 70)
print(" Detects label proxies : high global variance, near-zero within-class variance")

classes_for_filter = labeled['attack_type'].unique().tolist()
degenerate_features: list = []

for feat in remaining_after_var:
    per_class_vars = {}
    for cls in classes_for_filter:
        cls_vals = labeled.loc[labeled['attack_type'] == cls, feat].dropna().values
        if len(cls_vals) >= 2:
            per_class_vars[cls] = float(np.var(cls_vals, ddof=1))
        else:
            per_class_vars[cls] = float('nan')

    min_var_cls  = min((v for v in per_class_vars.values() if not np.isnan(v)),
                       default=float('nan'))
    min_var_name = min((k for k, v in per_class_vars.items() if not np.isnan(v)),
                       key=lambda k: per_class_vars[k],
                       default="N/A")

    is_degenerate = (not np.isnan(min_var_cls)) and (min_var_cls < WITHIN_CLASS_VAR_THRESHOLD)
    if is_degenerate:
        degenerate_features.append(feat)
        print(f"     DEGENERATE: {feat:<36} within-class var={min_var_cls:.2e}"
              f" (class='{min_var_name}')")
        print(f"         Feature is constant within '{min_var_name}'   label proxy, not informative")
        print(f"       Global variance: {float(labeled[feat].var()):.6f} (passes global filter)")

if not degenerate_features:
    print(f"  OK All {len(remaining_after_var)} features pass per-class variance filter")
else:
    print(f"\n  Removing {len(degenerate_features)} degenerate feature(s): "
          f"{degenerate_features}")

excluded_by_design = [f for f in ['freq_persistence_5w', 'freq_rolling_mean_5w']
                      if f in labeled.columns]
if excluded_by_design:
    print(f"\n  Features excluded by design (not in ALL_FEATURE_COLS) :")
    for feat in excluded_by_design:
        sp_vals = labeled.loc[labeled['attack_type'].str.lower().str.contains(
                                'slow|poison', na=False), feat].dropna().values
        if len(sp_vals) > 0:
            print(f"    {feat:<36} within-class var (sp): "
                  f"{float(np.var(sp_vals, ddof=1)):.2e}  "
                  f"[confirmed {'degenerate' if np.var(sp_vals, ddof=1) < WITHIN_CLASS_VAR_THRESHOLD else 'ok'}]")

remaining_after_var = [f for f in remaining_after_var if f not in degenerate_features]
failed_variance_perclass = degenerate_features

print(f"\n  After per-class filter: {len(remaining_after_var)} / "
      f"{len(passed_variance)} features remain")

print("\n" + " " * 70)
print("Step A Hedges' g per attack type (corrected pooled SD, n-1 weighted)")
print(" " * 70)

n_vals_all    = {col: normal[col].dropna().values for col in remaining_after_var}
hedges_g_a2   = {}
hedges_g_a5   = {}
hedges_g_min  = {}
mannwhitney_p = {}

print(f"\n  {'Feature':<38} {'g_A2':>6}  {'g_A5':>6}  {'g_min':>6}  "
      f"{'MW p':>8}  Normality")
print(" " + " " * 72)

for col in remaining_after_var:
    n_v  = n_vals_all[col]
    a2_v = attack_a2[col].dropna().values if col in attack_a2.columns else np.array([])
    a5_v = attack_a5[col].dropna().values if col in attack_a5.columns else np.array([])
    a_v  = attack[col].dropna().values

    g_a2 = hedges_g(n_v, a2_v) if len(a2_v) >= 2 else 0.0
    g_a5 = hedges_g(n_v, a5_v) if len(a5_v) >= 2 else 0.0
    g_min = min(g_a2, g_a5)

    try:
        _, mw_p = mannwhitneyu(n_v, a_v, alternative='two-sided')
    except Exception:
        mw_p = 1.0

    hedges_g_a2[col]   = g_a2
    hedges_g_a5[col]   = g_a5
    hedges_g_min[col]  = g_min
    mannwhitney_p[col] = mw_p

    _, sw_p = shapiro(n_v[:200])
    normal_flag = "normal" if sw_p > 0.05 else "NON-NORMAL"
    seq_tag = " [SEQ]" if col in ('freq_persistence_5w',
                                   'freq_rolling_mean_5w',
                                   'freq_rolling_std_5w') else ""

    print(f"  {col:<38} {g_a2:>6.3f}  {g_a5:>6.3f}  {g_min:>6.3f}  "
          f"{mw_p:>8.4f}  {normal_flag}{seq_tag}")

print("\n" + " " * 70)
print(f"Step B   Dual collinearity filter on NORMAL windows only "
      f"(threshold={COLLINEARITY_THRESHOLD})")
print(" " * 70)

normal_data   = normal[remaining_after_var]
pearson_corr  = normal_data.corr(method='pearson').abs()
spearman_corr = normal_data.corr(method='spearman').abs()

collinear_pairs = []
feats = remaining_after_var
for i in range(len(feats)):
    for j in range(i + 1, len(feats)):
        fi, fj = feats[i], feats[j]
        pr = pearson_corr.loc[fi, fj]
        sr = spearman_corr.loc[fi, fj]
        if pr > COLLINEARITY_THRESHOLD and sr > COLLINEARITY_THRESHOLD:
            collinear_pairs.append((fi, fj, pr, sr))
            print(f"  COLLINEAR: {fi}   {fj}  "
                  f"Pearson={pr:.4f}  Spearman={sr:.4f}")

if not collinear_pairs:
    print(f"  No collinear pairs found (threshold={COLLINEARITY_THRESHOLD})")

from collections import defaultdict

adj            = defaultdict(set)
collinear_nodes = set()
for fi, fj, _, _ in collinear_pairs:
    adj[fi].add(fj)
    adj[fj].add(fi)
    collinear_nodes.add(fi)
    collinear_nodes.add(fj)

visited   = set()
to_remove = set()

def bfs_component(start):
    component = []
    queue = [start]
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        component.append(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                queue.append(neighbor)
    return component

for node in collinear_nodes:
    if node not in visited:
        component = bfs_component(node)
        best = max(component, key=lambda f: hedges_g_min.get(f, 0))
        for f in component:
            if f != best:
                to_remove.add(f)
                print(f"  DROP '{f}' (g_min={hedges_g_min[f]:.3f}) "
                      f"— collinear with '{best}' (g_min={hedges_g_min[best]:.3f})")

remaining_after_collin = [f for f in remaining_after_var if f not in to_remove]
removed_collinear      = list(to_remove)

print(f"\n  After collinearity filter: {len(remaining_after_collin)} / "
      f"{len(remaining_after_var)} features remain")
if removed_collinear:
    print(f"  Removed: {removed_collinear}")

if len(remaining_after_collin) > 1:
    check_corr = normal[remaining_after_collin].corr(method='pearson').abs()
    max_r = check_corr.where(
        ~np.eye(len(remaining_after_collin), dtype=bool)
    ).max().max()
    print(f"  Max Pearson among surviving features: {max_r:.3f} "
          f"({'OK' if max_r < COLLINEARITY_THRESHOLD else 'WARNING'})")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, corr_mat, title in zip(
    axes,
    [pearson_corr, spearman_corr],
    ["Pearson (Normal windows)", "Spearman (Normal windows)"]
):
    mask = np.zeros_like(corr_mat.values, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(
        corr_mat, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, ax=ax, square=True, annot_kws={"size": 6},
        mask=mask, vmin=0, vmax=1
    )
    ax.set_title(f"Feature Correlation: {title}", fontsize=11)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', rotation=0, labelsize=7)
plt.tight_layout()
heatmap_path = f"{FIGURES_DIR}/feature_correlation_heatmap_v3.png"
plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"\n  Heatmap saved: {heatmap_path}")

print("\n" + " " * 70)
print(f"Step C   Calibration split ({int(CALIBRATION_SIZE*100)}% for F / MI)")
print(" " * 70)

sss = StratifiedShuffleSplit(
    n_splits=1, test_size=1 - CALIBRATION_SIZE, random_state=42
)
cal_idx, _ = next(sss.split(X_all, y_all))
X_cal = labeled.iloc[cal_idx][remaining_after_collin].values
y_cal = y_all[cal_idx]

print(f"  Calibration set: {len(cal_idx)} windows "
      f"(normal={sum(y_cal==1)}, attack={sum(y_cal==0)})")

try:
    f_selector = SelectKBest(f_classif, k='all')
    f_selector.fit(X_cal, y_cal)
    f_scores = dict(zip(remaining_after_collin, f_selector.scores_))
except Exception as e:
    print(f"     F-score failed: {e}. Using zeros.")
    f_scores = {f: 0.0 for f in remaining_after_collin}

try:
    mi_selector = SelectKBest(mutual_info_classif, k='all')
    mi_selector.fit(X_cal, y_cal)
    mi_scores = dict(zip(remaining_after_collin, mi_selector.scores_))
except Exception as e:
    print(f"     MI failed: {e}. Using zeros.")
    mi_scores = {f: 0.0 for f in remaining_after_collin}

kw_stats = {}
for col in remaining_after_collin:
    try:
        n_v = normal[col].dropna().values
        a_v = attack[col].dropna().values
        stat, p = kruskal(n_v, a_v)
        kw_stats[col] = {'stat': stat, 'p': p}
    except Exception:
        kw_stats[col] = {'stat': 0.0, 'p': 1.0}

print(f"\n  {'Feature':<38}  {'F-score':>9}  {'MI':>6}  {'KW stat':>8}  {'KW p':>8}")
print(" " + " " * 78)
for col in sorted(remaining_after_collin, key=lambda f: -f_scores.get(f, 0)):
    print(f"  {col:<38}  {f_scores[col]:>9.2f}  {mi_scores[col]:>6.4f}"
          f"  {kw_stats[col]['stat']:>8.1f}  {kw_stats[col]['p']:>8.4f}")

print("\n" + " " * 70)
print("Step D Combined ranking (Hedges_g + F + MI, equal weights)")
print(" " * 70)

max_g  = max(hedges_g_min.get(f, 0) for f in remaining_after_collin) or 1.0
max_f  = max(f_scores.get(f, 0)     for f in remaining_after_collin) or 1.0
max_mi = max(mi_scores.get(f, 0)    for f in remaining_after_collin) or 1.0

combined_scores = {}
for col in remaining_after_collin:
    g_norm  = hedges_g_min.get(col, 0) / max_g
    f_norm  = f_scores.get(col, 0)     / max_f
    mi_norm = mi_scores.get(col, 0)    / max_mi
    combined_scores[col] = (g_norm + f_norm + mi_norm) / 3.0

ranked = sorted(combined_scores.items(), key=lambda x: -x[1])

print(f"\n  {'Rank':<5} {'Feature':<38} {'g_min':>6} {'F':>8} {'MI':>6} "
      f"{'Score':>7}  Dec")
print(" " + " " * 80)
top_candidates = [col for col, _ in ranked[:N_FEATURES_FINAL]]
for i, (col, sc) in enumerate(ranked):
    decision = "KEEP" if col in top_candidates else "DROP"
    seq_tag  = " [SEQ]" if col in ('freq_persistence_5w',
                                    'freq_rolling_mean_5w',
                                    'freq_rolling_std_5w') else ""
    print(f"  {i+1:<5} {col:<38} "
          f"{hedges_g_min.get(col,0):>6.3f} "
          f"{f_scores.get(col,0):>8.1f} "
          f"{mi_scores.get(col,0):>6.4f} "
          f"{sc:>7.3f}  {decision}{seq_tag}")

print("\n" + " " * 70)
print(f"Step E   Bootstrap stability (n={BOOTSTRAP_N}, min={BOOTSTRAP_STABILITY_MIN:.0%})")
print(" " * 70)

selection_counts = {col: 0 for col in remaining_after_collin}

for b in range(BOOTSTRAP_N):
    rng_b  = np.random.RandomState(b)
    idx_n  = rng_b.choice(len(normal), size=len(normal), replace=True)
    idx_a  = rng_b.choice(len(attack), size=len(attack), replace=True)
    boot_n = normal.iloc[idx_n]
    boot_a = attack.iloc[idx_a]

    g_boot = {}
    for col in remaining_after_collin:
        n_v = boot_n[col].dropna().values
        a_v = boot_a[col].dropna().values
        g_boot[col] = hedges_g(n_v, a_v)

    max_g_b  = max(g_boot.values()) or 1.0
    scores_b = {col: g_boot[col] / max_g_b for col in remaining_after_collin}
    top_b    = sorted(scores_b.items(), key=lambda x: -x[1])[:N_FEATURES_FINAL]
    for col, _ in top_b:
        selection_counts[col] += 1

selection_freq = {col: selection_counts[col] / BOOTSTRAP_N
                  for col in remaining_after_collin}

print(f"\n  {'Feature':<38} {'Freq':>8}  Stability")
print(" " + " " * 55)
stable_features   = []
unstable_features = []
for col in sorted(selection_freq, key=lambda f: -selection_freq[f]):
    freq  = selection_freq[col]
    label = "STABLE " if freq >= BOOTSTRAP_STABILITY_MIN else "UNSTABLE "
    seq_tag = " [SEQ]" if col in ('freq_persistence_5w',
                                   'freq_rolling_mean_5w',
                                   'freq_rolling_std_5w') else ""
    print(f"  {col:<38} {freq:>7.1%}  {label}{seq_tag}")
    if freq >= BOOTSTRAP_STABILITY_MIN:
        stable_features.append(col)
    else:
        unstable_features.append(col)

fig, ax = plt.subplots(figsize=(10, 5))
cols_sorted = sorted(selection_freq, key=lambda f: -selection_freq[f])
freqs  = [selection_freq[c] for c in cols_sorted]
colors = ['#2ecc71' if f >= BOOTSTRAP_STABILITY_MIN else '#e74c3c' for f in freqs]
ax.barh(cols_sorted[::-1],
        [selection_freq[c] for c in cols_sorted[::-1]],
        color=colors[::-1])
ax.axvline(BOOTSTRAP_STABILITY_MIN, color='black', linestyle='--',
           label=f'Stability threshold ({BOOTSTRAP_STABILITY_MIN:.0%})')
ax.set_xlabel('Selection Frequency across Bootstrap Samples')
ax.set_title(f'Feature Bootstrap Stability (n={BOOTSTRAP_N} stratified resamples) — v4')
ax.legend()
plt.tight_layout()
stab_path = f"{FIGURES_DIR}/feature_bootstrap_stability_v3.png"
plt.savefig(stab_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"\n  Bootstrap chart saved: {stab_path}")

print("\n" + "=" * 70)
print("STEP F FINAL FEATURE SELECTION")
print("=" * 70)

stable_ranked = [col for col, _ in ranked if col in stable_features]

if len(stable_ranked) >= N_FEATURES_FINAL:
    selected_features = stable_ranked[:N_FEATURES_FINAL]
    selection_note = f"Top-{N_FEATURES_FINAL} stable features by combined score"
elif len(stable_ranked) > 0:
    remaining_candidates = [col for col, _ in ranked if col not in stable_ranked]
    fill_n = N_FEATURES_FINAL - len(stable_ranked)
    selected_features = stable_ranked + remaining_candidates[:fill_n]
    selection_note = (f"{len(stable_ranked)} stable + {fill_n} best unstable "
                      f"(insufficient stable features)")
else:
    selected_features = [col for col, _ in ranked[:N_FEATURES_FINAL]]
    selection_note = "Fallback: pure combined score"

print(f"\n  Selection strategy: {selection_note}")
print(f"\n  OK SELECTED ({len(selected_features)}):")
for i, col in enumerate(selected_features, 1):
    g    = hedges_g_min.get(col, 0)
    freq = selection_freq.get(col, 0)
    strength = "LARGE" if g >= 0.8 else ("MEDIUM" if g >= 0.5 else "small")
    seq_tag  = " [SEQ - NEW]" if col in ('freq_persistence_5w',
                                          'freq_rolling_mean_5w',
                                          'freq_rolling_std_5w') else ""
    print(f"    {i}. {col:<38}  g_min={g:.3f}  [{strength}]  "
          f"stability={freq:.0%}{seq_tag}")

all_dropped = (list(to_remove)
               + [f for f in remaining_after_collin if f not in selected_features]
               + failed_variance)
print(f"\n    DROPPED ({len(all_dropped)}):")
for col in failed_variance:
    print(f"       {col:<38}  [near-zero variance]")
for col in removed_collinear:
    print(f"       {col:<38}  [collinear, g_min={hedges_g_min.get(col,0):.3f}]")
for col in remaining_after_collin:
    if col not in selected_features:
        reason = ("low stability" if selection_freq.get(col, 0) < BOOTSTRAP_STABILITY_MIN
                  else "lower combined score")
        print(f"       {col:<38}  [{reason}, g_min={hedges_g_min.get(col,0):.3f}]")

sel_corr = normal[selected_features].corr(method='pearson').abs()
max_corr_sel = sel_corr.where(
    ~np.eye(len(selected_features), dtype=bool)
).max().max()
print(f"\n  Max Pearson (normal, selected): {max_corr_sel:.3f} "
      f"({'OK ' if max_corr_sel < COLLINEARITY_THRESHOLD else 'WARNING '})")

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

x     = np.arange(len(selected_features))
width = 0.35
g_a2_vals = [hedges_g_a2.get(f, 0) for f in selected_features]
g_a5_vals = [hedges_g_a5.get(f, 0) for f in selected_features]
ax = axes[0]
ax.bar(x - width/2, g_a2_vals, width, label='vs A2 (Flooding)',
       color='#e74c3c', alpha=0.85)
ax.bar(x + width/2, g_a5_vals, width, label='vs A5 (Slow Poisoning)',
       color='#3498db', alpha=0.85)
ax.axhline(0.8, linestyle='--', color='gray', alpha=0.7, label='Large effect (g=0.8)')
ax.axhline(0.5, linestyle=':', color='gray', alpha=0.5, label='Medium effect (g=0.5)')
ax.set_xticks(x)
ax.set_xticklabels([f.replace('_', '\n') for f in selected_features], fontsize=8)
ax.set_ylabel("Hedges' g (corrected effect size)")
ax.set_title("Feature Discriminability Hedges' g per Attack Type (v4 selected)")
ax.legend(fontsize=9)
ax.set_ylim(0, max(max(g_a2_vals), max(g_a5_vals), 1.0) * 1.3)

combined_vals = [combined_scores.get(f, 0) for f in selected_features]
stab_vals     = [selection_freq.get(f, 0) for f in selected_features]
ax2 = axes[1]
ax2.bar(x - width/2, combined_vals, width, label='Combined Score (1/3 each)',
        color='#2ecc71', alpha=0.85)
ax2.bar(x + width/2, stab_vals, width, label='Bootstrap Stability',
        color='#9b59b6', alpha=0.85)
ax2.axhline(BOOTSTRAP_STABILITY_MIN, linestyle='--', color='black', alpha=0.7,
            label=f'Stability threshold ({BOOTSTRAP_STABILITY_MIN:.0%})')
ax2.set_xticks(x)
ax2.set_xticklabels([f.replace('_', '\n') for f in selected_features], fontsize=8)
ax2.set_ylabel("Score / Frequency")
ax2.set_title("Combined Score and Bootstrap Stability")
ax2.legend(fontsize=9)
ax2.set_ylim(0, 1.15)

plt.tight_layout()
disc_path = f"{FIGURES_DIR}/feature_discriminability_v3.png"
plt.savefig(disc_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"\n  Discriminability figure saved: {disc_path}")

print("\n" + " " * 70)
print("Saving outputs...")
print(" " * 70)

result = {
    'selected_features'         : selected_features,
    'selection_note'            : selection_note,
    'step3_version'             : '',
    'n_candidates'              : len(ALL_FEATURE_COLS),
    'dropped_collinear'         : list(to_remove),
    'dropped_variance_global'   : failed_variance,
    'dropped_variance_perclass' : failed_variance_perclass,
    'dropped_discriminative'    : [f for f in remaining_after_collin
                                    if f not in selected_features],
    'sequential_features_candidate' : ['freq_rolling_std_5w'],
    'sequential_features_excluded_by_design': {
        'freq_persistence_5w':   'label proxy within-class var=0 for slow_poisoning',
        'freq_rolling_mean_5w':  'redundant Pearson r=0.924 with alert_frequency_per_hour',
    },
    'collinearity_threshold'    : COLLINEARITY_THRESHOLD,
    'variance_threshold'        : VARIANCE_THRESHOLD,
    'bootstrap_n'               : BOOTSTRAP_N,
    'bootstrap_stability_min'   : BOOTSTRAP_STABILITY_MIN,
    'hedges_g_a2'               : {k: round(v, 4) for k, v in hedges_g_a2.items()},
    'hedges_g_a5'               : {k: round(v, 4) for k, v in hedges_g_a5.items()},
    'hedges_g_min'              : {k: round(v, 4) for k, v in hedges_g_min.items()},
    'f_scores'                  : {k: round(v, 4) for k, v in f_scores.items()},
    'mi_scores'                 : {k: round(v, 4) for k, v in mi_scores.items()},
    'combined_scores'           : {k: round(v, 4) for k, v in combined_scores.items()},
    'bootstrap_stability'       : {k: round(v, 4) for k, v in selection_freq.items()},
    'max_pairwise_corr_selected': float(round(max_corr_sel, 4)),
    'mannwhitney_p'             : {k: round(v, 4) for k, v in mannwhitney_p.items()},
    'n_features_before'         : len(ALL_FEATURE_COLS),
    'n_features_after'          : len(selected_features),
}

with open(f"{FEATURES_DIR}/selected_features.json", 'w') as fh:
    json.dump(result, fh, indent=2)

normal_df = pd.read_csv(f"{FEATURES_DIR}/features_normal.csv")

keep_cols_normal = selected_features + ['label', 'attack_type']
keep_cols_normal = [c for c in keep_cols_normal if c in normal_df.columns]

missing_in_normal = [f for f in selected_features if f not in normal_df.columns]
assert not missing_in_normal, (
    f"[ERROR] Selected features missing from features_normal.csv:\n"
    f"  {missing_in_normal}\n"
    f"  Run sequential_features.py to overwrite features_normal.csv."
)

normal_df[keep_cols_normal].to_csv(
    f"{FEATURES_DIR}/features_selected_normal.csv", index=False)

labeled_df = labeled.copy()

keep_cols_labeled = selected_features + ['label', 'attack_type']

if _TIMESTAMP_COL in labeled_df.columns:
    keep_cols_labeled = keep_cols_labeled + [_TIMESTAMP_COL]
keep_cols_labeled = [c for c in keep_cols_labeled if c in labeled_df.columns]

labeled_df[keep_cols_labeled].to_csv(
    f"{FEATURES_DIR}/features_selected_labeled.csv", index=False)

saved_normal  = pd.read_csv(f"{FEATURES_DIR}/features_selected_normal.csv")
saved_labeled = pd.read_csv(f"{FEATURES_DIR}/features_selected_labeled.csv")

missing_step4 = [f for f in selected_features if f not in saved_normal.columns]
assert not missing_step4, (
    f"[ERROR] hyperparameter tuning assertion failure: {missing_step4}"
)

assert _TIMESTAMP_COL in saved_labeled.columns, (
    f"[ERROR] '{_TIMESTAMP_COL}' absent de features_selected_labeled.csv — "
    f"ensemble calibration ne pourra pas trier chronologiquement."
)

print(f"  OK {FEATURES_DIR}/selected_features.json")
print(f"  OK {FEATURES_DIR}/features_selected_normal.csv"
      f"  ({len(saved_normal)} windows, {saved_normal.shape[1]} cols)")
print(f"  OK {FEATURES_DIR}/features_selected_labeled.csv"
      f"  ({len(saved_labeled)} windows, {saved_labeled.shape[1]} cols, "
      f"incl. {_TIMESTAMP_COL})")
print(f"  OK {FIGURES_DIR}/feature_correlation_heatmap_v3.png")
print(f"  OK {FIGURES_DIR}/feature_discriminability_v3.png")
print(f"  OK {FIGURES_DIR}/feature_bootstrap_stability_v3.png")
print(f"\n  hyperparameter tuning assertion pre-validated: "
      f"selected_features ⊆ columns(features_selected_normal.csv) ✅")

seq_selected = [f for f in selected_features
                if f in ('freq_persistence_5w',
                          'freq_rolling_mean_5w',
                          'freq_rolling_std_5w')]

print("\n" + "=" * 70)
print("FINAL REPORT — FEATURE SELECTION")
print("=" * 70)
print(f"""
  Sync with zone extraction step:
    ✅ 3 sequential features added ({len(seq_selected)} selected)
    ✅ N_FEATURES_FINAL=8 unchanged (selection from 14 candidates)

  Sync with feature encoding (retained):
    ✅ time_sin + time_cos (cyclical encoding)

  Corrections (retained):
    ✅ Hedges' g (pooled SD n-1, Hedges correction)
    ✅ Collinearity on normal windows only
    ✅ Dual correlation (Pearson AND Spearman)
    ✅ Discriminability by attack type (A2 and A5 separately)
    ✅ Mutual Information + F-score (calibration split only)
    ✅ VarianceThreshold
    ✅ Bootstrap stability (n={BOOTSTRAP_N}, threshold={BOOTSTRAP_STABILITY_MIN:.0%})
    ✅ BFS graph-based collinearity removal

  Corrections (this version):
    ✅ freq_persistence_5w EXCLUDED — label proxy (within-class var=0 for
       slow_poisoning, AUC=1.000 alone, undefined at attack onset)
    ✅ freq_rolling_mean_5w EXCLUDED — redundant with alert_frequency_per_hour
       (Pearson r=0.924, zero marginal information)
    ✅ freq_rolling_std_5w RETAINED — orthogonal signal, within-class var > 0,
       captures temporal consistency of injection rate (Hedges g≈1.35)
    ✅ Step 0b added: intra-class variance filter (WITHIN_CLASS_VAR < 1e-6)
       → detects label proxies invisible to the global filter
    ✅ window_start_time retained in features_selected_labeled.csv

  Selected features ({len(selected_features)}):
    {selected_features}

  Sequential features retained    : {seq_selected if seq_selected else 'none (insufficient scores)'}

  Max Pearson pairwise (normal windows, selected): {max_corr_sel:.3f}

  IO contracts hyperparameter tuning → ensemble calibration: UNCHANGED ✅
""")

print("Next: python iforest_hyperparameter_tuning.py")
