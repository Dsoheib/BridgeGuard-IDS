
"""
BridgeGuard IForest Hyperparameter Tuning
==============================================

Selects Isolation Forest structural hyperparameters (n_estimators, max_features,
max_samples) via 5-fold cross-validation on normal training data only, maximising
the Lower Confidence Bound (LCB = _n 2 _n). No attack data is used during
tuning; post-selection diagnostics are purely informational.

SCALER STRATEGY

 The scaler is fitted on X_normal_TRAIN (first 60% chronologically).
 The same scaler (feature_scaler_selected.pkl) must be used by all
 downstream scripts. Never refit the scaler. The MD5 hash stored in
 best_params.json allows downstream verification.

PAPER (Section IV)

 "IForest structural hyperparameters (n_estimators, max_features, max_samples)
 were selected via 5-fold cross-validation on normal training data only,
 maximising the Lower Confidence Bound LCB = _n 2 _n, where _n and _n
 are the mean and standard deviation of IForest anomaly scores on held-out
 normal validation folds. This one-class objective is equivalent to minimising
 the SVDD hypersphere radius. Temporal fold ordering was preserved
 (KFold shuffle=False). No attack data or threshold assumption was used
 in hyperparameter selection."
"""

import hashlib
import json
import os
import pickle
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)

FEATURES_DIR     = "bridgeguard_features"
MODELS_DIR       = "bridgeguard_models"
N_FOLDS          = 5
RANDOM_STATE     = 42
LCB_K            = 2.0
STABILITY_WEIGHT = 1.0
TRAIN_FRACTION   = 0.60

os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 70)
print("BridgeGuard IForest Hyperparameter Tuning (LCB Compactness, Temporally Ordered CV)")
print("=" * 70)
print(f"\n  Objective  : LCB =  _n   {LCB_K}  _n  (Lower Confidence Bound)")
print(f"  Stability  : LCB_final = mean(LCB_folds)   {STABILITY_WEIGHT} std(LCB_folds)")
print(f"  CV order   : KFold(shuffle=False)   temporal order preserved")
print(f"  Scaler     : fit on X_normal_TRAIN ({int(TRAIN_FRACTION*100)}%) only\n")

with open(f"{FEATURES_DIR}/selected_features.json") as f:
    sel = json.load(f)
SELECTED   = sel["selected_features"]
N_FEATURES = len(SELECTED)

normal_df  = pd.read_csv(f"{FEATURES_DIR}/features_selected_normal.csv")
labeled_df = pd.read_csv(f"{FEATURES_DIR}/features_selected_labeled.csv")

csv_cols     = set(normal_df.columns)
missing_feats = [f for f in SELECTED if f not in csv_cols]
extra_old    = [f for f in ['time_of_day_deviation'] if f in csv_cols]

if missing_feats:
    raise RuntimeError(
        f"\n[ERROR] Features in selected_features.json not found in CSV:\n"
        f"  Missing: {missing_feats}\n"
        f"  ➜ Re-run attack_aware_feature_selection.py with the updated ALL_FEATURE_COLS\n"
        f"    (time_sin + time_cos instead of time_of_day_deviation)"
    )
if extra_old:
    print(f"  WARNING  Old feature 'time_of_day_deviation' found in CSV but not in SELECTED.")
    print(f"      This is harmless   it will be ignored during selection.\n")

print(f"  OK Feature coherence check passed: {N_FEATURES} features all present in CSV")
if 'time_sin' in SELECTED and 'time_cos' in SELECTED:
    print(f"  OK Cyclical time features (time_sin, time_cos) confirmed in SELECTED")
print()

attack_df = labeled_df[labeled_df["label"] == 0].reset_index(drop=True)
a2_raw    = attack_df[attack_df["attack_type"] == "flooding"][SELECTED].values
a5_raw    = attack_df[attack_df["attack_type"] == "slow_poisoning"][SELECTED].values

X_normal = normal_df[SELECTED].values

split_idx    = int(len(X_normal) * TRAIN_FRACTION)
X_normal_tr  = X_normal[:split_idx]
X_normal_val = X_normal[split_idx:]
n_train      = len(X_normal_tr)
n_total      = len(X_normal)

print(f"  Normal total : {n_total}  (train={n_train}, held-out={len(X_normal_val)})")
print(f"  Attack       : {len(a2_raw)} flooding  |  {len(a5_raw)} slow poisoning")
print(f"  Features ({N_FEATURES}): {SELECTED}")
print(f"\n  WARNING  Attack data loaded   accessed only AFTER HP selection (zero leakage).\n")

global_scaler = StandardScaler()
global_scaler.fit(X_normal_tr)
X_normal_tr_sc = global_scaler.transform(X_normal_tr)

scaler_path = f"{MODELS_DIR}/feature_scaler_selected.pkl"
with open(scaler_path, "wb") as f:
    pickle.dump(global_scaler, f)
with open(scaler_path, "rb") as f:
    scaler_md5 = hashlib.md5(f.read()).hexdigest()
print(f"  OK Scaler fitted on X_normal_TRAIN ({n_train} windows)")
print(f"     MD5: {scaler_md5}  (verify in IForest training/6 with same hash)\n")

PARAM_GRID = {
    "n_estimators"  : [100, 150, 200, 300, 500],
    "max_feat_frac" : [0.6, 0.8, 1.0],
    "ms_fraction"   : [0.70, 0.85, 1.00],
}

combos   = list(product(
    PARAM_GRID["n_estimators"],
    PARAM_GRID["max_feat_frac"],
    PARAM_GRID["ms_fraction"],
))
n_combos = len(combos)

print(f"  Grid search : {n_combos} configurations")
print(f"  Objective   : LCB =  _n   {LCB_K}  _n  [one-class, no attack data]")
print(f"  CV          : {N_FOLDS}-fold KFold(shuffle=False) on X_normal_TRAIN\n")

kf = KFold(n_splits=N_FOLDS, shuffle=False)
fold_splits = list(kf.split(X_normal_tr_sc))

first_n, first_mf, first_ms = combos[0]
best_obj = -np.inf
best_cfg = {
    "n_estimators"  : first_n,
    "max_feat_frac" : first_mf,
    "ms_fraction"   : first_ms,
    "mean_lcb"      : -np.inf,
    "std_lcb"       : np.inf,
    "final_obj"     : -np.inf,
}

results = []

for i, (n_est, mf_frac, ms_frac) in enumerate(combos):

    lcb_folds = []

    for train_idx, val_idx in fold_splits:
        X_tr_sc = X_normal_tr_sc[train_idx]
        X_vl_sc = X_normal_tr_sc[val_idx]

        n_fold_train = len(X_tr_sc)

        max_samp = max(2, int(ms_frac * n_fold_train))

        max_samp = min(max_samp, n_fold_train)

        try:
            clf = IsolationForest(
                contamination = "auto",
                n_estimators  = n_est,
                max_samples   = max_samp,
                max_features  = mf_frac,
                random_state  = RANDOM_STATE,
                n_jobs        = 1,
            )
            clf.fit(X_tr_sc)
            scores_vl = clf.score_samples(X_vl_sc)
            mu_fold   = scores_vl.mean()
            sig_fold  = scores_vl.std()
            lcb       = mu_fold - LCB_K * sig_fold

        except Exception as e:
            lcb = -np.inf

        lcb_folds.append(lcb)

    lcb_arr  = np.array(lcb_folds)
    mean_lcb = lcb_arr.mean()
    std_lcb  = lcb_arr.std()
    final_obj = mean_lcb - STABILITY_WEIGHT * std_lcb

    result = {
        "n_estimators"  : n_est,
        "max_feat_frac" : mf_frac,
        "ms_fraction"   : ms_frac,
        "mean_lcb"      : round(mean_lcb, 5),
        "std_lcb"       : round(std_lcb, 5),
        "final_obj"     : round(final_obj, 5),
    }
    results.append(result)

    if final_obj > best_obj:
        best_obj = final_obj
        best_cfg = result.copy()

    if (i + 1) % 15 == 0 or (i + 1) == n_combos:
        print(f"  [{i+1:3d}/{n_combos}]  Best LCB={best_obj:.4f}  "
              f"n_est={best_cfg['n_estimators']}  "
              f"mf={best_cfg['max_feat_frac']}  "
              f"ms={best_cfg['ms_fraction']:.2f}")

results_df = pd.DataFrame(results).sort_values("final_obj", ascending=False)

print(f"\n{'='*70}")
print(f"TOP 10   LCB Compactness (  {LCB_K} , temporal CV, no leakage)")
print(f"{'='*70}")
print(f"{'Rank':>4}  {'n_est':>5}  {'mf':>4}  {'ms':>4}  "
      f"{'LCB_mean':>9}  {'LCB_std':>8}  {'Final_obj':>10}")
print("-" * 60)
for rank, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
    print(f"  {rank:2d}   {int(row.n_estimators):>5}  "
          f"{row.max_feat_frac:>4.1f}  "
          f"{row.ms_fraction:>4.2f}  "
          f"{row.mean_lcb:>9.4f}  "
          f"{row.std_lcb:>8.4f}  "
          f"{row.final_obj:>10.4f}")

top5_vals = results_df["final_obj"].head(5).values
obj_range = top5_vals[0] - top5_vals[-1]
if obj_range < 0.005:
    print(f"\n  WARNING  FLAT OPTIMUM (top-5 range = {obj_range:.5f} < 0.005)")
    print(f"     IForest insensitive to HPs in this domain.")
    print(f"     Selecting by minimum std_lcb (most stable).")
    best_cfg = results_df.sort_values(
        ["final_obj", "std_lcb"], ascending=[False, True]
    ).iloc[0].to_dict()

ms_abs_train = max(2, min(int(best_cfg["ms_fraction"] * n_train), n_train))

best_clf = IsolationForest(
    contamination = "auto",
    n_estimators  = int(best_cfg["n_estimators"]),
    max_samples   = ms_abs_train,
    max_features  = best_cfg["max_feat_frac"],
    random_state  = RANDOM_STATE,
    n_jobs        = 1,
)
best_clf.fit(X_normal_tr_sc)

scores_tr = best_clf.score_samples(X_normal_tr_sc)
mu_full   = scores_tr.mean()
sig_full  = scores_tr.std()
lcb_full  = mu_full - LCB_K * sig_full

print(f"\n{'='*70}")
print("SELECTED CONFIGURATION")
print(f"{'='*70}")
print(f"""
  n_estimators  : {int(best_cfg['n_estimators'])}
  max_features  : {best_cfg['max_feat_frac']}
  ms_fraction   : {best_cfg['ms_fraction']:.2f}  (IForest training computes abs from its n_train)
  ms_abs_train  : {ms_abs_train}  (= {best_cfg['ms_fraction']:.0%} × {n_train} train windows)
  contamination : 'auto'  → threshold set in IForest training via FPR-target percentile

  Normal TRAIN distribution:
    μ_n  = {mu_full:+.5f}   (negative, sklearn convention)
    σ_n  = {sig_full:.5f}
    LCB  = μ − {LCB_K}·σ = {lcb_full:+.5f}

  CV cross-folds:
    LCB mean : {best_cfg['mean_lcb']:.5f}
    LCB std  : {best_cfg['std_lcb']:.5f}
    Final obj: {best_cfg['final_obj']:.5f}
""")

print(f"{'='*70}")
print("POST-SELECTION DIAGNOSTIC (attacks read-only, does not affect HP selection)")
print(f"{'='*70}\n")

X_a2_sc = global_scaler.transform(a2_raw)
X_a5_sc = global_scaler.transform(a5_raw)
sc_a2   = best_clf.score_samples(X_a2_sc)
sc_a5   = best_clf.score_samples(X_a5_sc)

thr_lcb      = lcb_full
thr_empirical = float(np.percentile(scores_tr, 2))

fpr_at_lcb       = (scores_tr  < thr_lcb).mean()
fpr_at_empirical = (scores_tr  < thr_empirical).mean()
tpr_a2_lcb       = (sc_a2      < thr_lcb).mean()
tpr_a5_lcb       = (sc_a5      < thr_lcb).mean()
tpr_a2_emp       = (sc_a2      < thr_empirical).mean()
tpr_a5_emp       = (sc_a5      < thr_empirical).mean()
sep_a2           = mu_full - sc_a2.mean()
sep_a5           = mu_full - sc_a5.mean()

all_sc  = np.concatenate([scores_tr, sc_a2, sc_a5])
all_lbl = np.array([1]*len(scores_tr) + [0]*len(sc_a2) + [0]*len(sc_a5))
diag_auc_train = roc_auc_score(all_lbl, all_sc)

scores_val = best_clf.score_samples(global_scaler.transform(X_normal_val))

print(f"  Normal TRAIN :  ={mu_full:+.5f}   ={sig_full:.5f}")
print(f"  Normal VAL   :  ={scores_val.mean():+.5f}   ={scores_val.std():.5f}  "
      f"{' stable' if abs(scores_val.mean()-mu_full)<0.05 else ' distribution shift'}")
print(f"  A2 (Flooding)   :  ={sc_a2.mean():+.5f}   ={sc_a2.std():.5f}  sep={sep_a2:.4f}")
print(f"  A5 (SlowPoison) :  ={sc_a5.mean():+.5f}   ={sc_a5.std():.5f}  sep={sep_a5:.4f}")
print()
print(f"     Threshold comparison                                           ")
print(f"  LCB threshold (  {LCB_K} )   = {thr_lcb:+.5f}")
print(f"  Empirical 2-pct threshold    = {thr_empirical:+.5f}")
print()
print(f"  At LCB threshold:")
print(f"    FPR         = {fpr_at_lcb*100:.2f}%  "
      f"{' ' if fpr_at_lcb < 0.05 else ' >5%: distribution is non-Gaussian, use empirical thr'}")
print(f"    TPR A2      = {tpr_a2_lcb*100:.1f}%  {' ' if tpr_a2_lcb > 0.80 else ' <80%'}")
print(f"    TPR A5      = {tpr_a5_lcb*100:.1f}%  {' ' if tpr_a5_lcb > 0.80 else ' <80%'}")
print()
print(f"  At empirical 2-pct threshold:")
print(f"    FPR         = {fpr_at_empirical*100:.2f}%  (by construction  2%)")
print(f"    TPR A2      = {tpr_a2_emp*100:.1f}%  {' ' if tpr_a2_emp > 0.80 else ' <80%'}")
print(f"    TPR A5      = {tpr_a5_emp*100:.1f}%  {' ' if tpr_a5_emp > 0.80 else ' <80%'}")
print()
print(f"  WARNING  diag_auc={diag_auc_train:.4f} computed on TRAINING SET   DO NOT REPORT")
print(f"     (Training-set AUC is optimistic. Final AUC reported from LSTM training test set.)")

if sep_a2 < 0.05 or sep_a5 < 0.05:
    print(f"\n  WARNING  Weak separation. Check feature selection features (Hedges' g < 0.5 for some features).")

output = {

    "n_estimators"               : int(best_cfg["n_estimators"]),
    "max_features"               : best_cfg["max_feat_frac"],

    "ms_fraction"                : best_cfg["ms_fraction"],
    "ms_abs_for_n_train"         : ms_abs_train,
    "n_train_this_step"          : n_train,
    "contamination"              : "auto",
    "selected_features"          : SELECTED,

    "cv_lcb_mean"                : round(best_cfg["mean_lcb"], 5),
    "cv_lcb_std"                 : round(best_cfg["std_lcb"], 5),
    "cv_final_obj"               : round(best_cfg["final_obj"], 5),
    "mu_n_train"                 : round(float(mu_full), 5),
    "sigma_n_train"              : round(float(sig_full), 5),
    "lcb_full"                   : round(float(lcb_full), 5),
    "lcb_k"                      : LCB_K,
    "stability_weight"           : STABILITY_WEIGHT,

    "diag_auc_TRAINING_SET_ONLY" : round(float(diag_auc_train), 5),
    "diag_DO_NOT_REPORT_AUC"     : True,
    "diag_sep_a2"                : round(float(sep_a2), 5),
    "diag_sep_a5"                : round(float(sep_a5), 5),
    "diag_fpr_at_lcb"            : round(float(fpr_at_lcb), 5),
    "diag_fpr_at_empirical_2pct" : round(float(fpr_at_empirical), 5),
    "diag_thr_lcb"               : round(float(thr_lcb), 5),
    "diag_thr_empirical_2pct"    : round(float(thr_empirical), 5),
    "diag_tpr_a2_lcb"            : round(float(tpr_a2_lcb), 5),
    "diag_tpr_a5_lcb"            : round(float(tpr_a5_lcb), 5),
    "diag_tpr_a2_emp"            : round(float(tpr_a2_emp), 5),
    "diag_tpr_a5_emp"            : round(float(tpr_a5_emp), 5),
    "diag_mu_normal_val"         : round(float(scores_val.mean()), 5),

    "scaler_md5"                 : scaler_md5,
    "scaler_fitted_on"           : f"X_normal_TRAIN ({n_train} windows, first {int(TRAIN_FRACTION*100)}%)",
    "scaler_note"                : (
        "Load feature_scaler_selected.pkl in IForest training/6/7. "
        "Never refit. Verify MD5 matches before use."
    ),

    "tuning_version"             : "1.0",
    "objective_formula"          : f"mean(LCB_folds) - {STABILITY_WEIGHT}*std(LCB_folds)",
    "lcb_formula"                : f"mu_fold - {LCB_K}*sigma_fold",
    "cv_strategy"                : "KFold(shuffle=False) temporal order preserved",
    "n_jobs"                     : 1,
    "contamination_strategy"     : "determined-in-train_iforest-via-calibration-percentile",
    "leakage_free"               : True,
    "circular_logic"             : False,
    "max_samples_strategy"       : "ms_fraction saved; IForest training computes abs from its n_train",
    "sign_bug_v4_fixed"          : True,
    "random_state"               : RANDOM_STATE,
    "n_folds_cv"                 : N_FOLDS,
}

with open(f"{MODELS_DIR}/best_params.json", "w") as f:
    json.dump(output, f, indent=2)

results_df.to_csv(f"{MODELS_DIR}/grid_search_results.csv", index=False)

print(f"\n{'='*70}")
print(f"OK  best_params.json          [ms_fraction saved, scaler_md5={scaler_md5[:8]}...]")
print(f"OK  grid_search_results.csv   [{n_combos} configurations]")
print(f"OK  feature_scaler_selected.pkl  [fit on {n_train} normal TRAIN windows]")
print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  CRITICAL NOTES FOR IForest training                                  │
  │                                                             │
  │  1. Load feature_scaler_selected.pkl — NEVER REFIT         │
  │     Verify MD5={scaler_md5[:16]}...                         │
  │                                                             │
  │  2. Compute ms_abs = int(ms_fraction * len(X_train))       │
  │     Do NOT use ms_abs_for_n_train from JSON                 │
  │     (IForest training train set may differ from hyperparameter tuning n_train)         │
  │                                                             │
  │  3. Set threshold via np.percentile(scores_calib, X)        │
  │     targeting FPR ≤ 2% — never use clf.predict()           │
  │                                                             │
  │  4. diag_auc={diag_auc_train:.4f} is TRAINING AUC — DO NOT report  │
  │     Report only LSTM training test set AUC                          │
  └─────────────────────────────────────────────────────────────┘
""")
print(f"Next: python train_iforest.py")
print(f"{'='*70}")
