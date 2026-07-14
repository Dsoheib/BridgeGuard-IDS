
"""
BridgeGuard IForest Training — Final Calibrated Model
======================================================

Trains the Isolation Forest on the normal-traffic partition using a
stratified 3-way chronological split (60 % train / 20 % calibration /
20 % held-out test).  The anomaly threshold is set empirically on the
calibration partition at the 2nd percentile of calibration scores,
guaranteeing FPR ≤ 2 % without any distributional assumption.
Anomaly scores are mapped to P(normal | score) ∈ [0, 1] via Min-Max
scaling over the [1st, 99th] percentile range of calibration scores.

Outputs
-------
bridgeguard_models/isolation_forest_optimized.pkl
bridgeguard_models/feature_scaler_step5.pkl
bridgeguard_models/feature_scaler_selected.pkl
bridgeguard_models/iforest_optimized_calibration.json
bridgeguard_models/iforest_step5_calibration_v2.png
"""

import hashlib
import json
import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

FEATURES_DIR      = "bridgeguard_features"
MODELS_DIR        = "bridgeguard_models"
TRAIN_FRAC        = 0.60
CALIB_FRAC        = 0.20
TEST_FRAC         = 0.20
FPR_TARGET        = 0.02
PROB_LO_PCTILE    = 1
PROB_HI_PCTILE    = 99
RANDOM_STATE      = 42

os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 70)
print("BridgeGuard IForest Training — Final Calibrated Model")
print("=" * 70)
print(f"\n  Split       : Train={TRAIN_FRAC:.0%} / Calib={CALIB_FRAC:.0%} / Test={TEST_FRAC:.0%}")
print(f"  Threshold   : empirical percentile (FPR target   {FPR_TARGET:.0%})")
print(f"  Probability : Min-Max [{PROB_LO_PCTILE}th, {PROB_HI_PCTILE}th] percentile")
print(f"  predict()   : NEVER used for threshold (contamination is cosmetic)\n")

with open(f"{MODELS_DIR}/best_params.json") as f:
    bp = json.load(f)

SELECTED = bp["selected_features"]

MS_FRACTION = bp.get("ms_fraction", None)
MS_ABSOLUTE = bp.get("max_samples", None)

if MS_FRACTION is not None:
    _ms_mode = f"ms_fraction={MS_FRACTION} (calculated on X_train by IForest training)"
elif MS_ABSOLUTE is not None:
    _ms_mode = f"max_samples={MS_ABSOLUTE} (legacy hyperparameter tuning — backward-compatible)"
    warnings.warn(
        "[IForest training] 'max_samples' found in best_params.json (legacy hyperparameter tuning). "
        "Migrate to hyperparameter tuning that saves 'ms_fraction' for better robustness.",
        UserWarning
    )
else:
    MS_FRACTION = 1.0
    _ms_mode = "ms_fraction=1.0 (fallback: neither ms_fraction nor max_samples found)"
    warnings.warn(
        "[IForest training] Neither 'ms_fraction' nor 'max_samples' found in best_params.json. "
        "Falling back to ms_fraction=1.0 (all X_train samples used).",
        UserWarning
    )

print(f"OK  best_params.json loaded")
print(f"    n_estimators={bp['n_estimators']}  max_features={bp['max_features']}")
print(f"    max_samples  : {_ms_mode}")
print(f"    contamination={bp['contamination']}")
print(f"    Features ({len(SELECTED)}): {', '.join(SELECTED)}")

normal_df  = pd.read_csv(f"{FEATURES_DIR}/features_selected_normal.csv")
labeled_df = pd.read_csv(f"{FEATURES_DIR}/features_selected_labeled.csv")

attack_df  = labeled_df[labeled_df["label"] == 0].reset_index(drop=True)
a2_df      = attack_df[attack_df["attack_type"] == "flooding"]
a5_df      = attack_df[attack_df["attack_type"] == "slow_poisoning"]

X_normal   = normal_df[SELECTED].values
X_a2       = a2_df[SELECTED].values
X_a5       = a5_df[SELECTED].values
X_attack   = attack_df[SELECTED].values

print(f"\nOK  Data : {len(X_normal)} normal | {len(X_a2)} flooding | {len(X_a5)} slow_poisoning")

n       = len(X_normal)
n_train = int(n * TRAIN_FRAC)
n_calib = int(n * CALIB_FRAC)

rng         = np.random.RandomState(RANDOM_STATE)
idx_shuffle = rng.permutation(n)
X_normal_sh = X_normal[idx_shuffle]

X_train     = X_normal_sh[:n_train]
X_calib     = X_normal_sh[n_train : n_train + n_calib]
X_test_norm = X_normal_sh[n_train + n_calib :]

print(f"\nOK  3-Way Split :")
print(f"    Train      : {len(X_train):>3} windows  (IForest training)")
print(f"    Calibration: {len(X_calib):>3} windows  (threshold + prob mapping)")
print(f"    Test       : {len(X_test_norm):>3} windows  (FPR final, never seen)")
print(f"    Attack     : {len(X_attack):>3} windows  (TPR, never seen)")

if MS_FRACTION is not None:
    max_samples_final = min(int(MS_FRACTION * len(X_train)), len(X_train))
else:

    max_samples_final = min(int(MS_ABSOLUTE), len(X_train))

print(f"\nOK  max_samples resolved: {max_samples_final} "
      f"({'ms_fraction=' + str(MS_FRACTION) + ' x ' + str(len(X_train)) if MS_FRACTION is not None else 'absolute max_samples (backward-compatible)'})")

scaler_path_step4 = f"{MODELS_DIR}/feature_scaler_selected.pkl"
if os.path.exists(scaler_path_step4):
    with open(scaler_path_step4, "rb") as f:
        scaler_step4 = pickle.load(f)
    print(f"\nOK  Scaler (hyperparameter tuning) loaded (mean={scaler_step4.mean_.mean():.4f})")
    print(f"    WARNING  IForest training fits its own scaler on X_train only.")
    print(f"    The hyperparameter tuning scaler (fitted on ALL normal) is kept separately.")

scaler = StandardScaler()
scaler.fit(X_train)

X_train_sc     = scaler.transform(X_train)
X_calib_sc     = scaler.transform(X_calib)
X_test_norm_sc = scaler.transform(X_test_norm)
X_a2_sc        = scaler.transform(X_a2)
X_a5_sc        = scaler.transform(X_a5)
X_attack_sc    = scaler.transform(X_attack)

print(f"\nOK  Scaler fitted on X_train ({len(X_train)} windows)")
print(f"      features : {scaler.mean_.round(3).tolist()}")

print(f"\n   Training final IForest...")
iforest = IsolationForest(
    contamination = bp["contamination"],
    n_estimators  = int(bp["n_estimators"]),
    max_samples   = max_samples_final,
    max_features  = float(bp["max_features"]),
    random_state  = RANDOM_STATE,
    n_jobs        = 1,
)
iforest.fit(X_train_sc)
print(f"OK  Training complete (max_samples={max_samples_final})")

scores_train     = iforest.score_samples(X_train_sc)
scores_calib     = iforest.score_samples(X_calib_sc)
scores_test_norm = iforest.score_samples(X_test_norm_sc)
scores_a2        = iforest.score_samples(X_a2_sc)
scores_a5        = iforest.score_samples(X_a5_sc)
scores_attack    = iforest.score_samples(X_attack_sc)

threshold = float(np.percentile(scores_calib, FPR_TARGET * 100))

fpr_calib = (scores_calib < threshold).mean()
fpr_test  = (scores_test_norm < threshold).mean()

lcb_step4 = bp.get("threshold_2sigma", None)

print(f"\n{'='*70}")
print("THRESHOLD CALIBRATION (Calibration Partition only)")
print(f"{'='*70}")
print(f"  Normal scores (calibration) :  μ={scores_calib.mean():.5f}  σ={scores_calib.std():.5f}")
print(f"  Seuil empirique (FPR={FPR_TARGET:.0%}) : {threshold:.5f}")
print(f"    = percentile({FPR_TARGET*100:.0f}) of calibration scores")
if lcb_step4 is not None:
    print(f"  LCB hyperparameter tuning (μ−2σ, reference)  : {lcb_step4:.5f}")
    diff = abs(threshold - lcb_step4)
    print(f"  Empirical difference vs LCB   : {diff:.5f}"
          f"  {' <0.05' if diff < 0.05 else ' >0.05 distribution skewed'}")
print(f"\n  FPR on Calibration        : {fpr_calib*100:.2f}%  (target ≤ {FPR_TARGET*100:.0f}%)")
print(f"  FPR on Test (unbiased)    : {fpr_test*100:.2f}%  "
      f"{' ' if fpr_test <= FPR_TARGET + 0.01 else ' > FPR target'}")

score_lo    = float(np.percentile(scores_calib, PROB_LO_PCTILE))
score_hi    = float(np.percentile(scores_calib, PROB_HI_PCTILE))
score_range = score_hi - score_lo

def score_to_prob(scores, lo=score_lo, hi=score_hi):
    s = np.asarray(scores, dtype=float)
    return np.clip((s - lo) / (hi - lo + 1e-12), 0.0, 1.0)

probs_train     = score_to_prob(scores_train)
probs_calib     = score_to_prob(scores_calib)
probs_test_norm = score_to_prob(scores_test_norm)
probs_a2        = score_to_prob(scores_a2)
probs_a5        = score_to_prob(scores_a5)
probs_attack    = score_to_prob(scores_attack)

tpr_a2  = (scores_a2 < threshold).mean()
tpr_a5  = (scores_a5 < threshold).mean()
tpr_all = (scores_attack < threshold).mean()

all_sc  = np.concatenate([scores_calib, scores_attack])
all_lbl = np.array([1]*len(scores_calib) + [0]*len(scores_attack))
try:
    auc_diag = roc_auc_score(all_lbl, all_sc)
except Exception:
    auc_diag = float("nan")

print(f"\n{'='*70}")
print("EVALUATION RESULTS (Complete distribution)")
print(f"{'='*70}")
print(f"""
  Prob mapping : Min-Max over [{score_lo:.4f}, {score_hi:.4f}] (calibration P[{PROB_LO_PCTILE}%, {PROB_HI_PCTILE}%])
  Threshold    : {threshold:.5f} (empirical percentile {FPR_TARGET*100:.0f}%)

  ┌─────────────────────────────────────────────────────────────┐
  │  Partition        N     FPR/TPR   P(Normal) μ    P(Normal) σ│
  ├─────────────────────────────────────────────────────────────┤
  │  Train (normal)  {len(scores_train):>4}  FPR:{(scores_train<threshold).mean()*100:5.1f}%   {probs_train.mean():.3f}         {probs_train.std():.3f}       │
  │  Calib (normal)  {len(scores_calib):>4}  FPR:{fpr_calib*100:5.1f}%   {probs_calib.mean():.3f}         {probs_calib.std():.3f}       │
  │  Test  (normal)  {len(scores_test_norm):>4}  FPR:{fpr_test*100:5.1f}%   {probs_test_norm.mean():.3f}         {probs_test_norm.std():.3f}       │
  │  A2    (attack)  {len(scores_a2):>4}  TPR:{tpr_a2*100:5.1f}%   {probs_a2.mean():.3f}         {probs_a2.std():.3f}       │
  │  A5    (attack)  {len(scores_a5):>4}  TPR:{tpr_a5*100:5.1f}%   {probs_a5.mean():.3f}         {probs_a5.std():.3f}       │
  │  All attacks     {len(scores_attack):>4}  TPR:{tpr_all*100:5.1f}%   {probs_attack.mean():.3f}         {probs_attack.std():.3f}       │
  └─────────────────────────────────────────────────────────────┘
  AUC diagnostic        : {auc_diag:.4f}  {' >0.90' if auc_diag>0.90 else ' <0.90'}
  FPR Test (unbiased)   : {fpr_test*100:.2f}%  {' ' if fpr_test<=FPR_TARGET+0.01 else ' >FPR_TARGET'}
  TPR A2                : {tpr_a2*100:.1f}%   {' >80%' if tpr_a2>0.80 else ' <80%'}
  TPR A5                : {tpr_a5*100:.1f}%   {' >80%' if tpr_a5>0.80 else ' <80%'}
""")

if fpr_test > FPR_TARGET + 0.05:
    print(f"  WARNING  FPR test ({fpr_test*100:.1f}%) exceeds FPR calib by >5 pts.")
    print(f"        Test distribution differs from calibration distribution. Risk of overfitting on calib.")
if tpr_a2 < 0.50:
    print(f"  WARNING  Low TPR A2 ({tpr_a2*100:.1f}%). Feature selection may be insufficient for flooding detection.")
if tpr_a5 < 0.50:
    print(f"  WARNING  Low TPR A5 ({tpr_a5*100:.1f}%). Feature selection may be insufficient for slow poisoning detection.")

print(f"   Generating plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("BridgeGuard IForest Calibration: Score Distributions and Probability Mapping", fontsize=13, fontweight="bold")

ax = axes[0, 0]
ax.hist(scores_train, bins=30, alpha=0.5, color="#2E86AB",
        label=f"Train normal (n={len(scores_train)})", edgecolor="black", linewidth=0.5)
ax.hist(scores_calib, bins=15, alpha=0.6, color="#3BB273",
        label=f"Calib normal (n={len(scores_calib)})", edgecolor="black", linewidth=0.5)
ax.hist(scores_test_norm, bins=10, alpha=0.6, color="#7B2D8B",
        label=f"Test normal  (n={len(scores_test_norm)})", edgecolor="black", linewidth=0.5)
ax.hist(scores_a2, bins=30, alpha=0.45, color="#C73E1D",
        label=f"A2 flooding  (n={len(scores_a2)})", edgecolor="black", linewidth=0.5)
ax.hist(scores_a5, bins=30, alpha=0.45, color="#F4A261",
        label=f"A5 slow      (n={len(scores_a5)})", edgecolor="black", linewidth=0.5)
ax.axvline(threshold, color="black", linestyle="--", lw=2,
           label=f"Empirical threshold τ={threshold:.3f}")
ax.set_title("Score Distributions by Partition")
ax.set_xlabel("Anomaly Score (sklearn, negative)")
ax.set_ylabel("Frequency")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis="y")

ax = axes[0, 1]
score_range_plot = np.linspace(
    min(scores_attack.min(), scores_calib.min()) - 0.05,
    scores_calib.max() + 0.05, 300
)
probs_curve = score_to_prob(score_range_plot)
ax.plot(score_range_plot, probs_curve, color="#2E86AB", lw=2.5,
        label="P(Normal|score) Empirical Min-Max")
ax.axvline(threshold, color="black", linestyle="--", lw=1.5,
           label=f"τ={threshold:.3f}")
ax.axvline(score_lo, color="gray", linestyle=":", lw=1.5,
           label=f"P{PROB_LO_PCTILE} (P=0): {score_lo:.3f}")
ax.axvline(score_hi, color="gray", linestyle="-.", lw=1.5,
           label=f"P{PROB_HI_PCTILE} (P=1): {score_hi:.3f}")
ax.axhline(0.8, color="green", linestyle=":", alpha=0.5, label="Zone FORWARD (P 0.8)")
ax.axhline(0.5, color="orange", linestyle=":", alpha=0.5, label="Zone DROP (P 0.5)")
ax.scatter(scores_calib, probs_calib,
           color="#3BB273", s=20, alpha=0.6, zorder=5, label="Calib normal")
ax.scatter(scores_a2[:min(50, len(scores_a2))],
           probs_a2[:min(50, len(probs_a2))],
           color="#C73E1D", s=20, alpha=0.6, zorder=5, label="A2 (50 pts)")
ax.scatter(scores_a5[:min(50, len(scores_a5))],
           probs_a5[:min(50, len(probs_a5))],
           color="#F4A261", s=20, alpha=0.6, zorder=5, label="A5 (50 pts)")
ax.set_title("Score P(Normal): Empirical Min-Max Mapping")
ax.set_xlabel("Anomaly Score")
ax.set_ylabel("P(Normal | score)")
ax.set_ylim(-0.05, 1.10)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
data_boxes = [probs_train, probs_test_norm, probs_a2, probs_a5]
labels_box  = [f"Train\nN={len(probs_train)}",
               f"Test\nN={len(probs_test_norm)}",
               f"A2\nN={len(probs_a2)}",
               f"A5\nN={len(probs_a5)}"]
colors_box  = ["#2E86AB", "#7B2D8B", "#C73E1D", "#F4A261"]
bp_plot = ax.boxplot(data_boxes, patch_artist=True, notch=False,
                     medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp_plot["boxes"], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticklabels(labels_box, fontsize=9)
ax.axhline(0.8, color="green", linestyle="--", alpha=0.5, label="FORWARD (0.8)")
ax.axhline(0.5, color="orange", linestyle="--", alpha=0.5, label="DROP (0.5)")
ax.set_title("P(Normal) Distribution by Class")
ax.set_ylabel("P(Normal)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

ax = axes[1, 1]
partitions = ["Train\n(normal)", "Calib\n(normal)", "Test\n(normal)", "A2\n(attack)", "A5\n(attack)"]
rates = [
    (scores_train < threshold).mean() * 100,
    fpr_calib * 100,
    fpr_test  * 100,
    tpr_a2    * 100,
    tpr_a5    * 100,
]
colors_bar = ["#2E86AB", "#3BB273", "#7B2D8B", "#C73E1D", "#F4A261"]
bars = ax.bar(partitions, rates, color=colors_bar, alpha=0.8, edgecolor="black", linewidth=0.8)
ax.axhline(FPR_TARGET * 100, color="red", linestyle="--", lw=2,
           label=f"FPR target ({FPR_TARGET*100:.0f}%)")
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{rate:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_title("FPR (Normal) and TPR (Attack) by Partition")
ax.set_ylabel("Rate (%)")
ax.set_ylim(0, max(rates) * 1.2 + 5)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plot_path = f"{MODELS_DIR}/iforest_step5_calibration_v2.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"OK  Plot saved: {plot_path}")

with open(f"{MODELS_DIR}/isolation_forest_optimized.pkl", "wb") as f:
    pickle.dump(iforest, f)

with open(f"{MODELS_DIR}/feature_scaler_step5.pkl", "wb") as f:
    pickle.dump(scaler, f)

scaler_selected_path = f"{MODELS_DIR}/feature_scaler_selected.pkl"
with open(scaler_selected_path, "wb") as f:
    pickle.dump(scaler, f)
with open(scaler_selected_path, "rb") as f:
    scaler_step5_md5 = hashlib.md5(f.read()).hexdigest()

_step4_md5 = bp.get("scaler_md5", "")
if _step4_md5 and _step4_md5 != scaler_step5_md5:
    print(f"\n  Scaler chain updated:")
    print(f"    hyperparameter tuning (HP tuning) MD5 : {_step4_md5}")
    print(f"    IForest training (inference)  MD5 : {scaler_step5_md5}")
    print(f"    feature_scaler_selected.pkl now holds the IForest training scaler.")
    print(f"    Downstream scripts must verify against scaler_step5_md5 in")
    print(f"    iforest_optimized_calibration.json, not scaler_md5 in best_params.json.")

calibration = {

    "threshold_empirical"        : float(threshold),
    "threshold_fpr_target"       : FPR_TARGET,
    "threshold_percentile_used"  : FPR_TARGET * 100,
    "prob_score_lo"              : float(score_lo),
    "prob_score_hi"              : float(score_hi),
    "prob_mapping_method"        : f"min-max P[{PROB_LO_PCTILE},{PROB_HI_PCTILE}] empirical",

    "scores_calib_mean"          : float(scores_calib.mean()),
    "scores_calib_std"           : float(scores_calib.std()),
    "scores_calib_p02"           : float(np.percentile(scores_calib, 2)),
    "scores_calib_p98"           : float(np.percentile(scores_calib, 98)),

    "fpr_calib"                  : float(fpr_calib),
    "fpr_test_unbiased"          : float(fpr_test),

    "tpr_a2_distribution"        : float(tpr_a2),
    "tpr_a5_distribution"        : float(tpr_a5),
    "tpr_all_attacks"            : float(tpr_all),
    "auc_diagnostic"             : float(auc_diag),

    "split_train_n"              : len(X_train),
    "split_calib_n"              : len(X_calib),
    "split_test_n"               : len(X_test_norm),
    "split_attack_n"             : len(X_attack),
    "split_fracs"                : [TRAIN_FRAC, CALIB_FRAC, TEST_FRAC],

    "max_samples_resolved"       : max_samples_final,
    "ms_fraction_used"           : MS_FRACTION,
    "ms_absolute_fallback"       : MS_ABSOLUTE,

    "selected_features"          : SELECTED,
    "hyperparameters"            : {k: v for k, v in bp.items() if k != "selected_features"},

    "scaler_step5_md5"           : scaler_step5_md5,
    "scaler_step4_md5"           : bp.get("scaler_md5", ""),
    "scaler_note"                : (
        "feature_scaler_selected.pkl holds the IForest training scaler (fitted on X_train). "
        "Verify scaler_step5_md5 in ensemble calibration/stats, not scaler_md5 from best_params.json."
    ),

    "calibration_version"        : "v3-ms-fraction-sync-step4v6",
    "leakage_free"               : True,
    "gaussian_assumption"        : False,
    "magic_number_sigmoid_scale" : False,
    "evaluation_on_distribution" : True,
    "random_state"               : RANDOM_STATE,
}

with open(f"{MODELS_DIR}/iforest_optimized_calibration.json", "w") as f:
    json.dump(calibration, f, indent=2)

print(f"\n{'='*70}")
print(f"OK  isolation_forest_optimized.pkl")
print(f"OK  feature_scaler_step5.pkl  (fitted on X_train={len(X_train)} windows)")
print(f"OK  feature_scaler_selected.pkl  [MD5={scaler_step5_md5[:16]}...]  (inference scaler)")
print(f"OK  iforest_optimized_calibration.json")
print(f"\n   PAPER SUMMARY :")
print(f"    Threshold = {threshold:.5f}  (percentile {FPR_TARGET*100:.0f}% of calibration normal scores)")
print(f"    FPR test (unbiased) = {fpr_test*100:.2f}%")
print(f"    TPR A2 (full dist.)   = {tpr_a2*100:.1f}%  |  TPR A5 = {tpr_a5*100:.1f}%")
print(f"    P(Normal) mapping : [{score_lo:.4f}, {score_hi:.4f}] → [0, 1]")
print(f"    max_samples = {max_samples_final}  ({_ms_mode})")
print(f"\nNext: python train_lstm.py")
print(f"{'='*70}")
