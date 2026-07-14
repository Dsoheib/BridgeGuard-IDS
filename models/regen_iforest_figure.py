
"""
Standalone regeneration of iforest_step5_calibration_v2.png
using saved model artifacts no retraining required.
Run from the project root: python models/regen_iforest_figure.py
"""

import json
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES_DIR = "bridgeguard_features"
MODELS_DIR   = "bridgeguard_models"
OUT_PATH     = "figures/iforest_step5_calibration_v2.png"

with open(f"{MODELS_DIR}/iforest_optimized_calibration.json") as f:
    cal = json.load(f)

with open(f"{MODELS_DIR}/isolation_forest_optimized.pkl", "rb") as f:
    iforest = pickle.load(f)

with open(f"{MODELS_DIR}/feature_scaler_step5.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(f"{MODELS_DIR}/best_params.json") as f:
    bp = json.load(f)

SELECTED      = bp["selected_features"]
threshold     = cal["threshold_empirical"]
score_lo      = cal["prob_score_lo"]
score_hi      = cal["prob_score_hi"]
FPR_TARGET    = cal["threshold_fpr_target"]

def score_to_prob(scores, lo=score_lo, hi=score_hi):
    s = np.asarray(scores, dtype=float)
    return np.clip((s - lo) / (hi - lo + 1e-12), 0.0, 1.0)

normal_df  = pd.read_csv(f"{FEATURES_DIR}/features_selected_normal.csv")
labeled_df = pd.read_csv(f"{FEATURES_DIR}/features_selected_labeled.csv")

attack_df  = labeled_df[labeled_df["label"] == 0].reset_index(drop=True)
a2_df      = attack_df[attack_df["attack_type"] == "flooding"]
a5_df      = attack_df[attack_df["attack_type"] == "slow_poisoning"]

X_normal = normal_df[SELECTED].values
X_a2     = a2_df[SELECTED].values
X_a5     = a5_df[SELECTED].values
X_attack = attack_df[SELECTED].values

TRAIN_FRAC = cal["split_fracs"][0]
CALIB_FRAC = cal["split_fracs"][1]
n          = len(X_normal)
n_train    = int(n * TRAIN_FRAC)
n_calib    = int(n * CALIB_FRAC)

rng         = np.random.RandomState(42)
idx_shuffle = rng.permutation(n)
X_normal_sh = X_normal[idx_shuffle]

X_train     = X_normal_sh[:n_train]
X_calib     = X_normal_sh[n_train : n_train + n_calib]
X_test_norm = X_normal_sh[n_train + n_calib :]

X_train_sc     = scaler.transform(X_train)
X_calib_sc     = scaler.transform(X_calib)
X_test_norm_sc = scaler.transform(X_test_norm)
X_a2_sc        = scaler.transform(X_a2)
X_a5_sc        = scaler.transform(X_a5)
X_attack_sc    = scaler.transform(X_attack)

scores_train     = iforest.score_samples(X_train_sc)
scores_calib     = iforest.score_samples(X_calib_sc)
scores_test_norm = iforest.score_samples(X_test_norm_sc)
scores_a2        = iforest.score_samples(X_a2_sc)
scores_a5        = iforest.score_samples(X_a5_sc)
scores_attack    = iforest.score_samples(X_attack_sc)

probs_train     = score_to_prob(scores_train)
probs_calib     = score_to_prob(scores_calib)
probs_test_norm = score_to_prob(scores_test_norm)
probs_a2        = score_to_prob(scores_a2)
probs_a5        = score_to_prob(scores_a5)

fpr_calib = (scores_calib     < threshold).mean()
fpr_test  = (scores_test_norm < threshold).mean()
tpr_a2    = (scores_a2        < threshold).mean()
tpr_a5    = (scores_a5        < threshold).mean()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "BridgeGuard IForest Calibration: Score Distributions and Probability Mapping",
    fontsize=13, fontweight="bold"
)

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
        label=f"A5 slow poisoning (n={len(scores_a5)})", edgecolor="black", linewidth=0.5)
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
        label="P(Normal|score) empirical Min-Max")
ax.axvline(threshold, color="black", linestyle="--", lw=1.5,
           label=f"τ={threshold:.3f}")
ax.axvline(score_lo, color="gray", linestyle=":", lw=1.5,
           label=f"P1 (P=0): {score_lo:.3f}")
ax.axvline(score_hi, color="gray", linestyle="-.", lw=1.5,
           label=f"P99 (P=1): {score_hi:.3f}")
ax.axhline(0.8, color="green", linestyle=":", alpha=0.5, label="FORWARD zone (P 0.8)")
ax.axhline(0.5, color="orange", linestyle=":", alpha=0.5, label="DROP zone (P 0.5)")
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
data_boxes  = [probs_train, probs_test_norm, probs_a2, probs_a5]
labels_box  = [f"Train\nN={len(probs_train)}", f"Test\nN={len(probs_test_norm)}",
               f"A2\nN={len(probs_a2)}", f"A5\nN={len(probs_a5)}"]
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
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_PATH}")
