
"""
Standalone regeneration of fig5_ensemble_evaluation.png
Uses saved model artifacts no retraining required.
Run from the project root: python evaluation/regen_ensemble_figure.py
"""

import json, os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.optimize import minimize_scalar
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             f1_score, precision_recall_curve,
                             average_precision_score)
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow import keras

MODELS_DIR      = "bridgeguard_models"
FEATURES_DIR    = "bridgeguard_features"
OUT_PATH        = "figures/fig5_ensemble_evaluation.png"
SEQ_LEN         = 10
MAX_GAP_MINUTES = 120
CALIB_START     = 0.60
CALIB_END       = 0.80
THRESHOLD_DROP  = 0.50
RANDOM_STATE    = 42
np.random.seed(RANDOM_STATE)

import tensorflow as tf
tf.random.set_seed(RANDOM_STATE)

with open(f"{FEATURES_DIR}/selected_features.json") as f:
    SELECTED = json.load(f)["selected_features"]

with open(f"{MODELS_DIR}/iforest_optimized_calibration.json") as f:
    cal = json.load(f)
PROB_LO = cal["prob_score_lo"]
PROB_HI = cal["prob_score_hi"]

iforest = pickle.load(open(f"{MODELS_DIR}/isolation_forest_optimized.pkl", "rb"))
scaler_iforest = pickle.load(open(f"{MODELS_DIR}/feature_scaler_selected.pkl", "rb"))

scaler_lstm_path = f"{MODELS_DIR}/feature_scaler_lstm.pkl"
scaler_lstm = (pickle.load(open(scaler_lstm_path, "rb"))
               if os.path.exists(scaler_lstm_path) else scaler_iforest)

lstm = keras.models.load_model(f"{MODELS_DIR}/lstm_model_selected.keras")
print("LSTM model loaded.")

platt_if   = pickle.load(open(f"{MODELS_DIR}/platt_iforest.pkl", "rb"))
platt_lstm = pickle.load(open(f"{MODELS_DIR}/platt_lstm.pkl",    "rb"))

with open(f"{MODELS_DIR}/lstm_temperature.json") as f:
    T_opt = json.load(f).get("T_star", 1.0)

with open(f"{MODELS_DIR}/final_paper_metrics.json") as f:
    saved = json.load(f)

best_delta       = saved["calibration"]["gating_delta"]
best_alpha       = saved["calibration"]["gating_alpha"]
threshold_youden = saved["calibration"]["threshold_youden"]
adv_results      = saved["adversarial"]["results"]
baseline_auc     = saved["test_zone_c"]["ensemble_drop_thresh"]["auc"]

labeled = pd.read_csv(f"{FEATURES_DIR}/features_selected_labeled.csv")
TIMESTAMP_COL = next(
    (c for c in ["window_start", "hour_window", "timestamp"] if c in labeled.columns),
    None
)
if TIMESTAMP_COL:
    labeled = labeled.sort_values(TIMESTAMP_COL).reset_index(drop=True)

def build_stratified_zones(df, ts_col):
    zone_a_parts, zone_b_parts, zone_c_parts = [], [], []
    for cls in df["attack_type"].unique():
        cls_df = df[df["attack_type"] == cls].copy()
        if ts_col:
            cls_df = cls_df.sort_values(ts_col).reset_index(drop=True)
        n   = len(cls_df)
        n_a = int(n * CALIB_START)
        n_b = int(n * CALIB_END)
        zone_a_parts.append(cls_df.iloc[:n_a])
        zone_b_parts.append(cls_df.iloc[n_a:n_b])
        zone_c_parts.append(cls_df.iloc[n_b:])
    def merge_sort(parts):
        m = pd.concat(parts, ignore_index=True)
        return m.sort_values(ts_col).reset_index(drop=True) if ts_col else m
    return merge_sort(zone_a_parts), merge_sort(zone_b_parts), merge_sort(zone_c_parts)

_, zone_b, zone_c = build_stratified_zones(labeled, TIMESTAMP_COL)
print(f"Zone B: {len(zone_b)} windows   Zone C: {len(zone_c)} windows")

def iforest_prob(df_zone):
    X = scaler_iforest.transform(df_zone[SELECTED].values.astype(np.float32))
    s = iforest.score_samples(X)
    return np.clip((s - PROB_LO) / (PROB_HI - PROB_LO + 1e-12), 0.0, 1.0)

def lstm_prob(df_zone):
    X  = scaler_lstm.transform(df_zone[SELECTED].values.astype(np.float32))
    ts = df_zone[TIMESTAMP_COL] if TIMESTAMP_COL else None
    seqs, ends = [], []
    for i in range(len(X) - SEQ_LEN + 1):
        if ts is not None:
            sl = ts.iloc[i:i+SEQ_LEN]
            try:
                gaps = pd.to_datetime(sl).diff().dropna().dt.total_seconds() / 60.0
                if gaps.max() > MAX_GAP_MINUTES:
                    continue
            except Exception:
                pass
        seqs.append(X[i:i+SEQ_LEN])
        ends.append(i + SEQ_LEN - 1)
    if not seqs:
        return np.array([]), np.array([], dtype=int)
    seqs = np.array(seqs, dtype=np.float32)
    probs = lstm.predict(seqs, batch_size=32, verbose=0).flatten()
    return probs, np.array(ends, dtype=int)

def apply_platt(clf, probs):
    cal = clf.predict_proba(probs.reshape(-1, 1))

    if np.corrcoef(probs, cal[:, 1])[0, 1] < 0:
        return cal[:, 0]
    return cal[:, 1]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def logit(p):
    return np.log(np.clip(p, 1e-7, 1-1e-7) / np.clip(1-p, 1e-7, 1))

def apply_temperature(probs, T):
    return sigmoid(logit(probs) / T)

def ece_score(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for i in range(n_bins):
        m = (probs >= bins[i]) & (probs < bins[i+1])
        if m.sum():
            ece += m.sum() * abs(labels[m].mean() - probs[m].mean())
    return ece / max(len(probs), 1)

def confidence_gated_ensemble(p_if_cal, p_lstm_cal, delta, alpha):
    p_ens = np.copy(p_lstm_cal)
    disagree = np.abs(p_lstm_cal - p_if_cal) >= delta
    c2 = disagree & (p_lstm_cal < 0.5) & (p_if_cal >= 0.5)
    p_ens[c2] = (1.0 - alpha) * p_lstm_cal[c2] + alpha * p_if_cal[c2]
    c4 = disagree & (p_if_cal < 0.5) & (p_lstm_cal < 0.5)
    p_ens[c4] = 0.6 * p_lstm_cal[c4] + 0.4 * p_if_cal[c4]
    return p_ens

print("Running IForest inference...")
if_b_raw = iforest_prob(zone_b)
if_c_raw = iforest_prob(zone_c)

print("Running LSTM inference...")
lm_b_raw, lm_b_ends = lstm_prob(zone_b)
lm_c_raw, lm_c_ends = lstm_prob(zone_c)

if_b_aln = if_b_raw[lm_b_ends];  y_b = zone_b["label"].values.astype(float)[lm_b_ends]
if_c_aln = if_c_raw[lm_c_ends];  y_c = zone_c["label"].values.astype(float)[lm_c_ends]
print(f"Zone B aligned: {len(y_b)}  (N={(y_b==1).sum()}, A={(y_b==0).sum()})")
print(f"Zone C aligned: {len(y_c)}  (N={(y_c==1).sum()}, A={(y_c==0).sum()})")

if_b_platt = apply_platt(platt_if,   if_b_aln)
lm_b_platt = apply_platt(platt_lstm, lm_b_raw)
if_c_platt = apply_platt(platt_if,   if_c_aln)
lm_c_platt = apply_platt(platt_lstm, lm_c_raw)

lm_b_cal = apply_temperature(lm_b_platt, T_opt)
lm_c_cal = apply_temperature(lm_c_platt, T_opt)

ece_if_b_raw = ece_score(if_b_aln,   y_b)
ece_if_b_cal = ece_score(if_b_platt, y_b)
ece_lm_b_raw = ece_score(lm_b_raw,   y_b)
ece_lm_b_cal = ece_score(lm_b_cal,   y_b)

gate_grid = {}
for delta in np.arange(0.05, 0.70, 0.05):
    for alpha in [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00]:
        p_b = confidence_gated_ensemble(if_b_platt, lm_b_cal, delta, alpha)
        fa  = float(((p_b < 0.5) & (y_b == 1)).sum()) / max((y_b == 1).sum(), 1)
        f1  = f1_score((y_b==0).astype(int), (p_b<0.5).astype(int), zero_division=0)
        try:
            auc = roc_auc_score(y_b, p_b)
        except Exception:
            auc = 0.0
        gate_grid[(round(delta, 2), round(alpha, 2))] = {"f1": f1, "fpr_norm": fa, "auc": auc}

en_c = confidence_gated_ensemble(if_c_platt, lm_c_cal, best_delta, best_alpha)
best_f1_gate = gate_grid.get((round(best_delta, 2), round(best_alpha, 2)), {}).get("f1", 0)
best_auc_cal = gate_grid.get((round(best_delta, 2), round(best_alpha, 2)), {}).get("auc", 0)

auc_c = roc_auc_score(y_c, en_c)
print(f"Ensemble AUC Zone C: {auc_c:.4f}  (paper: {baseline_auc:.4f})")

auc_if_paper   = saved["test_zone_c"]["iforest_only"]["auc"]
auc_lstm_paper = saved["test_zone_c"]["lstm_only"]["auc"]
auc_ens_paper  = baseline_auc

mask_n_c = y_c == 1
mask_a_c = y_c == 0
y_attack_c   = (y_c == 0).astype(float)
p_attack_if  = 1.0 - if_c_platt
p_attack_lstm= 1.0 - lm_c_cal
p_attack_ens = 1.0 - en_c

auprc_if   = average_precision_score(y_attack_c, p_attack_if)
auprc_lstm = average_precision_score(y_attack_c, p_attack_lstm)
auprc_ens  = average_precision_score(y_attack_c, p_attack_ens)

bins_bc = np.linspace(0, 1, 51)
def bhatt_coeff(scores, mask_n, mask_a):
    h_n, _ = np.histogram(scores[mask_n], bins=bins_bc, density=True)
    h_a, _ = np.histogram(scores[mask_a], bins=bins_bc, density=True)
    h_n /= (h_n.sum() + 1e-12); h_a /= (h_a.sum() + 1e-12)
    return np.sqrt(h_n * h_a), h_n, h_a

ov_if,   _, _ = bhatt_coeff(if_c_platt, mask_n_c, mask_a_c)
ov_lstm, _, _ = bhatt_coeff(lm_c_cal,   mask_n_c, mask_a_c)
ov_ens,  _, _ = bhatt_coeff(en_c,       mask_n_c, mask_a_c)
bc_if   = ov_if.sum()
bc_lstm = ov_lstm.sum()
bc_ens  = ov_ens.sum()
bin_centers = (bins_bc[:-1] + bins_bc[1:]) / 2

matplotlib.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "savefig.facecolor":"white",
})

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.patch.set_facecolor("white")
fig.suptitle("BridgeGuard Ensemble Evaluation (Platt + Confidence-Gated)",
             fontsize=13, fontweight="bold")
for ax in axes.flat:
    ax.set_facecolor("white")

ax = axes[0, 0]
for probs, label, color in [
    (if_c_platt, f"IForest·Platt (AUC={auc_if_paper:.4f})", "#2E86AB"),
    (lm_c_cal,   f"LSTM          (AUC={auc_lstm_paper:.4f})", "#3BB273"),
    (en_c,       f"Ensemble      (AUC={auc_ens_paper:.4f})", "#C73E1D"),
]:
    fpr_r, tpr_r, _ = roc_curve(y_c, probs)
    ax.plot(fpr_r, tpr_r, lw=2, label=label)
ax.plot([0,1],[0,1], "k--", alpha=0.3)
ax.set_title("ROC Curves (Zone C test)")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for probs, label, color in [
    (if_b_aln,   "IForest raw (Zone B)",           "#F4A261"),
    (if_b_platt, "IForest Platt (Zone B)",          "#E76F51"),
    (lm_b_raw,   "LSTM raw (Zone B)",               "#2E86AB"),
    (lm_b_cal,   f"LSTM Platt+T*={T_opt:.2f} (B)", "#C73E1D"),
]:
    bins_e = np.linspace(0, 1, 11)
    bm, bf = [], []
    for j in range(10):
        m = (probs >= bins_e[j]) & (probs < bins_e[j+1])
        if m.sum():
            bm.append(probs[m].mean()); bf.append(y_b[m].mean())
    ax.plot(bm, bf, "o-", label=label, color=color)
ax.plot([0,1],[0,1], "k--", alpha=0.5, label="Perfect calibration")
ax.set_title(f"Reliability Diagram  IForest & LSTM\n"
             f"(ECE LSTM: {ece_lm_b_raw:.3f}→{ece_lm_b_cal:.3f}  "
             f"IF: {ece_if_b_raw:.3f}→{ece_if_b_cal:.3f})")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

ax = axes[0, 2]
deltas_plot = sorted(set(d for d, a in gate_grid.keys()))
f1_vals  = [gate_grid.get((round(d,2), round(best_alpha,2)), {}).get("f1",  0) for d in deltas_plot]
auc_vals = [gate_grid.get((round(d,2), round(best_alpha,2)), {}).get("auc", 0) for d in deltas_plot]
ax2 = ax.twinx()
ax.plot(deltas_plot, f1_vals,  "o-",  color="#C73E1D", lw=2, label="F1 attack")
ax2.plot(deltas_plot, auc_vals, "s--", color="#2E86AB", lw=2, label="AUC")
ax.axvline(best_delta, color="black", linestyle=":", lw=1.5,
           label=f"δ*={best_delta:.2f}  α*={best_alpha:.2f}")
ax.set_xlabel(" (agreement gap)"); ax.set_ylabel("F1 (attack)", color="#C73E1D")
ax2.set_ylabel("AUC", color="#2E86AB")
ax.set_title(f"Confidence-Gating Grid  α={best_alpha:.2f}\n"
             f"Best: δ={best_delta:.2f}  F1={best_f1_gate:.3f}")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
cm_plot = confusion_matrix(y_c, (en_c >= threshold_youden).astype(int), labels=[0,1])
sns.heatmap(cm_plot, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Attack","Normal"], yticklabels=["Attack","Normal"])
ax.set_title(f"Confusion Matrix (Ensemble, τ={threshold_youden:.3f} Youden)\n"
             f"[positive=Attack | P(normal)<τ → Attack]")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_facecolor("white")

ax = axes[1, 1]
ax.hist(en_c[y_c==1], bins=25, alpha=0.7, color="#2E86AB",
        label=f"Normal (n={(y_c==1).sum()})", edgecolor="black", lw=0.5)
ax.hist(en_c[y_c==0], bins=25, alpha=0.7, color="#C73E1D",
        label=f"Attack (n={(y_c==0).sum()})", edgecolor="black", lw=0.5)
ax.axvline(THRESHOLD_DROP,   color="orange", linestyle="--", lw=2,
           label=f"DROP τ={THRESHOLD_DROP}")
ax.axvline(threshold_youden, color="green",  linestyle=":",  lw=2,
           label=f"Youden τ={threshold_youden:.3f}")
ax.set_title("P(Normal) Distribution Zone C")
ax.set_xlabel("P(Normal)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

ax = axes[1, 2]
adv_labels = list(adv_results.keys())
adv_aucs   = [v["auc"] for v in adv_results.values()]
adv_colors = ["green" if v["robust"] else "red" for v in adv_results.values()]
ax.bar(adv_labels, adv_aucs, color=adv_colors, alpha=0.8, edgecolor="black")
ax.axhline(baseline_auc, color="navy",   linestyle="--", lw=2,
           label=f"Baseline AUC={baseline_auc:.3f}")
ax.axhline(0.85, color="orange", linestyle=":", lw=1.5,
           label="Min robust threshold (0.85)")
for j, v in enumerate(adv_aucs):
    ax.text(j, v+0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
ax.set_title("AUC vs Feature Noise (Adversarial Robustness Zone C)")
ax.set_ylabel("AUC"); ax.set_ylim([0, 1.1])
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

ax = axes[2, 0]
for p_atk, label, color in [
    (p_attack_if,   f"IForest·Platt (AP={auprc_if:.3f})",  "#2E86AB"),
    (p_attack_lstm, f"LSTM (AP={auprc_lstm:.3f})",          "#3BB273"),
    (p_attack_ens,  f"Ensemble (AP={auprc_ens:.3f})",       "#C73E1D"),
]:
    prec_r, rec_r, _ = precision_recall_curve(y_attack_c, p_atk)
    ax.plot(rec_r, prec_r, lw=2, label=label)
ax.axhline(y_attack_c.mean(), color="gray", linestyle="--", alpha=0.5, lw=1,
           label=f"No-skill ({y_attack_c.mean():.2f})")
ax.set_xlabel("Recall (Attack)"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve (AUPRC)\n[positive=attack]")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[2, 1]
ax.plot(bin_centers, ov_if,   lw=1.5, alpha=0.85, label=f"IForest·Platt  BC={bc_if:.3f}",  color="#2E86AB")
ax.plot(bin_centers, ov_lstm, lw=1.5, alpha=0.85, label=f"LSTM           BC={bc_lstm:.3f}", color="#3BB273")
ax.plot(bin_centers, ov_ens,  lw=1.5, alpha=0.85, label=f"Ensemble       BC={bc_ens:.3f}",  color="#C73E1D")
ax.set_xlabel("P(Normal)"); ax.set_ylabel(" (p_norm p_attack) [overlap]")
ax.set_title("Bhattacharyya Overlap (Zone C)\n[lower = better separation]")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

ax = axes[2, 2]
ax.axis("off")
m = saved["test_zone_c"]
ed = m["ensemble_drop_thresh"]
summary = (
    f"Zone C: {m['n_windows']} windows  "
    f"({m['n_normal']}N / {m['n_attack']}A)\n"
    f"Split: stratified chronological last 20%\n\n"
    f"Ensemble (DROP τ={THRESHOLD_DROP}):\n"
    f"  AUC    = {ed['auc']:.4f}\n"
    f"  TPR    = {ed['tpr_attack']*100:.1f}%\n"
    f"  FPR    = {ed['fpr_normal']*100:.2f}%\n\n"
    f"Confidence-Gated params:\n"
    f"  δ* = {best_delta:.2f}   α* = {best_alpha:.2f}\n"
    f"  Youden τ = {threshold_youden:.4f}\n\n"
    f"Calibration (Zone B):\n"
    f"  IForest Platt ECE: {ece_if_b_raw:.3f}→{ece_if_b_cal:.3f}\n"
    f"  LSTM Platt ECE:    {ece_lm_b_raw:.3f}→{ece_lm_b_cal:.3f}\n"
    f"  T*={T_opt:.4f}"
)
ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f0f4f8", alpha=0.8))
ax.set_title("Evaluation Summary (Zone C)")

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUT_PATH}")

import shutil
latex_path = "figures-latex/fig5_ensemble_evaluation.png"
os.makedirs(os.path.dirname(latex_path), exist_ok=True)
shutil.copy(OUT_PATH, latex_path)
print(f"Copied: {latex_path}")
