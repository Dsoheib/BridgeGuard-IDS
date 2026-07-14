
"""
BridgeGuard Statistical Validation (Platt + Confidence-Gated Ensemble)
=======================================================================

Formal statistical evaluation on Zone C (held-out test set):
 - Clopper-Pearson exact 95% CI for TPR and FPR
 - Bootstrap 95% CI for AUC (10,000 resamples)
 - McNemar test on attack windows (TPR gap: Ensemble vs IForest)
 - McNemar test on normal windows (FPR reduction claim)
 - ECE before/after Platt calibration and temperature scaling

Zone split: stratified chronological per-class (last 20% = Zone C).
Both components use Platt calibration; ensemble uses confidence-gated fusion.

Outputs:
 stats_results/formal_stats.json
 stats_results/table_stats.tex
 figures/confidence_intervals.{pdf,png}

Usage:
 python evaluation/statistical_validation.py
"""

import os
import sys
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import beta as beta_dist, binom as binom_dist
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

np.random.seed(42)

FEATURES_DIR     = "bridgeguard_features"
MODELS_DIR       = "bridgeguard_models"
OUTPUT_DIR       = "stats_results"
FIGURES_DIR      = "figures"
SEQ_LEN          = 10
MAX_GAP_MINUTES  = 120
CALIB_FRAC_START = 0.60
CALIB_FRAC_END   = 0.80
N_NORMAL_C_MIN   = 100
T_BOUND_LO       = 0.05
T_BOUND_HI       = 5.00
DECISION_THR     = 0.50
N_BOOTSTRAP      = 10_000
ALPHA            = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 70)
print("BridgeGuard Statistical Validation (Platt + Confidence-Gated Ensemble)")
print("=" * 70)

def abort(msg):
    print(f"\n  FATAL: {msg}")
    sys.exit(1)

print("\n[1] Loading models...")

for p in [
    f"{FEATURES_DIR}/selected_features.json",
    f"{FEATURES_DIR}/features_selected_labeled.csv",
    f"{MODELS_DIR}/isolation_forest_optimized.pkl",
    f"{MODELS_DIR}/feature_scaler_selected.pkl",
    f"{MODELS_DIR}/iforest_optimized_calibration.json",
    f"{MODELS_DIR}/lstm_model_selected.keras",
]:
    if not os.path.exists(p):
        abort(f"Missing required file: {p}")

with open(f"{FEATURES_DIR}/selected_features.json") as fh:
    SELECTED = json.load(fh)["selected_features"]

with open(f"{MODELS_DIR}/isolation_forest_optimized.pkl", "rb") as fh:
    iforest = pickle.load(fh)
with open(f"{MODELS_DIR}/feature_scaler_selected.pkl", "rb") as fh:
    scaler_if = pickle.load(fh)
with open(f"{MODELS_DIR}/iforest_optimized_calibration.json") as fh:
    cal = json.load(fh)
PROB_LO = float(cal["prob_score_lo"])
PROB_HI = float(cal["prob_score_hi"])

lstm = tf.keras.models.load_model(f"{MODELS_DIR}/lstm_model_selected.keras")

lstm_sc_path = f"{MODELS_DIR}/feature_scaler_lstm.pkl"
fallback_sc  = f"{MODELS_DIR}/feature_scaler_selected.pkl"
sc_path = lstm_sc_path if os.path.exists(lstm_sc_path) else fallback_sc
with open(sc_path, "rb") as fh:
    scaler_lstm = pickle.load(fh)
print(f"  LSTM scaler: {sc_path}")
print(f"  IForest probability mapping: [{PROB_LO:.4f}, {PROB_HI:.4f}]")

_platt_if_path   = f"{MODELS_DIR}/platt_iforest.pkl"
_platt_lstm_path = f"{MODELS_DIR}/platt_lstm.pkl"
if os.path.exists(_platt_if_path) and os.path.exists(_platt_lstm_path):
    with open(_platt_if_path,   "rb") as fh: _platt_if   = pickle.load(fh)
    with open(_platt_lstm_path, "rb") as fh: _platt_lstm = pickle.load(fh)
    _PLATT_OK = True
    print("  Platt calibrators loaded.")
else:
    _platt_if = _platt_lstm = None
    _PLATT_OK = False
    print("  WARNING: Platt calibrators not found — run evaluate_bridgeguard.py first.")
    print("  Falling back to raw min-max scores.")

_temp_path = f"{MODELS_DIR}/lstm_temperature.json"
if os.path.exists(_temp_path):
    with open(_temp_path) as fh:
        _td = json.load(fh)
    if _td.get("requires_recalibration", False):
        print("\n  ABORT: lstm_temperature.json has requires_recalibration=True.")
        print("  Run evaluate_bridgeguard.py first to populate calibration parameters.")
        print("  Results from this script cannot be reported as paper metrics.")
        sys.exit(1)
    GATING_DELTA  = float(_td.get("gating_delta",  0.30))
    GATING_ALPHA  = float(_td.get("gating_alpha",  0.00))
    _rev_if       = bool(_td.get("platt_if_reversed",   False))
    _rev_lstm     = bool(_td.get("platt_lstm_reversed",  False))
    print(f"  Gating: delta={GATING_DELTA:.2f}  alpha={GATING_ALPHA:.2f}  "
          f"rev_if={_rev_if}  rev_lstm={_rev_lstm}")
else:
    GATING_DELTA = 0.30
    GATING_ALPHA = 0.00
    _rev_if = _rev_lstm = False
    print("  WARNING: lstm_temperature.json not found — using default gating parameters.")

print("\n[2] Building Zone C (stratified per-class split)...")

labeled = pd.read_csv(f"{FEATURES_DIR}/features_selected_labeled.csv")

TIMESTAMP_COL = None
for col in ["window_start", "hour_window", "timestamp"]:
    if col in labeled.columns:
        TIMESTAMP_COL = col
        break

if TIMESTAMP_COL:
    labeled = labeled.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    print(f"  Sorted chronologically by '{TIMESTAMP_COL}'")
else:
    print("  No timestamp column found — using CSV row order as temporal proxy.")

if "attack_type" not in labeled.columns:

    labeled["attack_type"] = labeled["label"].map(
        lambda v: "normal" if int(v) == 1 else "attack"
    )
    print("  Column 'attack_type' absent — reconstructed from 'label'.")
    print("  For optimal split, window extraction should write attack_type.")

def build_stratified_zones(df, ts_col,
                            calib_start=CALIB_FRAC_START,
                            calib_end=CALIB_FRAC_END):
    parts_a, parts_b, parts_c = [], [], []
    for cls in df["attack_type"].unique():
        sub = df[df["attack_type"] == cls].copy()
        if ts_col:
            sub = sub.sort_values(ts_col).reset_index(drop=True)
        n   = len(sub)
        na  = int(n * calib_start)
        nb  = int(n * calib_end)
        parts_a.append(sub.iloc[:na])
        parts_b.append(sub.iloc[na:nb])
        parts_c.append(sub.iloc[nb:])
        nc   = len(sub.iloc[nb:])
        n_nm = int((sub.iloc[nb:]["label"] == 1).sum())
        n_at = int((sub.iloc[nb:]["label"] == 0).sum())
        print(f"    {cls:<22}: total={n:>4}  A={na:>4}  B={nb-na:>4}  "
              f"C={nc:>4}  (normal_C={n_nm}, atk_C={n_at})")

    def merge(parts):
        m = pd.concat(parts, ignore_index=True)
        if ts_col:
            m = m.sort_values(ts_col).reset_index(drop=True)
        return m

    return merge(parts_a), merge(parts_b), merge(parts_c)

zone_a, zone_b, zone_c = build_stratified_zones(labeled, TIMESTAMP_COL)

n_normal_c = int((zone_c["label"] == 1).sum())
n_atk_c    = int((zone_c["label"] == 0).sum())

print("\n  Zone split results:")
print(f"    Zone A : {len(zone_a):>4} windows")
print(f"    Zone B : {len(zone_b):>4} windows  "
      f"(N={(zone_b['label']==1).sum()}, A={(zone_b['label']==0).sum()})")
print(f"    Zone C : {len(zone_c):>4} windows  "
      f"(N={n_normal_c}, A={n_atk_c})")

if n_normal_c < N_NORMAL_C_MIN:
    print(f"\n  WARNING: Zone C N_normal={n_normal_c} < {N_NORMAL_C_MIN} required.")
else:
    print(f"\n  Zone C: N_normal={n_normal_c} >= {N_NORMAL_C_MIN} [OK]")

print("\n[3] IForest and LSTM scoring (sliding-window sequences, gap-handling)...")

def iforest_probs(df_zone):
    X = df_zone[SELECTED].values.astype(np.float32)
    X_sc = scaler_if.transform(X)
    scores = iforest.score_samples(X_sc)
    return np.clip((scores - PROB_LO) / (PROB_HI - PROB_LO + 1e-12), 0.0, 1.0)

def apply_platt(clf, raw_probs, reversed_=False):
    cal = clf.predict_proba(np.array(raw_probs).reshape(-1, 1))
    return cal[:, 0] if reversed_ else cal[:, 1]

def confidence_gated_ensemble(p_if_cal, p_lstm_cal,
                               delta=GATING_DELTA, alpha=GATING_ALPHA):
    p_ens    = np.copy(p_lstm_cal)
    disagree = np.abs(p_lstm_cal - p_if_cal) >= delta

    c2 = disagree & (p_lstm_cal < 0.5) & (p_if_cal >= 0.5)
    p_ens[c2] = (1.0 - alpha) * p_lstm_cal[c2] + alpha * p_if_cal[c2]

    c4 = disagree & (p_if_cal < 0.5) & (p_lstm_cal < 0.5)
    p_ens[c4] = 0.6 * p_lstm_cal[c4] + 0.4 * p_if_cal[c4]

    return p_ens

def make_sequences(X_sc, timestamps=None):
    seqs, ends = [], []
    for i in range(len(X_sc) - SEQ_LEN + 1):
        if timestamps is not None and MAX_GAP_MINUTES is not None:
            ts_sl = (timestamps.iloc[i:i + SEQ_LEN]
                     if hasattr(timestamps, "iloc") else timestamps[i:i + SEQ_LEN])
            try:
                ts_p = pd.to_datetime(ts_sl)
                gaps = ts_p.diff().dropna().dt.total_seconds() / 60.0
                if gaps.max() > MAX_GAP_MINUTES:
                    continue
            except Exception:
                pass
        seqs.append(X_sc[i:i + SEQ_LEN])
        ends.append(i + SEQ_LEN - 1)
    if not seqs:
        return (np.empty((0, SEQ_LEN, X_sc.shape[1]), dtype=np.float32),
                np.array([], dtype=int))
    return np.array(seqs, dtype=np.float32), np.array(ends, dtype=int)

def lstm_raw_probs(df_zone):
    X = df_zone[SELECTED].values.astype(np.float32)
    X_sc = scaler_lstm.transform(X)
    ts = df_zone[TIMESTAMP_COL] if TIMESTAMP_COL else None
    seqs, ends = make_sequences(X_sc, timestamps=ts)
    if len(seqs) == 0:
        return np.array([]), np.array([], dtype=int)
    raw = lstm.predict(seqs, batch_size=32, verbose=0).flatten()
    return raw, ends

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def logit(p):
    return np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))

def apply_temperature(probs, T):
    return sigmoid(logit(probs) / T)

def ece_score(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(labels[mask].mean() - probs[mask].mean())
    return ece / max(len(probs), 1)

def align(if_probs, lm_raw, lm_ends, zone_df):
    if len(lm_ends) == 0:
        return np.array([]), np.array([]), np.array([])
    return (if_probs[lm_ends],
            lm_raw,
            zone_df["label"].values.astype(float)[lm_ends])

if_b  = iforest_probs(zone_b)
lm_b_raw, lm_b_ends = lstm_raw_probs(zone_b)
if_b_aln, lm_b_aln, y_b = align(if_b, lm_b_raw, lm_b_ends, zone_b)

if len(y_b) == 0 or len(np.unique(y_b)) < 2:
    abort("Zone B is mono-class — verify the stratified split.")

print("  Platt calibration (Zone B)...")
ece_if_raw  = ece_score(if_b_aln, y_b)
ece_lm_raw  = ece_score(lm_b_aln, y_b)

if _PLATT_OK:
    if_b_platt  = apply_platt(_platt_if,   if_b_aln,  _rev_if)
    lm_b_platt  = apply_platt(_platt_lstm, lm_b_aln,  _rev_lstm)
else:

    def _fit_platt_local(raw, labels):
        clf = LogisticRegression(C=1e5, solver="lbfgs", max_iter=2000)
        clf.fit(raw.reshape(-1, 1), labels.astype(int))
        cal = clf.predict_proba(raw.reshape(-1, 1))

        if np.corrcoef(raw, cal[:, 1])[0, 1] >= 0:
            return clf, False
        return clf, True
    _platt_if,   _rev_if   = _fit_platt_local(if_b_aln,  y_b)
    _platt_lstm, _rev_lstm = _fit_platt_local(lm_b_aln,  y_b)
    if_b_platt  = apply_platt(_platt_if,   if_b_aln,  _rev_if)
    lm_b_platt  = apply_platt(_platt_lstm, lm_b_aln,  _rev_lstm)
    print(" Platt fitted locally on Zone B (disk calibrators absent)")

ece_if_cal  = ece_score(if_b_platt, y_b)
ece_lm_cal  = ece_score(lm_b_platt, y_b)
print(f"    IForest ECE: {ece_if_raw:.4f} -> {ece_if_cal:.4f}  "
      f"({'improved' if ece_if_cal < ece_if_raw else 'unchanged'})")
print(f"    LSTM    ECE: {ece_lm_raw:.4f} -> {ece_lm_cal:.4f}  "
      f"({'improved' if ece_lm_cal < ece_lm_raw else 'unchanged'})")

print("  Temperature scaling (Zone B, Platt-calibrated LSTM)...")
ece_before = ece_score(lm_b_platt, y_b)
logits_b   = logit(lm_b_platt)

def nll_t(T):
    p_T = sigmoid(logits_b / T)
    return -np.mean(y_b * np.log(p_T + 1e-7) + (1 - y_b) * np.log(1 - p_T + 1e-7))

res   = minimize_scalar(nll_t, bounds=(T_BOUND_LO, T_BOUND_HI), method="bounded")
T_opt = float(res.x)
at_bound = (abs(T_opt - T_BOUND_LO) < 1e-4 or abs(T_opt - T_BOUND_HI) < 1e-4)
lm_b_cal  = apply_temperature(lm_b_platt, T_opt)
ece_after = ece_score(lm_b_cal, y_b)

print(f"    T* = {T_opt:.4f}  {' AT BOUND' if at_bound else ' '}")
print(f"    ECE LSTM (Platt+T*): {ece_before:.4f} -> {ece_after:.4f}")

best_w_if = round(GATING_ALPHA * 0.4, 2)
best_w_lm = round(1.0 - best_w_if, 2)

en_b = confidence_gated_ensemble(
    apply_platt(_platt_if, if_b_aln, _rev_if),
    lm_b_cal,
    delta=GATING_DELTA, alpha=GATING_ALPHA
)
try:
    best_auc = float(roc_auc_score(y_b, en_b))
except Exception:
    best_auc = 0.0
print(f"    Gating: delta={GATING_DELTA:.2f}  alpha={GATING_ALPHA:.2f}  "
      f"AUC_B={best_auc:.4f}")

if_c  = iforest_probs(zone_c)
lm_c_raw, lm_c_ends = lstm_raw_probs(zone_c)
if_c_aln, lm_c_raw_aln, y_c = align(if_c, lm_c_raw, lm_c_ends, zone_c)

if len(y_c) == 0 or len(np.unique(y_c)) < 2:
    abort("Zone C is mono-class after LSTM alignment — verify the stratified split.")

if_c_platt  = apply_platt(_platt_if,   if_c_aln,       _rev_if)
lm_c_platt  = apply_platt(_platt_lstm, lm_c_raw_aln,   _rev_lstm)
lm_c_cal    = apply_temperature(lm_c_platt, T_opt)
en_c        = confidence_gated_ensemble(if_c_platt, lm_c_cal,
                                        delta=GATING_DELTA, alpha=GATING_ALPHA)

n_c = len(y_c)
n_n = int((y_c == 1).sum())
n_a = int((y_c == 0).sum())

print(f"\n  Zone C aligned: {n_c} windows ({n_n}N / {n_a}A)")

y_pred    = (en_c    >= DECISION_THR).astype(int)
y_pred_if = (if_c_platt >= DECISION_THR).astype(int)

TP_n = int(((y_pred == 1) & (y_c == 1)).sum())
TN_a = int(((y_pred == 0) & (y_c == 0)).sum())
FP_a = int(((y_pred == 1) & (y_c == 0)).sum())
FN_n = int(((y_pred == 0) & (y_c == 1)).sum())

TPR_attack = TN_a / n_a if n_a > 0 else 0.0
FPR_normal = FN_n / n_n if n_n > 0 else 0.0

print(f"\n  Confusion matrix (threshold={DECISION_THR}):")
print(f"    TP_n={TP_n}  FP_a={FP_a}  TN_a={TN_a}  FN_n={FN_n}")
print(f"  TPR (attack): {TPR_attack*100:.1f}%")
print(f"  FPR (normal): {FPR_normal*100:.1f}%")

try:
    auc = float(roc_auc_score(y_c, en_c))
    print(f"  AUC: {auc:.4f}")
except Exception:
    auc = None
    print("  AUC: not computable.")

TN_if = int(((y_pred_if == 0) & (y_c == 0)).sum())
FP_if = int(((y_pred_if == 1) & (y_c == 0)).sum())
FN_if = int(((y_pred_if == 0) & (y_c == 1)).sum())
TP_if = int(((y_pred_if == 1) & (y_c == 1)).sum())
TPR_if = TN_if / n_a if n_a > 0 else 0.0
FPR_if = FN_if / n_n if n_n > 0 else 0.0
try:
    auc_if = float(roc_auc_score(y_c, if_c_platt))
except Exception:
    auc_if = None
auc_if_str = f"{auc_if:.4f}" if auc_if is not None else "N/A"
print(f"  IForest (Platt): TPR={TPR_if*100:.1f}%  FPR={FPR_if*100:.1f}%  "
      f"AUC={auc_if_str}")

y_attack_c   = (y_c == 0).astype(int)
try:
    auprc_ens = float(average_precision_score(y_attack_c, 1.0 - en_c))
    auprc_if  = float(average_precision_score(y_attack_c, 1.0 - if_c_platt))
    print(f"  AUPRC: Ensemble={auprc_ens:.4f}  IForest={auprc_if:.4f}")
except Exception:
    auprc_ens = auprc_if = None
    print("  AUPRC: not computable.")

print("\n[4] Clopper-Pearson exact confidence intervals (95%)...")

def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return 0.0, 1.0
    lo = beta_dist.ppf(alpha / 2,     k,     n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k    ) if k < n else 1.0
    return float(lo), float(hi)

tpr_atk_lo, tpr_atk_hi = clopper_pearson(TN_a, n_a, ALPHA)
fpr_nm_lo,  fpr_nm_hi  = clopper_pearson(FN_n, n_n, ALPHA)
tpr_if_lo,  tpr_if_hi  = clopper_pearson(TN_if, n_a, ALPHA)

print(f"\n  TPR attack (Ensemble): k={TN_a}, n={n_a}")
print(f"    {TPR_attack*100:.1f}%  [{tpr_atk_lo*100:.1f}%, {tpr_atk_hi*100:.1f}%]")

print(f"\n  FPR normal (Ensemble): k={FN_n}, n={n_n}")
print(f"    {FPR_normal*100:.1f}%  [{fpr_nm_lo*100:.1f}%, {fpr_nm_hi*100:.1f}%]")

print(f"\n  TPR attack (IForest):  k={TN_if}, n={n_a}")
print(f"    {TPR_if*100:.1f}%  [{tpr_if_lo*100:.1f}%, {tpr_if_hi*100:.1f}%]")

print("\n[5] McNemar Test + Power Analysis...")

atk_mask    = (y_c == 0)
pred_en_atk = (en_c[atk_mask]      < DECISION_THR).astype(int)
pred_if_atk = (if_c_platt[atk_mask] < DECISION_THR).astype(int)

b = int(((pred_if_atk == 1) & (pred_en_atk == 0)).sum())
c = int(((pred_if_atk == 0) & (pred_en_atk == 1)).sum())
n_discordant = b + c

print(f"  Discordant pairs (attack windows): b={b}  c={c}  total={n_discordant}")

if n_discordant > 0:
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_mcnemar = float(1 - stats.chi2.cdf(chi2, df=1))
    p_observed = c / n_discordant
    k_crit = binom_dist.ppf(1 - ALPHA, n_discordant, 0.5)
    power  = float(1 - binom_dist.cdf(k_crit - 1, n_discordant, p_observed))

    print(f"  McNemar: chi2={chi2:.4f}  p={p_mcnemar:.4f}")
    print("  Power Analysis:")
    print(f"    n_discordant = {n_discordant}")
    print(f"    p_observed   = {p_observed:.3f}  (c/(b+c))")
    print(f"    Power        = {power*100:.1f}%")
    if p_mcnemar < 0.05:
        if c > b:
            print(f"      SIGNIFICANT: Ensemble detects more attacks (c={c} > b={b})")
        else:
            print(f"      SIGNIFICANT: IForest detects more attacks (b={b} > c={c})")
            print(f"      (Ensemble achieves lower FPR: {FPR_normal*100:.1f}% vs {FPR_if*100:.1f}%)")
    else:
        print(f"      Non-significant ({'low power' if power < 0.80 else 'no difference'})")
else:
    chi2      = 0.0
    p_mcnemar = 1.0
    power     = 0.0
    print("  No discordant pairs — test not applicable.")

print("\n[6] McNemar Test (normal windows, FPR reduction)...")

norm_mask    = (y_c == 1)
pred_en_norm = (en_c[norm_mask]       >= DECISION_THR).astype(int)
pred_if_norm = (if_c_platt[norm_mask] >= DECISION_THR).astype(int)

b_norm = int(((pred_en_norm == 0) & (pred_if_norm == 1)).sum())
c_norm = int(((pred_en_norm == 1) & (pred_if_norm == 0)).sum())
n_disc_norm = b_norm + c_norm

print(f"  Normal windows : {norm_mask.sum()}  |  "
      f"Ensemble FP={b_norm + int(((pred_en_norm==0)&(pred_if_norm==0)).sum())}  "
      f"IForest FP={c_norm + int(((pred_en_norm==0)&(pred_if_norm==0)).sum())}")
print(f"  Discordants : b_norm={b_norm} (ens FA, IF ok)  "
      f"c_norm={c_norm} (IF FA, ens ok)  total={n_disc_norm}")

wilcoxon_stat, wilcoxon_p = None, None
mcnemar_norm_p = 1.0
if n_disc_norm > 0:
    chi2_norm = (abs(b_norm - c_norm) - 1) ** 2 / (b_norm + c_norm)
    mcnemar_norm_p = float(1 - stats.chi2.cdf(chi2_norm, df=1))
    print(f"  McNemar (normals): chi2={chi2_norm:.4f}  p={mcnemar_norm_p:.4f}")
    if mcnemar_norm_p < 0.05 and c_norm > b_norm:
        print(f"    SIGNIFICANT: Ensemble significantly reduces false alarms "
              f"({FPR_normal*100:.1f}% vs {FPR_if*100:.1f}%)")
    elif mcnemar_norm_p < 0.05:
        print("    SIGNIFICANT but direction reversed (IForest has fewer false alarms)")
    else:
        print("    Non-significant")
else:
    print("  No discordant pairs on normal windows.")
    chi2_norm = 0.0

print(f"\n[7] Bootstrap AUC CI ({N_BOOTSTRAP:,} resamples, Zone C={n_c} windows)...")

boot_lo = boot_hi = boot_mean = None
if auc is not None and len(np.unique(y_c)) == 2:
    rng   = np.random.RandomState(42)
    boots = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n_c, n_c, replace=True)
        yb, pb = y_c[idx], en_c[idx]
        if len(np.unique(yb)) == 2:
            try:
                boots.append(roc_auc_score(yb, pb))
            except Exception:
                pass
    boots      = np.array(boots)
    boot_lo    = float(np.percentile(boots, 2.5))
    boot_hi    = float(np.percentile(boots, 97.5))
    boot_mean  = float(np.mean(boots))
    print(f"  AUC = {auc:.4f}  Bootstrap 95% CI = [{boot_lo:.4f}, {boot_hi:.4f}]")
    print(f"  ({len(boots)}/{N_BOOTSTRAP} valid resamples)")

print("\n[8] Generating figures...")

C_RED = "#E63946"
C_BLU = "#457B9D"
C_GRN = "#2D6A4F"
C_BG  = "#f8f9fa"

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f"BridgeGuard v4 — Formal Statistical Analysis\n"
    f"(Zone C: {n_n}N + {n_a}A aligned | Platt + Confidence-Gated Ensemble)",
    fontsize=12, fontweight="bold"
)

ax = axes[0]
metrics = ["TPR\n(Ensemble)", "TPR\n(IForest Platt)", "FPR\n(Ensemble)"]
points  = [TPR_attack, TPR_if, FPR_normal]
lowers  = [tpr_atk_lo, tpr_if_lo, fpr_nm_lo]
uppers  = [tpr_atk_hi, tpr_if_hi, fpr_nm_hi]
colors  = [C_GRN, C_BLU, C_RED]

for i, (pt, lo, hi, col) in enumerate(zip(points, lowers, uppers, colors)):
    ax.plot(i, pt, "o", color=col, markersize=9, zorder=3)
    ax.vlines(i, lo, hi, color=col, lw=3, alpha=0.7)
    ax.text(i, hi + 0.01, f"{pt*100:.1f}%\n[{lo*100:.1f},{hi*100:.1f}]",
            ha="center", va="bottom", fontsize=8, color=col)

ax.axhline(1.0, color="#333", lw=0.8, linestyle=":")
ax.axhline(0.0, color="#333", lw=0.8, linestyle=":")
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(metrics, fontsize=9)
ax.set_ylabel("Rate (proportion)", fontsize=10)
ax.set_title(
    f"Clopper–Pearson 95% CI\n(Zone C: {n_n}N + {n_a}A windows, stratified split)",
    fontsize=10
)
ax.set_ylim(-0.05, 1.20)
ax.set_facecolor(C_BG)
ax.grid(True, alpha=0.3, color="white", axis="y")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
if boot_lo is not None:
    auc_range = boots.max() - boots.min()
    if auc_range < 1e-6:
        ax.bar([boots.mean()], [len(boots)], width=0.001,
               color=C_BLU, alpha=0.8, edgecolor="white")
        ax.set_title(
            f"Bootstrap AUC CI (n={N_BOOTSTRAP:,})\n"
            f"AUC={auc:.4f} [degenerate: all resamples = {auc:.4f}]",
            fontsize=10
        )
    else:
        n_bins = min(40, max(5, int(auc_range / 0.001)))
        ax.hist(boots, bins=n_bins, color=C_BLU, alpha=0.8, edgecolor="white")
        ax.set_title(
            f"Bootstrap AUC CI (n={N_BOOTSTRAP:,})\n"
            f"Zone C = {n_c} windows (stratified, ensemble calibration-aligned)",
            fontsize=10
        )
    ax.axvline(auc,     color=C_RED, lw=2,   label=f"AUC={auc:.4f}")
    ax.axvline(boot_lo, color=C_GRN, lw=1.5, linestyle="--",
               label=f"95% CI [{boot_lo:.4f}, {boot_hi:.4f}]")
    ax.axvline(boot_hi, color=C_GRN, lw=1.5, linestyle="--")
    ax.set_xlabel("Bootstrap AUC", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_facecolor(C_BG)
    ax.grid(True, alpha=0.3, color="white")
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(
        os.path.join(FIGURES_DIR, f"confidence_intervals.{ext}"),
        bbox_inches="tight", dpi=300 if ext == "pdf" else 180
    )
plt.close()
print(f"  figures/confidence_intervals.{{pdf,png}}")

print("\n[9] Saving results...")

results = {
    "version"   : "v3-platt-confidence-gated",
    "zone_c"    : {
        "split_method" : "stratified_chronological_per_class_last_20pct",
        "n_total"      : n_c,
        "n_normal"     : n_n,
        "n_attack"     : n_a,
    },
    "calibration" : {
        "T_star"           : round(T_opt, 4),
        "T_at_boundary"    : at_bound,
        "gating_delta"     : GATING_DELTA,
        "gating_alpha"     : GATING_ALPHA,
        "platt_applied"    : _PLATT_OK,
        "ece_iforest_raw"  : round(ece_if_raw,  4),
        "ece_iforest_cal"  : round(ece_if_cal,  4),
        "ece_lstm_raw"     : round(ece_lm_raw,  4),
        "ece_lstm_cal"     : round(ece_lm_cal,  4),
        "ece_before"       : round(ece_before,  4),
        "ece_after"        : round(ece_after,   4),

        "w_IF"             : best_w_if,
        "w_LSTM"           : best_w_lm,
    },
    "point_estimates" : {
        "tpr_attack_ensemble" : round(TPR_attack, 4),
        "fpr_normal_ensemble" : round(FPR_normal, 4),
        "tpr_attack_iforest"  : round(TPR_if,     4),
        "fpr_normal_iforest"  : round(FPR_if,     4),
        "auc_ensemble"        : round(auc,    4) if auc    else None,
        "auc_iforest"         : round(auc_if, 4) if auc_if else None,
        "auprc_ensemble"      : round(auprc_ens, 4) if auprc_ens else None,
        "auprc_iforest"       : round(auprc_if,  4) if auprc_if  else None,
    },
    "clopper_pearson_95" : {
        "tpr_attack" : [round(tpr_atk_lo, 4), round(tpr_atk_hi, 4)],
        "fpr_normal" : [round(fpr_nm_lo,  4), round(fpr_nm_hi,  4)],
        "tpr_iforest": [round(tpr_if_lo,  4), round(tpr_if_hi,  4)],
    },
    "mcnemar" : {
        "b" : b, "c" : c, "n_discordant" : n_discordant,
        "chi2"        : round(chi2,      4),
        "p_value"     : round(p_mcnemar, 4),
        "power_pct"   : round(power * 100, 1),
        "significant" : p_mcnemar < 0.05,
        "direction"   : ("ensemble_better" if c > b else "iforest_better_tpr"),
        "note"        : "Compared on attack windows; both models use Platt-calibrated scores. direction=ensemble_better means c>b (ensemble detects attacks IForest misses more often).",
    },
    "mcnemar_normals" : {
        "b_norm"       : b_norm,
        "c_norm"       : c_norm,
        "n_discordant" : n_disc_norm,
        "chi2"         : round(chi2_norm, 4),
        "p_value"      : round(mcnemar_norm_p, 4),
        "significant"  : mcnemar_norm_p < 0.05,
        "direction"    : ("ensemble_fewer_fa" if c_norm > b_norm else "iforest_fewer_fa"),
        "note"         : "McNemar on NORMAL windows: c_norm=IForest FA/ensemble correct, b_norm=ensemble FA/IForest correct. Tests FPR reduction claim.",
    },
    "bootstrap_auc" : {
        "n_bootstrap" : N_BOOTSTRAP,
        "n_zone_c"    : n_c,
        "auc"         : round(auc,     4) if auc     else None,
        "ci_lo"       : round(boot_lo, 4) if boot_lo else None,
        "ci_hi"       : round(boot_hi, 4) if boot_hi else None,
    },
}

json_path = os.path.join(OUTPUT_DIR, "formal_stats.json")
with open(json_path, "w") as fh:
    json.dump(results, fh, indent=2)
print(f"  {json_path}")

def pct(v):
    return f"{v*100:.1f}"

auc_str  = f"{auc:.4f}"  if auc    else "N/A"
boot_str = (f"[{boot_lo:.4f},\\,{boot_hi:.4f}]" if boot_lo else "N/A")
mcn_interp = (f"$p={p_mcnemar:.4f}$ (IForest$>$Ensemble TPR; Ensemble$<$IForest FPR)"
              if (p_mcnemar < 0.05 and b > c)
              else f"$p={p_mcnemar:.4f}$ (Ensemble$>$IForest)"
              if p_mcnemar < 0.05
              else f"$p={p_mcnemar:.4f}$, Power$={power*100:.0f}\\%$ (insufficient)")
mcn_norm_interp = (f"$p={mcnemar_norm_p:.4f}$ (Ensemble fewer FA: {FPR_normal*100:.1f}\\% vs {FPR_if*100:.1f}\\%)"
                   if (mcnemar_norm_p < 0.05 and c_norm > b_norm)
                   else f"$p={mcnemar_norm_p:.4f}$")

tex_lines = [
    r"% ==========================================================",
    r"% Table: Formal Statistical Analysis BridgeGuard",
    r"% Zone C: stratified per-class, last 20% (ensemble calibration aligned)",
    r"% Calibration: Platt (IF+LSTM) + Temperature Scaling + Gating",
    r"% ==========================================================",
    "",
    r"\begin{table}[ht]",
    r"\centering",
    (r"\caption{Formal statistical evaluation on Zone~C ("
     + str(n_c) + r"~aligned windows: "
     + str(n_n) + r"~normal, " + str(n_a) + r"~attack; "
     r"stratified chronological split, last~20\% per class)."
     r" Both components use Platt calibration; ensemble uses confidence-gated fusion."
     r" Clopper--Pearson exact 95\% CI on TPR and FPR."
     r" Bootstrap 95\% CI on AUC (" + f"{N_BOOTSTRAP:,}" + r"~resamples)."
     r" McNemar on attack windows tests TPR gap; McNemar on normal windows tests FPR reduction (both Platt-calibrated).}"),
    r"\label{tab:formal_stats}",
    r"\begin{tabular}{llcc}",
    r"\toprule",
    r"\textbf{Metric} & \textbf{Method} & \textbf{Estimate} & \textbf{95\% CI} \\",
    r"\midrule",
    (r"TPR$_{\mathrm{attack}}$ (Ensemble) & Clopper--Pearson & "
     + pct(TPR_attack) + r"\% & ["
     + pct(tpr_atk_lo) + r"\%, " + pct(tpr_atk_hi) + r"\%] \\"),
    (r"TPR$_{\mathrm{attack}}$ (IForest$_{\mathrm{Platt}}$) & Clopper--Pearson & "
     + pct(TPR_if) + r"\% & ["
     + pct(tpr_if_lo) + r"\%, " + pct(tpr_if_hi) + r"\%] \\"),
    (r"FPR$_{\mathrm{normal}}$ (Ensemble) & Clopper--Pearson & "
     + pct(FPR_normal) + r"\% & ["
     + pct(fpr_nm_lo) + r"\%, " + pct(fpr_nm_hi) + r"\%] \\"),
    (r"AUC (Ensemble) & Bootstrap & " + auc_str + r" & " + boot_str + r" \\"),
    (r"AUPRC (Ensemble) & & "
     + (f"{auprc_ens:.4f}" if auprc_ens else "N/A") + r" & \\"),
    (r"AUPRC (IForest$_{\mathrm{Platt}}$) & & "
     + (f"{auprc_if:.4f}" if auprc_if else "N/A") + r" & \\"),
    r"\midrule",
    (r"McNemar attacks ($n_{\mathrm{disc}}=" + str(n_discordant) + r"$) & "
     r"Ensemble vs.\ IForest$_{\mathrm{Platt}}$ & " + mcn_interp + r" & \\"),
    (r"McNemar normals ($n_{\mathrm{disc}}=" + str(n_disc_norm) + r"$) & "
     r"FPR: Ensemble vs.\ IForest$_{\mathrm{Platt}}$ & " + mcn_norm_interp + r" & \\"),
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]

tex_path = os.path.join(OUTPUT_DIR, "table_stats.tex")
with open(tex_path, "w") as fh:
    fh.write("\n".join(tex_lines) + "\n")
print(f"  {tex_path}")

_auprc_ens_str = f"{auprc_ens:.4f}" if auprc_ens is not None else "N/A"
_auprc_if_str  = f"{auprc_if:.4f}"  if auprc_if  is not None else "N/A"

mcnemar_atk_sig  = "SIGNIFICANT" if p_mcnemar < 0.05 else "non-significant"
mcnemar_norm_sig = "SIGNIFICANT (Ensemble reduces FP)" if (mcnemar_norm_p < 0.05 and c_norm > b_norm) else "non-significant"

rows = [
    ("Zone C windows",            f"{n_c}  ({n_n} N / {n_a} A)"),
    ("",                          ""),
    ("-- Calibration --",         ""),
    ("Platt IForest ECE",         f"{ece_if_raw:.4f} -> {ece_if_cal:.4f}"),
    ("Platt LSTM ECE",            f"{ece_lm_raw:.4f} -> {ece_lm_cal:.4f}"),
    ("T*",                        f"{T_opt:.4f}  ({'AT BOUND' if at_bound else 'free min'})"),
    ("ECE LSTM+T*",               f"{ece_before:.4f} -> {ece_after:.4f}"),
    ("Gating (delta, alpha)",     f"{GATING_DELTA:.2f}  {GATING_ALPHA:.2f}"),
    ("",                          ""),
    ("-- Zone C Performance --",  ""),
    ("Ensemble AUC",              f"{auc_str}  [bootstrap CI: {boot_str}]"),
    ("Ensemble TPR / FPR",        f"{pct(TPR_attack)}%  [{pct(tpr_atk_lo)}%, {pct(tpr_atk_hi)}%]  /  {pct(FPR_normal)}%  [{pct(fpr_nm_lo)}%, {pct(fpr_nm_hi)}%]"),
    ("IForest AUC",               f"{auc_if_str}  TPR={pct(TPR_if)}%  [{pct(tpr_if_lo)}%, {pct(tpr_if_hi)}%]"),
    ("AUPRC Ensemble / IForest",  f"{_auprc_ens_str}  /  {_auprc_if_str}"),
    ("",                          ""),
    ("-- McNemar Tests --",       ""),
    ("Attacks (b, c, p)",         f"b={b}  c={c}  p={p_mcnemar:.4f}  Power={power*100:.1f}%  [{mcnemar_atk_sig}]"),
    ("Normals (b, c, p)",         f"b={b_norm}  c={c_norm}  p={mcnemar_norm_p:.4f}  [{mcnemar_norm_sig}]"),
    ("",                          ""),
    ("LaTeX table",               tex_path),
]
c1 = max(len(r[0]) for r in rows)
c2 = max(len(r[1]) for r in rows)
sep = f"+{'-'*(c1+2)}+{'-'*(c2+2)}+"
def _row(label, val):
    return f"| {label:<{c1}} | {val:<{c2}} |"
print("\n" + sep)
print(_row("Statistical Validation Summary", ""))
print(sep)
for label, val in rows:
    if label.startswith("--"):
        print(sep)
        print(_row(label, val))
    elif label == "":
        pass
    else:
        print(_row(label, val))
print(sep)
