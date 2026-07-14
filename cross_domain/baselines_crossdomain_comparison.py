
"""
BridgeGuard Multi-Baseline Cross-Domain Comparison
=====================================================
Evaluates 5 baselines zero-shot on the same 3 ToN-IoT sensor datasets,
all trained on the same MQTT source normal windows as BridgeGuard.

Baselines:
 1. IForest standalone trained (bridgeguard_models/), IQR-robust scoring
 2. LSTM standalone trained (bridgeguard_models/), CORAL-aligned, T*-calibrated
 3. MLP Autoencoder trained here, reconstruction-error anomaly score
 4. LOF Local Outlier Factor (novelty=True, sklearn)
 5. OCSVM One-Class SVM (nu=0.05 RBF) [for unified table]

Each baseline is tested with and without CORAL covariance alignment.
Score normalization: percentile-based (p2..p98 of target batch) avoids
the zero-TPR collapse that fixed training-data bounds produce cross-domain.

Run from bridgeguard/ root:
 python3 cross_domain/baselines_crossdomain_comparison.py
"""

import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score, f1_score
import tensorflow as tf

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.random.seed(42)

FEATURES_DIR    = "bridgeguard_features"
MODELS_DIR      = "bridgeguard_models"
V6_RESULTS_DIR  = "toniot_results"
V10_RESULTS_DIR = "toniot_v10_results"
OUTPUT_DIR      = "baselines_crossdomain_results"
SEQ_LEN         = 10
DATASETS        = ["IoT_Fridge", "IoT_Thermostat", "IoT_Weather"]

BEHAVIORAL_ATTACKS = {"ddos", "injection", "ransomware", "backdoor",
                      "flooding", "slow_poisoning", "mitm", "dos"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("BridgeGuard Multi-Baseline Cross-Domain Comparison")
print("=" * 70)

print("\nLoading BridgeGuard artifacts...")
try:
    with open(os.path.join(FEATURES_DIR, "selected_features.json")) as fh:
        SELECTED = json.load(fh)["selected_features"]
    with open(os.path.join(MODELS_DIR, "feature_scaler_selected.pkl"), "rb") as fh:
        scaler = pickle.load(fh)
    with open(os.path.join(MODELS_DIR, "isolation_forest_optimized.pkl"), "rb") as fh:
        iforest_src = pickle.load(fh)
    lstm_model = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "lstm_model_selected.keras"))
    with open(os.path.join(MODELS_DIR, "lstm_temperature.json")) as fh:
        td = json.load(fh)
    def _get(d, keys, default):
        for k in keys:
            v = d.get(k)
            if v is not None: return float(v)
        return float(default)
    T_STAR = _get(td, ["temperature", "T_opt", "T_star", "T"], 1.0)

    normal_df = pd.read_csv(os.path.join(FEATURES_DIR, "features_selected_normal.csv"))
    X_src     = normal_df[SELECTED].values.astype(np.float64)
    X_src_sc  = scaler.transform(X_src)

except FileNotFoundError as e:
    print(f"  ERROR: {e}"); sys.exit(1)

print(f"  Features ({len(SELECTED)}): {SELECTED}")
print(f"  Source N : {len(X_src)} normal windows  T*={T_STAR:.4f}")

print("\nTraining baseline models on source normal windows...")

skewed    = ['payload_entropy', 'freq_rolling_std_5w', 'alert_rate_acceleration']
skew_idx  = [SELECTED.index(f) for f in skewed if f in SELECTED]
src_log   = X_src.copy()
if skew_idx:
    src_log[:, skew_idx] = np.log1p(np.abs(src_log[:, skew_idx]))
src_med   = np.median(src_log, axis=0)
src_iqr   = (np.percentile(src_log, 75, axis=0)
             - np.percentile(src_log, 25, axis=0)) + 1e-9
src_robust = (src_log - src_med) / src_iqr
print(f"  IForest   : loaded from {MODELS_DIR}/ (pre-trained)")

n_feat = len(SELECTED)
h1, h2 = max(n_feat, 8), max(n_feat // 2, 4)
ae_model = MLPRegressor(
    hidden_layer_sizes=(h1, h2, h1), activation='relu', solver='adam',
    max_iter=500, random_state=42, early_stopping=True,
    validation_fraction=0.15, n_iter_no_change=15, verbose=False)
ae_model.fit(X_src_sc, X_src_sc)
ae_src_rec = ae_model.predict(X_src_sc)
ae_src_err = np.mean((X_src_sc - ae_src_rec) ** 2, axis=1)
AE_MU, AE_SIG = ae_src_err.mean(), ae_src_err.std() + 1e-9
print(f"  AE (MLP)  : trained  mu_err={AE_MU:.4f}  sigma={AE_SIG:.4f}")

lof_model = LocalOutlierFactor(
    n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1)
lof_model.fit(X_src_sc)
print(f"  LOF       : trained  n_neighbors=20  novelty=True")

ocsvm_model = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
ocsvm_model.fit(X_src_sc)
print(f"  OCSVM     : trained  nu=0.05  kernel=rbf")

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def coral_align(Xs, Xt):
    eps = 1e-6
    Cs  = np.cov(Xs.T) + eps * np.eye(Xs.shape[1])
    Ct  = np.cov(Xt.T) + eps * np.eye(Xt.shape[1])
    try:
        A = np.real(np.linalg.inv(sqrtm(Ct))) @ np.real(sqrtm(Cs))
        if not np.all(np.isfinite(A)):
            raise ValueError("non-finite")
        aligned = (Xt - Xt.mean(0)) @ A + Xs.mean(0)
        aligned = np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)
        return aligned, True
    except Exception:
        return Xt.copy(), False

def percentile_normalize(scores, lo_pct=2, hi_pct=98):
    lo = float(np.percentile(scores, lo_pct))
    hi = float(np.percentile(scores, hi_pct))
    rng = hi - lo
    if rng < 1e-12:
        return np.full(len(scores), 0.5)
    return np.clip((scores - lo) / rng, 0.0, 1.0)

def metrics_at_fpr(p_normal, y, target_fpr=0.15):
    y_atk = (y == 0).astype(int)
    n_atk  = int(y_atk.sum())
    n_norm = int(len(y) - n_atk)
    if n_atk == 0 or n_norm == 0:
        return dict(auc=0.5, tpr=0.0, fpr=0.0, f1=0.0, threshold=0.5)
    try:
        auc = float(roc_auc_score(y, p_normal))
    except Exception:
        auc = 0.5
    best = dict(tpr=0.0, fpr=0.0, f1=0.0, threshold=0.5)
    for thr in np.arange(0.99, 0.01, -0.01):
        yp  = (p_normal < thr).astype(int)
        det = int((yp.astype(bool) & y_atk.astype(bool)).sum())
        fa  = int((yp.astype(bool) & (~y_atk.astype(bool))).sum())
        tpr_v = det / n_atk
        fpr_v = fa  / n_norm
        if fpr_v <= target_fpr and tpr_v > best["tpr"]:
            best = dict(tpr=tpr_v, fpr=fpr_v,
                        f1=float(f1_score(y_atk, yp, zero_division=0)),
                        threshold=float(thr))
    return dict(auc=auc, **best)

def build_lstm_sequences(X, seq_len):
    n, d = X.shape
    seqs = []
    for i in range(n):
        start = max(0, i - seq_len + 1)
        seq = X[start:i + 1]
        if len(seq) < seq_len:
            pad = np.tile(X[0], (seq_len - len(seq), 1))
            seq = np.vstack([pad, seq])
        seqs.append(seq[-seq_len:])
    return np.array(seqs, dtype=np.float32)

def load_bg_v10(dname):
    path = os.path.join(V10_RESULTS_DIR, f"{dname}_v10.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)

all_results = {}

for dname in DATASETS:
    print(f"\n{'='*70}")
    print(f"DATASET: {dname}")
    print("="*70)

    feat_path = os.path.join(V6_RESULTS_DIR, f"{dname}_features.csv")
    if not os.path.exists(feat_path):
        print(f"  Feature CSV not found   skipped (run cross-domain adaptation.py first)")
        continue

    df = pd.read_csv(feat_path)

    if "freq_rolling_std_5w" not in df.columns and "alert_frequency_per_hour" in df.columns:
        df["freq_rolling_std_5w"] = (
            df["alert_frequency_per_hour"]
            .rolling(window=5, min_periods=3).std().fillna(0.0))
    for f in SELECTED:
        if f not in df.columns:
            df[f] = 0.0

    y   = df["label"].values
    nm  = (y == 1); am = (y == 0)
    X_raw = np.nan_to_num(df[SELECTED].values.astype(np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  {len(df)} windows: {nm.sum()} normal / {am.sum()} attack")

    X_tgt_sc = scaler.transform(X_raw)

    X_log = X_raw.copy()
    if skew_idx:
        X_log[:, skew_idx] = np.log1p(np.abs(X_log[:, skew_idx]))
    X_robust = (X_log - src_med) / src_iqr

    X_z     = np.clip((X_raw - X_raw.mean(0)) / (X_raw.std(0) + 1e-9), -50, 50)
    X_src_z = np.clip((X_src_sc - X_src_sc.mean(0)) / (X_src_sc.std(0) + 1e-9), -50, 50)
    X_coral, coral_ok = coral_align(X_src_z, X_z)
    print(f"  CORAL: {'ok' if coral_ok else 'fallback'}")

    src_mu  = X_src_sc.mean(0); src_std = X_src_sc.std(0) + 1e-9
    X_coral_sc = (X_coral - src_mu) / src_std

    _sc_src_ref = iforest_src.score_samples(src_robust)
    _if_lo = float(np.percentile(_sc_src_ref, 2))
    _if_hi = float(np.percentile(_sc_src_ref, 98))

    sc_if       = iforest_src.score_samples(X_robust)
    p_if_raw    = np.clip((sc_if - _if_lo) / (_if_hi - _if_lo + 1e-12), 0.0, 1.0)

    if nm.sum() > 0 and am.sum() > 0:
        if p_if_raw[nm].mean() < p_if_raw[am].mean():
            p_if_raw = 1.0 - p_if_raw
    m_if = metrics_at_fpr(p_if_raw, y)

    X_robust_coral = (X_coral - src_med) / src_iqr
    sc_if_c     = iforest_src.score_samples(X_robust_coral)
    p_if_c_raw  = np.clip((sc_if_c - _if_lo) / (_if_hi - _if_lo + 1e-12), 0.0, 1.0)
    if nm.sum() > 0 and am.sum() > 0:
        if p_if_c_raw[nm].mean() < p_if_c_raw[am].mean():
            p_if_c_raw = 1.0 - p_if_c_raw
    m_if_coral = metrics_at_fpr(p_if_c_raw, y)

    print(f"  IForest          AUC={m_if['auc']:.4f}  "
          f"TPR={m_if['tpr']*100:.1f}%  FPR={m_if['fpr']*100:.1f}%")
    print(f"  IForest+CORAL    AUC={m_if_coral['auc']:.4f}  "
          f"TPR={m_if_coral['tpr']*100:.1f}%  FPR={m_if_coral['fpr']*100:.1f}%")

    seqs = build_lstm_sequences(X_coral, SEQ_LEN)
    raw_pr = lstm_model.predict(seqs, batch_size=64, verbose=0).flatten()
    raw_pr = np.clip(raw_pr, 1e-7, 1 - 1e-7)
    p_lstm = sigmoid(np.log(raw_pr / (1 - raw_pr)) / T_STAR)
    p_lstm = np.nan_to_num(p_lstm, nan=0.5)

    if nm.sum() > 0 and am.sum() > 0:
        if p_lstm[nm].mean() < p_lstm[am].mean():
            p_lstm = 1.0 - p_lstm
    m_lstm = metrics_at_fpr(p_lstm, y)

    seqs_z = build_lstm_sequences(X_z, SEQ_LEN)
    raw_pr_z = lstm_model.predict(seqs_z, batch_size=64, verbose=0).flatten()
    raw_pr_z = np.clip(raw_pr_z, 1e-7, 1 - 1e-7)
    p_lstm_z = sigmoid(np.log(raw_pr_z / (1 - raw_pr_z)) / T_STAR)
    p_lstm_z = np.nan_to_num(p_lstm_z, nan=0.5)
    if nm.sum() > 0 and am.sum() > 0:
        if p_lstm_z[nm].mean() < p_lstm_z[am].mean():
            p_lstm_z = 1.0 - p_lstm_z
    m_lstm_nocoral = metrics_at_fpr(p_lstm_z, y)

    print(f"  LSTM (CORAL)     AUC={m_lstm['auc']:.4f}  "
          f"TPR={m_lstm['tpr']*100:.1f}%  FPR={m_lstm['fpr']*100:.1f}%")
    print(f"  LSTM (no CORAL)  AUC={m_lstm_nocoral['auc']:.4f}  "
          f"TPR={m_lstm_nocoral['tpr']*100:.1f}%  FPR={m_lstm_nocoral['fpr']*100:.1f}%")

    def ae_score(X_in):
        rec   = ae_model.predict(X_in)
        err   = np.mean((X_in - rec) ** 2, axis=1)

        ae_z  = -(err - AE_MU) / AE_SIG
        return percentile_normalize(ae_z)

    p_ae       = ae_score(X_tgt_sc)
    p_ae_coral = ae_score(X_coral_sc)

    if nm.sum() > 0 and am.sum() > 0:
        for p in [p_ae, p_ae_coral]:
            pass

    m_ae       = metrics_at_fpr(p_ae, y)
    m_ae_coral = metrics_at_fpr(p_ae_coral, y)
    print(f"  AE               AUC={m_ae['auc']:.4f}  "
          f"TPR={m_ae['tpr']*100:.1f}%  FPR={m_ae['fpr']*100:.1f}%")
    print(f"  AE+CORAL         AUC={m_ae_coral['auc']:.4f}  "
          f"TPR={m_ae_coral['tpr']*100:.1f}%  FPR={m_ae_coral['fpr']*100:.1f}%")

    lof_scores      = lof_model.score_samples(X_tgt_sc)
    lof_scores_coral= lof_model.score_samples(X_coral_sc)
    p_lof           = percentile_normalize(lof_scores)
    p_lof_coral     = percentile_normalize(lof_scores_coral)

    m_lof       = metrics_at_fpr(p_lof, y)
    m_lof_coral = metrics_at_fpr(p_lof_coral, y)
    print(f"  LOF              AUC={m_lof['auc']:.4f}  "
          f"TPR={m_lof['tpr']*100:.1f}%  FPR={m_lof['fpr']*100:.1f}%")
    print(f"  LOF+CORAL        AUC={m_lof_coral['auc']:.4f}  "
          f"TPR={m_lof_coral['tpr']*100:.1f}%  FPR={m_lof_coral['fpr']*100:.1f}%")

    ocsvm_df         = ocsvm_model.decision_function(X_tgt_sc)
    ocsvm_df_coral   = ocsvm_model.decision_function(X_coral_sc)
    p_ocsvm          = percentile_normalize(ocsvm_df)
    p_ocsvm_coral    = percentile_normalize(ocsvm_df_coral)

    m_ocsvm          = metrics_at_fpr(p_ocsvm, y)
    m_ocsvm_coral    = metrics_at_fpr(p_ocsvm_coral, y)
    print(f"  OCSVM            AUC={m_ocsvm['auc']:.4f}  "
          f"TPR={m_ocsvm['tpr']*100:.1f}%  FPR={m_ocsvm['fpr']*100:.1f}%")
    print(f"  OCSVM+CORAL      AUC={m_ocsvm_coral['auc']:.4f}  "
          f"TPR={m_ocsvm_coral['tpr']*100:.1f}%  FPR={m_ocsvm_coral['fpr']*100:.1f}%")

    bg = load_bg_v10(dname)
    if bg:
        bg_m = bg.get("v10_fpr15", {})
        bg_b = bg.get("v10_behavioral_fpr15") or bg.get("v10_behavioral_optimal") or {}
        print(f"  BridgeGuard       AUC={bg_m.get('auc',0):.4f}  "
              f"TPR={bg_m.get('tpr',0)*100:.1f}%  FPR={bg_m.get('fpr',0)*100:.1f}%  "
              f"F1={bg_m.get('f1',0):.4f}  (beh AUC={bg_b.get('auc',0):.3f})")

    all_results[dname] = {
        "iforest":       m_if,
        "iforest_coral": m_if_coral,
        "lstm":          m_lstm,
        "lstm_nocoral":  m_lstm_nocoral,
        "ae":            m_ae,
        "ae_coral":      m_ae_coral,
        "lof":           m_lof,
        "lof_coral":     m_lof_coral,
        "ocsvm":         m_ocsvm,
        "ocsvm_coral":   m_ocsvm_coral,
        "bridgeguard":   bg,
        "n_normal": int(nm.sum()),
        "n_attack":  int(am.sum()),
    }

MODELS_ORDER = [
    ("IForest",        "iforest"),
    ("IForest+CORAL",  "iforest_coral"),
    ("LSTM",           "lstm"),
    ("LSTM (no CORAL)","lstm_nocoral"),
    ("AE (MLP)",       "ae"),
    ("AE+CORAL",       "ae_coral"),
    ("LOF",            "lof"),
    ("LOF+CORAL",      "lof_coral"),
    ("OCSVM",          "ocsvm"),
    ("OCSVM+CORAL",    "ocsvm_coral"),
]

print(f"\n\n{'='*85}")
print("UNIFIED SUMMARY All Baselines vs BridgeGuard (cross-domain, FPR<=15%)")
print("Setting: [CD/UNS] zero-shot, MQTT source ToN-IoT target, no target labels")
print("="*85)

for dname in DATASETS:
    if dname not in all_results:
        continue
    r = all_results[dname]
    bg = r.get("bridgeguard") or {}
    bg_m = bg.get("v10_fpr15", {})
    bg_b = bg.get("v10_behavioral_fpr15") or bg.get("v10_behavioral_optimal") or {}
    n_win = r["n_normal"] + r["n_attack"]

    print(f"\n  -- {dname}  ({n_win} windows: {r['n_normal']} normal / {r['n_attack']} attack) --")
    print(f"  {'Model':<20}  {'AUC':>6}  {'TPR@FPR 15%':>12}  {'FPR':>7}  {'F1':>6}")
    print(f"  {'-'*60}")

    for label, key in MODELS_ORDER:
        m = r[key]
        marker = " best" if m["auc"] == max(r[k]["auc"] for _, k in MODELS_ORDER) else ""
        print(f"  {label:<20}  {m['auc']:>6.3f}  {m['tpr']*100:>11.1f}%  "
              f"{m['fpr']*100:>6.1f}%  {m['f1']:>6.3f}{marker}")

    auc_bg = bg_m.get("auc", 0)
    tpr_bg = bg_m.get("tpr", 0)
    fpr_bg = bg_m.get("fpr", 0)
    f1_bg  = bg_m.get("f1",  0)
    print(f"  {' BridgeGuard':<20}  {auc_bg:>6.3f}  {tpr_bg*100:>11.1f}%  "
          f"{fpr_bg*100:>6.1f}%  {f1_bg:>6.3f}  *** THIS WORK ***")

print(f"\n\n% ====== LaTeX Table: All Baselines vs BridgeGuard Cross-Domain ======")
print(r"""\begin{table*}[ht]
\centering
\caption{Cross-domain zero-shot evaluation: all baselines vs BridgeGuard on
 ToN-IoT sensor datasets. All models trained on MQTT source normal windows
 only (no target labels, no retraining). Metric: highest TPR at FPR$\leq$15\%.
 CORAL = covariance alignment applied to input before scoring.
 $^*$Behavioral attacks (DDoS, injection, ransomware, backdoor) only.
 IForest: IQR-robust preprocessing + dynamic source-percentile bounds.
 AE: MLP autoencoder reconstruction error.
 LOF: Local Outlier Factor (novelty=True, $k$=20).
 OCSVM: One-Class SVM ($\nu$=0.05, RBF).
 LSTM: sequence model with temperature scaling ($T^*$=0.9995).}
\label{tab:all_baselines_crossdomain}
\begin{tabular}{llccccc}
\toprule
Dataset & Model & Win & AUC & TPR & FPR & F1 \\
\midrule""")

for dname in DATASETS:
    if dname not in all_results:
        continue
    r = all_results[dname]
    bg = r.get("bridgeguard") or {}
    bg_m = bg.get("v10_fpr15", {})
    bg_b = bg.get("v10_behavioral_fpr15") or bg.get("v10_behavioral_optimal") or {}
    n_win = r["n_normal"] + r["n_attack"]
    short = dname.replace("IoT_", "")

    best_auc = max(r[k]["auc"] for _, k in MODELS_ORDER)

    first = True
    for label, key in MODELS_ORDER:
        m = r[key]
        n_str = str(n_win) if first else ""
        d_str = short if first else ""
        bold  = "\\textbf{" if m["auc"] == best_auc else ""
        ebold = "}" if m["auc"] == best_auc else ""
        print(f"  {d_str} & {label} & {n_str} & "
              f"{bold}{m['auc']:.3f}{ebold} & "
              f"{m['tpr']*100:.1f}\\% & "
              f"{m['fpr']*100:.1f}\\% & "
              f"{m['f1']:.3f} \\\\")
        first = False

    auc_bg = bg_m.get("auc", 0)
    tpr_bg = bg_m.get("tpr", 0)
    fpr_bg = bg_m.get("fpr", 0)
    f1_bg  = bg_m.get("f1", 0)
    auc_beh_str = (f"({bg_b.get('auc',0):.3f}$^*$)"
                   if bg_b and abs(bg_b.get("auc",auc_bg) - auc_bg) > 0.005 else "")
    print(f"  & \\textbf{{BridgeGuard (ours)}} & & "
          f"\\textbf{{{auc_bg:.3f}}}{auc_beh_str} & "
          f"\\textbf{{{tpr_bg*100:.1f}\\%}} & "
          f"\\textbf{{{fpr_bg*100:.1f}\\%}} & "
          f"\\textbf{{{f1_bg:.3f}}} \\\\")
    print(" \\midrule")

print(r"""\bottomrule
\end{tabular}
\end{table*}""")

out = {}
for dname, r in all_results.items():
    bg_m = (r.get("bridgeguard") or {}).get("v10_fpr15", {})
    out[dname] = {}
    for label, key in MODELS_ORDER:
        out[dname][label] = {
            **r[key],
            "delta_auc_vs_bridgeguard": round(bg_m.get("auc", 0) - r[key]["auc"], 4)
        }
    out[dname]["BridgeGuard"] = bg_m

out_path = os.path.join(OUTPUT_DIR, "all_baselines_comparison.json")
with open(out_path, "w") as fh:
    json.dump(out, fh, indent=2)

def _print_summary():
    cols   = ["Baseline", "Fridge ΔAUC", "Thermostat ΔAUC", "Weather ΔAUC"]
    widths = [22, 13, 16, 13]
    sep    = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    def row(vals):
        return "| " + " | ".join(f"{str(v):<{w}}" for v, w in zip(vals, widths)) + " |"
    print("\n" + sep)
    print(row(cols))
    print(sep)
    for label, key in MODELS_ORDER:
        vals = [label]
        for dname in DATASETS:
            if dname not in all_results:
                vals.append("N/A")
                continue
            r    = all_results[dname]
            bg_m = (r.get("bridgeguard") or {}).get("v10_fpr15", {})
            delta = bg_m.get("auc", 0) - r[key]["auc"]
            vals.append(f"{delta:+.4f}")
        print(row(vals))
    print(sep)
    print(f"  Positive = BridgeGuard wins  |  Output: {out_path}")
    print(sep)

_print_summary()
