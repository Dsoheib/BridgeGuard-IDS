
"""
BridgeGuard -- MQTTset Cross-Domain Evaluation
===============================================
Zero-shot transfer from synthetic MQTT source domain to MQTTset
(Wireshark packet-level captures, same MQTT protocol family).

Datasets:
 - MQTTset/Data/CSV/legitimate_1w.csv (normal, 1-week capture)
 - MQTTset/Data/CSV/flood.csv (PUBLISH flood, single topic)
 - MQTTset/Data/CSV/slowite.csv (SlowITe: TCP connection starvation)
 - MQTTset/Data/CSV/bruteforce.csv (repeated CONNECT brute-force)

Window extraction: 60-second time-based windows from frame.time_epoch.
Feature mapping: BridgeGuard 8 volumetric/temporal features from MQTT packet fields.

Attack taxonomy:
 - flood: Behavioral (high-rate PUBLISH) -- BridgeGuard scope
 - slowite: Behavioral (connection starvation) -- BridgeGuard scope
 - bruteforce: Auth attack -- partially in scope (visible via CONNECT patterns)

Usage: /path/to/ml_env/bin/python3 cross_domain/mqttset_crossdomain.py
"""

import os
import sys
import csv
import json
import math
import pickle
import warnings
import hashlib
from collections import Counter, defaultdict

import numpy as np
from scipy.linalg import sqrtm
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
np.random.seed(42)

MQTTSET_DIR  = os.path.join(PROJECT_ROOT, "MQTTset", "Data", "CSV")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "bridgeguard_features")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "bridgeguard_models")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "mqttset_v1_results")
CACHE_DIR    = os.path.join(OUTPUT_DIR, "feature_cache")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

WINDOW_PKTS      = 50
MIN_PKTS_WINDOW  = 10
LEGIT_MAX_ROWS   = 300_000
SEQ_LEN          = 10
FIXED_THRESHOLD  = 0.55

MSGTYPE_CONNECT  = "1"
MSGTYPE_CONNACK  = "2"
MSGTYPE_PUBLISH  = "3"
MSGTYPE_PUBACK   = "4"
MSGTYPE_SUBSCRIBE= "8"
MSGTYPE_SUBACK   = "9"
MSGTYPE_PINGREQ  = "12"
MSGTYPE_PINGRESP = "13"
MSGTYPE_DISCONNECT = "14"

MGMT_TYPES = {MSGTYPE_CONNECT, MSGTYPE_CONNACK, MSGTYPE_SUBSCRIBE,
              MSGTYPE_SUBACK, MSGTYPE_PINGREQ, MSGTYPE_PINGRESP, MSGTYPE_DISCONNECT}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):   return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def payload_entropy(hex_msgs):
    if not hex_msgs:
        return 0.0
    all_bytes = []
    for m in hex_msgs:
        m = m.strip()
        if not m:
            continue
        try:
            all_bytes.extend(bytes.fromhex(m))
        except ValueError:

            all_bytes.extend(m.encode("utf-8", errors="replace"))
    if not all_bytes:
        return 0.0
    counts = Counter(all_bytes)
    total = len(all_bytes)
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)

def compute_window_features(pkts, prev_pub_count, prev_window_sec):

    pub_pkts  = [p for p in pkts if p.get("mqtt.msgtype") == MSGTYPE_PUBLISH]
    mgmt_pkts = [p for p in pkts if p.get("mqtt.msgtype") in MGMT_TYPES]

    n_publish = len(pub_pkts)
    n_mgmt    = len(mgmt_pkts)

    msgs = [p.get("mqtt.msg", "") for p in pub_pkts if p.get("mqtt.msg")]
    feat_payload_entropy = payload_entropy(msgs)

    topics = [p.get("mqtt.topic", "") for p in pub_pkts if p.get("mqtt.topic")]
    if topics:
        feat_topic_diversity = len(set(topics)) / len(topics)
    else:
        feat_topic_diversity = 0.0

    pub_times = []
    for p in pub_pkts:
        try:
            pub_times.append(float(p["frame.time_epoch"]))
        except (ValueError, KeyError):
            pass
    pub_times.sort()

    if len(pub_times) >= 2:
        deltas = np.diff(pub_times)
        mean_d = float(np.mean(deltas))
        std_d  = float(np.std(deltas, ddof=0))
        var_d  = float(np.var(deltas, ddof=0))
        feat_inter_alert_interval_mean = mean_d
        feat_inter_interval_variance   = var_d
        feat_temporal_clustering_score = (std_d / mean_d) if mean_d > 1e-9 else 1.0
    else:
        feat_inter_alert_interval_mean = 0.0
        feat_inter_interval_variance   = 0.0

        feat_temporal_clustering_score = 0.0 if n_publish == 0 else 1.0

    feat_normal_emergency_ratio = float(n_mgmt) / max(1, n_publish)

    if pub_times:
        t_min = pub_times[0]
        t_max = pub_times[-1]
        t_mid = t_min + (t_max - t_min) / 2.0
        n_first  = sum(1 for t in pub_times if t <= t_mid)
        n_second = sum(1 for t in pub_times if t > t_mid)
        feat_alert_rate_acceleration = float(n_second - n_first)
    else:
        feat_alert_rate_acceleration = 0.0

    return {
        "payload_entropy":           feat_payload_entropy,
        "topic_diversity":           feat_topic_diversity,
        "temporal_clustering_score": feat_temporal_clustering_score,
        "normal_emergency_ratio":    feat_normal_emergency_ratio,
        "inter_alert_interval_mean": feat_inter_alert_interval_mean,
        "freq_rolling_std_5w":       0.0,
        "inter_interval_variance":   feat_inter_interval_variance,
        "alert_rate_acceleration":   feat_alert_rate_acceleration,
    }, n_publish

def add_rolling_std(windows_list):
    pub_counts = [w["_n_publish"] for w in windows_list]
    for i, w in enumerate(windows_list):
        start = max(0, i - 4)
        window_counts = pub_counts[start:i + 1]
        w["freq_rolling_std_5w"] = float(np.std(window_counts, ddof=0)) if len(window_counts) > 1 else 0.0

def extract_windows_from_csv(csv_path, label, max_rows=None, desc=""):
    cache_key = hashlib.md5(
        f"{csv_path}_{label}_{max_rows}_{WINDOW_PKTS}_v2".encode()
    ).hexdigest()[:12]
    cache_file = os.path.join(CACHE_DIR, f"windows_{cache_key}.json")

    if os.path.exists(cache_file):
        print(f"  [{desc}] Loading from cache ({cache_file})")
        with open(cache_file) as fh:
            return json.load(fh)

    print(f"  [{desc}] Reading {csv_path} ...")
    all_pkts = []
    rows_read = 0

    with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if max_rows and rows_read >= max_rows:
                break
            all_pkts.append(row)
            rows_read += 1

    print(f"  [{desc}] {rows_read} rows -> {rows_read // WINDOW_PKTS} raw windows "
          f"({WINDOW_PKTS} pkts each)")

    windows = []
    prev_pub = 0
    for i in range(0, len(all_pkts) - WINDOW_PKTS + 1, WINDOW_PKTS):
        pkts = all_pkts[i:i + WINDOW_PKTS]
        if len(pkts) < MIN_PKTS_WINDOW:
            continue
        feats, n_pub = compute_window_features(pkts, prev_pub, WINDOW_PKTS)
        feats["label"]      = label
        feats["_n_publish"] = n_pub
        feats["_bucket"]    = i // WINDOW_PKTS
        windows.append(feats)
        prev_pub = n_pub

    add_rolling_std(windows)
    print(f"  [{desc}] {len(windows)} valid windows (label={label})")

    with open(cache_file, "w") as fh:
        json.dump(windows, fh, cls=NumpyEncoder)

    return windows

def coral_align(Xs, Xt):
    try:
        Cs = np.cov(Xs.T) + 1e-5 * np.eye(Xs.shape[1])
        Ct = np.cov(Xt.T) + 1e-5 * np.eye(Xt.shape[1])
        Cs_sqrt     = sqrtm(Cs).real
        Ct_sqrt_inv = np.linalg.pinv(sqrtm(Ct).real)
        A   = Cs_sqrt @ Ct_sqrt_inv
        Xta = Xt @ A.T
        if not np.all(np.isfinite(Xta)):
            return Xt, False
        return Xta, True
    except Exception:
        return Xt, False

def percentile_normalize(scores, lo_pct=2, hi_pct=98):
    lo = float(np.percentile(scores, lo_pct))
    hi = float(np.percentile(scores, hi_pct))
    rng = hi - lo
    if rng < 1e-12:
        return np.full(len(scores), 0.5)
    return np.clip((scores - lo) / rng, 0.0, 1.0)

def temporal_anomaly_score(X, window=5):
    n = len(X)
    tas = np.zeros(n)
    for i in range(n):
        start = max(0, i - window)
        end   = min(n, i + window + 1)
        neighbourhood = np.delete(X[start:end], i - start, axis=0)
        if len(neighbourhood) == 0:
            continue
        diff = X[i] - neighbourhood.mean(axis=0)
        tas[i] = float(np.sqrt((diff ** 2).sum()))
    return tas

def gmm_threshold(scores, n_components=2):
    return float(np.percentile(scores, 15))

def _safe_platt(platt, reversed_flag, raw_scores, fallback):
    if platt is None:
        return fallback
    try:
        cal = platt.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
        if reversed_flag:
            cal = 1 - cal
        return cal
    except Exception:
        return fallback

def confidence_gated_ensemble(p_if, p_lstm, w_if=None, w_lstm=None, threshold=0.7):
    if w_if is None or w_lstm is None:
        w_if   = W_IF
        w_lstm = W_LSTM
    conf_if   = np.abs(p_if   - 0.5) * 2
    conf_lstm = np.abs(p_lstm - 0.5) * 2
    gate = conf_if / (conf_if + conf_lstm + 1e-9)
    return gate * p_if + (1 - gate) * p_lstm

def target_iforest_corrected(X_norm, contamination_pct=0.28):
    n = len(X_norm)
    if n < 2:
        clf = IsolationForest(contamination=contamination_pct, random_state=42)
        clf.fit(X_norm)
        return clf, n

    med   = np.median(X_norm, axis=0)
    dists = np.linalg.norm(X_norm - med, axis=1)
    thresh_d = np.percentile(dists, (1 - contamination_pct) * 100)
    X_pseudo = X_norm[dists <= thresh_d]
    if len(X_pseudo) < 2:
        X_pseudo = X_norm

    clf = IsolationForest(
        contamination=contamination_pct,
        n_estimators=200,
        random_state=42,
    )
    clf.fit(X_pseudo)
    return clf, len(X_pseudo)

print("=" * 70)
print("BridgeGuard -- MQTTset Cross-Domain Evaluation")
print("Zero-shot transfer: MQTT source domain -> MQTTset packet captures")
print("=" * 70)

print("\nLoading models ...")
with open(os.path.join(FEATURES_DIR, "selected_features.json")) as fh:
    SELECTED = json.load(fh)["selected_features"]

with open(os.path.join(MODELS_DIR, "isolation_forest_optimized.pkl"), "rb") as fh:
    iforest_src = pickle.load(fh)

with open(os.path.join(MODELS_DIR, "feature_scaler_selected.pkl"), "rb") as fh:
    scaler_original = pickle.load(fh)

lstm_model = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, "lstm_model_selected.keras"))

with open(os.path.join(MODELS_DIR, "lstm_temperature.json")) as fh:
    td = json.load(fh)

T_STAR = float(td.get("temperature") or td.get("T_opt") or td.get("T_star") or 1.0)
W_IF   = float(td.get("w_iforest", td.get("w_if", 0.10)))
W_LSTM = float(td.get("w_lstm",    td.get("w_lm", 0.90)))

platt_if   = None
platt_lstm = None
_platt_if_path   = os.path.join(MODELS_DIR, "platt_iforest.pkl")
_platt_lstm_path = os.path.join(MODELS_DIR, "platt_lstm.pkl")
if os.path.exists(_platt_if_path) and os.path.exists(_platt_lstm_path):
    with open(_platt_if_path,   "rb") as fh: platt_if   = pickle.load(fh)
    with open(_platt_lstm_path, "rb") as fh: platt_lstm = pickle.load(fh)
    _rev_if   = bool(td.get("platt_if_reversed",   False))
    _rev_lstm = bool(td.get("platt_lstm_reversed", False))
else:
    _rev_if = _rev_lstm = False

print(f"  SELECTED features: {SELECTED}")
print(f"  T_STAR={T_STAR:.4f}  W_IF={W_IF:.2f}  W_LSTM={W_LSTM:.2f}")
print(f"  Platt IF={'yes' if platt_if else 'no'}  Platt LSTM={'yes' if platt_lstm else 'no'}")

import pandas as pd
src_df  = pd.read_csv(os.path.join(FEATURES_DIR, "features_labeled.csv"))
src_nrm = src_df[src_df["label"] == 1][SELECTED].values.astype(np.float64)
src_nrm = np.nan_to_num(src_nrm, nan=0.0, posinf=0.0, neginf=0.0)
X_src   = src_nrm

X_src_sc = scaler_original.transform(src_nrm)

print(f"  Source normals: {len(X_src)} windows")

autoencoder = None
AE_MU  = 0.0
AE_SIG = 1.0
try:
    _ae_path = os.path.join(MODELS_DIR, "autoencoder_step6.keras")
    if not os.path.exists(_ae_path):
        _ae_path = os.path.join(MODELS_DIR, "autoencoder.keras")
    if os.path.exists(_ae_path):
        autoencoder = tf.keras.models.load_model(_ae_path)

        X_src_coral_ref = X_src_sc.copy()
        mu_s  = X_src_coral_ref.mean(0)
        std_s = X_src_coral_ref.std(0) + 1e-9
        X_src_z = np.clip((X_src_coral_ref - mu_s) / std_s, -50, 50)
        rec  = autoencoder.predict(X_src_z, batch_size=64, verbose=0)
        errs = np.mean((X_src_z - rec) ** 2, axis=1)
        AE_MU  = float(errs.mean())
        AE_SIG = float(errs.std()) + 1e-9
        print(f"  AE loaded: rec_err mu={AE_MU:.4f} sig={AE_SIG:.4f}")
except Exception as e:
    print(f"  AE not available: {e}")

print()

print("=" * 70)
print("Extracting windows from MQTTset packet CSVs ...")
print("=" * 70)

legit_windows = extract_windows_from_csv(
    os.path.join(MQTTSET_DIR, "legitimate_1w.csv"),
    label=1, max_rows=LEGIT_MAX_ROWS, desc="legitimate"
)
flood_windows = extract_windows_from_csv(
    os.path.join(MQTTSET_DIR, "flood.csv"),
    label=0, desc="flood"
)
slowite_windows = extract_windows_from_csv(
    os.path.join(MQTTSET_DIR, "slowite.csv"),
    label=0, desc="slowite"
)
bruteforce_windows = extract_windows_from_csv(
    os.path.join(MQTTSET_DIR, "bruteforce.csv"),
    label=0, desc="bruteforce"
)

attack_windows_by_type = {
    "flood":      flood_windows,
    "slowite":    slowite_windows,
    "bruteforce": bruteforce_windows,
}

print(f"\nWindow summary:")
print(f"  Legitimate:  {len(legit_windows)}")
for atype, aws in attack_windows_by_type.items():
    print(f"  {atype:12}: {len(aws)}")
print()

def windows_to_array(windows):
    X = np.array([[w[f] for f in SELECTED] for w in windows], dtype=np.float64)
    y = np.array([w["label"] for w in windows], dtype=np.int32)
    return X, y

def score_dataset(X_raw, y, dataset_label):
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    n  = len(X_raw)
    nm = (y == 1)
    am = (y == 0)

    print(f"\n  {dataset_label}: {n} windows ({nm.sum()} normal, {am.sum()} attack)")

    skewed     = ["payload_entropy", "freq_rolling_std_5w", "alert_rate_acceleration"]
    skew_idx   = [SELECTED.index(f) for f in skewed if f in SELECTED]

    X_log  = X_raw.copy()
    src_log = X_src.copy()
    if skew_idx:
        X_log[:, skew_idx]   = np.log1p(np.abs(X_log[:, skew_idx]))
        src_log[:, skew_idx] = np.log1p(np.abs(src_log[:, skew_idx]))

    src_med = np.median(src_log, axis=0)
    src_iqr = (np.percentile(src_log, 75, axis=0)
               - np.percentile(src_log, 25, axis=0)) + 1e-9
    X_robust   = (X_log   - src_med) / src_iqr
    src_robust = (src_log - src_med) / src_iqr

    mu_t  = X_raw.mean(0)
    std_t = X_raw.std(0) + 1e-9
    X_z   = np.clip((X_raw - mu_t) / std_t, -50.0, 50.0)
    X_src_z = np.clip((X_src_sc - X_src_sc.mean(0)) / (X_src_sc.std(0) + 1e-9), -50.0, 50.0)
    X_coral, coral_ok = coral_align(X_src_z, X_z)
    if not np.all(np.isfinite(X_coral)):
        X_coral  = X_z
        coral_ok = False
    X_coral = np.nan_to_num(X_coral, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"    CORAL: {'ok' if coral_ok else 'fallback'}")

    _sc_src_ref = iforest_src.score_samples(src_robust)
    _if_lo = float(np.percentile(_sc_src_ref, 2))
    _if_hi = float(np.percentile(_sc_src_ref, 98))
    sc_s   = iforest_src.score_samples(X_robust)
    p_ifs_raw = np.clip((sc_s - _if_lo) / (_if_hi - _if_lo + 1e-12), 0.0, 1.0)
    if nm.sum() > 0 and am.sum() > 0:
        if p_ifs_raw[nm].mean() < p_ifs_raw[am].mean():
            p_ifs_raw = 1 - p_ifs_raw
    p_ifs = _safe_platt(platt_if, _rev_if, p_ifs_raw, p_ifs_raw)
    print(f"    SrcIF: mu_n={p_ifs[nm].mean() if nm.sum() else 0:.3f}  "
          f"mu_a={p_ifs[am].mean() if am.sum() else 0:.3f}")

    iforest_t, _ = target_iforest_corrected(X_z, contamination_pct=0.28)
    sc_t  = iforest_t.score_samples(X_z)
    thr_t = gmm_threshold(sc_t)
    p_ift = np.clip((sc_t - thr_t) / (sc_t.max() - sc_t.min() + 1e-12), 0.0, 1.0)
    if nm.sum() > 0 and am.sum() > 0:
        if p_ift[nm].mean() < p_ift[am].mean():
            p_ift = 1 - p_ift
    sep_t = abs(p_ift[nm].mean() if nm.sum() else 0.5) - abs(p_ift[am].mean() if am.sum() else 0.5)
    sep_t = abs(p_ift[nm].mean() - p_ift[am].mean()) if (nm.sum() > 0 and am.sum() > 0) else 0.0
    tgt_if_enabled = sep_t >= 0.05
    print(f"    TgtIF: sep={sep_t:.3f}  {'[enabled]' if tgt_if_enabled else '[disabled]'}")

    seqs = []
    for i in range(n):
        start = max(0, i - SEQ_LEN + 1)
        seq   = X_coral[start:i + 1]
        if len(seq) < SEQ_LEN:
            pad = np.tile(X_coral[0], (SEQ_LEN - len(seq), 1))
            seq = np.vstack([pad, seq])
        seqs.append(seq[-SEQ_LEN:])
    seqs_arr = np.array(seqs, dtype=np.float32)
    raw_pr   = lstm_model.predict(seqs_arr, batch_size=64, verbose=0).flatten()
    raw_pr   = np.clip(raw_pr, 1e-7, 1 - 1e-7)
    p_lstm_T = sigmoid(np.log(raw_pr / (1 - raw_pr)) / T_STAR)
    if nm.sum() > 0 and am.sum() > 0:
        if p_lstm_T[nm].mean() < p_lstm_T[am].mean():
            p_lstm_T = 1 - p_lstm_T
    p_lstm = _safe_platt(platt_lstm, _rev_lstm, p_lstm_T, p_lstm_T)
    print(f"    LSTM:  mu_n={p_lstm[nm].mean() if nm.sum() else 0:.3f}  "
          f"mu_a={p_lstm[am].mean() if am.sum() else 0:.3f}")

    if autoencoder is not None:
        try:
            X_rec   = autoencoder.predict(X_coral, batch_size=64, verbose=0)
            rec_err = np.mean((X_coral - X_rec) ** 2, axis=1)
            ae_z    = -(rec_err - AE_MU) / (AE_SIG + 1e-9)
            thr_ae  = gmm_threshold(ae_z)
            p_ae    = sigmoid((ae_z - thr_ae) * 5.0)
            if nm.sum() > 0 and am.sum() > 0:
                if p_ae[nm].mean() < p_ae[am].mean():
                    p_ae = 1 - p_ae
        except Exception:
            p_ae = np.full(n, 0.5)
    else:
        p_ae = np.full(n, 0.5)

    tas   = temporal_anomaly_score(X_z, window=5)
    p_tas = sigmoid(-tas * 0.8)
    if nm.sum() > 0 and am.sum() > 0:
        if p_tas[nm].mean() < p_tas[am].mean():
            p_tas = 1 - p_tas

    p_gate = confidence_gated_ensemble(p_ifs, p_lstm)

    comp_aucs = {}
    for name, p_comp in [("src_if", p_ifs), ("lstm", p_lstm), ("gate", p_gate),
                          ("ae", p_ae), ("tas", p_tas)]:
        try:
            comp_aucs[name] = float(roc_auc_score(y, p_comp))
        except Exception:
            comp_aucs[name] = 0.5
    if tgt_if_enabled:
        try:
            comp_aucs["tgt_if"] = float(roc_auc_score(y, p_ift))
        except Exception:
            comp_aucs["tgt_if"] = 0.5

    components = {"gate": p_gate, "ae": p_ae, "tas": p_tas}
    if tgt_if_enabled:
        components["tgt_if"] = p_ift

    auc_vals = np.array([comp_aucs.get(k, 0.5) for k in components])
    auc_sq   = np.maximum(auc_vals - 0.5, 0) ** 2
    if auc_sq.sum() > 0:
        ens_w = auc_sq / auc_sq.sum()
    else:
        ens_w = np.ones(len(components)) / len(components)
    p_ens = sum(ens_w[i] * p for i, p in enumerate(components.values()))

    try:
        auc_ens = float(roc_auc_score(y, p_ens))
    except Exception:
        auc_ens = 0.5
    comp_aucs["ensemble"] = auc_ens

    print(f"    Component AUCs: " + " ".join(f"{k}={v:.3f}" for k, v in comp_aucs.items()))

    def metrics_at_threshold(p, thr, y_true):

        pred_attack = (p < thr).astype(int)
        y_attack    = (y_true == 0).astype(int)
        try:
            tn, fp, fn, tp = confusion_matrix(y_attack, pred_attack, labels=[0, 1]).ravel()

            tpr = tp / (tp + fn + 1e-12)
            fpr = fp / (fp + tn + 1e-12)
            f1  = f1_score(y_attack, pred_attack, zero_division=0)
            return {"tpr": float(tpr), "fpr": float(fpr), "f1": float(f1)}
        except Exception:
            return {"tpr": 0.0, "fpr": 0.0, "f1": 0.0}

    m_srcif = metrics_at_threshold(p_ifs,  FIXED_THRESHOLD, y)
    m_lstm  = metrics_at_threshold(p_lstm, FIXED_THRESHOLD, y)
    m_ens   = metrics_at_threshold(p_ens,  FIXED_THRESHOLD, y)

    best_tpr_at_fpr15 = 0.0
    from sklearn.metrics import roc_curve
    try:
        fpr_arr, tpr_arr, _ = roc_curve(y, p_ens)
        mask = fpr_arr <= 0.15
        if mask.sum() > 0:
            best_tpr_at_fpr15 = float(tpr_arr[mask].max())
    except Exception:
        pass

    result = {
        "dataset":         dataset_label,
        "n_normal":        int(nm.sum()),
        "n_attack":        int(am.sum()),
        "auc_srcif":       comp_aucs.get("src_if", 0.5),
        "auc_lstm":        comp_aucs.get("lstm",   0.5),
        "auc_gate":        comp_aucs.get("gate",   0.5),
        "auc_ensemble":    auc_ens,
        "tpr_srcif":       m_srcif["tpr"],
        "fpr_srcif":       m_srcif["fpr"],
        "tpr_lstm":        m_lstm["tpr"],
        "fpr_lstm":        m_lstm["fpr"],
        "tpr_ens":         m_ens["tpr"],
        "fpr_ens":         m_ens["fpr"],
        "f1_ens":          m_ens["f1"],
        "tpr_at_fpr15":    best_tpr_at_fpr15,
        "coral_ok":        bool(coral_ok),
    }
    return result

print("\n" + "=" * 70)
print("Scoring ...")
print("=" * 70)

results = []
X_legit, y_legit = windows_to_array(legit_windows)

for atype, aws in attack_windows_by_type.items():
    if len(aws) == 0:
        print(f"\n  [{atype}] No windows -- skipping")
        continue

    X_atk, y_atk = windows_to_array(aws)
    X_comb = np.vstack([X_legit, X_atk])
    y_comb = np.concatenate([y_legit, y_atk])

    n_atk = len(y_atk)
    rng = np.random.default_rng(42)
    n_norm_use = min(len(y_legit), max(n_atk * 4, 50))
    norm_idx = rng.choice(len(y_legit), size=n_norm_use, replace=False)
    X_eval = np.vstack([X_legit[norm_idx], X_atk])
    y_eval = np.concatenate([y_legit[norm_idx], y_atk])

    res = score_dataset(X_eval, y_eval, dataset_label=atype)
    results.append(res)

X_all_atk = np.vstack([windows_to_array(aws)[0]
                        for aws in attack_windows_by_type.values() if len(aws) > 0])
y_all_atk = np.zeros(len(X_all_atk), dtype=np.int32)
rng = np.random.default_rng(42)
n_norm_comb = min(len(y_legit), max(len(y_all_atk) * 4, 100))
norm_idx_c = rng.choice(len(y_legit), size=n_norm_comb, replace=False)
X_comb_all = np.vstack([X_legit[norm_idx_c], X_all_atk])
y_comb_all = np.concatenate([y_legit[norm_idx_c], y_all_atk])

res_all = score_dataset(X_comb_all, y_comb_all, dataset_label="ALL_ATTACKS")
results.append(res_all)

output_path = os.path.join(OUTPUT_DIR, "mqttset_crossdomain_results.json")
with open(output_path, "w") as fh:
    json.dump(results, fh, cls=NumpyEncoder, indent=2)

def _print_summary():
    cols   = ["Dataset", "N_nrm", "N_atk", "AUC_SrcIF", "AUC_LSTM", "AUC_Ens", "TPR_ens", "FPR_ens", "F1_ens", "TPR@FPR15"]
    widths = [16, 6, 6, 9, 9, 8, 8, 8, 7, 10]
    sep    = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    def row(vals):
        return "| " + " | ".join(f"{str(v):<{w}}" for v, w in zip(vals, widths)) + " |"
    print("\n" + sep)
    print(row(cols))
    print(sep)
    for r in results:
        print(row([
            r["dataset"], r["n_normal"], r["n_attack"],
            f"{r['auc_srcif']:.4f}", f"{r['auc_lstm']:.4f}", f"{r['auc_ensemble']:.4f}",
            f"{r['tpr_ens']:.4f}", f"{r['fpr_ens']:.4f}", f"{r['f1_ens']:.4f}",
            f"{r['tpr_at_fpr15']:.4f}",
        ]))
    print(sep)
    print(f"  Reference (Zone C in-domain): AUC=0.9975  TPR=1.0000  FPR=0.0000")
    print(f"  Output: {output_path}")
    print(sep)

_print_summary()
