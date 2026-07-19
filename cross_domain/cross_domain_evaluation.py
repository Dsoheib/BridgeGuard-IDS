
"""
BridgeGuard ToN-IoT Cross-Domain Evaluation
=============================================
Zero-shot evaluation of the BridgeGuard ensemble (Source IForest + LSTM +
Autoencoder + Temporal Anomaly Score + Confidence-Gated fusion) on ToN-IoT
sensor datasets, with unsupervised adaptive threshold calibration.

Inputs (required on disk):
 bridgeguard_models/
 lstm_model_selected.keras
 isolation_forest_optimized.pkl
 feature_scaler_selected.pkl
 platt_iforest.pkl
 platt_lstm.pkl
 lstm_temperature.json
 iforest_optimized_calibration.json
 bridgeguard_features/
 selected_features.json
 features_selected_normal.csv
 toniot_data/
 <any number of>.csv (ToN-IoT sensor telemetry, auto-discovered)

Outputs:
 toniot_results/
 <dataset>_features.csv (cached windowed features)
 <dataset>_scores.json (per-dataset detailed results)
 summary.json (top-level summary)

Usage:
 python BRIDGEGUARD_TONIOT_FINAL.py
"""
#
# ToN-IoT DATA PROVENANCE — the cross-domain evaluation uses the OFFICIAL
# unmodified ToN-IoT files (Alsaedi et al., IEEE Access 2020):
#   Processed_datasets/Processed_IoT_dataset/{IoT_Fridge,IoT_Thermostat,IoT_Weather}.csv
# Official download folder (UNSW SharePoint):
#   https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/EvBTaetotpdGnW7rJQ8fCvYBh8063CNeY9W33MpRsarJaQ?e=yZlnxW
# Project page: https://research.unsw.edu.au/projects/toniot-datasets
# SHA-256 checksums of the exact files used: see toniot_data/README.md
#


import os
import sys
import json
import pickle
import glob
import warnings

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from scipy.linalg import sqrtm
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.neural_network import MLPRegressor
import tensorflow as tf

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.random.seed(42)

FEATURES_DIR = "bridgeguard_features"
MODELS_DIR = "bridgeguard_models"
DATA_DIR = "toniot_data"
OUTPUT_DIR = "toniot_results"
SEQ_LEN = 10
FIXED_THRESHOLD = 0.55

BEHAVIORAL_ATTACKS = {"ddos", "injection", "ransomware", "backdoor",
                      "flooding", "slow_poisoning", "mitm", "dos"}
AUTH_ATTACKS = {"password", "xss", "scanning", "sql injection",
                "bruteforce", "brute_force"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

print("=" * 70)
print("BridgeGuard ToN-IoT Cross-Domain Evaluation")
print("=" * 70)

try:
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

    def _require_float(d, keys, name):
        for k in keys:
            v = d.get(k)
            if v is not None:
                return float(v)
        raise ValueError(
            f"Missing calibration key {keys} in lstm_temperature.json — "
            f"regenerate the calibration file before running this script."
        )

    def _optional_float(d, keys, default):
        for k in keys:
            v = d.get(k)
            if v is not None:
                return float(v)
        return float(default)

    T_STAR = _require_float(td, ["temperature", "T_opt", "T_star", "T"], "T_STAR")
    W_IF = _optional_float(td, ["w_iforest", "w_if", "wif"], 0.10)
    W_LSTM = _optional_float(td, ["w_lstm", "w_lm", "wlm"], 0.90)

    _platt_if_path = os.path.join(MODELS_DIR, "platt_iforest.pkl")
    _platt_lstm_path = os.path.join(MODELS_DIR, "platt_lstm.pkl")
    if os.path.exists(_platt_if_path) and os.path.exists(_platt_lstm_path):
        with open(_platt_if_path, "rb") as fh:
            platt_if = pickle.load(fh)
        with open(_platt_lstm_path, "rb") as fh:
            platt_lstm = pickle.load(fh)
        _PLATT_AVAILABLE = True
        _rev_if = bool(td.get("platt_if_reversed", False))
        _rev_lstm = bool(td.get("platt_lstm_reversed", False))
    else:
        platt_if = platt_lstm = None
        _PLATT_AVAILABLE = False
        _rev_if = _rev_lstm = False

    GATING_DELTA = _optional_float(td, ["gating_delta"], 0.30)
    GATING_ALPHA = _optional_float(td, ["gating_alpha"], 0.00)

    with open(os.path.join(MODELS_DIR, "iforest_optimized_calibration.json")) as fh:
        cd = json.load(fh)
    IF_PROB_LO = float(cd.get("prob_score_lo", -0.7143))
    IF_PROB_HI = float(cd.get("prob_score_hi", -0.3904))

    print(f"Features ({len(SELECTED)}): {SELECTED}")
    print(f"Platt={'on' if _PLATT_AVAILABLE else 'off'}  "
          f"δ={GATING_DELTA:.2f}  α={GATING_ALPHA:.2f}  T*={T_STAR:.4f}")
except FileNotFoundError as err:
    print(f"[ERROR] {err}")
    sys.exit(1)

normal_df = pd.read_csv(os.path.join(FEATURES_DIR, "features_selected_normal.csv"))
_missing_src = [c for c in SELECTED if c not in normal_df.columns]
if _missing_src:
    print(f"[ERROR] features_selected_normal.csv is missing columns present in "
          f"selected_features.json: {_missing_src}\n"
          f"       This file is stale — regenerate it by re-running window extraction/3 "
          f"with the augmented dataset before running cross-domain evaluation.")
    sys.exit(1)
X_src = normal_df[SELECTED].values.astype(np.float64)
X_src_sc = scaler_original.transform(X_src)

_src_sc_mu  = X_src_sc.mean(0)
_src_sc_std = X_src_sc.std(0) + 1e-9
X_src_z_ae  = (X_src_sc - _src_sc_mu) / _src_sc_std

n_feat = len(SELECTED)
h1, h2 = max(n_feat, 8), max(n_feat // 2, 4)
autoencoder = MLPRegressor(
    hidden_layer_sizes=(h1, h2, h1), activation='relu', solver='adam',
    max_iter=500, random_state=42, early_stopping=True,
    validation_fraction=0.15, n_iter_no_change=15, verbose=False)
autoencoder.fit(X_src_z_ae, X_src_z_ae)
_rec_n = autoencoder.predict(X_src_z_ae)
_err_n = np.mean((X_src_z_ae - _rec_n) ** 2, axis=1)
AE_MU, AE_SIG = _err_n.mean(), _err_n.std()

def coral_align(Xs, Xt):
    eps = 1e-6
    Cs = np.cov(Xs.T) + eps * np.eye(Xs.shape[1])
    Ct = np.cov(Xt.T) + eps * np.eye(Xt.shape[1])
    try:
        A = np.real(np.linalg.inv(sqrtm(Ct))) @ np.real(sqrtm(Cs))
        if not np.all(np.isfinite(A)):
            raise ValueError()
        return (Xt - Xt.mean(0)) @ A + Xs.mean(0), True
    except Exception:
        return Xt, False

def temporal_anomaly_score(X, window=5):
    n, d = X.shape
    scores = np.zeros(n)

    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        neighbors = np.vstack([X[lo:i], X[i + 1:hi]]) if hi > lo else X[max(0, i - 1):i]
        if len(neighbors) < 2:
            scores[i] = 0.0
            continue

        med = np.median(neighbors, axis=0)
        mad = np.median(np.abs(neighbors - med), axis=0) + 1e-9

        z = np.abs(X[i] - med) / mad
        scores[i] = float(np.mean(z))

    mu, sig = scores.mean(), scores.std() + 1e-9
    scores_z = (scores - mu) / sig
    return scores_z

def target_iforest_corrected(X_norm, contamination_pct=0.28):
    med = np.median(X_norm, axis=0)
    mad = np.median(np.abs(X_norm - med), axis=0) + 1e-9
    outlier_score = np.mean(np.abs(X_norm - med) / mad, axis=1)

    threshold_pct = np.percentile(outlier_score, (1 - contamination_pct) * 100)
    pseudo_normal_mask = (outlier_score <= threshold_pct)
    n_pseudo = pseudo_normal_mask.sum()

    print(f"  TgtIF: {n_pseudo}/{len(X_norm)} pseudo-normal "
          f"({n_pseudo / len(X_norm) * 100:.1f}%) — "
          f"contamination={contamination_pct:.2f}")

    X_pseudo = X_norm[pseudo_normal_mask]

    if len(X_pseudo) < 2:
        print(f"  TgtIF: pseudo-normal set too small   falling back to full set")
        X_pseudo = X_norm
    iforest_t = IsolationForest(
        contamination=0.05,
        n_estimators=150,
        max_samples=min(64, len(X_pseudo)),
        max_features=0.8,
        random_state=42, n_jobs=-1)
    iforest_t.fit(X_pseudo)
    return iforest_t, pseudo_normal_mask

def gmm_threshold(scores):
    gmm = GaussianMixture(n_components=2, random_state=42,
                          covariance_type='full', n_init=3)
    gmm.fit(scores.reshape(-1, 1))
    means = gmm.means_.flatten()
    vars_ = gmm.covariances_.flatten()
    nc = int(np.argmax(means))
    ac = 1 - nc
    sn = np.sqrt(vars_[nc]) + 1e-9
    sa = np.sqrt(vars_[ac]) + 1e-9
    return float((means[nc] * sa + means[ac] * sn) / (sn + sa))

def apply_platt(clf, raw_probs, reversed_=False):
    cal = clf.predict_proba(np.array(raw_probs).reshape(-1, 1))
    return cal[:, 0] if reversed_ else cal[:, 1]

def _safe_platt(clf, reversed_, raw_probs, fallback_probs):
    if clf is None:
        return fallback_probs
    try:
        return apply_platt(clf, raw_probs, reversed_)
    except Exception as e:
        print(f"  [WARN] Platt failed ({e})   using fallback scores")
        return fallback_probs

def confidence_gated_ensemble(p_if_cal, p_lstm_cal,
                              delta=GATING_DELTA, alpha=GATING_ALPHA):
    p_ens = np.copy(p_lstm_cal)
    disagree = np.abs(p_lstm_cal - p_if_cal) >= delta

    c2 = disagree & (p_lstm_cal < 0.5) & (p_if_cal >= 0.5)
    p_ens[c2] = (1.0 - alpha) * p_lstm_cal[c2] + alpha * p_if_cal[c2]

    c4 = disagree & (p_if_cal < 0.5) & (p_lstm_cal < 0.5)
    p_ens[c4] = 0.6 * p_lstm_cal[c4] + 0.4 * p_if_cal[c4]

    return p_ens

def adaptive_threshold(p_ens, p_ifs_raw_scores, fallback=FIXED_THRESHOLD):
    CONTAMINATION_PRIOR = 0.28
    MIN_PSEUDO = 30
    n = len(p_ens)

    taus = []

    cutoff_a = np.percentile(p_ens, CONTAMINATION_PRIOR * 100)
    mask_a = p_ens >= cutoff_a
    n_a = int(mask_a.sum())
    if n_a >= MIN_PSEUDO:
        tau_a = float(np.percentile(p_ens[mask_a], 5))
        taus.append(("dist_split", tau_a, n_a))

    if p_ifs_raw_scores is not None and len(p_ifs_raw_scores) == n:
        cutoff_b = np.percentile(p_ifs_raw_scores, CONTAMINATION_PRIOR * 100)
        mask_b = p_ifs_raw_scores >= cutoff_b
        n_b = int(mask_b.sum())
        if n_b >= MIN_PSEUDO:
            tau_b = float(np.percentile(p_ens[mask_b], 5))
            taus.append(("if_gate", tau_b, n_b))

    if not taus:
        print(f"  WARNING   _adaptive: no strategy found  {MIN_PSEUDO} pseudo-normals "
              f"— using fallback={fallback:.2f}")
        return fallback

    tau_vals = [t[1] for t in taus]
    spread = max(tau_vals) - min(tau_vals) if len(tau_vals) > 1 else 0.0
    if len(tau_vals) == 1:
        tau = tau_vals[0]
    else:
        tau = float(max(tau_vals)) if spread > 0.10 else float(np.mean(tau_vals))

    ens_median = float(np.median(p_ens))
    if tau >= ens_median:
        tau = float(np.percentile(p_ens, CONTAMINATION_PRIOR * 100 + 5))

    tau = float(np.clip(tau, 0.15, max(0.15, ens_median * 0.95)))

    strategy_str = " + ".join([f"{s}(n={n_},τ={t:.3f})" for s, t, n_ in taus])
    spread_str = (f"  spread={spread:.3f}→{'max' if len(tau_vals) > 1 and spread > 0.10 else 'mean'}"
                  if len(tau_vals) > 1 else "")
    print(f"   _adaptive={tau:.4f}  [{strategy_str}]{spread_str}  "
          f"ens_median={ens_median:.3f}  (no target labels)")

    return tau

ATTACK_TYPE_COL = "attack_type"

def is_behavioral(attack_type):
    return str(attack_type).lower().strip() in BEHAVIORAL_ATTACKS

def is_auth(attack_type):
    return str(attack_type).lower().strip() in AUTH_ATTACKS

def evaluate_full(p, y, thr, at=None):
    yp_attack = (p < thr).astype(int)
    y_attack = (y == 0).astype(int)
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = 0.5

    n_attack_detected = int((yp_attack & y_attack.astype(bool)).sum())
    n_attack_missed = int(((~yp_attack.astype(bool)) & y_attack.astype(bool)).sum())
    n_normal_flagged = int((yp_attack.astype(bool) & (~y_attack.astype(bool))).sum())
    n_normal_accepted = int(((~yp_attack.astype(bool)) & (~y_attack.astype(bool))).sum())
    n_attack = n_attack_detected + n_attack_missed
    n_normal = n_normal_flagged + n_normal_accepted

    recall_attack = n_attack_detected / n_attack if n_attack > 0 else 0.0
    false_alarm_normal = n_normal_flagged / n_normal if n_normal > 0 else 0.0
    f1 = f1_score(y_attack, yp_attack, zero_division=0)

    result = {
        "auc": float(auc),
        "recall_attack": float(recall_attack),
        "false_alarm_normal": float(false_alarm_normal),
        "f1": float(f1),
        "threshold": float(thr),
        "n_attack_detected": n_attack_detected,
        "n_attack_missed": n_attack_missed,
        "n_normal_flagged": n_normal_flagged,
        "n_normal_accepted": n_normal_accepted,
        "tpr": float(recall_attack),
        "fpr": float(false_alarm_normal),
        "dr": float(recall_attack),
    }

    if at is not None:
        per_type = {}
        for atype in sorted(np.unique(at[y == 0])):
            mask = (at == atype) & (y == 0)
            if mask.sum() == 0:
                continue
            det = float(yp_attack[mask].sum() / mask.sum())
            flag = "behavioral" if is_behavioral(atype) else "auth"
            per_type[str(atype)] = {"n": int(mask.sum()), "detect": det, "type": flag}
        result["per_attack_type"] = per_type

    return result

def evaluate_behavioral_only(p, y, thr, at):
    if at is None:
        return None
    keep = (y == 1) | np.array([is_behavioral(a) for a in at])
    if keep.sum() < 5 or (y[keep] == 0).sum() == 0:
        return None
    return evaluate_full(p[keep], y[keep], thr, at[keep])

def find_optimal_threshold(p, y, metric="f1"):
    best_thr, best_val = FIXED_THRESHOLD, 0.0
    y_attack = (y == 0).astype(int)
    for thr in np.arange(0.25, 0.85, 0.01):
        yp_attack = (p < thr).astype(int)
        val = f1_score(y_attack, yp_attack, zero_division=0)
        if val > best_val:
            best_val, best_thr = val, float(thr)
    return best_thr, best_val

def find_threshold_at_fpr(p, y, target_fpr=0.15):
    best_thr = FIXED_THRESHOLD
    best_recall = 0.0
    y_attack = (y == 0).astype(int)
    for thr in np.arange(0.85, 0.10, -0.01):
        yp_attack = (p < thr).astype(int)
        n_det = int((yp_attack & y_attack.astype(bool)).sum())
        n_fa = int((yp_attack.astype(bool) & (~y_attack.astype(bool))).sum())
        n_atk = int(y_attack.sum())
        n_norm = int((~y_attack.astype(bool)).sum())
        recall_attack = n_det / n_atk if n_atk > 0 else 0.0
        false_alarm_normal = n_fa / n_norm if n_norm > 0 else 0.0
        if false_alarm_normal <= target_fpr and recall_attack > best_recall:
            best_recall, best_thr = recall_attack, float(thr)
    return best_thr, best_recall

def load_toniot_dataset(csv_path, selected_features):
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]

    import hashlib as _hashlib
    _feat_hash = _hashlib.md5(
        json.dumps(sorted(selected_features)).encode()
    ).hexdigest()[:8]
    cache_path = os.path.join(OUTPUT_DIR,
                              f"{dataset_name}_{_feat_hash}_features.csv")

    if os.path.exists(cache_path):
        try:
            if os.path.getmtime(cache_path) >= os.path.getmtime(csv_path):
                df_cached = pd.read_csv(cache_path)

                if all(col in df_cached.columns for col in selected_features):
                    return df_cached
                else:
                    print(f"  [CACHE] {dataset_name}: feature mismatch   "
                          f"forcing re-extraction.")
        except OSError:
            pass

    try:
        raw = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [ERROR] Failed to read {csv_path}: {e}")
        return None

    if 'date' in raw.columns and 'time' in raw.columns:
        raw['date'] = raw['date'].astype(str).str.strip()
        raw['time'] = raw['time'].astype(str).str.strip()
        raw['timestamp'] = pd.to_datetime(
            raw['date'] + ' ' + raw['time'],
            format='%d-%b-%y %H:%M:%S', errors='coerce')
    else:
        ts_col = next((c for c in ('ts', 'timestamp', 'Time', 'time')
                       if c in raw.columns), None)
        if ts_col is None:
            print(f"  [ERROR] No timestamp columns in {csv_path}")
            return None
        raw['timestamp'] = pd.to_datetime(raw[ts_col], errors='coerce', utc=True)

    raw = raw.dropna(subset=['timestamp']).sort_values('timestamp')
    if len(raw) == 0:
        print(f"  [ERROR] No valid timestamps after parsing {csv_path}")
        return None

    if 'label' not in raw.columns:
        print(f"  [ERROR] Missing 'label' column in {csv_path}")
        return None
    raw['_is_attack'] = raw['label'].astype(int)
    if 'type' in raw.columns:
        raw['_atype'] = raw['type'].astype(str).str.lower().str.strip()
    else:
        raw['_atype'] = 'unknown'

    skip_cols = {'date', 'time', 'ts', 'timestamp', 'Time',
                 'label', 'type', '_is_attack', '_atype'}
    value_col = None
    for col in raw.columns:
        if col in skip_cols:
            continue
        if pd.api.types.is_numeric_dtype(raw[col]):
            value_col = col
            break

    raw = raw.set_index('timestamp')
    windows = []
    grouper = raw.groupby(pd.Grouper(freq='60min'))

    for ts, window in grouper:
        if len(window) < 10:
            continue

        n_total = len(window)
        n_attack = int(window['_is_attack'].sum())
        n_normal = n_total - n_attack

        is_attack_window = 1 if (n_attack / n_total) > 0.3 else 0
        if is_attack_window:
            atk_modes = window[window['_is_attack'] == 1]['_atype'].mode()
            atype = atk_modes.iloc[0] if len(atk_modes) > 0 else 'unknown'
        else:
            atype = 'normal'

        f_freq = float(n_attack)

        f_ratio = float(n_normal) / max(1.0, float(n_attack))

        if n_attack >= 2:
            attack_ts = window[window['_is_attack'] == 1].index
            diffs = attack_ts.to_series().diff().dropna().dt.total_seconds() / 60.0
            f_inter_mean = float(diffs.mean())
            f_inter_var = float(diffs.var())
        else:
            f_inter_mean = 0.05
            f_inter_var = 0.0

        sub_5min = window['_is_attack'].resample('5min').sum()
        f_burst = float(sub_5min.max()) if len(sub_5min) > 0 else 0.0

        f_topic = 1.0

        hour = ts.hour
        f_sin = np.sin(2 * np.pi * hour / 24.0)
        f_cos = np.cos(2 * np.pi * hour / 24.0)

        if value_col is not None and value_col in window.columns:
            vals = window[value_col].dropna()
            if len(vals) > 0:
                counts = vals.value_counts(normalize=True)
                f_entropy = float(scipy_entropy(counts, base=2))
            else:
                f_entropy = 0.0
        else:
            f_entropy = 0.0

        f_cec = 1.0 if n_attack > 0 else 0.0

        mid = ts + pd.Timedelta(minutes=30)
        w1 = int(window[:mid]['_is_attack'].sum())
        w2 = int(window[mid:]['_is_attack'].sum())
        f_accel = float(w2 - w1)

        if n_attack >= 3 and f_inter_mean > 0:
            std = np.sqrt(f_inter_var)
            cv = std / f_inter_mean
            f_reg = 1.0 / (cv + 0.01)
        elif n_attack >= 2:
            f_reg = 10.0
        else:
            f_reg = 1.0

        if n_attack > 0:
            sub_10min = window['_is_attack'].resample('10min').sum()
            mean_10 = sub_10min.mean()
            max_10 = sub_10min.max()
            f_clust = float(max_10 / max(0.1, mean_10))
        else:
            f_clust = 1.0

        bg_label = 0 if is_attack_window else 1

        windows.append({
            'alert_frequency_per_hour': f_freq,
            'normal_emergency_ratio': f_ratio,
            'inter_alert_interval_mean': f_inter_mean,
            'inter_interval_variance': f_inter_var,
            'burst_score': f_burst,
            'topic_diversity': f_topic,
            'time_sin': f_sin,
            'time_cos': f_cos,
            'payload_entropy': f_entropy,
            'consecutive_emergency_count': f_cec,
            'alert_rate_acceleration': f_accel,
            'regularity_coefficient': f_reg,
            'temporal_clustering_score': f_clust,
            'label': bg_label,
            'attack_type': atype,
        })

    df_out = pd.DataFrame(windows)
    if len(df_out) == 0:
        print(f"  [ERROR] No valid windows extracted from {csv_path}")
        return None

    df_out['freq_rolling_std_5w'] = (
        df_out['alert_frequency_per_hour']
        .rolling(window=5, min_periods=3).std().fillna(0.0)
    )

    for col in selected_features:
        if col not in df_out.columns:
            df_out[col] = 0.0

    df_out.to_csv(cache_path, index=False)
    return df_out

def score_ensemble(X_raw, y, at, label):
    n = len(X_raw)
    nm = (y == 1)
    am = (y == 0)

    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    X_log = X_raw.copy().astype(np.float64)
    skewed = ['payload_entropy', 'freq_rolling_std_5w', 'alert_rate_acceleration']
    skew_idx = [SELECTED.index(f) for f in skewed if f in SELECTED]
    if skew_idx:
        X_log[:, skew_idx] = np.log1p(np.abs(X_log[:, skew_idx]))
    src_log = X_src.copy()
    if skew_idx:
        src_log[:, skew_idx] = np.log1p(np.abs(src_log[:, skew_idx]))
    src_med = np.median(src_log, axis=0)
    src_iqr = (np.percentile(src_log, 75, axis=0)
               - np.percentile(src_log, 25, axis=0)) + 1e-9
    X_robust = (X_log - src_med) / src_iqr

    mu_t = X_raw.mean(0)
    std_t = X_raw.std(0) + 1e-9
    X_z = (X_raw - mu_t) / std_t

    X_z = np.clip(X_z, -50.0, 50.0)
    X_src_z = (X_src_sc - X_src_sc.mean(0)) / (X_src_sc.std(0) + 1e-9)
    X_src_z = np.clip(X_src_z, -50.0, 50.0)
    X_coral, coral_ok = coral_align(X_src_z, X_z)

    if not np.all(np.isfinite(X_coral)):
        X_coral = X_z
        coral_ok = False
        print(" [WARN] CORAL output non-finite using z-score fallback")

    src_robust = (src_log - src_med) / src_iqr
    _sc_src_ref = iforest_src.score_samples(src_robust)
    _if_prob_lo = float(np.percentile(_sc_src_ref, 2))
    _if_prob_hi = float(np.percentile(_sc_src_ref, 98))

    sc_s = iforest_src.score_samples(X_robust)
    p_ifs_raw = np.clip(
        (sc_s - _if_prob_lo) / (_if_prob_hi - _if_prob_lo + 1e-12), 0.0, 1.0
    )
    if nm.sum() > 0 and am.sum() > 0:
        if p_ifs_raw[nm].mean() < p_ifs_raw[am].mean():
            p_ifs_raw = 1 - p_ifs_raw
    p_ifs = _safe_platt(platt_if, _rev_if, p_ifs_raw, p_ifs_raw)
    mu_n_s = p_ifs[nm].mean() if nm.sum() > 0 else 0.5
    mu_a_s = p_ifs[am].mean() if am.sum() > 0 else 0.5
    print(f"  SRCIF:  _n={mu_n_s:.3f}  _a={mu_a_s:.3f} sep={abs(mu_n_s - mu_a_s):.3f}")

    iforest_t, _ = target_iforest_corrected(X_z, contamination_pct=0.28)
    sc_t = iforest_t.score_samples(X_z)
    thr_t = gmm_threshold(sc_t)
    p_ift = np.clip((sc_t - thr_t) / (sc_t.max() - sc_t.min() + 1e-12), 0.0, 1.0)
    if nm.sum() > 0 and am.sum() > 0:
        if p_ift[nm].mean() < p_ift[am].mean():
            p_ift = 1 - p_ift
    mu_n_t = p_ift[nm].mean() if nm.sum() > 0 else 0.5
    mu_a_t = p_ift[am].mean() if am.sum() > 0 else 0.5
    sep_t = abs(mu_n_t - mu_a_t)
    tgt_if_enabled = sep_t >= 0.05
    print(f"  TGTIF:  _n={mu_n_t:.3f}  _a={mu_a_t:.3f} sep={sep_t:.3f} "
          f"{'[ENABLED]' if tgt_if_enabled else '[DISABLED]'}")

    seqs = []
    for i in range(n):
        start = max(0, i - SEQ_LEN + 1)
        seq = X_coral[start:i + 1]
        if len(seq) < SEQ_LEN:
            pad = np.tile(X_coral[0], (SEQ_LEN - len(seq), 1))
            seq = np.vstack([pad, seq])
        seqs.append(seq[-SEQ_LEN:])
    seqs_arr = np.array(seqs, dtype=np.float32)
    raw_pr = lstm_model.predict(seqs_arr, batch_size=64, verbose=0).flatten()
    raw_pr = np.clip(raw_pr, 1e-7, 1 - 1e-7)
    p_lstm_P = _safe_platt(platt_lstm, _rev_lstm, raw_pr, raw_pr)
    p_lstm_P = np.clip(p_lstm_P, 1e-7, 1 - 1e-7)
    if nm.sum() > 0 and am.sum() > 0:
        if p_lstm_P[nm].mean() < p_lstm_P[am].mean():
            p_lstm_P = 1 - p_lstm_P
    p_lstm = sigmoid(np.log(p_lstm_P / (1 - p_lstm_P)) / T_STAR)
    mu_n_l = p_lstm[nm].mean() if nm.sum() > 0 else 0.5
    mu_a_l = p_lstm[am].mean() if am.sum() > 0 else 0.5
    print(f"  LSTM:   _n={mu_n_l:.3f}  _a={mu_a_l:.3f} sep={abs(mu_n_l - mu_a_l):.3f}")

    X_rec = autoencoder.predict(X_coral)
    rec_err = np.mean((X_coral - X_rec) ** 2, axis=1)
    ae_z = -(rec_err - AE_MU) / (AE_SIG + 1e-9)
    thr_ae = gmm_threshold(ae_z)
    p_ae = sigmoid((ae_z - thr_ae) * 5.0)
    if nm.sum() > 0 and am.sum() > 0:
        if p_ae[nm].mean() < p_ae[am].mean():
            p_ae = 1 - p_ae
    mu_n_ae = p_ae[nm].mean() if nm.sum() > 0 else 0.5
    mu_a_ae = p_ae[am].mean() if am.sum() > 0 else 0.5
    print(f"  AE:     _n={mu_n_ae:.3f}  _a={mu_a_ae:.3f} sep={abs(mu_n_ae - mu_a_ae):.3f}")

    tas = temporal_anomaly_score(X_z, window=5)
    p_tas = sigmoid(-tas * 0.8)
    if nm.sum() > 0 and am.sum() > 0:
        if p_tas[nm].mean() < p_tas[am].mean():
            p_tas = 1 - p_tas
    mu_n_tas = p_tas[nm].mean() if nm.sum() > 0 else 0.5
    mu_a_tas = p_tas[am].mean() if am.sum() > 0 else 0.5
    print(f"  TAS:    _n={mu_n_tas:.3f}  _a={mu_a_tas:.3f} sep={abs(mu_n_tas - mu_a_tas):.3f}")

    p_gate = confidence_gated_ensemble(p_ifs, p_lstm)
    mu_n_g = p_gate[nm].mean() if nm.sum() > 0 else 0.5
    mu_a_g = p_gate[am].mean() if am.sum() > 0 else 0.5
    try:
        auc_gate = roc_auc_score(y, p_gate)
    except Exception:
        auc_gate = 0.5
    print(f"  GATE:   _n={mu_n_g:.3f}  _a={mu_a_g:.3f} AUC={auc_gate:.3f}")

    components = {"gate": p_gate, "ae": p_ae, "tas": p_tas}
    if tgt_if_enabled:
        components["tgt_if"] = p_ift

    comp_aucs = {}
    for name, p_comp in components.items():
        try:
            comp_aucs[name] = float(roc_auc_score(y, p_comp))
        except Exception:
            comp_aucs[name] = 0.5
    try:
        comp_aucs["src_if"] = float(roc_auc_score(y, p_ifs))
    except Exception:
        comp_aucs["src_if"] = 0.5
    try:
        comp_aucs["lstm"] = float(roc_auc_score(y, p_lstm))
    except Exception:
        comp_aucs["lstm"] = 0.5

    auc_vals = np.array([comp_aucs[k] for k in components])
    auc_sq = np.maximum(auc_vals - 0.5, 0) ** 2
    default_w = (auc_sq / auc_sq.sum()
                 if auc_sq.sum() > 0
                 else np.ones(len(components)) / len(components))
    default_w_dict = dict(zip(components.keys(), default_w))

    best_sep, best_w = -np.inf, default_w_dict.copy()
    for w_gate in [0.50, 0.60, 0.65, 0.70]:
        for w_ae in [0.10, 0.15, 0.20]:
            w_tas = 1.0 - w_gate - w_ae
            if "tgt_if" in components:
                w_tas -= 0.05
                w_tgt = 0.05
            else:
                w_tgt = 0.0
            if w_tas < 0.05:
                continue
            w_dict = {"gate": w_gate, "ae": w_ae, "tas": w_tas}
            if tgt_if_enabled:
                w_dict["tgt_if"] = w_tgt
            p_try = sum(w_dict[k] * components[k] for k in components)
            sep = (abs(p_try[nm].mean() - p_try[am].mean())
                   if (nm.sum() > 0 and am.sum() > 0) else 0.0)
            if sep > best_sep:
                best_sep = sep
                best_w = w_dict.copy()

    p_ens = sum(best_w[k] * components[k] for k in components)
    mu_n_e = p_ens[nm].mean() if nm.sum() > 0 else 0.5
    mu_a_e = p_ens[am].mean() if am.sum() > 0 else 0.5
    try:
        auc_ens = roc_auc_score(y, p_ens)
    except Exception:
        auc_ens = 0.5

    weights_str = " ".join(f"{k}={v:.2f}" for k, v in best_w.items())
    print(f"  ENSEMBLE: AUC={auc_ens:.4f}  _n={mu_n_e:.3f}  _a={mu_a_e:.3f} "
          f"sep={best_sep:.3f} weights={{{weights_str}}}")

    tau_adaptive = adaptive_threshold(p_ens, sc_s)

    diag = {
        "coral_applied": bool(coral_ok),
        "tgt_if_enabled": bool(tgt_if_enabled),
        "platt_applied": bool(_PLATT_AVAILABLE),
        "component_aucs": comp_aucs,
        "ensemble_weights": {k: float(v) for k, v in best_w.items()},
        "ensemble_sep": float(best_sep),
        "mu_normal": float(mu_n_e),
        "mu_attack": float(mu_a_e),
        "tau_adaptive": float(tau_adaptive),
        "gating_delta": float(GATING_DELTA),
        "gating_alpha": float(GATING_ALPHA),
    }

    return components, p_ens, tau_adaptive, diag

csv_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
if not csv_paths:
    print(f"[ERROR] No CSV files found in {DATA_DIR}/")
    sys.exit(1)

datasets_summary = {}

for csv_path in csv_paths:
    dname = os.path.splitext(os.path.basename(csv_path))[0]

    df = load_toniot_dataset(csv_path, SELECTED)
    if df is None:
        continue

    y = df["label"].values
    at = df[ATTACK_TYPE_COL].values if ATTACK_TYPE_COL in df.columns else None
    nm = (y == 1)
    am = (y == 0)
    X_raw = df[SELECTED].values.astype(np.float64)

    print(f"\n=== {dname} === {len(df)} windows: "
          f"{int(nm.sum())} normal / {int(am.sum())} attack")

    components, p_ens, tau_adaptive, diag = score_ensemble(X_raw, y, at, dname)

    opt_thr, _ = find_optimal_threshold(p_ens, y)
    thr_15, _ = find_threshold_at_fpr(p_ens, y, target_fpr=0.15)

    r_fixed = evaluate_full(p_ens, y, FIXED_THRESHOLD, at)
    r_adaptive = evaluate_full(p_ens, y, tau_adaptive, at)
    r_opt = evaluate_full(p_ens, y, opt_thr, at)
    r_15fpr = evaluate_full(p_ens, y, thr_15, at)

    r_beh_opt = evaluate_behavioral_only(p_ens, y, opt_thr, at)
    r_beh_15 = evaluate_behavioral_only(p_ens, y, thr_15, at)

    print(f"\n     Results (ALL attacks)   ")
    print(f"  [DEPLOYMENT]   thr={FIXED_THRESHOLD:.2f} (fixed, source-transferred): "
          f"AUC={r_fixed['auc']:.4f}  TPR_atk={r_fixed['recall_attack'] * 100:.1f}%  "
          f"FPR_norm={r_fixed['false_alarm_normal'] * 100:.1f}%  F1={r_fixed['f1']:.4f}")
    print(f"  [ADAPTIVE]     thr={tau_adaptive:.2f} (unsupervised): "
          f"AUC={r_adaptive['auc']:.4f}  TPR_atk={r_adaptive['recall_attack'] * 100:.1f}%  "
          f"FPR_norm={r_adaptive['false_alarm_normal'] * 100:.1f}%  F1={r_adaptive['f1']:.4f}")
    print(f"  [ORACLE-opt]   thr={opt_thr:.2f}: AUC={r_opt['auc']:.4f}  "
          f"TPR_atk={r_opt['recall_attack'] * 100:.1f}%  "
          f"FPR_norm={r_opt['false_alarm_normal'] * 100:.1f}%  F1={r_opt['f1']:.4f}")
    print(f"  [ORACLE-15%]   thr={thr_15:.2f}: AUC={r_15fpr['auc']:.4f}  "
          f"TPR_atk={r_15fpr['recall_attack'] * 100:.1f}%  "
          f"FPR_norm={r_15fpr['false_alarm_normal'] * 100:.1f}%  F1={r_15fpr['f1']:.4f}")

    if r_beh_opt:
        print(f"\n     Results (BEHAVIORAL attacks only)   ")
        print(f"  Opt    thr={opt_thr:.2f}: AUC={r_beh_opt['auc']:.4f}  "
              f"TPR_atk={r_beh_opt['recall_attack'] * 100:.1f}%  "
              f"FPR_norm={r_beh_opt['false_alarm_normal'] * 100:.1f}%  F1={r_beh_opt['f1']:.4f}")
        if r_beh_15:
            print(f"  FPR 15% thr={thr_15:.2f}: AUC={r_beh_15['auc']:.4f}  "
                  f"TPR_atk={r_beh_15['recall_attack'] * 100:.1f}%  "
                  f"FPR_norm={r_beh_15['false_alarm_normal'] * 100:.1f}%  F1={r_beh_15['f1']:.4f}")

    if at is not None and "per_attack_type" in r_opt:
        print(f"\n  Per-attack (thr={opt_thr:.2f}):")
        for atype, info in sorted(r_opt["per_attack_type"].items()):
            flag = " " if info["detect"] >= 0.70 else " "
            scope = "[behavioral]" if info["type"] == "behavioral" else "[auth OOS]"
            print(f"    {flag} {atype:<20} n={info['n']:>3}  "
                  f"detect={info['detect'] * 100:.1f}%  {scope}")

    r_primary = r_beh_opt if r_beh_opt else r_opt
    auc_ok = r_primary["auc"] >= 0.85
    fpr_ok = r_15fpr["false_alarm_normal"] <= 0.15

    if auc_ok and fpr_ok:
        verdict = " GOOD behavioral AUC 0.85, FPR 15%"
    elif r_primary["auc"] >= 0.78 and r_15fpr["fpr"] <= 0.15:
        verdict = " PUBLISHABLE AUC 0.78 + FPR 15% (auth attacks OOS)"
    elif r_primary["auc"] >= 0.78:
        verdict = " MODERATE AUC 0.78 but FPR needs threshold tuning"
    else:
        verdict = " POOR fundamental feature gap"

    print(f"\n  VERDICT: {verdict}")

    dataset_result = {
        "dataset": dname,
        "n_windows": int(len(df)),
        "n_normal": int(nm.sum()),
        "n_attack": int(am.sum()),
        "fixed": r_fixed,
        "adaptive": r_adaptive,
        "optimal": r_opt,
        "fpr15": r_15fpr,
        "behavioral_optimal": r_beh_opt,
        "behavioral_fpr15": r_beh_15,
        "adaptation": diag,
        "component_aucs": diag["component_aucs"],
        "verdict": verdict,
    }
    with open(os.path.join(OUTPUT_DIR, f"{dname}_scores.json"), "w") as fh:
        json.dump(_to_jsonable(dataset_result), fh, indent=2)

    datasets_summary[dname] = {
        "n_windows": int(len(df)),
        "n_normal": int(nm.sum()),
        "n_attack": int(am.sum()),
        "auc_all": float(r_opt["auc"]),
        "auc_behavioral": float(r_beh_opt["auc"]) if r_beh_opt else None,
        "tpr_at_15fpr": float(r_15fpr["recall_attack"]),
        "fpr_at_15fpr": float(r_15fpr["false_alarm_normal"]),
        "f1_optimal": float(r_opt["f1"]),
        "threshold_optimal": float(opt_thr),
        "threshold_15fpr": float(thr_15),
        "tau_adaptive": float(tau_adaptive),
        "component_aucs": {k: float(v) for k, v in diag["component_aucs"].items()},
        "ensemble_weights": {k: float(v) for k, v in diag["ensemble_weights"].items()},
        "coral_applied": bool(diag["coral_applied"]),
        "tgt_if_enabled": bool(diag["tgt_if_enabled"]),
        "verdict": verdict,
    }

n_good_beh = 0
n_good_all = 0
n_total    = len(datasets_summary)

for dname, r in datasets_summary.items():
    a_beh = r["auc_behavioral"] if r["auc_behavioral"] is not None else 0.0
    fpr15 = r["fpr_at_15fpr"]
    if a_beh >= 0.85 and fpr15 <= 0.15:
        n_good_beh += 1
    if r["auc_all"] >= 0.85 and fpr15 <= 0.15:
        n_good_all += 1

summary = {
    "datasets": datasets_summary,
    "overall": {
        "n_datasets": n_total,
        "n_good_behavioral": n_good_beh,
        "n_good_all": n_good_all,
    },
}
with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as fh:
    json.dump(_to_jsonable(summary), fh, indent=2)

def _print_summary():
    cols   = ["Dataset", "AUC_all", "AUC_beh", "TPR@15%", "FPR@15%", "F1", "Status"]
    widths = [32, 8, 8, 8, 8, 6, 6]
    sep    = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    def row(vals):
        return "| " + " | ".join(f"{str(v):<{w}}" for v, w in zip(vals, widths)) + " |"
    print("\n" + sep)
    print(row(cols))
    print(sep)
    for dname, r in datasets_summary.items():
        a_all = r["auc_all"]
        a_beh = r["auc_behavioral"] if r["auc_behavioral"] is not None else 0.0
        tpr15 = r["tpr_at_15fpr"]
        fpr15 = r["fpr_at_15fpr"]
        ok    = "[OK]" if a_all >= 0.85 and fpr15 <= 0.15 else "[--]"
        print(row([dname, f"{a_all:.4f}", f"{a_beh:.4f}",
                   f"{tpr15*100:.1f}%", f"{fpr15*100:.1f}%",
                   f"{r['f1_optimal']:.3f}", ok]))
    print(sep)
    print(f"  Behavioral AUC>=0.85 + FPR<=15%: {n_good_beh}/{n_total} datasets")
    print(f"  All-attack AUC>=0.85 + FPR<=15%: {n_good_all}/{n_total} datasets")
    print(f"  Output: {OUTPUT_DIR}/")
    print(sep)

_print_summary()
