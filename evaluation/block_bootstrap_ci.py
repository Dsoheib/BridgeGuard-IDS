
"""
Block Bootstrap CI BridgeGuard Task 1.3
==========================================

Computes block bootstrap confidence intervals (b=6, n=10,000 resamples)
for all primary metrics reported in the manuscript, and compares them
against the naive Clopper-Pearson / asymptotic CIs.

Background:
 Temporal autocorrelation analysis of Zone C windows shows |r| < 0.1
 beyond lag 6, motivating a block size b=6. Naive CIs (Clopper-Pearson
 for proportions, Wilson for AUC) assume i.i.d. observations; block
 bootstrap is the correct choice when temporal dependence exists.

Usage:
 python evaluation/block_bootstrap_ci.py

Environment overrides:
 BG_FEATURES_DIR path to bridgeguard_features/ (default: bridgeguard_features)
 BG_MODELS_DIR path to bridgeguard_models/ (default: bridgeguard_models)
 BG_SEQ_LEN LSTM sequence length (default: 10)
 BG_BLOCK_SIZE block size for bootstrap (default: 6)
 BG_N_BOOTSTRAP number of resamples (default: 10000)

Produces:
 BLOCK_BOOTSTRAP_REPORT.md comparison table (block BB vs CP)
 block_bootstrap_results.json full CI data
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

FEATURES_DIR = os.getenv("BG_FEATURES_DIR", "bridgeguard_features")
MODELS_DIR   = os.getenv("BG_MODELS_DIR",   "bridgeguard_models")
SEQ_LEN      = int(os.getenv("BG_SEQ_LEN",       "10"))
BLOCK_SIZE   = int(os.getenv("BG_BLOCK_SIZE",     "6"))
N_BOOTSTRAP  = int(os.getenv("BG_N_BOOTSTRAP",    "10000"))
RANDOM_SEED  = int(os.getenv("BG_SEED",           "42"))

CALIB_START  = 0.60
CALIB_END    = 0.80
DECISION_THR = 0.50
MAX_GAP_MIN  = 120

OUTPUT_JSON  = "block_bootstrap_results.json"
OUTPUT_MD    = "BLOCK_BOOTSTRAP_REPORT.md"

def build_stratified_zone_c(df, ts_col=None,
                             calib_start=CALIB_START, calib_end=CALIB_END):
    parts_b, parts_c = [], []
    for cls in (df["attack_type"].unique() if "attack_type" in df.columns else [None]):
        sub = df[df["attack_type"] == cls].copy() if cls is not None else df.copy()
        if ts_col and ts_col in sub.columns:
            sub = sub.sort_values(ts_col).reset_index(drop=True)
        n = len(sub)
        parts_b.append(sub.iloc[int(n * calib_start): int(n * calib_end)])
        parts_c.append(sub.iloc[int(n * calib_end):])
    return pd.concat(parts_b).reset_index(drop=True), \
           pd.concat(parts_c).reset_index(drop=True)

def make_sequences(X_sc, seq_len=SEQ_LEN, timestamps=None, max_gap_min=MAX_GAP_MIN):
    n = len(X_sc)
    seqs, ends = [], []
    i = seq_len - 1
    while i < n:
        start = i - seq_len + 1
        if timestamps is not None:
            ts = pd.to_datetime(timestamps)
            gap_ok = True
            for j in range(start, i):
                diff = (ts.iloc[j + 1] - ts.iloc[j]).total_seconds() / 60.0
                if diff > max_gap_min:
                    gap_ok = False
                    break
            if not gap_ok:
                i += 1
                continue
        seqs.append(X_sc[start:i + 1])
        ends.append(i)
        i += seq_len
    if not seqs:
        return np.empty((0, seq_len, X_sc.shape[1]), dtype=np.float32), []
    return np.array(seqs, dtype=np.float32), ends

def block_bootstrap_ci(scores, labels, metric_fn, block_size=BLOCK_SIZE,
                        n_resamples=N_BOOTSTRAP, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.RandomState(RANDOM_SEED)

    n = len(scores)

    blocks = []
    for start in range(0, n - block_size + 1, block_size):
        blocks.append((start, start + block_size))

    last = blocks[-1][1] if blocks else 0
    if last < n:
        blocks.append((last, n))

    n_blocks = len(blocks)
    observed = metric_fn(scores, labels)

    boot_stats = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        chosen = rng.choice(n_blocks, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(blocks[c][0], blocks[c][1]) for c in chosen])

        idx = idx[:n] if len(idx) >= n else np.resize(idx, n)
        try:
            boot_stats[b] = metric_fn(scores[idx], labels[idx])
        except Exception:
            boot_stats[b] = np.nan

    boot_stats = boot_stats[np.isfinite(boot_stats)]
    ci_lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return {
        "observed": float(observed),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "std_boot": float(boot_stats.std()),
        "n_valid_resamples": len(boot_stats),
    }

def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    if n == 0:
        return {"ci_lo": 0.0, "ci_hi": 1.0}
    lo = float(beta.ppf(alpha / 2,     k,     n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(1 - alpha / 2, k + 1, n - k))     if k < n else 1.0
    return {"ci_lo": lo, "ci_hi": hi}

def auc_metric(scores, labels):
    from sklearn.metrics import roc_auc_score

    u = np.unique(labels)
    if len(u) < 2:
        return np.nan
    return float(roc_auc_score(labels, scores))

def auprc_metric(scores, labels):
    from sklearn.metrics import average_precision_score
    u = np.unique(labels)
    if len(u) < 2:
        return np.nan

    return float(average_precision_score(1 - labels, 1 - scores))

def tpr_metric(scores, labels, thr=DECISION_THR):
    pred_attack = (scores < thr).astype(int)
    true_attack = (labels == 0).astype(int)
    tp = int((pred_attack & true_attack).sum())
    fn = int(((1 - pred_attack) & true_attack).sum())
    return tp / max(tp + fn, 1)

def fpr_metric(scores, labels, thr=DECISION_THR):
    pred_attack = (scores < thr).astype(int)
    true_normal = (labels == 1).astype(int)
    fp = int((pred_attack & true_normal).sum())
    tn = int(((1 - pred_attack) & true_normal).sum())
    return fp / max(fp + tn, 1)

def apply_platt(clf, raw_probs, reversed_=False):
    if clf is None:
        return raw_probs
    cal = clf.predict_proba(np.array(raw_probs).reshape(-1, 1))
    return cal[:, 0] if reversed_ else cal[:, 1]

def apply_temperature(p, T):
    p_c = np.clip(p, 1e-7, 1 - 1e-7)
    return 1.0 / (1.0 + np.exp(-np.log(p_c / (1 - p_c)) / T))

def confidence_gated_ensemble(p_if, p_lstm, delta=0.30, alpha=0.00):
    p_ens = np.copy(p_lstm)
    disagree = np.abs(p_lstm - p_if) >= delta
    c2 = disagree & (p_lstm < 0.5) & (p_if >= 0.5)
    p_ens[c2] = (1 - alpha) * p_lstm[c2] + alpha * p_if[c2]
    c4 = disagree & (p_if < 0.5) & (p_lstm < 0.5)
    p_ens[c4] = 0.6 * p_lstm[c4] + 0.4 * p_if[c4]
    return p_ens

def check_autocorrelation(residuals, max_lag=15):
    n = len(residuals)
    r = residuals - residuals.mean()
    acf = {}
    c0 = np.dot(r, r) / n
    for lag in range(1, min(max_lag + 1, n)):
        acf[lag] = float(np.dot(r[:-lag], r[lag:]) / (n * c0))
    return acf

def write_report(results, acf_info, output_path):
    lines = [
        "# Block Bootstrap CI Report BridgeGuard Task 1.3",
        "",
        f"Block size b={BLOCK_SIZE} (matching autocorrelation horizon |r|<0.1 beyond lag {BLOCK_SIZE})",
        f"Resamples: n={N_BOOTSTRAP:,}",
        f"Confidence level: 95%",
        "",
        "---",
        "",
        "## Autocorrelation Check (Zone C ensemble residuals)",
        "",
        "| Lag | r |",
        "|-----|---|",
    ]
    for lag, r in acf_info.items():
        lines.append(f"| {lag} | {r:+.4f} |")

    lines += [
        "",
        f"Block size b={BLOCK_SIZE} justified: first lag with |r| < 0.1 "
        f"should be lag {BLOCK_SIZE} or earlier.",
        "",
        "---",
        "",
        "## Comparison Table: Naive CI vs Block Bootstrap CI",
        "",
        "| Metric | Observed | Naive CI lo | Naive CI hi | Block BB lo | Block BB hi | "
        "Naive width | BB width | Width ratio |",
        "|--------|----------|-------------|-------------|-------------|-------------|"
        "------------|----------|-------------|",
    ]

    metrics_order = ["AUC", "AUPRC", "TPR", "FPR"]
    for m in metrics_order:
        if m not in results:
            continue
        d = results[m]
        obs   = d["observed"]
        n_lo  = d["naive_ci_lo"]
        n_hi  = d["naive_ci_hi"]
        b_lo  = d["bb_ci_lo"]
        b_hi  = d["bb_ci_hi"]
        n_w   = n_hi - n_lo
        b_w   = b_hi - b_lo
        ratio = b_w / n_w if n_w > 0 else float("nan")
        lines.append(
            f"| {m} | {obs:.4f} | {n_lo:.4f} | {n_hi:.4f} | "
            f"{b_lo:.4f} | {b_hi:.4f} | {n_w:.4f} | {b_w:.4f} | {ratio:.2f}x |"
        )

    lines += [
        "",
        "---",
        "",
        "## Implications",
        "",
        "- **Ratio > 1.0:** Block bootstrap produces wider CIs than naive "
        "temporal dependence inflates effective variance. Report block BB as primary.",
        "- **Ratio 1.0:** Temporal dependence negligible Clopper-Pearson acceptable.",
        "- **Ratio < 1.0:** Unlikely; would indicate negative autocorrelation (anti-persistence).",
        "",
        "## Recommended Manuscript Language",
        "",
        "All confidence intervals are computed via non-overlapping block bootstrap "
        f"(b={BLOCK_SIZE}, n={N_BOOTSTRAP:,} resamples) to account for temporal "
        "autocorrelation in Zone C windows (|r| < 0.1 beyond lag {BLOCK_SIZE}). "
        "Clopper-Pearson exact binomial CIs are reported parenthetically for comparison.",
        "",
        "## Updated Table 11 Values",
        "",
        "| Metric | Point estimate | Block BB 95% CI |",
        "|--------|----------------|-----------------|",
    ]
    for m in metrics_order:
        if m not in results:
            continue
        d = results[m]
        lines.append(f"| {m} | {d['observed']:.4f} | [{d['bb_ci_lo']:.4f}, {d['bb_ci_hi']:.4f}] |")

    with open(output_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Report written   {output_path}")

def main():
    rng = np.random.RandomState(RANDOM_SEED)

    feat_path = os.path.join(FEATURES_DIR, "features_selected_labeled.csv")
    sel_path  = os.path.join(FEATURES_DIR, "selected_features.json")
    if not os.path.exists(feat_path):
        sys.exit(f"ERROR: {feat_path} not found — run window extraction–3 first.")

    with open(sel_path) as fh:
        selected = json.load(fh)
        if isinstance(selected, dict):
            selected = selected["selected_features"]

    df_all = pd.read_csv(feat_path)
    ts_col = "window_start" if "window_start" in df_all.columns else None
    zone_b, zone_c = build_stratified_zone_c(df_all, ts_col)
    print(f"  Zone B: {len(zone_b)} windows  |  Zone C: {len(zone_c)} windows")
    print(f"  NOTE: block bootstrap uses non-overlapping LSTM sequences (step=SEQ_LEN={SEQ_LEN}).")
    print(f"        statistical_validation.py uses sliding-window sequences (step=1) on the same")
    print(f"        Zone C rows, yielding ~10x more aligned predictions. Both are valid   the")
    print(f"        non-overlapping design here avoids inflated autocorrelation from sequence overlap.")

    with open(os.path.join(MODELS_DIR, "isolation_forest_optimized.pkl"), "rb") as fh:
        iforest = pickle.load(fh)
    with open(os.path.join(MODELS_DIR, "feature_scaler_selected.pkl"), "rb") as fh:
        scaler_if = pickle.load(fh)
    with open(os.path.join(MODELS_DIR, "iforest_optimized_calibration.json")) as fh:
        cal = json.load(fh)

    lstm_sc_path = os.path.join(MODELS_DIR, "feature_scaler_lstm.pkl")
    sc_path = lstm_sc_path if os.path.exists(lstm_sc_path) \
              else os.path.join(MODELS_DIR, "feature_scaler_selected.pkl")
    with open(sc_path, "rb") as fh:
        scaler_lstm = pickle.load(fh)

    with open(os.path.join(MODELS_DIR, "lstm_temperature.json")) as fh:
        td = json.load(fh)

    if td.get("requires_recalibration", False):
        sys.exit("ERROR: lstm_temperature.json has requires_recalibration=True. "
                 "Run ensemble calibration first.")

    with open(os.path.join(MODELS_DIR, "platt_iforest.pkl"),  "rb") as fh:
        platt_if = pickle.load(fh)
    with open(os.path.join(MODELS_DIR, "platt_lstm.pkl"),     "rb") as fh:
        platt_lstm = pickle.load(fh)

    prob_lo      = float(cal["prob_score_lo"])
    prob_hi      = float(cal["prob_score_hi"])
    T_star       = float(td["temperature"])
    gating_delta = float(td.get("gating_delta", 0.30))
    gating_alpha = float(td.get("gating_alpha", 0.00))
    rev_if       = bool(td.get("platt_if_reversed",   False))
    rev_lstm     = bool(td.get("platt_lstm_reversed", False))

    try:
        import tensorflow as tf
        lstm = tf.keras.models.load_model(
            os.path.join(MODELS_DIR, "lstm_model_selected.keras"), compile=False)
    except ImportError:
        sys.exit("ERROR: TensorFlow not installed cannot load LSTM.")

    print(" Computing Zone C predictions...")

    X_c_raw = zone_c[selected].values.astype(np.float32)
    X_c_if  = scaler_if.transform(X_c_raw)
    scores_c = iforest.score_samples(X_c_if)
    p_if_raw = np.clip((scores_c - prob_lo) / (prob_hi - prob_lo + 1e-12), 0.0, 1.0)
    p_if_platt = apply_platt(platt_if, p_if_raw, rev_if)

    X_c_lstm = scaler_lstm.transform(X_c_raw).astype(np.float32)
    ts_c = zone_c[ts_col] if ts_col and ts_col in zone_c.columns else None
    seqs_c, ends_c = make_sequences(X_c_lstm, timestamps=ts_c)

    if len(seqs_c) == 0:
        sys.exit("ERROR: No valid sequences in Zone C. Check gap settings.")

    lm_c_raw   = lstm.predict(seqs_c, batch_size=32, verbose=0).flatten()
    lm_c_platt = apply_platt(platt_lstm, lm_c_raw, rev_lstm)
    lm_c_cal   = apply_temperature(lm_c_platt, T_star)

    p_if_aln   = p_if_platt[ends_c]
    y_c        = zone_c["label"].values[ends_c]

    p_ens = confidence_gated_ensemble(p_if_aln, lm_c_cal,
                                       delta=gating_delta, alpha=gating_alpha)

    print(f"  Zone C aligned: {len(y_c)} windows  "
          f"(N={(y_c==1).sum()}, A={(y_c==0).sum()})")

    residuals = p_ens - y_c.astype(float)
    acf_info  = check_autocorrelation(residuals, max_lag=15)
    print(" Autocorrelation (ensemble residuals):")
    for lag in range(1, min(BLOCK_SIZE + 3, 16)):
        print(f"    lag {lag:2d}: r = {acf_info.get(lag, 0):+.4f}")

    def tpr_fn(s, l): return tpr_metric(s, l)
    def fpr_fn(s, l): return fpr_metric(s, l)

    results = {}

    print(f"\n  Running block bootstrap for AUC ({N_BOOTSTRAP:,} resamples)...")
    bb_auc = block_bootstrap_ci(p_ens, y_c, auc_metric, rng=rng)

    auc_obs = bb_auc["observed"]

    n_pos = int((y_c == 0).sum())
    n_neg = int((y_c == 1).sum())
    q1 = auc_obs / (2 - auc_obs)
    q2 = 2 * auc_obs**2 / (1 + auc_obs)
    se_auc = np.sqrt((auc_obs*(1-auc_obs) + (n_pos-1)*(q1-auc_obs**2) +
                      (n_neg-1)*(q2-auc_obs**2)) / (n_pos*n_neg))
    z95 = 1.96
    results["AUC"] = {
        "observed": auc_obs,
        "naive_ci_lo": float(np.clip(auc_obs - z95 * se_auc, 0, 1)),
        "naive_ci_hi": float(np.clip(auc_obs + z95 * se_auc, 0, 1)),
        "bb_ci_lo": bb_auc["ci_lo"],
        "bb_ci_hi": bb_auc["ci_hi"],
        "bb_std": bb_auc["std_boot"],
    }

    print(f"  Running block bootstrap for AUPRC ({N_BOOTSTRAP:,} resamples)...")
    bb_auprc = block_bootstrap_ci(p_ens, y_c, auprc_metric, rng=rng)
    auprc_obs = bb_auprc["observed"]
    results["AUPRC"] = {
        "observed": auprc_obs,
        "naive_ci_lo": float(np.clip(auprc_obs - z95 * bb_auprc["std_boot"], 0, 1)),
        "naive_ci_hi": float(np.clip(auprc_obs + z95 * bb_auprc["std_boot"], 0, 1)),
        "bb_ci_lo": bb_auprc["ci_lo"],
        "bb_ci_hi": bb_auprc["ci_hi"],
        "bb_std": bb_auprc["std_boot"],
    }

    print(f"  Running block bootstrap for TPR ({N_BOOTSTRAP:,} resamples)...")
    bb_tpr = block_bootstrap_ci(p_ens, y_c, tpr_fn, rng=rng)
    tpr_obs = bb_tpr["observed"]
    n_att   = int((y_c == 0).sum())
    tp      = int(round(tpr_obs * n_att))
    cp_tpr  = clopper_pearson(tp, n_att)
    results["TPR"] = {
        "observed": tpr_obs,
        "naive_ci_lo": cp_tpr["ci_lo"],
        "naive_ci_hi": cp_tpr["ci_hi"],
        "bb_ci_lo": bb_tpr["ci_lo"],
        "bb_ci_hi": bb_tpr["ci_hi"],
        "bb_std": bb_tpr["std_boot"],
    }

    print(f"  Running block bootstrap for FPR ({N_BOOTSTRAP:,} resamples)...")
    bb_fpr = block_bootstrap_ci(p_ens, y_c, fpr_fn, rng=rng)
    fpr_obs = bb_fpr["observed"]
    n_nor   = int((y_c == 1).sum())
    fp      = int(round(fpr_obs * n_nor))
    cp_fpr  = clopper_pearson(fp, n_nor)
    results["FPR"] = {
        "observed": fpr_obs,
        "naive_ci_lo": cp_fpr["ci_lo"],
        "naive_ci_hi": cp_fpr["ci_hi"],
        "bb_ci_lo": bb_fpr["ci_lo"],
        "bb_ci_hi": bb_fpr["ci_hi"],
        "bb_std": bb_fpr["std_boot"],
    }

    with open(OUTPUT_JSON, "w") as fh:
        json.dump({"block_size": BLOCK_SIZE, "n_resamples": N_BOOTSTRAP,
                   "autocorrelation": acf_info, "metrics": results}, fh, indent=2)

    write_report(results, acf_info, OUTPUT_MD)

    def _print_summary():
        rows = [("Metric", "Observed", "Naive CI 95%", "Block-Bootstrap CI 95%", "BB std")]
        for name, r in results.items():
            obs  = f"{r['observed']:.4f}"
            nci  = f"[{r['naive_ci_lo']:.4f}, {r['naive_ci_hi']:.4f}]"
            bbi  = f"[{r['bb_ci_lo']:.4f}, {r['bb_ci_hi']:.4f}]"
            bbs  = f"{r['bb_std']:.4f}"
            rows.append((name, obs, nci, bbi, bbs))
        widths = [max(len(r[i]) for r in rows) for i in range(5)]
        sep    = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        def row(vals):
            return "| " + " | ".join(f"{str(v):<{w}}" for v, w in zip(vals, widths)) + " |"
        print("\n" + sep)
        print(row(rows[0]))
        print(sep)
        for r in rows[1:]:
            print(row(r))
        print(sep)
        print(f"  Block size: {BLOCK_SIZE}  |  Resamples: {N_BOOTSTRAP:,}")
        print(f"  JSON: {OUTPUT_JSON}  |  Report: {OUTPUT_MD}")
        print(sep)

    _print_summary()

if __name__ == "__main__":
    main()
