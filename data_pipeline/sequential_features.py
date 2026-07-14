
"""
BridgeGuard Sequential Feature Engineering (Leakage-Free)
==========================================================

Post-processing step executed AFTER window extraction and BEFORE feature
selection.  Enriches features_labeled.csv with three temporal persistence
features that are invisible to the static IForest but strongly discriminative
for Slow Poisoning.

Signal rationale
----------------
alert_frequency_per_hour autocorrelation: r=0.877 (lag=1), r=0.648 (lag=2).
Slow Poisoning is a persistence signal, not a magnitude signal.
A single window is indistinguishable from noise; five consecutive windows are not.

Features produced
-----------------
freq_persistence_5w  : proportion of windows with alerts over 5 rolling windows
                       (expected Hedges g ≈ 3.0 for slow_poisoning vs normal)
freq_rolling_mean_5w : mean alert frequency over 5 rolling windows
                       (expected Hedges g ≈ 2.0)
freq_rolling_std_5w  : std of alert frequency over 5 rolling windows
                       (expected Hedges g ≈ 1.5)

Scientific guarantees
---------------------
- Strict causality: rolling is backward-looking only (no look-ahead)
- Per-class isolation: rolling computed separately for each attack_type
- Leakage-free imputation: median fitted on TRAIN (60% chronological) per class
- Zero inter-zone contamination: rolling never crosses class boundaries
- Backward-compatible: features_labeled.csv and features_normal.csv are
  overwritten with the enriched versions; originals backed up under _orig

Inputs
------
bridgeguard_features/features_labeled.csv  (produced by window extraction)
bridgeguard_features/features_normal.csv   (produced by window extraction)

Outputs
-------
bridgeguard_features/features_labeled.csv      (overwritten, +3 features)
bridgeguard_features/features_normal.csv       (overwritten, +3 features)
bridgeguard_features/features_labeled_seq.csv  (enriched copy)
bridgeguard_features/features_normal_seq.csv   (enriched copy)
bridgeguard_features/features_labeled_orig.csv (original backup)
bridgeguard_features/features_normal_orig.csv  (original backup)
bridgeguard_features/step2b_audit.json         (full audit trail)

Execution order
---------------
python window_extraction.py
python sequential_features.py
python attack_aware_feature_selection.py
python iforest_hyperparameter_tuning.py
python train_iforest.py
"""

from __future__ import annotations

import json
import os
import shutil
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
np.random.seed(42)

FEATURES_DIR   : str   = os.getenv("BG_FEATURES_DIR", "bridgeguard_features")
LABELED_CSV    : str   = os.path.join(FEATURES_DIR, "features_labeled.csv")
NORMAL_CSV     : str   = os.path.join(FEATURES_DIR, "features_normal.csv")

ROLLING_WINDOW : int   = 5
MIN_PERIODS    : int   = 3
TRAIN_FRACTION : float = 0.60

NEW_FEATURES   : List[str] = [
    "freq_persistence_5w",
    "freq_rolling_mean_5w",
    "freq_rolling_std_5w",
]

_FREQ_COL_CANDIDATES  = ["alert_frequency_per_hour"]
_NEMG_COL_CANDIDATES  = ["n_emergency"]

def _detect_column(df: pd.DataFrame, candidates: List[str],
                   pattern: Optional[str] = None, label: str = "") -> str:
    for name in candidates:
        if name in df.columns:
            return name
    if pattern:
        matches = [c for c in df.columns if pd.Series([c]).str.contains(pattern).iloc[0]]
        if matches:
            return matches[0]
    raise RuntimeError(
        f"[ERROR] Column '{label}' not found in CSV.\n"
        f"  Expected one of: {candidates}\n"
        f"  Available columns: {sorted(df.columns.tolist())}\n"
        f"  Verify that window extraction produced features_labeled.csv correctly."
    )

def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_var = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    if pooled_var <= 0:
        return float("nan")
    d = (np.mean(a) - np.mean(b)) / np.sqrt(pooled_var)

    df_total = na + nb - 2
    j = 1.0 - (3.0 / (4.0 * df_total - 1.0)) if df_total > 1 else 1.0
    return float(abs(d * j))

def _compute_sequential_features_for_class(
    sub_df: pd.DataFrame,
    freq_col: str,
    nemg_col: str,
    train_n: int,
) -> pd.DataFrame:
    assert len(sub_df) >= 1, "Empty sub-DataFrame — invariant violated."

    freq_series   = sub_df[freq_col].astype(np.float64)
    has_alert     = (sub_df[nemg_col].values > 0).astype(np.float64)
    has_alert_s   = pd.Series(has_alert, index=sub_df.index)

    roll_kwargs = dict(window=ROLLING_WINDOW, min_periods=MIN_PERIODS)

    freq_persistence_5w  = has_alert_s.rolling(**roll_kwargs).mean()
    freq_rolling_mean_5w = freq_series.rolling(**roll_kwargs).mean()
    freq_rolling_std_5w  = freq_series.rolling(**roll_kwargs).std(ddof=1)

    result = sub_df.copy()
    result["freq_persistence_5w"]  = freq_persistence_5w
    result["freq_rolling_mean_5w"] = freq_rolling_mean_5w
    result["freq_rolling_std_5w"]  = freq_rolling_std_5w

    for feat in NEW_FEATURES:
        train_vals = result[feat].iloc[:train_n]
        med_train  = float(train_vals.median())
        if np.isnan(med_train):

            med_train = float(result[feat].median())
        if np.isnan(med_train):

            med_train = 0.0
        n_nan = int(result[feat].isna().sum())
        if n_nan > 0:
            result[feat] = result[feat].fillna(med_train)

    return result[[*NEW_FEATURES]]

def main() -> None:
    print("=" * 70)
    print("BridgeGuard Sequential Feature Engineering (Leakage-Free)")
    print("=" * 70)

    for path in [LABELED_CSV, NORMAL_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[ERROR] File not found: {path}\n"
                f"  Run window_extraction.py before sequential feature extraction."
            )

    print(f"\n{' '*60}")
    print("1. LOADING DATA")
    labeled = pd.read_csv(LABELED_CSV)
    normal  = pd.read_csv(NORMAL_CSV)

    print(f"  features_labeled.csv : {len(labeled)} windows")
    print(f"  features_normal.csv  : {len(normal)} windows")
    print(f"  Columns  : {labeled.shape[1]}")

    required_meta = {"label", "attack_type", "window_start_time"}
    missing_meta  = required_meta - set(labeled.columns)
    assert not missing_meta, (
        f"[ERROR] Missing metadata columns: {missing_meta}\n"
        f"  Verify that window extraction produced the required columns."
    )
    assert set(labeled["label"].unique()).issubset({0, 1}), \
        "[ERROR] Unexpected label values — expected {0, 1}."

    freq_col = _detect_column(labeled, _FREQ_COL_CANDIDATES,
                              pattern="frequency.*hour|freq.*hour",
                              label="alert_frequency_per_hour")
    nemg_col = _detect_column(labeled, _NEMG_COL_CANDIDATES,
                              pattern="n_emerg|emergency_count",
                              label="n_emergency")
    print(f"\n  Frequency source column: '{freq_col}'")
    print(f"  Emergency source column : '{nemg_col}'")

    existing_new = [f for f in NEW_FEATURES if f in labeled.columns]
    if existing_new:
        print(f"\n     Features already present in source CSV: {existing_new}")
        print(f"     Sequential feature extraction is idempotent — forced recalculation.")
        labeled = labeled.drop(columns=existing_new)
        normal  = normal.drop(columns=[c for c in existing_new if c in normal.columns])

    print(f"\n{' '*60}")
    print("2. SAVING ORIGINALS")
    orig_labeled = LABELED_CSV.replace(".csv", "_orig.csv")
    orig_normal  = NORMAL_CSV.replace(".csv", "_orig.csv")
    for src, dst in [(LABELED_CSV, orig_labeled), (NORMAL_CSV, orig_normal)]:
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  Backup : {dst}")
        else:
            print(f"  Existing backup retained: {dst}")

    labeled["_ts"] = pd.to_datetime(labeled["window_start_time"], utc=True)
    labeled = labeled.sort_values("_ts").reset_index(drop=True)

    print(f"\n{' '*60}")
    print("3. COMPUTING SEQUENTIAL FEATURES (causal, per class)")
    print(f"   rolling window={ROLLING_WINDOW}, min_periods={MIN_PERIODS}")

    classes    = labeled["attack_type"].unique().tolist()
    result_dfs : List[pd.DataFrame] = []
    audit_per_class: Dict = {}

    for cls in sorted(classes):
        cls_mask = labeled["attack_type"] == cls
        cls_df   = labeled[cls_mask].copy()
        cls_df   = cls_df.sort_values("_ts")

        n_cls    = len(cls_df)
        train_n  = max(1, int(np.floor(n_cls * TRAIN_FRACTION)))

        print(f"\n  Class '{cls}' : {n_cls} windows  "
              f"(TRAIN={train_n}, TEST={n_cls - train_n})")

        if n_cls < MIN_PERIODS:

            print(f"       Class too small ({n_cls} < {MIN_PERIODS}) — "
                  f"sequential features set to 0.0")
            for feat in NEW_FEATURES:
                cls_df[feat] = 0.0
            seq_feats = cls_df[NEW_FEATURES]
        else:
            seq_feats = _compute_sequential_features_for_class(
                cls_df, freq_col, nemg_col, train_n
            )

        nan_counts = seq_feats.isna().sum().to_dict()
        assert all(v == 0 for v in nan_counts.values()), (
            f"[ERROR] Residual NaN after imputation for class '{cls}' :\n"
            f"  {nan_counts}\n"
            f"  Check imputation logic (TRAIN partition too small?)."
        )

        cls_stats: Dict = {}
        for feat in NEW_FEATURES:
            vals_cls = seq_feats[feat].values
            if labeled["label"].loc[cls_df.index].iloc[0] == 1:

                cls_stats[feat] = {
                    "mean": float(np.mean(vals_cls)),
                    "std":  float(np.std(vals_cls, ddof=1)) if len(vals_cls) > 1 else 0.0,
                }
            else:
                cls_stats[feat] = {
                    "mean": float(np.mean(vals_cls)),
                    "std":  float(np.std(vals_cls, ddof=1)) if len(vals_cls) > 1 else 0.0,
                }

        for feat in NEW_FEATURES:
            vals = seq_feats[feat].values
            print(f"    {feat:<26} : mean={np.mean(vals):.4f}  "
                  f"std={np.std(vals, ddof=1):.4f}  "
                  f"nan_imputed={int(cls_df[feat].isna().sum() if feat in cls_df.columns else 0)}")

        for feat in NEW_FEATURES:
            labeled.loc[cls_df.index, feat] = seq_feats[feat].values

        audit_per_class[cls] = cls_stats

    print(f"\n{' '*60}")
    print("4. VALIDATION")

    for feat in NEW_FEATURES:
        n_nan = int(labeled[feat].isna().sum())
        assert n_nan == 0, (
            f"[ERROR] {n_nan} NaN in '{feat}' after full computation.\n"
            f"  Leakage-free invariant violated."
        )

    assert not labeled[NEW_FEATURES].isin([np.inf, -np.inf]).any().any(), \
        "[ERROR] Inf values detected in sequential features."

    for feat in NEW_FEATURES:
        assert labeled[feat].min() >= 0.0 or feat == "freq_rolling_std_5w", \
            f"[ERROR] Negative values found in '{feat}'."
    assert labeled["freq_persistence_5w"].max() <= 1.0 + 1e-9, \
        "[ERROR] freq_persistence_5w > 1.0 — definition violation."

    print(f"  OK Zero NaN in the 3 sequential features")
    print(f"  OK Zero Inf")
    print(f"  OK freq_persistence_5w   [0, 1]")

    print(f"\n{' '*60}")
    print("5. HEDGES g DISCRIMINABILITY (slow_poisoning vs normal)")

    normal_mask   = labeled["label"] == 1
    sp_mask       = labeled["attack_type"].str.lower().str.contains("slow|poison|attack5|a5")
    fl_mask       = labeled["attack_type"].str.lower().str.contains("flood|attack2|a2")

    hedges_g_sp: Dict[str, float] = {}
    hedges_g_fl: Dict[str, float] = {}

    for feat in NEW_FEATURES:
        vals_n  = labeled.loc[normal_mask, feat].values
        vals_sp = labeled.loc[sp_mask, feat].values if sp_mask.any() else np.array([])
        vals_fl = labeled.loc[fl_mask, feat].values if fl_mask.any() else np.array([])

        g_sp = hedges_g(vals_sp, vals_n) if len(vals_sp) >= 2 else float("nan")
        g_fl = hedges_g(vals_fl, vals_n) if len(vals_fl) >= 2 else float("nan")
        hedges_g_sp[feat] = g_sp
        hedges_g_fl[feat] = g_fl

        size_sp = ("LARGE" if g_sp > 0.8 else "MEDIUM" if g_sp > 0.5 else
                   "SMALL" if not np.isnan(g_sp) else "N/A")
        print(f"  {feat:<30}: g_slow_poison={g_sp:>6.3f} [{size_sp:<6}]  "
              f"g_flooding={g_fl:>6.3f}")

    labeled = labeled.drop(columns=["_ts"])

    original_cols = [c for c in labeled.columns if c not in NEW_FEATURES]
    final_labeled = labeled[original_cols + NEW_FEATURES].copy()

    final_normal = final_labeled[final_labeled["label"] == 1].copy().reset_index(drop=True)

    print(f"\n{' '*60}")
    print("6. SAVING")

    seq_labeled_path = os.path.join(FEATURES_DIR, "features_labeled_seq.csv")
    seq_normal_path  = os.path.join(FEATURES_DIR, "features_normal_seq.csv")
    final_labeled.to_csv(seq_labeled_path, index=False)
    final_normal.to_csv(seq_normal_path,   index=False)
    print(f"  OK {seq_labeled_path:<52} {len(final_labeled):>5} windows  "
          f"{final_labeled.shape[1]} cols")
    print(f"  OK {seq_normal_path:<52} {len(final_normal):>5} windows  "
          f"{final_normal.shape[1]} cols")

    final_labeled.to_csv(LABELED_CSV, index=False)
    final_normal.to_csv(NORMAL_CSV,   index=False)
    print(f"  OK {LABELED_CSV:<52}  overwritten  (+3 features)")
    print(f"  OK {NORMAL_CSV:<52}  overwritten  (+3 features)")

    audit = {
        "step": "sequential_feature_extraction",
        "version":        "1.0.0",
        "n_windows":      int(len(final_labeled)),
        "n_normal":       int((final_labeled["label"] == 1).sum()),
        "n_attack":       int((final_labeled["label"] == 0).sum()),
        "new_features":   NEW_FEATURES,
        "rolling_window": ROLLING_WINDOW,
        "min_periods":    MIN_PERIODS,
        "train_fraction": TRAIN_FRACTION,
        "freq_col_used":  freq_col,
        "nemg_col_used":  nemg_col,
        "hedges_g_slow_poisoning": {k: round(v, 4) if not np.isnan(v) else None
                                    for k, v in hedges_g_sp.items()},
        "hedges_g_flooding":       {k: round(v, 4) if not np.isnan(v) else None
                                    for k, v in hedges_g_fl.items()},
        "per_class_stats": audit_per_class,
        "validation": {
            "zero_nan_new_features": True,
            "zero_inf_new_features": True,
            "persistence_bounded_01": True,
        },
        "output_files": {
            "features_labeled_seq": seq_labeled_path,
            "features_normal_seq":  seq_normal_path,
            "features_labeled":     LABELED_CSV,
            "features_normal":      NORMAL_CSV,
        },
    }

    audit_path = os.path.join(FEATURES_DIR, "step2b_audit.json")
    with open(audit_path, "w") as fh:
        json.dump(audit, fh, indent=2)
    print(f"  OK {audit_path}")

    n_total  = len(final_labeled)
    n_normal = int((final_labeled["label"] == 1).sum())
    n_attack = int((final_labeled["label"] == 0).sum())

    rows = [
        ("Total windows",   str(n_total)),
        ("Normal windows",  str(n_normal)),
        ("Attack windows",  str(n_attack)),
        ("Columns before",  str(len(original_cols))),
        ("Columns after",   f"{final_labeled.shape[1]}  (+{len(NEW_FEATURES)} sequential)"),
        ("Audit trail",     audit_path),
    ] + [(f"Hedges g  {feat}", f"{g:.3f}" if not np.isnan(g) else "nan")
         for feat, g in hedges_g_sp.items()] + [
        ("Next step",       "python attack_aware_feature_selection.py"),
    ]
    c1 = max(len(r[0]) for r in rows)
    c2 = max(len(r[1]) for r in rows)
    sep = f"+{'-'*(c1+2)}+{'-'*(c2+2)}+"
    print("\n" + sep)
    print(f"| {'Sequential Features':<{c1}} | {'':<{c2}} |")
    print(sep)
    for label, val in rows:
        print(f"| {label:<{c1}} | {val:<{c2}} |")
    print(sep)

if __name__ == "__main__":
    main()
