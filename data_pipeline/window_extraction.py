
"""
Sliding Window Extraction BridgeGuard Data Pipeline
======================================================

Extracts behavioral feature windows from raw MQTT capture CSVs
(normal traffic, flooding attack, slow-poisoning attack) and produces
the labeled feature dataset used for model training and evaluation.

OUTPUT FILES:
 $FEATURES_DIR/features_labeled.csv full dataset (chronological order)
 $FEATURES_DIR/features_normal.csv normal windows only
 $FEATURES_DIR/features_train.csv chronological train split (60%)
 $FEATURES_DIR/features_test.csv chronological test split (40%)
 $FEATURES_DIR/real_dataset_stats.json audit trail
"""

from __future__ import annotations

import importlib
import json
import os
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")

_BASELINE_DIR: str = os.getenv("BG_BASELINE_DIR", "baseline_data")
_FEATURES_DIR: str = os.getenv("BG_FEATURES_DIR", "bridgeguard_features")

NORMAL_CSV:  str = os.getenv("BG_NORMAL_CSV",  os.path.join(_BASELINE_DIR, "interleaved/normal_interleaved.csv"))
ATTACK2_CSV: str = os.getenv("BG_ATTACK2_CSV", os.path.join(_BASELINE_DIR, "interleaved/attack2_interleaved.csv"))
ATTACK5_CSV: str = os.getenv("BG_ATTACK5_CSV", os.path.join(_BASELINE_DIR, "interleaved/attack5_interleaved.csv"))

OUTPUT_LABELED: str = os.path.join(_FEATURES_DIR, "features_labeled.csv")
OUTPUT_NORMAL:  str = os.path.join(_FEATURES_DIR, "features_normal.csv")
OUTPUT_TRAIN:   str = os.path.join(_FEATURES_DIR, "features_train.csv")
OUTPUT_TEST:    str = os.path.join(_FEATURES_DIR, "features_test.csv")

PRESERVE_TEMPORAL_ORDER: bool = os.getenv("BG_PRESERVE_ORDER", "1") == "1"
TEST_SIZE: float = float(os.getenv("BG_TEST_SIZE", "0.40"))

os.makedirs(_FEATURES_DIR, exist_ok=True)

@dataclass(frozen=True)
class WindowConfig:
    window_size_min: int
    step_min:        int
    min_emergency:   int
    label:           int
    attack_type:     str
    deduplicate:     bool = False

    def __post_init__(self) -> None:
        if self.step_min >= self.window_size_min:
            raise ValueError(
                f"step_min={self.step_min} must be < window_size_min={self.window_size_min}"
            )
        if self.step_min <= 0 or self.window_size_min <= 0:
            raise ValueError("step_min and window_size_min must be > 0")

    @property
    def overlap_pct(self) -> int:
        return int(100 * (1 - self.step_min / self.window_size_min))

    @property
    def window_ns(self) -> int:
        return self.window_size_min * 60 * 1_000_000_000

    @property
    def step_ns(self) -> int:
        return self.step_min * 60 * 1_000_000_000

CFG_NORMAL = WindowConfig(
    window_size_min = int(os.getenv("BG_WINDOW_MIN",  "60")),
    step_min        = int(os.getenv("BG_STEP_NORMAL", "10")),
    min_emergency   = 0,
    label           = 1,
    attack_type     = "normal",
    deduplicate     = False,
)
CFG_ATTACK2 = WindowConfig(
    window_size_min = int(os.getenv("BG_WINDOW_MIN",  "60")),
    step_min        = int(os.getenv("BG_STEP_ATTACK", "15")),
    min_emergency   = 1,
    label           = 0,
    attack_type     = "flooding",
    deduplicate     = True,
)
CFG_ATTACK5 = WindowConfig(
    window_size_min = int(os.getenv("BG_WINDOW_MIN",  "60")),
    step_min        = int(os.getenv("BG_STEP_ATTACK", "15")),
    min_emergency   = 1,
    label           = 0,
    attack_type     = "slow_poisoning",
    deduplicate     = True,
)

_RNG = np.random.default_rng(42)

_CV_EPSILON: float = float(os.getenv("BG_CV_EPSILON", "0.01"))

class FeatureDict(TypedDict):
    alert_frequency_per_hour:    float
    normal_emergency_ratio:      float
    inter_alert_interval_mean:   float
    inter_interval_variance:     float
    burst_score:                 float
    topic_diversity:             float
    time_sin:                    float
    time_cos:                    float
    payload_entropy:             float
    consecutive_emergency_count: float
    alert_rate_acceleration:     float
    regularity_coefficient:      float
    temporal_clustering_score:   float

ALL_FEATURES: List[str] = list(FeatureDict.__annotations__.keys())

def _to_ns(s: pd.Series) -> np.ndarray:
    return s.dt.tz_convert(None).astype("datetime64[ns]").values.view("int64")

def _cyclical_time(ts: pd.Timestamp) -> Tuple[float, float]:
    if pd.isna(ts):
        return 0.0, 0.0
    angle = 2.0 * np.pi * (ts.hour + ts.minute / 60.0) / 24.0
    return float(np.sin(angle)), float(np.cos(angle))

def _payload_entropy_safe(messages: pd.Series) -> float:
    clean = messages.dropna()
    if clean.empty:
        return 0.0
    full_blob: bytes = b"\x00".join(
        s.encode("utf-8", errors="replace")
        for s in clean.astype(str)
    )
    if not full_blob:
        return 0.0
    arr = np.frombuffer(full_blob, dtype=np.uint8)
    _, counts = np.unique(arr, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.dot(probs, np.log2(probs + 1e-12)))

def load_csv(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File not found: {filepath}\n"
            f"  → Have you run the dataset interleaving script?"
        )

    _parquet_path = filepath.rsplit(".", 1)[0] + ".parquet"
    _use_cache    = os.getenv("BG_CACHE_PARQUET", "1") == "1"
    _has_parquet  = importlib.util.find_spec("pyarrow") is not None

    if (_use_cache and _has_parquet
            and os.path.exists(_parquet_path)
            and os.path.getmtime(_parquet_path) >= os.path.getmtime(filepath)):
        df = pd.read_parquet(_parquet_path)
        print(f"    [cache] Loaded Parquet: {_parquet_path}")
        return df

    _CSV_ENGINE = "pyarrow" if _has_parquet else "c"
    _OPT_DTYPES = {
        "type":    "category",
        "topic":   "category",
        "message": "string",
    }

    try:
        df = pd.read_csv(filepath, dtype=_OPT_DTYPES, engine=_CSV_ENGINE)
    except Exception:
        try:
            df = pd.read_csv(filepath, engine="c")
        except Exception as exc:
            raise FileNotFoundError(f"Read error {filepath}: {exc}") from exc

    if df.empty:
        raise ValueError(f"Empty CSV: {filepath}")
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing 'timestamp' column in {filepath}")

    _f64_cols = df.select_dtypes(include=["float64"]).columns.tolist()
    if _f64_cols:
        df[_f64_cols] = df[_f64_cols].astype("float32")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if _use_cache and _has_parquet:
        try:
            df.to_parquet(_parquet_path, index=False)
            print(f"    [cache] Wrote Parquet: {_parquet_path}")
        except Exception as exc:
            warnings.warn(f"[load_csv] Parquet cache not written ({exc}) — CSV fallback retained")

    return df

def extract_features(
    window_df: pd.DataFrame,
    cfg: WindowConfig,
) -> FeatureDict:
    if window_df.empty:
        return {f: 0.0 for f in ALL_FEATURES}

    emg_mask = window_df["type"].values == "emergency"
    nrm_mask = ~emg_mask

    emg = window_df[emg_mask]
    n_emg = int(emg_mask.sum())
    n_nrm = int(nrm_mask.sum())

    alert_frequency_per_hour = float(n_emg)

    normal_emergency_ratio = n_nrm / max(1, n_emg)

    inter_alert_interval_mean: float
    inter_alert_interval_std:  float
    if n_emg >= 2:
        ts_emg_s = _to_ns(emg["timestamp"]) / 1e9 / 60.0
        intervals = np.diff(ts_emg_s)
        inter_alert_interval_mean = float(intervals.mean())
        inter_alert_interval_std  = float(intervals.std(ddof=1)) if len(intervals) > 1 else 0.0
    else:
        inter_alert_interval_mean = np.nan
        inter_alert_interval_std  = 0.0
    inter_interval_variance = inter_alert_interval_std ** 2

    if n_emg > 0:
        bin5_ns = 5 * 60 * 1_000_000_000
        bins    = _to_ns(emg["timestamp"]) // bin5_ns
        _, bin_counts_5 = np.unique(bins, return_counts=True)
        burst_score = float(bin_counts_5.max())
    else:
        burst_score = 0.0

    if n_emg > 0 and "topic" in window_df.columns:
        topic_diversity = float(emg["topic"].nunique())
    else:
        topic_diversity = 0.0

    ref_ts: pd.Timestamp
    if n_emg > 0:
        ref_ts = emg["timestamp"].iloc[0]
    else:
        ref_ts = window_df["timestamp"].iloc[0]
    time_sin, time_cos = _cyclical_time(ref_ts)

    if n_emg > 0 and "message" in window_df.columns:
        payload_entropy = _payload_entropy_safe(emg["message"])
    else:
        payload_entropy = 0.0

    consecutive_emergency_count = 1.0 if n_emg > 0 else 0.0

    if n_emg >= 4:
        ts_min_ns = int(_to_ns(window_df["timestamp"])[0])
        half_ns   = cfg.window_ns // 2
        mid_ns    = ts_min_ns + half_ns
        emg_ns    = _to_ns(emg["timestamp"])
        n_first   = int((emg_ns <  mid_ns).sum())
        n_second  = int((emg_ns >= mid_ns).sum())
        alert_rate_acceleration = float(n_second - n_first)
    else:
        alert_rate_acceleration = 0.0

    regularity_coefficient: float
    if (n_emg >= 3
            and inter_alert_interval_std > 0
            and not np.isnan(inter_alert_interval_mean)
            and inter_alert_interval_mean != 0.0):
        cv = inter_alert_interval_std / inter_alert_interval_mean
        regularity_coefficient = 1.0 / max(cv, _CV_EPSILON)
    elif n_emg >= 2:
        regularity_coefficient = 1.0 / _CV_EPSILON
    else:
        regularity_coefficient = np.nan

    if n_emg >= 3:
        bin10_ns = 10 * 60 * 1_000_000_000
        bins10   = _to_ns(emg["timestamp"]) // bin10_ns
        _, bin_counts = np.unique(bins10, return_counts=True)
        bin_counts = bin_counts.astype(float)
        mean_d     = bin_counts.mean()
        temporal_clustering_score = float(bin_counts.max() / mean_d) if mean_d > 0 else 1.0
    else:
        temporal_clustering_score = 1.0

    feats: Dict[str, float] = {
        "alert_frequency_per_hour":    alert_frequency_per_hour,
        "normal_emergency_ratio":      normal_emergency_ratio,
        "inter_alert_interval_mean":   inter_alert_interval_mean,
        "inter_interval_variance":     inter_interval_variance,
        "burst_score":                 burst_score,
        "topic_diversity":             topic_diversity,
        "time_sin":                    time_sin,
        "time_cos":                    time_cos,
        "payload_entropy":             payload_entropy,
        "consecutive_emergency_count": consecutive_emergency_count,
        "alert_rate_acceleration":     alert_rate_acceleration,
        "regularity_coefficient":      regularity_coefficient,
        "temporal_clustering_score":   temporal_clustering_score,
    }

    for k, v in feats.items():
        if isinstance(v, float) and np.isinf(v):
            feats[k] = np.nan

    return feats

def extract_sliding_windows(
    filepath: str,
    cfg:      WindowConfig,
) -> List[Dict]:
    print(f"\n  Loading {filepath}...")
    df = load_csv(filepath)

    ts_ns: np.ndarray = _to_ns(df["timestamp"])
    assert (np.diff(ts_ns) >= 0).all(), "Timestamps not sorted after load_csv check CSV"

    t_min_ns = int(ts_ns[0])
    t_max_ns = int(ts_ns[-1])

    overlap_pct = cfg.overlap_pct
    n_expected  = max(1, (t_max_ns - t_min_ns - cfg.window_ns) // cfg.step_ns + 1)

    print(f"  {len(df):,} messages  |  "
          f"Window={cfg.window_size_min}min  Step={cfg.step_min}min  "
          f"Overlap={overlap_pct}%  (~{n_expected} windows estimated)")

    n_windows = max(0, (t_max_ns - t_min_ns - cfg.window_ns) // cfg.step_ns + 1)
    starts_ns  = t_min_ns + np.arange(n_windows, dtype=np.int64) * cfg.step_ns
    starts_ns  = starts_ns[starts_ns + cfg.window_ns <= t_max_ns]
    ends_ns    = starts_ns + cfg.window_ns

    i_starts = np.searchsorted(ts_ns, starts_ns, side="left").astype(np.int32)
    i_ends   = np.searchsorted(ts_ns, ends_ns,   side="left").astype(np.int32)

    valid_mask = i_starts < i_ends
    i_starts   = i_starts[valid_mask]
    i_ends     = i_ends[valid_mask]
    starts_ns  = starts_ns[valid_mask]
    ends_ns    = ends_ns[valid_mask]

    _cols: Dict[str, List] = {f: [] for f in ALL_FEATURES}
    _meta: Dict[str, List] = {
        "label": [], "attack_type": [],
        "window_start_time": [], "window_end_time": [],
        "n_messages": [], "n_emergency": [],
    }

    for i_start, i_end, t_start_ns, t_end_ns in zip(
            i_starts, i_ends, starts_ns, ends_ns):
        window_view: pd.DataFrame = df.iloc[int(i_start):int(i_end)]

        n_emergency = int((window_view["type"].values == "emergency").sum())
        if n_emergency < cfg.min_emergency:
            continue

        feats = extract_features(window_view, cfg)

        for f in ALL_FEATURES:
            _cols[f].append(feats.get(f, np.nan))

        _meta["label"].append(cfg.label)
        _meta["attack_type"].append(cfg.attack_type)
        _meta["window_start_time"].append(
            pd.Timestamp(int(t_start_ns), unit="ns", tz="UTC").isoformat())
        _meta["window_end_time"].append(
            pd.Timestamp(int(t_end_ns), unit="ns", tz="UTC").isoformat())
        _meta["n_messages"].append(int(i_end) - int(i_start))
        _meta["n_emergency"].append(n_emergency)

    n_valid = len(_meta["label"])
    if n_valid == 0:
        rows: List[Dict] = []
    else:
        _df_out = pd.DataFrame({**_meta, **_cols})
        rows = _df_out.to_dict("records")

    print(f"  Extracted: {n_valid} windows (min_emergency {cfg.min_emergency})")
    return rows

def deduplicate_windows(rows: List[Dict]) -> List[Dict]:
    if not rows:
        return rows
    seen: set = set()
    kept: List[Dict] = []
    for row in rows:
        key = row.get("window_start_time", id(row))
        if key not in seen:
            seen.add(key)
            kept.append(row)
    n_removed = len(rows) - len(kept)
    if n_removed:
        print(f"  Dedup: {n_removed} windows removed   {len(kept)} retained")
    return kept

def chronological_split_per_class(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["_ts"] = pd.to_datetime(df["window_start_time"], utc=True)

    train_parts: List[pd.DataFrame] = []
    test_parts:  List[pd.DataFrame] = []

    for cls in df["attack_type"].unique():
        cls_df    = df[df["attack_type"] == cls].sort_values("_ts").reset_index(drop=True)
        split_idx = int(len(cls_df) * (1.0 - test_size))
        train_parts.append(cls_df.iloc[:split_idx])
        test_parts.append(cls_df.iloc[split_idx:])
        print(f"  {cls:<20}: total={len(cls_df):>4}  "
              f"train={split_idx:>4}  test={len(cls_df) - split_idx:>4}")

    train_df = pd.concat(train_parts, ignore_index=True).drop("_ts", axis=1)
    test_df  = pd.concat(test_parts,  ignore_index=True).drop("_ts", axis=1)
    return train_df, test_df

class TrainFittedPreprocessor:

    def __init__(
        self,
        feature_cols:    List[str],
        clip_percentile: float = 99.0,
    ) -> None:
        self.feature_cols    = feature_cols
        self.clip_percentile = clip_percentile
        self.medians_:    Dict[str, float] = {}
        self.clip_upper_: Dict[str, float] = {}

    def fit(self, train_df: pd.DataFrame) -> "TrainFittedPreprocessor":
        for col in self.feature_cols:
            if col not in train_df.columns:
                continue
            series = train_df[col].replace([np.inf, -np.inf], np.nan)
            self.medians_[col]    = float(series.median())
            self.clip_upper_[col] = float(series.quantile(self.clip_percentile / 100.0))
        return self

    def transform(self, df: pd.DataFrame, label: str = "") -> pd.DataFrame:
        df = df.copy()
        n_imputed = n_clipped = 0
        for col in self.feature_cols:
            if col not in df.columns:
                continue
            col_data = df[col].replace([np.inf, -np.inf], np.nan)
            nan_mask = col_data.isna()
            if nan_mask.any():
                col_data = col_data.fillna(self.medians_.get(col, 0.0))
                n_imputed += int(nan_mask.sum())
            clip_val = self.clip_upper_.get(col)
            if clip_val is not None:
                clip_mask = col_data > clip_val
                if clip_mask.any():
                    col_data = col_data.clip(upper=clip_val)
                    n_clipped += int(clip_mask.sum())
            df[col] = col_data
        if n_imputed or n_clipped:
            print(f"  Preprocessor [{label}]: imputed={n_imputed}  clipped={n_clipped}")
        return df

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        self.fit(train_df)
        return self.transform(train_df, label="TRAIN")

    def to_dict(self) -> Dict:
        return {"medians": self.medians_, "clip_upper": self.clip_upper_}

def balance_train(
    train_df: pd.DataFrame,
    rng:      np.random.Generator,
) -> pd.DataFrame:
    a2_train = train_df[train_df["attack_type"] == "flooding"]
    a5_train = train_df[train_df["attack_type"] == "slow_poisoning"]

    if len(a2_train) == 0 or len(a5_train) == 0:
        print(" Balancing skipped (one attack class absent from train)")
        return train_df

    n_bal  = min(len(a2_train), len(a5_train))
    a2_bal = a2_train.iloc[rng.choice(len(a2_train), n_bal, replace=False)]
    a5_bal = a5_train.iloc[rng.choice(len(a5_train), n_bal, replace=False)]
    normal = train_df[train_df["label"] == 1]

    n_dropped = (len(a2_train) - n_bal) + (len(a5_train) - n_bal)
    print(f"  Balancing (TRAIN): A2={n_bal}, A5={n_bal}, Normal={len(normal)}"
          f"  ({n_dropped} excess windows dropped)")
    return pd.concat([normal, a2_bal, a5_bal], ignore_index=True)

def validate_dataset_integrity(df: pd.DataFrame, stage: str = "") -> None:
    tag = f"[{stage}] " if stage else ""

    nans = int(df[ALL_FEATURES].isna().sum().sum())
    if nans > 0:
        raise ValueError(
            f"{tag}FATAL: {nans} NaN values detected after imputation. "
            f"Check TrainFittedPreprocessor (undefined median?)."
        )

    feat_f64 = df[ALL_FEATURES].astype("float64")
    infs = int(np.isinf(feat_f64.values).sum())
    if infs > 0:
        raise ValueError(f"{tag}FATAL: {infs} infinite values detected.")

    bad_labels = set(df["label"].unique()) - {0, 1}
    if bad_labels:
        raise ValueError(f"{tag}FATAL: Invalid labels detected: {bad_labels}")

    n_normal = int((df["label"] == 1).sum())
    n_attack = int((df["label"] == 0).sum())
    if n_attack > 0:
        ratio = n_normal / n_attack
        if ratio > 10.0:
            warnings.warn(
                f"{tag}⚠️  Normal:attack ratio = {ratio:.1f}:1 > 10:1. "
                f"Consider class_weight or under-sampling in downstream training.",
                stacklevel=2,
            )

    print(f"  OK {tag}Integrity check: {len(df)} windows, "
          f"{n_normal}N/{n_attack}A, 0 NaN, 0 Inf, labels={{0,1}}")

def _hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    if pooled < 1e-10:
        return 0.0
    g_raw = abs(a.mean() - b.mean()) / pooled
    return g_raw * (1.0 - 3.0 / (4.0 * (n1 + n2 - 2) - 1.0))

def main() -> None:

    _FILE_HASH = __import__('hashlib').md5(open(__file__, 'rb').read()).hexdigest()[:8]
    sep = "\n" + " " * 70 + "\n"

    print("=" * 70)
    print(f"BridgeGuard   Sliding Window Extraction  [{_FILE_HASH}]")
    print("=" * 70)
    print(f"  BG_BASELINE_DIR : {_BASELINE_DIR}")
    print(f"  BG_FEATURES_DIR : {_FEATURES_DIR}")
    print(f"  Window          : {CFG_NORMAL.window_size_min} min")
    print(f"  Step normal     : {CFG_NORMAL.step_min} min  ({CFG_NORMAL.overlap_pct}% overlap)")
    print(f"  Step attack     : {CFG_ATTACK2.step_min} min  ({CFG_ATTACK2.overlap_pct}% overlap)")

    print(f"{sep}1 3. PARALLEL EXTRACTION (ThreadPoolExecutor, 3 workers)")

    _N_WORKERS = int(os.getenv("BG_N_WORKERS", "3"))
    _tasks = [
        (NORMAL_CSV,  CFG_NORMAL,  "NORMAL TRAFFIC",          False),
        (ATTACK2_CSV, CFG_ATTACK2, "ATTACK 2 FLOODING",        True),
        (ATTACK5_CSV, CFG_ATTACK5, "ATTACK 5 SLOW POISONING",  True),
    ]
    _results: Dict[str, List[Dict]] = {}

    with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
        future_map = {
            pool.submit(extract_sliding_windows, path, cfg): (name, dedup)
            for path, cfg, name, dedup in _tasks
        }
        for future in as_completed(future_map):
            name, dedup = future_map[future]
            rows_out = future.result()
            if dedup:
                rows_out = deduplicate_windows(rows_out)
            _results[name] = rows_out
            print(f"  OK {name:<30} : {len(rows_out)} windows")

    normal_rows  = _results["NORMAL TRAFFIC"]
    attack2_rows = _results["ATTACK 2 FLOODING"]
    attack5_rows = _results["ATTACK 5 SLOW POISONING"]

    print(f"{sep}4. DATASET ASSEMBLY")
    all_rows = normal_rows + attack2_rows + attack5_rows
    df_raw = pd.DataFrame(all_rows)

    for f in ALL_FEATURES:
        if f not in df_raw.columns:
            df_raw[f] = np.nan
        df_raw[f] = pd.to_numeric(df_raw[f], errors="coerce")

    print(f"  Total: {len(df_raw)} windows  "
          f"(normal={len(normal_rows)}, A2={len(attack2_rows)}, A5={len(attack5_rows)})")

    print(f"{sep}5. CHRONOLOGICAL SPLIT (per class)")
    train_raw, test_raw = chronological_split_per_class(df_raw, test_size=TEST_SIZE)
    print(f"\n  Train: {len(train_raw)} windows  |  Test: {len(test_raw)} windows")

    print(f"{sep}6. BALANCING (train only)")
    train_balanced = balance_train(train_raw, _RNG)

    print(f"{sep}7. IMPUTATION + CLIPPING (fit=TRAIN, apply=TRAIN+TEST)")
    preprocessor = TrainFittedPreprocessor(feature_cols=ALL_FEATURES, clip_percentile=99.0)
    train_clean  = preprocessor.fit_transform(train_balanced)
    test_clean   = preprocessor.transform(test_raw, label="TEST")
    print(f"\n  Fitted medians: {preprocessor.medians_}")

    print(f"{sep}7b. DATASET INTEGRITY CHECK")
    validate_dataset_integrity(train_clean, stage="TRAIN")
    validate_dataset_integrity(test_clean,  stage="TEST")

    print(f"{sep}8. TEMPORAL ORDER: PRESERVE_TEMPORAL_ORDER={PRESERVE_TEMPORAL_ORDER}")
    if PRESERVE_TEMPORAL_ORDER:
        train_out = train_clean.sort_values("window_start_time").reset_index(drop=True)
        test_out  = test_clean.sort_values("window_start_time").reset_index(drop=True)
        print(" Temporal order preserved compatible with LSTM TimeSeriesSplit")
    else:
        train_out = train_clean.sample(frac=1, random_state=42).reset_index(drop=True)
        test_out  = test_clean.sample(frac=1, random_state=43).reset_index(drop=True)
        print(" Shuffled use only if the downstream model does not rely on temporal order")

    print(f"{sep}9. SAVING OUTPUTS")
    df_full = pd.concat([train_out, test_out], ignore_index=True)
    df_full.to_csv(OUTPUT_LABELED, index=False)
    df_full[df_full["label"] == 1].to_csv(OUTPUT_NORMAL, index=False)
    train_out.to_csv(OUTPUT_TRAIN, index=False)
    test_out.to_csv(OUTPUT_TEST,  index=False)

    print(f"  {OUTPUT_LABELED:<50} {len(df_full):>5} windows")
    print(f"  {OUTPUT_NORMAL:<50} {(df_full['label'] == 1).sum():>5} windows")
    print(f"  {OUTPUT_TRAIN:<50} {len(train_out):>5} windows")
    print(f"  {OUTPUT_TEST:<50} {len(test_out):>5} windows")

    print(f"\n{'=' * 70}")
    print("FEATURE QUALITY DIAGNOSTIC (Hedges' g bias-corrected effect size)")
    print("=" * 70)

    n_df  = df_full[df_full["label"] == 1]
    a2_df = df_full[df_full["attack_type"] == "flooding"]
    a5_df = df_full[df_full["attack_type"] == "slow_poisoning"]

    print(f"\n  {'Feature':<35} {'g(N,A2)':>8} {'g(N,A5)':>8}  {'min(g)':>7}  Status")
    print(" " + " " * 67)
    strong_a2 = strong_a5 = 0
    for f in ALL_FEATURES:
        if f not in df_full.columns:
            continue
        ga2  = _hedges_g(n_df[f].dropna().values, a2_df[f].dropna().values)
        ga5  = _hedges_g(n_df[f].dropna().values, a5_df[f].dropna().values)
        gmin = min(ga2, ga5)
        ok2, ok5 = ga2 > 0.8, ga5 > 0.8
        if ok2: strong_a2 += 1
        if ok5: strong_a5 += 1
        status = "LARGE " if gmin >= 0.8 else ("MEDIUM" if gmin >= 0.5 else "small")
        print(f"  {f:<35} {ga2:>7.3f}{'*' if ok2 else ' '}  "
              f"{ga5:>7.3f}{'*' if ok5 else ' '}  {gmin:>7.3f}  {status}")
    print(f"\n  LARGE vs Flooding     : {strong_a2}/{len(ALL_FEATURES)}")
    print(f"  LARGE vs Slow Poison  : {strong_a5}/{len(ALL_FEATURES)}")

    stats = {
        "total":                int(len(df_full)),
        "normal":               int((df_full["label"] == 1).sum()),
        "attack2_flooding":     int(len(a2_df)),
        "attack5_slow":         int(len(a5_df)),
        "train":                int(len(train_out)),
        "test":                 int(len(test_out)),
        "window_size_min":      CFG_NORMAL.window_size_min,
        "step_normal_min":      CFG_NORMAL.step_min,
        "step_attack_min":      CFG_ATTACK2.step_min,
        "overlap_normal_pct":   CFG_NORMAL.overlap_pct,
        "overlap_attack_pct":   CFG_ATTACK2.overlap_pct,
        "test_size":            TEST_SIZE,
        "preserve_order":       PRESERVE_TEMPORAL_ORDER,
        "min_emergency_attack": CFG_ATTACK2.min_emergency,
        "preprocessor":         preprocessor.to_dict(),
        "source":               "100% real MQTT captures (interleaved)",
        "synthetic":            False,
    }
    stats_path = os.path.join(_FEATURES_DIR, "real_dataset_stats.json")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"\n  Audit trail saved: {stats_path}")

    rows = [
        ("Total windows",     str(stats["total"])),
        ("Normal windows",    str(stats["normal"])),
        ("Attack2 (flood)",   str(stats["attack2_flooding"])),
        ("Attack5 (slow)",    str(stats["attack5_slow"])),
        ("Train windows",     str(stats["train"])),
        ("Test windows",      str(stats["test"])),
        ("Features",          str(len(ALL_FEATURES))),
        ("Window size (min)", str(stats["window_size_min"])),
        ("Test size",         str(stats["test_size"])),
        ("Audit trail",       stats_path),
        ("Next step",         "python feature_selection.py"),
    ]
    c1 = max(len(r[0]) for r in rows)
    c2 = max(len(r[1]) for r in rows)
    sep = f"+{'-'*(c1+2)}+{'-'*(c2+2)}+"
    print("\n" + sep)
    print(f"| {'Sliding Window Extraction':<{c1}} | {'':<{c2}} |")
    print(sep)
    for label, val in rows:
        print(f"| {label:<{c1}} | {val:<{c2}} |")
    print(sep)

if __name__ == "__main__":
    main()
