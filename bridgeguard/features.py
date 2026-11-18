"""Per-window behavioral features of an alert stream (timing, volume, burstiness, topic and payload diversity)."""

from __future__ import annotations

import os
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")

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

