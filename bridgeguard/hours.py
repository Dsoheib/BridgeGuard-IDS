"""Complete hours: the non-overlapping scoring unit.

Hourly grid = every 6th 10-min-stride window from the zone start. A grid hour is complete if it
lies entirely inside one stretch of the zone's event stream (no gap > 1 h).
"""
import numpy as np
import pandas as pd

from .inject import stretches


def hourly_complete(windows: pd.DataFrame, events: pd.DataFrame, stride_min: int, window_min: int,
                    time_col: str = "timestamp", spans=None) -> pd.DataFrame:
    """events may be None when spans (the stretches, e.g. from featurize_stream) are given."""
    w = windows.copy()
    w["window_start_time"] = pd.to_datetime(w["window_start_time"])
    w["window_end_time"] = pd.to_datetime(w["window_end_time"])
    t0 = w["window_start_time"].min()
    step = (w["window_start_time"] - t0) / pd.Timedelta(minutes=stride_min)
    w = w[(step % (window_min // stride_min)) == 0].copy()
    if spans is None:
        spans = stretches(events[time_col])
    s = w["window_start_time"].values
    e = w["window_end_time"].values
    inside = np.zeros(len(w), dtype=bool)
    for a, b in spans:
        inside |= (s >= np.datetime64(a)) & (e <= np.datetime64(b))
    w = w[inside].copy()
    w["hour_index"] = ((w["window_start_time"] - t0) / pd.Timedelta(minutes=window_min)).astype(int)
    return w.reset_index(drop=True)
