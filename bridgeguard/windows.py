"""Class-agnostic windowing on a fixed grid.

Same window and stride for all traffic; every window with at least one message is kept.
Per-window features come from bridgeguard.features.extract_features.
Labels never enter the feature table: window labels (overlap with an episode) are a separate frame.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from bridgeguard.features import WindowConfig, extract_features


def grid(start: pd.Timestamp, end: pd.Timestamp, window_min: int, stride_min: int) -> pd.DataFrame:
    w, s = pd.Timedelta(minutes=window_min), pd.Timedelta(minutes=stride_min)
    starts = pd.date_range(start, end - w, freq=s)
    return pd.DataFrame({"window_start_time": starts, "window_end_time": starts + w})


def featurize(events: pd.DataFrame, g: pd.DataFrame, window_min: int, stride_min: int) -> pd.DataFrame:
    ev = events.copy()
    ev["timestamp"] = pd.to_datetime(ev["timestamp"], utc=True)
    ev = ev.sort_values("timestamp").reset_index(drop=True)
    cfg = WindowConfig(window_size_min=window_min, step_min=stride_min, label=0, attack_type="",
                       min_emergency=0)
    ts = ev["timestamp"].values
    gs = g["window_start_time"].dt.tz_localize("UTC").values
    ge = g["window_end_time"].dt.tz_localize("UTC").values
    i0, i1 = np.searchsorted(ts, gs, "left"), np.searchsorted(ts, ge, "left")
    rows = []
    for a, b, s, e in zip(i0, i1, g["window_start_time"], g["window_end_time"]):
        if a == b:
            continue
        view = ev.iloc[a:b]
        f = extract_features(view, cfg)
        f.update(window_start_time=s, window_end_time=e, n_messages=int(b - a),
                 n_emergency=int((view["type"].values == "emergency").sum()))
        rows.append(f)
    out = pd.DataFrame(rows)
    # sequential feature: rolling std of alert frequency over the last 5 windows of this stream
    out["freq_rolling_std_5w"] = out["alert_frequency_per_hour"].rolling(5, min_periods=3).std(ddof=1)
    return out


def featurize_stream(path, g: pd.DataFrame, window_min: int, stride_min: int, chunksize: int = 1_000_000,
                     max_gap=pd.Timedelta(hours=1)):
    """Same output as featurize, from a time-sorted event file read in chunks (only the open windows in memory).

    Also returns the stretches (no gap > max_gap) seen while streaming, for hourly_complete(spans=...).
    """
    cfg = WindowConfig(window_size_min=window_min, step_min=stride_min, label=0, attack_type="", min_emergency=0)
    gs = [np.datetime64(x) for x in g["window_start_time"].dt.tz_localize("UTC").dt.tz_convert(None)]
    ge = [np.datetime64(x) for x in g["window_end_time"].dt.tz_localize("UTC").dt.tz_convert(None)]
    rows, buf, k = [], None, 0
    spans, s0, prev = [], None, None

    def emit(upto):
        nonlocal k, buf
        while k < len(gs) and (upto is None or ge[k] <= upto):
            ts = buf["timestamp"].values
            a, b = np.searchsorted(ts, gs[k], "left"), np.searchsorted(ts, ge[k], "left")
            if a < b:
                view = buf.iloc[a:b]
                f = extract_features(view, cfg)
                f.update(window_start_time=g["window_start_time"].iloc[k], window_end_time=g["window_end_time"].iloc[k],
                         n_messages=int(b - a), n_emergency=int((view["type"].values == "emergency").sum()))
                rows.append(f)
            k += 1
            if k < len(gs):
                buf = buf.iloc[np.searchsorted(buf["timestamp"].values, gs[k], "left"):]

    source = pd.read_csv(path, chunksize=chunksize) if isinstance(path, (str, Path)) else path   # or chunk iterator
    for chunk in source:
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], utc=True)
        t = chunk["timestamp"].dt.tz_convert(None)
        d = t.diff()
        if prev is None:
            s0 = t.iloc[0]
        else:
            d.iloc[0] = t.iloc[0] - prev
        for i in np.flatnonzero((d > max_gap).values):
            spans.append((s0, t.iloc[i - 1] if i else prev))
            s0 = t.iloc[i]
        prev = t.iloc[-1]
        buf = chunk if buf is None else pd.concat([buf, chunk], ignore_index=True)
        emit(np.datetime64(prev))    # every window ending at or before the last timestamp read is complete
    emit(None)
    spans.append((s0, prev))
    out = pd.DataFrame(rows)
    out["freq_rolling_std_5w"] = out["alert_frequency_per_hour"].rolling(5, min_periods=3).std(ddof=1)
    return out, spans


def window_labels(windows: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    """Evaluation-only labels: a window is attack if it overlaps any episode of its copy."""
    s = windows["window_start_time"].values
    e = windows["window_end_time"].values
    lab = np.zeros(len(windows), dtype=int)
    epi = np.full(len(windows), -1)
    for _, r in episodes.iterrows():
        hit = (s < np.datetime64(pd.Timestamp(r["end"]))) & (e > np.datetime64(pd.Timestamp(r["start"])))
        lab[hit] = 1
        epi[hit] = r["episode"]
    return pd.DataFrame({"window_start_time": windows["window_start_time"], "attack": lab, "episode": epi})


def merged_alerts_per_episode(background: pd.DataFrame, injected: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    """Alerts inside each episode interval [start, end] after merging: background + injected (per hour).

    Closed interval: an A5-S episode ends exactly on its last alert, which [start, end) would drop.
    """
    bt = pd.to_datetime(background.loc[background["type"] == "emergency", "timestamp"])
    it = pd.to_datetime(injected["timestamp"])
    rows = []
    for _, r in episodes.iterrows():
        s, e = pd.Timestamp(r["start"]), pd.Timestamp(r["end"])
        h = (e - s).total_seconds() / 3600
        nb, ni = int(((bt >= s) & (bt <= e)).sum()), int(((it >= s) & (it <= e)).sum())
        rows.append({"attack": r["attack"], "copy": r["copy"], "episode": r["episode"], "hours": h,
                     "background_alerts": nb, "injected_alerts": ni,
                     "merged_rate_per_h": (nb + ni) / h, "injected_rate_per_h": ni / h})
    return pd.DataFrame(rows)
