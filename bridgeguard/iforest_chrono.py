"""Isolation forest trained on Zone A normal windows only, chronologically."""
import pandas as pd
from sklearn.ensemble import IsolationForest

from .labelguard import assert_label_free
from .zones import window_zone


def zone_a_windows(windows: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    s = pd.to_datetime(windows["window_start_time"]).dt.tz_localize(None)
    e = pd.to_datetime(windows["window_end_time"]).dt.tz_localize(None)
    keep = [window_zone(a, b, bounds) == "A" for a, b in zip(s, e)]
    return windows[keep]


def fit(windows: pd.DataFrame, features: list, bounds: dict, seed: int = 0):
    """Raises LabelLeakError if `windows` carries label columns; callers must strip them beforehand."""
    assert_label_free(windows, "iforest_chrono.fit")
    train = zone_a_windows(windows, bounds)
    X = train[features]
    model = IsolationForest(n_estimators=200, random_state=seed).fit(X.values)
    return model, train
