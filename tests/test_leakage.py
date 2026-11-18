"""Label-leakage tests: the label firewall, purged chronological zones, the lock box and label-free fitting.

Run from the repository root: python -m pytest tests -q
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bridgeguard import iforest_chrono, lockbox
from bridgeguard.labelguard import LABEL_COLUMNS, LabelLeakError, assert_label_free, strip_labels
from bridgeguard.zones import split_events, window_zone, zone_bounds

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "bridgeguard_features"


# ---------------------------------------------------------------- mechanisms

def test_labelguard_rejects_and_strips():
    df = pd.DataFrame({"f": [1.0], "label": [0], "attack_type": ["x"]})
    with pytest.raises(LabelLeakError):
        assert_label_free(df, "test")
    assert list(strip_labels(df).columns) == ["f"]


def test_zone_bounds_are_ordered_and_purged():
    t0, t1 = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-11")
    b = zone_bounds(t0, t1, (0.6, 0.2, 0.2), purge_minutes=60)
    assert b["A"][1] + pd.Timedelta("60min") == b["B"][0]
    assert b["B"][1] + pd.Timedelta("60min") == b["C"][0]
    ts = pd.date_range(t0, t1, freq="1min")
    parts = split_events(pd.DataFrame({"timestamp": ts}), "timestamp", b)
    assert parts["A"].timestamp.max() < parts["B"].timestamp.min() < parts["C"].timestamp.min()
    assert sum(len(p) for p in parts.values()) == len(ts) - 2 * 60  # purged minutes dropped
    assert window_zone(b["A"][1] - pd.Timedelta("30min"), b["A"][1] + pd.Timedelta("30min"), b) is None


def test_lockbox_locks_zone_c_and_detects_tampering(tmp_path):
    src = tmp_path / "events.csv"
    pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=1000, freq="1min"),
                  "v": np.arange(1000)}).to_csv(src, index=False)
    man = tmp_path / "manifest.json"
    lockbox.write_zones(src, "timestamp", tmp_path / "zones", man, (0.6, 0.2, 0.2), 60)
    with pytest.raises(PermissionError):
        lockbox.load_zone(man, "C")
    assert len(lockbox.load_zone(man, "C", allow_test=True)) > 0
    with open(tmp_path / "zones" / "zone_B.csv", "a") as f:
        f.write("2026-01-02,0\n")
    with pytest.raises(RuntimeError):
        lockbox.load_zone(man, "B")


# ---------------------------------------------------------------- chronological fit (development windows)

@pytest.mark.skipif(not (FEATURES / "features_normal.csv").exists(), reason="development windows not present")
def test_chronological_iforest_control():
    sel = pd.read_csv(FEATURES / "features_selected_normal.csv")
    win = pd.read_csv(FEATURES / "features_normal.csv")
    assert len(sel) == len(win)
    assert np.allclose(sel["payload_entropy"], win["payload_entropy"])  # same rows, same (time) order
    windows = pd.concat([win[["window_start_time", "window_end_time"]], sel], axis=1)
    s = pd.to_datetime(win["window_start_time"]).dt.tz_localize(None)
    e = pd.to_datetime(win["window_end_time"]).dt.tz_localize(None)
    b = zone_bounds(s.min(), e.max(), (0.6, 0.2, 0.2), 60)
    with pytest.raises(LabelLeakError):  # labeled input is rejected, not silently cleaned
        iforest_chrono.fit(windows, list(sel.columns), b)
    features = [c for c in sel.columns if c not in LABEL_COLUMNS]
    model, train = iforest_chrono.fit(strip_labels(windows), features, b)
    train_end = pd.to_datetime(train["window_end_time"]).dt.tz_localize(None).max()
    assert train_end <= b["A"][1] < b["C"][0]
    assert len(train) > 0 and hasattr(model, "score_samples")


def test_selection_rejects_labeled_windows():
    from bridgeguard import select as S
    df = pd.DataFrame({"f": [0.0, 1.0], "label": [1, 0]})
    with pytest.raises(LabelLeakError):
        S.prepare(df, exclude=[], impute={})


def test_training_refuses_zone_c():
    import run_dev
    with pytest.raises(PermissionError):
        run_dev.zone_hours({}, "C")
