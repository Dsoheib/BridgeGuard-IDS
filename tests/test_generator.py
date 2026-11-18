import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "testbed"))
from generator import band_entries  # noqa: E402


def test_fires_once_on_entry_and_rearms_after_60s_out():
    t = np.arange(300, dtype=float)
    v = np.full(300, 80.0)
    v[10:20] = 140       # entry at 10 -> fires
    v[40:50] = 140       # back after 20 s out -> no fire
    v[200:210] = 140     # back after 150 s out -> fires
    assert band_entries(t, v, "high", 131).tolist() == [10.0, 200.0]


def test_gap_counts_toward_rearm_and_record_start_in_band_does_not_fire():
    t = np.r_[np.arange(0, 10), np.arange(100, 110)].astype(float)    # 90 s gap
    v = np.r_[np.full(5, 140.0), np.full(5, 80.0), np.full(1, 80.0), np.full(9, 140.0)]
    # starts in band (no fire); out at 5..100; last in-band 4, out sample at 100 >= 64 -> armed; entry at 101 fires
    assert band_entries(t, v, "high", 131).tolist() == [101.0]


def test_low_band_and_boundary_values():
    t = np.arange(5, dtype=float)
    assert band_entries(t, np.array([95, 91, 95, 95, 95.0]), "low", 91).tolist() == [1.0]
