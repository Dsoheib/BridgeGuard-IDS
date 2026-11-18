"""Injector checks on synthetic stretches (fast) and on the built injection set (if present)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bridgeguard import inject
from bridgeguard.labelguard import LABEL_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
INJ = ROOT / "dev_data" / "injected"


def test_a5_script_matches_generator():
    rng = np.random.default_rng(0)
    for _ in range(200):
        ev, d = inject.a5_script(rng)
        waits = np.diff(ev["offset_s"])
        assert len(ev) == 20 and waits.min() >= 300 and waits.max() <= 600
        assert d == pd.Timedelta(seconds=ev["offset_s"].iloc[-1])
        assert set(ev["topic"]) <= {f"retirement/emergency/{k}" for k in inject.SCRIPT_WEIGHTS}


def test_poisson_event_count():
    # Gaps inside a fixed window are truncated (biased short), so check the count: E[N] = 120 / 9.
    rng = np.random.default_rng(1)
    counts = [len(inject.poisson(rng, 9.0)[0]) for _ in range(2000)]
    assert abs(np.mean(counts) - 120 / 9.0) < 0.3


def test_placement_inside_stretches_with_gaps():
    t0 = pd.Timestamp("2026-04-01")
    spans = [(t0 + pd.Timedelta(hours=h), t0 + pd.Timedelta(hours=h + 8)) for h in range(0, 200, 10)]
    rng = np.random.default_rng(2)
    durs = [pd.Timedelta(hours=2)] * 30
    starts = inject.place(durs, spans, rng)
    iv = sorted((s, s + d) for s, d in zip(starts, durs))
    assert all(any(a <= s and e <= b for a, b in spans) for s, e in iv)
    assert all(iv[i + 1][0] - iv[i][1] >= pd.Timedelta(hours=1) for i in range(len(iv) - 1))


WIN = ROOT / "dev_data" / "windows"


@pytest.mark.skipif(not (WIN / "zone_C_clean.csv").exists(), reason="run: run_dev.py --stage windows")
def test_windows_share_grid_and_carry_no_labels():
    clean = pd.read_csv(WIN / "zone_C_clean.csv", usecols=["window_start_time"])
    for f in WIN.glob("zone_*.csv"):
        cols = pd.read_csv(f, nrows=1).columns
        assert not set(cols) & LABEL_COLUMNS, f.name
    for f in WIN.glob("zone_C_*_copy*.csv"):
        starts = pd.read_csv(f, usecols=["window_start_time"])["window_start_time"]
        assert set(clean["window_start_time"]) <= set(starts), f.name   # same grid as the clean stream


@pytest.mark.skipif(not (INJ / "episodes.json").exists(), reason="run: run_dev.py --stage inject")
def test_built_injection_set():
    episodes = pd.DataFrame(json.loads((INJ / "episodes.json").read_text()))
    assert (episodes.groupby(["attack", "copy"]).size() == 30).all()
    zc = json.loads((ROOT / "lockbox" / "manifest.json").read_text())["zones"]["C"]
    assert (pd.to_datetime(episodes["start"]) >= pd.Timestamp(zc["start"])).all()
    assert (pd.to_datetime(episodes["end"]) <= pd.Timestamp(zc["end"])).all()
    for f in INJ.glob("*_copy*.csv"):
        ev = pd.read_csv(f, nrows=50000)
        assert not set(ev.columns) & LABEL_COLUMNS
        if not f.name.startswith("A2"):
            assert not ev["message"].astype(str).str.contains("FAKE ALERT").any()
