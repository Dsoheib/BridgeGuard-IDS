import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "testbed"))
import run_testbed as R  # noqa: E402


def test_replace_removes_only_that_beds_emergencies_in_window_and_keeps_order(tmp_path, monkeypatch):
    t0 = pd.Timestamp("2026-01-05")
    rows = []
    for s in range(0, 600):
        for bed in (0, 1):
            rows.append((t0 + pd.Timedelta(seconds=s), "normal", f"ward/bed{bed}/vitals", "{}", 80.0))
            if s % 50 == 0:
                rows.append((t0 + pd.Timedelta(seconds=s), "emergency", f"ward/bed{bed}/emergency/resp", "{}", 80.0))
    clean = pd.DataFrame(rows, columns=["timestamp", "type", "topic", "message", "heart_rate"])
    (tmp_path / "zones").mkdir()
    clean.to_csv(tmp_path / "zones" / "zone_C_events.csv.gz", index=False)
    monkeypatch.setattr(R, "ZONES", tmp_path / "zones")
    inj = pd.DataFrame({"timestamp": [t0 + pd.Timedelta(seconds=x) for x in (120.5, 130.0, 599.0)],
                        "type": "emergency", "topic": "ward/bed0/emergency/spo2", "message": "{}", "heart_rate": np.nan})
    removals = [(0, t0 + pd.Timedelta(seconds=100), t0 + pd.Timedelta(seconds=300))]
    out = pd.concat(list(R.merged_chunks(inj, removals, chunksize=97)), ignore_index=True)
    ts = pd.to_datetime(out["timestamp"])
    assert ts.is_monotonic_increasing
    bed0_emg = out[(out["topic"] == "ward/bed0/emergency/resp")]
    removed = clean[(clean["topic"] == "ward/bed0/emergency/resp") & (clean["timestamp"] >= removals[0][1])
                    & (clean["timestamp"] < removals[0][2])]
    assert len(removed) == 4                                   # seconds 100, 150, 200, 250
    assert len(bed0_emg) == len(clean[clean["topic"] == "ward/bed0/emergency/resp"]) - 4
    assert (out["type"] == "normal").sum() == (clean["type"] == "normal").sum()
    assert (out["topic"] == "ward/bed1/emergency/resp").sum() == (clean["topic"] == "ward/bed1/emergency/resp").sum()
    assert (out["topic"] == "ward/bed0/emergency/spo2").sum() == 3
