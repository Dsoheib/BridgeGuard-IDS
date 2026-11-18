import numpy as np
import pandas as pd

from bridgeguard import windows as W
from bridgeguard.hours import hourly_complete


def _events():
    rng = np.random.default_rng(0)
    t = pd.Timestamp("2026-01-05") + pd.to_timedelta(np.sort(rng.uniform(0, 8 * 3600, 6000)), unit="s")
    t = t[(t < pd.Timestamp("2026-01-05 03:00")) | (t >= pd.Timestamp("2026-01-05 04:30"))]   # 1.5 h gap
    emg = rng.random(len(t)) < 0.05
    return pd.DataFrame({"timestamp": t.strftime("%Y-%m-%d %H:%M:%S.%f"),
                         "type": np.where(emg, "emergency", "normal"),
                         "topic": np.where(emg, "ward/bed0/emergency/resp", "ward/bed0/vitals"),
                         "message": [f'{{"v":{i % 7}}}' for i in range(len(t))]})


def test_stream_matches_in_memory_featurize_and_hours(tmp_path):
    ev = _events()
    path = tmp_path / "ev.csv.gz"
    ev.to_csv(path, index=False)
    g = W.grid(pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-05 08:00"), 60, 10)
    a = W.featurize(ev, g, 60, 10)
    b, spans = W.featurize_stream(path, g, 60, 10, chunksize=257)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    assert len(spans) == 2
    ha, hb = hourly_complete(a, ev, 10, 60), hourly_complete(b, None, 10, 60, spans=spans)
    assert ha["window_start_time"].tolist() == hb["window_start_time"].tolist()
