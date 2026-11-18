import numpy as np
import pandas as pd

from bridgeguard import baselines as BL
from bridgeguard.perbed import BedCUSUM, BedLSTMAE


def _hours():
    # two beds, interleaved rows, same clock hours; bed 1 has a gap at hour 3
    rows = []
    for h in range(6):
        rows.append({"bed": 0, "hour_index": h, "alert_frequency_per_hour": 10.0})
        if h != 3:
            rows.append({"bed": 1, "hour_index": h, "alert_frequency_per_hour": 0.0})
    return pd.DataFrame(rows)


def test_bed_cusum_equals_cusum_run_on_each_bed_alone():
    hrs = _hours()
    m = BedCUSUM().fit(hrs)
    got = m.score(hrs)
    for b in (0, 1):
        sub = hrs[hrs["bed"] == b]
        ref = BL.CUSUM.score(m, sub)
        assert np.allclose(got[hrs["bed"].values == b], ref)


def test_bed_sequences_never_mix_beds():
    hrs = _hours()
    cfg = {"lstm_ae": {"seq_hours": 3}}
    m = BedLSTMAE(["alert_frequency_per_hour"], {}, 0, cfg)
    Z = np.where(hrs["bed"].values[:, None] == 0, 1.0, -1.0)
    seq, rows = m._seq(hrs, Z)
    for s, r in zip(seq, rows):
        assert np.all(s == Z[r, 0])                     # every hour in the sequence is from the scored row's bed
    # bed 0: hours 2..5 have 2 previous contiguous hours (4 rows); bed 1: only hour 2 (the gap at 3 breaks 4 and 5)
    assert len(rows) == 5


def test_ward_alarm_and_localized_detection():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "testbed"))
    import perbed as P
    t0 = pd.Timestamp("2026-01-05")
    rows = [{"bed": b, "hour_index": h, "window_start_time": t0 + pd.Timedelta(hours=h),
             "window_end_time": t0 + pd.Timedelta(hours=h + 1)} for h in range(4) for b in range(4)]
    hrs = pd.DataFrame(rows)
    p = np.full(len(hrs), 0.5)
    p[(hrs["bed"] == 2) & (hrs["hour_index"] == 1)] = 0.01        # alarm (<= 0.0125)
    p[(hrs["bed"] == 0) & (hrs["hour_index"] == 2)] = 0.02        # not an alarm at alpha/4
    m = P.bed_metrics(p, hrs, 0.05, 4, 0.05)
    assert m["ward_hours"] == 4 and m["ward_alarms"] == 1 and m["bed_hour_alarms"] == 1
    ep = pd.DataFrame([{"bed": 0, "start": str(t0 + pd.Timedelta(minutes=70)), "end": str(t0 + pd.Timedelta(minutes=110))},
                       {"bed": 2, "start": str(t0 + pd.Timedelta(minutes=70)), "end": str(t0 + pd.Timedelta(minutes=110))}])
    c = P.copy_bed_metrics(p, hrs, ep, 0.0125, 0.0, 40)
    assert c["episodes_detected"] == 1                              # bed 2 alarms in its episode; bed 0 does not
    assert c["attacked_bed_hours"] == 2 and c["episodes_covered"] == 2
