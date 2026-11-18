import json

import numpy as np
import pandas as pd

from bridgeguard import inject_testbed as T

CFG = {"seed": 1, "episodes_per_copy": 5, "episode_hours": 2, "a5p_rate_per_hour": 6.77, "a5u_rate_per_hour": 2.0,
       "mode": {"A5-S": "add", "A5-P": "add", "A5-U": "replace", "A5-M": "replace", "A5-R": "add", "A2": "add"}}


def _pool():
    rng = np.random.default_rng(0)
    t = pd.Timestamp("2026-01-05") + pd.to_timedelta(np.sort(rng.uniform(0, 48 * 3600, 400)), unit="s")
    alerts = rng.choice(["resp_high", "spo2_low", "hr_high"], size=400, p=[0.76, 0.17, 0.07])
    return pd.DataFrame({"timestamp": t, "bed": rng.integers(4, size=400), "alert": alerts,
                         "value": rng.uniform(20, 140, 400).round(1)})


def test_payload_is_testbed_json_and_matches_topic():
    pool = _pool()
    spans = [(pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-08"))]
    for attack in ("A5-S", "A5-P", "A5-U", "A5-M"):
        ev, eps = T.build(attack, spans, CFG, 0, pool, None, None, 4)
        assert (ev["type"] == "emergency").all()
        for topic, msg in zip(ev["topic"], ev["message"]):
            m = json.loads(msg)
            bed, sig = topic.split("/")[1], topic.split("/")[3]
            assert bed == f"bed{m['bed']}" and m["alert"].startswith(sig)
            assert not topic.startswith("retirement/")
        assert all(e["mode"] == CFG["mode"][attack] for e in eps)


def test_forged_values_come_from_the_pool_and_mimic_keeps_segment_values():
    pool = _pool()
    spans = [(pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-08"))]
    pairs = set(zip(pool["alert"], pool["value"]))
    for attack in ("A5-P", "A5-M"):
        ev, _ = T.build(attack, spans, CFG, 0, pool, None, None, 4)
        got = {(json.loads(m)["alert"], json.loads(m)["value"]) for m in ev["message"]}
        assert got <= pairs


def test_bed_first_places_each_episode_in_its_own_beds_stretch_with_global_gap():
    pool = _pool()
    t0 = pd.Timestamp("2026-02-01")
    bed_spans = {b: [(t0 + pd.Timedelta(hours=10 * b), t0 + pd.Timedelta(hours=10 * b + 200))] for b in range(4)}
    cfg = dict(CFG, episodes_per_copy=20)
    ev, eps = T.build_bed_first("A5-P", bed_spans, cfg, 0, pool, None, None, 4)
    iv = sorted((pd.Timestamp(e["start"]), pd.Timestamp(e["end"])) for e in eps)
    assert all(b[0] >= a[1] + pd.Timedelta(hours=1) for a, b in zip(iv, iv[1:]))
    for e in eps:
        a, z = bed_spans[e["bed"]][0]
        assert a <= pd.Timestamp(e["start"]) and pd.Timestamp(e["end"]) <= z
    assert set(ev["topic"].str.split("/").str[1]) <= {f"bed{e['bed']}" for e in eps}
