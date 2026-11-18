"""Zone A only: per-bed and ward alert counts per complete hour, and the ward-hour count each candidate attack would
produce if its alerts are added to one bed or replace that bed's alerts. No detector, no Zone B/C.

Attacked-hour counts are simulated from the injector's own processes (2,000 episodes each, uniform phase against the
hour grid, every complete hour an episode overlaps). Each attacked hour is paired with a random Zone A complete hour and
bed: added = ward + injected; replaced = ward - (bed alerts x overlap fraction) + injected.
"""
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridgeguard import inject  # noqa: E402

N_EPISODES, SEED = 2000, 7


def zone_a_counts(events_gz, hours_csv, n_beds):
    parts = []
    for ch in pd.read_csv(events_gz, usecols=["timestamp", "type", "topic"], chunksize=2_000_000):
        e = ch[ch["type"] == "emergency"]
        parts.append(pd.DataFrame({"t": pd.to_datetime(e["timestamp"]),
                                   "bed": e["topic"].str.split("/").str[1].str[3:].astype(int)}))
    em = pd.concat(parts, ignore_index=True)
    hs = pd.to_datetime(pd.read_csv(hours_csv)["window_start_time"]).values
    idx = np.searchsorted(hs, em["t"].values, "right") - 1
    inside = (idx >= 0) & (em["t"].values < hs[np.clip(idx, 0, None)] + np.timedelta64(1, "h"))
    bed = np.zeros((len(hs), n_beds), dtype=int)
    np.add.at(bed, (idx[inside], em["bed"].values[inside]), 1)
    return em, bed


def q(x):
    x = np.asarray(x, float)
    return {"mean": round(float(x.mean()), 2), "sd": round(float(x.std()), 2),
            **{f"p{p}": float(np.percentile(x, p)) for p in (5, 10, 50, 90, 95)}}


def episode_hours(offsets_s, length_s, rng):
    """Injected count and overlap fraction for each hour of the grid that the episode overlaps."""
    phase = rng.uniform(0, 3600)
    t = np.asarray(offsets_s) + phase
    first, last = 0, int((phase + length_s) // 3600)
    out = []
    for h in range(first, last + 1):
        lo, hi = max(h * 3600, phase), min((h + 1) * 3600, phase + length_s)
        if hi <= lo:
            continue
        out.append((int(((t >= h * 3600) & (t < (h + 1) * 3600)).sum()), (hi - lo) / 3600))
    return out


def main():
    cfg = yaml.safe_load(open("configs/protocol.yaml"))
    c, tb = cfg["inject"], cfg["testbed"]
    em, bed = zone_a_counts(Path(tb["out_dir"]) / "zone_A_events.csv.gz", "testbed_data/windows/zone_A_hours.csv",
                            tb["n_beds"])
    ward = bed.sum(axis=1)
    per_bed_rate = float(bed.mean())
    length = pd.Timedelta(hours=c["episode_hours"])

    def old_emergencies(path):
        d = pd.read_csv(path, usecols=["timestamp", "type", "topic", "heart_rate", "emergency_type", "message"])
        d = d[d["type"] == "emergency"].reset_index(drop=True)
        d["timestamp"] = pd.to_datetime(d["timestamp"])      # parsed once; replay() would re-parse per episode
        return d

    a5, a2 = old_emergencies(c["attack5_file"]), old_emergencies(c["attack2_file"])
    ref_bed = em.assign(timestamp=em["t"], topic="", heart_rate=np.nan, emergency_type="", message="")
    procs = {
        "A5-S script (20 alerts, 300-600 s waits)": inject.a5_script,
        "A5-R development-stream A5 timing, 2 h": partial(inject.replay, source=a5, length=length),
        f"A5-P Poisson at the Zone A per-bed mean ({per_bed_rate:.2f}/h)":
            partial(inject.poisson, mean_gap_min=60 / per_bed_rate, length=length),
        "A5-M one Zone A bed's alerts, 2 h segment": None,
        "A5-U Poisson 2.0/h": partial(inject.poisson, mean_gap_min=30.0, length=length),
        "A2 development-stream A2 timing, 2 h": partial(inject.replay, source=a2, length=length),
    }
    ref_by_bed = {b: ref_bed[ref_bed["bed"] == b].reset_index(drop=True) for b in range(tb["n_beds"])}
    rng = np.random.default_rng(SEED)
    res = {"zone_A_complete_hours": len(ward), "ward_alerts_per_hour": q(ward),
           "per_bed_alerts_per_hour": q(bed.ravel()), "per_bed_zero_hour_share": round(float((bed == 0).mean()), 3),
           "per_bed_mean_by_bed": [round(float(x), 2) for x in bed.mean(axis=0)], "attacks": {}}
    for name, make in procs.items():
        inj, frac = [], []
        for _ in range(N_EPISODES):
            if make is None:                                   # A5-M: a random bed's own alerts over a 2 h segment
                b = rng.integers(tb["n_beds"])
                ev, ln = inject.replay(rng, ref_by_bed[b], length)
            else:
                ev, ln = make(rng)
            for k, f in episode_hours(ev["offset_s"].values, ln.total_seconds(), rng):
                inj.append(k)
                frac.append(f)
        inj, frac = np.array(inj), np.array(frac)
        h = rng.integers(len(ward), size=len(inj))
        b = rng.integers(tb["n_beds"], size=len(inj))
        added = ward[h] + inj
        replaced = ward[h] - np.rint(bed[h, b] * frac).astype(int) + inj
        res["attacks"][name] = {
            "attacked_hours": len(inj), "injected_per_attacked_hour": q(inj),
            "ward_hour_if_added": q(added), "ward_hour_if_replaced": q(replaced),
            "added_share_above_zoneA_ward_p95": round(float((added > np.percentile(ward, 95)).mean()), 3),
            "replaced_share_above_zoneA_ward_p95": round(float((replaced > np.percentile(ward, 95)).mean()), 3),
            "replaced_share_below_zoneA_ward_median": round(float((replaced < np.median(ward)).mean()), 3)}
    Path("results/testbed/zone_A_bed_rates.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
