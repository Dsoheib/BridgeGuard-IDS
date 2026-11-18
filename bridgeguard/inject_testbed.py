"""Testbed attack injector (revision/TESTBED_SCORERS.md, "Testbed attacks"; configs/v2.yaml, testbed_inject).

Timing comes from the development injector's processes (inject.a5_script, inject.poisson, alert-anchored inject.replay)
and placement from inject.place. Only the payload differs:
- forged alerts (A5-S, A5-R, A5-P, A5-U) copy the signal, direction and value of a uniformly drawn Zone A emergency;
- A5-M keeps each replayed Zone A alert's signal and value and only changes the bed id;
- A2 floods use the same forged payload rule at the old flood timing.
Every injected row is an emergency on ward/bed<b>/emergency/<signal> with the testbed JSON. heart_rate is left empty:
the features never read it, and a forger has no access to the bed's vital sample.
Replace-mode episodes (A5-M, A5-U) also name the bed whose own emergencies inside [start, end] are removed.
"""
import json

import numpy as np
import pandas as pd

from . import inject

COLUMNS = ["timestamp", "type", "topic", "message", "heart_rate"]


def zone_a_alerts(events_gz, chunksize=2_000_000) -> pd.DataFrame:
    """Zone A emergencies with bed, signal, alert and value parsed from the testbed JSON."""
    parts = []
    for ch in pd.read_csv(events_gz, usecols=["timestamp", "type", "topic", "message"], chunksize=chunksize):
        parts.append(ch[ch["type"] == "emergency"])
    em = pd.concat(parts, ignore_index=True)
    msg = em["message"].map(json.loads)
    return pd.DataFrame({"timestamp": pd.to_datetime(em["timestamp"]), "bed": msg.map(lambda m: m["bed"]),
                         "alert": msg.map(lambda m: m["alert"]), "value": msg.map(lambda m: m["value"])})


def _rows(start, offsets_s, bed, alerts, values) -> pd.DataFrame:
    sig = [a.split("_")[0] for a in alerts]
    return pd.DataFrame({
        "timestamp": start + pd.to_timedelta(np.asarray(offsets_s, float), unit="s"),
        "type": "emergency",
        "topic": [f"ward/bed{bed}/emergency/{s}" for s in sig],
        "message": [json.dumps({"bed": int(bed), "alert": a, "value": float(v)}, separators=(",", ":"))
                    for a, v in zip(alerts, values)],
        "heart_rate": np.nan})


def forged(offsets_s, pool: pd.DataFrame, rng):
    """Payload for forged alerts: one uniformly drawn Zone A emergency per offset."""
    k = rng.integers(len(pool), size=len(offsets_s))
    return pool["alert"].values[k], pool["value"].values[k]


def episode_timing(attack, rng, cfg, a5_src, a2_src, pool_by_bed):
    """(offsets_s, length, alerts, values) for one episode; alerts/values None means forged payload."""
    length = pd.Timedelta(hours=cfg["episode_hours"])
    if attack == "A5-S":
        ev, ln = inject.a5_script(rng)
    elif attack == "A5-P":
        ev, ln = inject.poisson(rng, 60 / cfg["a5p_rate_per_hour"], length)
    elif attack == "A5-U":
        ev, ln = inject.poisson(rng, 60 / cfg["a5u_rate_per_hour"], length)
    elif attack == "A5-R":
        ev, ln = inject.replay(rng, a5_src, length)
    elif attack == "A2":
        ev, ln = inject.replay(rng, a2_src, length)
    elif attack == "A5-M":
        src = pool_by_bed[int(rng.integers(len(pool_by_bed)))]
        t = src["timestamp"]
        s = t.iloc[rng.integers(len(t))] - length * rng.random()            # alert-anchored, as inject.replay
        seg = src[(t >= s) & (t < s + length)]
        return (seg["timestamp"] - s).dt.total_seconds().values, length, seg["alert"].values, seg["value"].values
    else:
        raise ValueError(attack)
    return ev["offset_s"].values, ln, None, None


def place_bed_first(durations, beds, bed_spans, rng, gap=inject.GAP, tries=20000):
    """Per-bed study (PERBED_PROTOCOL 7.3): each episode inside one stretch of its own bed; one taken list per copy,
    so the >= 1 h gap holds across beds. Same sampling as inject.place, restricted to the drawn bed's stretches."""
    taken, starts = [], []
    for d, b in zip(durations, beds):
        fits = [(a, z) for a, z in bed_spans[b] if z - a >= d]
        if not fits:
            raise RuntimeError(f"bed {b} has no stretch of {d}")
        for _ in range(tries):
            a, z = fits[rng.integers(len(fits))]
            s = a + (z - a - d) * rng.random()
            if all(s + d + gap <= ts or s >= te + gap for ts, te in taken):
                taken.append((s, s + d))
                starts.append(s)
                break
        else:
            raise RuntimeError(f"could not place episode {len(starts) + 1} of {len(durations)}")
    return starts


def build_bed_first(attack, bed_spans, cfg, copy, pool, a5_src, a2_src, n_beds):
    """Per-bed study: like build(), but the bed is drawn first and the episode placed in that bed's stretches."""
    rng = np.random.default_rng([cfg["seed"], copy, sum(map(ord, attack))])
    pool_by_bed = [pool[pool["bed"] == b].reset_index(drop=True) for b in range(n_beds)]
    drawn = [episode_timing(attack, rng, cfg, a5_src, a2_src, pool_by_bed) for _ in range(cfg["episodes_per_copy"])]
    beds = [int(b) for b in rng.integers(n_beds, size=len(drawn))]
    starts = place_bed_first([d[1] for d in drawn], beds, bed_spans, rng)
    mode = cfg["mode"][attack]
    events, episodes = [], []
    for i, ((offs, ln, alerts, values), s, bed) in enumerate(zip(drawn, starts, beds)):
        if alerts is None:
            alerts, values = forged(offs, pool, rng)
        events.append(_rows(s, offs, bed, alerts, values))
        episodes.append({"attack": attack, "copy": copy, "episode": i, "bed": bed, "mode": mode,
                         "start": str(s), "end": str(s + ln), "n_injected": int(len(offs))})
    return pd.concat(events, ignore_index=True)[COLUMNS], episodes


def build(attack, spans, cfg, copy, pool, a5_src, a2_src, n_beds):
    """Events and episode records for one attack and one placement copy (seeded by seed, copy and attack)."""
    rng = np.random.default_rng([cfg["seed"], copy, sum(map(ord, attack))])
    pool_by_bed = [pool[pool["bed"] == b].reset_index(drop=True) for b in range(n_beds)]
    drawn = [episode_timing(attack, rng, cfg, a5_src, a2_src, pool_by_bed) for _ in range(cfg["episodes_per_copy"])]
    starts = inject.place([d[1] for d in drawn], spans, rng)
    mode = cfg["mode"][attack]
    events, episodes = [], []
    for i, ((offs, ln, alerts, values), s) in enumerate(zip(drawn, starts)):
        bed = int(rng.integers(n_beds))
        if alerts is None:
            alerts, values = forged(offs, pool, rng)
        events.append(_rows(s, offs, bed, alerts, values))
        episodes.append({"attack": attack, "copy": copy, "episode": i, "bed": bed, "mode": mode,
                         "start": str(s), "end": str(s + ln), "n_injected": int(len(offs))})
    return pd.concat(events, ignore_index=True)[COLUMNS], episodes
