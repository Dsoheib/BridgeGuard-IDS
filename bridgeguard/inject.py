"""Attack injector: bounded episodes placed into Zone C stretches (plan, Phase 1).

Each episode is a set of emergency events with offsets from its start. Episodes are placed
inside one stretch (no gap > 1 h) with >= 1 h between episodes. Event files carry no labels;
the episode list (start, end, attack, copy) is the only label source and is stored separately.
"""
import numpy as np
import pandas as pd

SCRIPT_WEIGHTS = {"heart": 0.5, "fall": 0.3, "inactivity": 0.2}      # attack5_slow_poisoning.py:68
SCRIPT_TEMPLATES = {                                                  # attack5_slow_poisoning.py:70-86
    "heart": ["Heart rate: {hr} bpm - Elevated", "Irregular heartbeat detected - Rate: {hr} bpm",
              "Heart rate spike: {hr} bpm"],
    "fall": ["Sudden acceleration detected - Possible fall", "Impact sensor triggered - Fall suspected",
             "Patient down - Fall detector activated"],
    "inactivity": ["No movement detected for 3 hours", "Extended inactivity period - Check patient",
                   "Inactivity alert - 4 hour threshold exceeded"],
}
EVENT_COLUMNS = ["offset_s", "topic", "heart_rate", "emergency_type", "message"]
GAP = pd.Timedelta(hours=1)


# ---------------------------------------------------------------- stretches and placement

def stretches(t: pd.Series, max_gap=GAP) -> list:
    t = pd.to_datetime(t).sort_values().reset_index(drop=True)
    breaks = list(np.where(t.diff() > max_gap)[0])
    starts, ends = [0] + breaks, [b - 1 for b in breaks] + [len(t) - 1]
    return [(t[s], t[e]) for s, e in zip(starts, ends)]


def place(durations, spans, rng, gap=GAP, tries=20000) -> list:
    """Random non-overlapping starts; each episode inside one stretch, >= gap from the others."""
    taken, starts = [], []
    for d in durations:
        fits = [(a, b) for a, b in spans if b - a >= d]
        for _ in range(tries):
            a, b = fits[rng.integers(len(fits))]
            s = a + (b - a - d) * rng.random()
            if all(s + d + gap <= ts or s >= te + gap for ts, te in taken):
                taken.append((s, s + d))
                starts.append(s)
                break
        else:
            raise RuntimeError(f"could not place episode {len(starts) + 1} of {len(durations)}")
    return starts


# ---------------------------------------------------------------- episode generators

def _script_event(rng):
    etype = rng.choice(list(SCRIPT_WEIGHTS), p=list(SCRIPT_WEIGHTS.values()))
    tmpl = SCRIPT_TEMPLATES[etype][rng.integers(3)]
    hr = int(rng.integers(45, 176)) if etype == "heart" else np.nan
    return f"retirement/emergency/{etype}", hr, etype, tmpl.format(hr=hr) if etype == "heart" else tmpl


def _script_events(offsets, rng) -> pd.DataFrame:
    rows = [(o, *_script_event(rng)) for o in offsets]
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def a5_script(rng, n_alerts=20, wait=(300, 600)):
    """A5-S: the generator's paper mode, 20 alerts, randint waits of 300-600 s."""
    offsets = np.concatenate([[0], np.cumsum(rng.integers(wait[0], wait[1] + 1, n_alerts - 1))])
    return _script_events(offsets.astype(float), rng), pd.Timedelta(seconds=float(offsets[-1]))


def poisson(rng, mean_gap_min, length=pd.Timedelta(hours=2)):
    """A5-P / A5-U: Poisson arrivals with the script's event model over a fixed length."""
    offsets, t = [], rng.exponential(mean_gap_min * 60)
    while t < length.total_seconds():
        offsets.append(t)
        t += rng.exponential(mean_gap_min * 60)
    return _script_events(offsets, rng), length


def replay(rng, source: pd.DataFrame, length=pd.Timedelta(hours=2)):
    """A5-R / A2: a bounded segment of a file's emergency events, with original rows."""
    t = pd.to_datetime(source["timestamp"])
    while True:
        s = t.iloc[rng.integers(len(t))] - length * rng.random()
        seg = source[(t >= s) & (t < s + length)]
        if len(seg):
            out = seg.copy()
            out["offset_s"] = (pd.to_datetime(out["timestamp"]) - s).dt.total_seconds()
            return out[EVENT_COLUMNS].reset_index(drop=True), length


def mimic(rng, reference: pd.DataFrame, length=pd.Timedelta(hours=2)):
    """A5-M: a replayed segment of Zone A normal emergencies (block resampling).

    Keeps burst structure, topic mix (incl. inactivity) and heart-rate fields of normal alerts.
    Resampling single gaps would not: it loses bursts and inflates the rate several-fold.
    """
    return replay(rng, reference, length)


# ---------------------------------------------------------------- build

def build(attack: str, make_episode, spans, n: int, copy: int, seed: int):
    """Return (events, episodes) for one attack and one placement copy."""
    rng = np.random.default_rng([seed, copy])
    drawn = [make_episode(rng) for _ in range(n)]
    starts = place([d for _, d in drawn], spans, rng)
    events, episodes = [], []
    for i, ((ev, d), s) in enumerate(zip(drawn, starts)):
        ts = s + pd.to_timedelta(ev["offset_s"], unit="s")
        events.append(pd.DataFrame({
            "timestamp": ts, "type": "emergency", "topic": ev["topic"].values,
            "heart_rate": ev["heart_rate"].values, "emergency_type": ev["emergency_type"].values,
            "hour_of_day": ts.dt.hour, "day_of_week": ts.dt.day_name(), "message": ev["message"].values}))
        episodes.append({"attack": attack, "copy": copy, "episode": i, "start": str(s),
                         "end": str(s + d), "n_events": len(ev)})
    return pd.concat(events, ignore_index=True).sort_values("timestamp"), episodes
