"""Chronological zone split of the event stream, with purge gaps between zones.

The split is made on the raw event stream, before any window or feature exists.
"""
import pandas as pd

ZONES = ("A", "B", "C")


def zone_bounds(t0: pd.Timestamp, t1: pd.Timestamp, fractions=(0.6, 0.2, 0.2),
                purge_minutes: int = 60) -> dict:
    """Return {zone: (start, end)}; half-open intervals with a purge gap before B and C."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    purge = pd.Timedelta(minutes=purge_minutes)
    span = t1 - t0
    cut_ab = t0 + span * fractions[0]
    cut_bc = cut_ab + span * fractions[1]
    return {
        "A": (t0, cut_ab),
        "B": (cut_ab + purge, cut_bc),
        "C": (cut_bc + purge, t1 + pd.Timedelta(microseconds=1)),
    }


def split_events(df: pd.DataFrame, time_col: str, bounds: dict) -> dict:
    t = pd.to_datetime(df[time_col])
    return {z: df[(t >= s) & (t < e)] for z, (s, e) in bounds.items()}


def window_zone(start: pd.Timestamp, end: pd.Timestamp, bounds: dict):
    """Zone that fully contains the window [start, end), or None if it straddles a boundary or purge."""
    for z, (s, e) in bounds.items():
        if start >= s and end <= e:
            return z
    return None
