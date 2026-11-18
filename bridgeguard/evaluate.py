"""Zone C evaluation under the frozen protocol (protocol/PROTOCOL.md).

Hour starts come from the clean Zone C complete-hour grid (133). Each injected copy is scored at
those same starts. An hour belongs to an episode if [hour_start, hour_end) overlaps [start, end].
"""
import numpy as np
import pandas as pd
from scipy.stats import beta, binom

from .detector import conformal_p


def clopper_pearson(k: int, n: int):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return [round(float(lo), 4), round(float(hi), 4)]


def at_starts(windows: pd.DataFrame, clean_hours: pd.DataFrame) -> pd.DataFrame:
    """Rows of a copy's windows at the clean grid's hour starts, in the clean order, with hour_index."""
    w = windows.copy()
    w["window_start_time"] = pd.to_datetime(w["window_start_time"])
    w["window_end_time"] = pd.to_datetime(w["window_end_time"])
    key = clean_hours[["window_start_time", "hour_index"]]
    out = key.merge(w, on="window_start_time", how="left")
    if out["window_end_time"].isna().any():
        raise RuntimeError("copy is missing windows at clean hour starts")
    return out


def episode_of_hour(hours: pd.DataFrame, episodes: pd.DataFrame) -> np.ndarray:
    s, e = hours["window_start_time"].values, hours["window_end_time"].values
    ep = np.full(len(hours), -1)
    for _, r in episodes.iterrows():
        hit = (s <= np.datetime64(pd.Timestamp(r["end"]))) & (e > np.datetime64(pd.Timestamp(r["start"])))
        if (ep[hit] >= 0).any():
            raise RuntimeError("an hour overlaps two episodes")
        ep[hit] = r["episode"]
    return ep


def following_hours(hours: pd.DataFrame, episodes: pd.DataFrame, minutes: int) -> np.ndarray:
    """Testbed rule: hours that start within `minutes` after an episode ends (end < start <= end + minutes)."""
    s = hours["window_start_time"].values
    out = np.zeros(len(hours), dtype=bool)
    for _, r in episodes.iterrows():
        end = np.datetime64(pd.Timestamp(r["end"]))
        out |= (s > end) & (s <= end + np.timedelta64(minutes, "m"))
    return out


def copy_metrics(p: np.ndarray, ep: np.ndarray, n_episodes: int, alpha: float, following=None) -> dict:
    alarm = p <= alpha
    attack = ep >= 0
    detected = {int(x) for x in ep[alarm & attack]}
    k_tp, n_att = int((alarm & attack).sum()), int(attack.sum())
    covered = len(set(ep[attack].tolist()))
    outside = ~attack if following is None else ~attack & ~following
    out = {"attack_hours": n_att, "tpr": round(k_tp / n_att, 4) if n_att else None,
           "tpr_ci": clopper_pearson(k_tp, n_att) if n_att else None,
           "episodes_covered": covered, "episodes_detected": len(detected),
           "campaign_recall": round(len(detected) / n_episodes, 4),
           "campaign_recall_ci": clopper_pearson(len(detected), n_episodes),
           "false_alarms_outside_episodes": int((alarm & outside).sum()), "hours_outside_episodes": int(outside.sum())}
    if following is not None:
        fol = following & ~attack
        out.update(following_hours=int(fol.sum()), alarms_in_following_hours=int((alarm & fol).sum()))
    return out


def clean_metrics(p: np.ndarray, alpha: float) -> dict:
    k = int((p <= alpha).sum())
    return {"hours": len(p), "false_alarms": k, "fpr": round(k / len(p), 4), "fpr_ci": clopper_pearson(k, len(p))}


def calibration_test(k: int, n: int, alpha: float, level: float) -> dict:
    """Calibration amendment: one-sided exact binomial test of H0 'false-alarm rate <= alpha'.

    Rejects (miscalibrated) when P(X >= k | n, alpha) <= level. Not rejecting does not show the rate is below alpha.
    """
    tail = float(binom.sf(k - 1, n, alpha))
    k_crit = next(j for j in range(n + 1) if binom.sf(j - 1, n, alpha) <= level)
    return {"false_alarms": k, "hours": n, "tail_p": round(tail, 4), "reject_at_or_above": k_crit,
            "rejected": tail <= level, "fpr_ci": clopper_pearson(k, n)}


def pvalues(scores: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    return conformal_p(scores, calibration)
