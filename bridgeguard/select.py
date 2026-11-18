"""Label-free feature selection on Zone A normal windows.

Primary rule: temporal stability, after non-degeneracy, then the conjunctive redundancy filter.
No labels and no attack data are read here.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .labelguard import assert_label_free


def prepare(windows: pd.DataFrame, exclude, impute) -> pd.DataFrame:
    assert_label_free(windows, "select.prepare")
    X = windows.drop(columns=[c for c in exclude if c in windows.columns])
    return X.fillna(impute).fillna(0.0)


def nonmode_share(x: pd.Series) -> float:
    return float((x != x.mode().iloc[0]).mean())


def max_block_ks(x: pd.Series, blocks: int) -> float:
    """Largest KS distance between one contiguous time block and the rest of Zone A."""
    parts = np.array_split(np.arange(len(x)), blocks)
    v = x.values
    return float(max(ks_2samp(v[p], np.delete(v, p)).statistic for p in parts))


def select(X: pd.DataFrame, min_nonmode_share: float, blocks: int, max_ks: float, redundancy: float,
           corr_rows=None):
    """corr_rows: boolean mask of the windows the redundancy filter uses (default: all windows)."""
    C = X if corr_rows is None else X[corr_rows]
    report = {}
    for c in X.columns:
        report[c] = {"nonmode_share": round(nonmode_share(X[c]), 4), "max_block_ks": round(max_block_ks(X[c], blocks), 4)}
        report[c]["degenerate"] = report[c]["nonmode_share"] < min_nonmode_share
        report[c]["unstable"] = report[c]["max_block_ks"] > max_ks
    candidates = sorted((c for c, r in report.items() if not r["degenerate"] and not r["unstable"]),
                        key=lambda c: report[c]["max_block_ks"])
    kept = []
    for c in candidates:
        twin = next((k for k in kept
                     if abs(C[c].corr(C[k])) > redundancy
                     and abs(C[c].corr(C[k], method="spearman")) > redundancy), None)
        if twin is None:
            kept.append(c)
        else:
            report[c]["redundant_with"] = twin
    return kept, report
