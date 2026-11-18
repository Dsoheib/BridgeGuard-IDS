"""Per-bed study (protocol/PERBED_PROTOCOL.md, 7.7): sequence scorers run within each bed.

Bed-hour tables carry a `bed` column and the clock `hour_index`, which repeats across beds. CUSUM and the LSTM-AE
must not chain hours of different beds, so they are applied per bed and the scores put back in row order.
Every other scorer treats bed-hours as independent rows and needs no change.
"""
import numpy as np

from . import baselines as BL


class BedCUSUM(BL.CUSUM):
    def score(self, hours):
        out = np.zeros(len(hours))
        beds = hours["bed"].values
        for b in np.unique(beds):
            idx = np.flatnonzero(beds == b)
            sub = hours.iloc[idx]
            order = np.argsort(sub["hour_index"].values, kind="stable")
            out[idx[order]] = super().score(sub.iloc[order])
        return out


class BedLSTMAE(BL.LSTMAE):
    def _seq(self, hours, Z):
        seqs, rows = [], []
        beds = hours["bed"].values
        for b in np.unique(beds):
            idx = np.flatnonzero(beds == b)
            s, r = super()._seq(hours.iloc[idx], Z[idx])
            if len(r):
                seqs.append(s)
                rows.append(idx[r])
        if not rows:
            return np.zeros((0, self.c["seq_hours"], Z.shape[1])), np.array([], dtype=int)
        return np.concatenate(seqs), np.concatenate(rows)


OVERRIDES = {"CUSUM (alert rate)": BedCUSUM, "LSTM-AE": BedLSTMAE}


def build(name, features, impute, seed, cfg):
    if name == "CUSUM (alert rate)":
        return BedCUSUM()
    if name == "LSTM-AE":
        return BedLSTMAE(features, impute, seed, cfg)
    return BL.build(name, features, impute, seed, cfg)
