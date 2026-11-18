"""Normal-only detectors and the single conformal score.

Isolation forest on standardized features + LSTM forecaster of the next complete hour.
Score = max of the two ranks against held-out Zone A normals; conformal p-value against Zone B.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .labelguard import assert_label_free


def matrix(hours: pd.DataFrame, features, impute) -> np.ndarray:
    assert_label_free(hours, "detector.matrix")
    return hours[features].fillna(impute).fillna(0.0).values.astype(float)


def sequences(hours: pd.DataFrame, X: np.ndarray, k: int):
    """(inputs, targets, row) for every hour whose previous k hours are present and contiguous."""
    idx = hours["hour_index"].values
    pos = {h: i for i, h in enumerate(idx)}
    xs, ys, rows = [], [], []
    for i, h in enumerate(idx):
        prev = [pos.get(h - j) for j in range(k, 0, -1)]
        if None not in prev:
            xs.append(X[prev])
            ys.append(X[i])
            rows.append(i)
    return np.array(xs), np.array(ys), np.array(rows, dtype=int)


def build_forecaster(k: int, d: int, units: int, seed: int):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    m = tf.keras.Sequential([tf.keras.Input((k, d)), tf.keras.layers.LSTM(units), tf.keras.layers.Dense(d)])
    m.compile(optimizer="adam", loss="mse")
    return m


class Detector:
    """use_lstm=False (frozen protocol): the score is the forest rank alone; the LSTM is ablation-only."""

    def __init__(self, features, impute, k, units, seed, use_lstm=False):
        self.features, self.impute, self.k, self.units, self.seed = features, impute, k, units, seed
        self.use_lstm = use_lstm
        self.lstm = None

    def fit(self, train: pd.DataFrame, holdout: pd.DataFrame, epochs=300, patience=20):
        Xt, Xh = matrix(train, self.features, self.impute), matrix(holdout, self.features, self.impute)
        self.scaler = StandardScaler().fit(Xt)
        Zt, Zh = self.scaler.transform(Xt), self.scaler.transform(Xh)
        self.forest = IsolationForest(n_estimators=300, random_state=self.seed).fit(Zt)
        info = {"train_hours": len(train), "holdout_hours": len(holdout)}
        if not self.use_lstm:
            self.ref_if = self._raw(holdout)[0]
            return info
        import tensorflow as tf
        st, yt, _ = sequences(train, Zt, self.k)
        sh, yh, _ = sequences(holdout, Zh, self.k)
        self.lstm = build_forecaster(self.k, Zt.shape[1], self.units, self.seed)
        stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        hist = self.lstm.fit(st, yt, validation_data=(sh, yh), epochs=epochs, batch_size=32,
                             callbacks=[stop], verbose=0)
        # rank references: held-out Zone A normals, never used to fit the forest
        self.ref_if, self.ref_lstm = self._raw(holdout)
        self.ref_lstm = self.ref_lstm[~np.isnan(self.ref_lstm)]
        return {**info, "train_sequences": len(st),
                "holdout_sequences": len(sh), "epochs_run": len(hist.history["loss"]),
                "best_val_loss": float(min(hist.history["val_loss"])),
                "train_loss_at_best": float(hist.history["loss"][int(np.argmin(hist.history["val_loss"]))])}

    def _raw(self, hours: pd.DataFrame):
        Z = self.scaler.transform(matrix(hours, self.features, self.impute))
        s_if = -self.forest.score_samples(Z)                       # higher = more anomalous
        s_lstm = np.full(len(hours), np.nan)
        if self.lstm is None:
            return s_if, s_lstm
        seq, y, rows = sequences(hours, Z, self.k)
        if len(rows):
            s_lstm[rows] = ((self.lstm.predict(seq, verbose=0) - y) ** 2).mean(axis=1)
        return s_if, s_lstm

    @staticmethod
    def _rank(x, ref):
        return np.searchsorted(np.sort(ref), x, side="right") / len(ref)

    def score(self, hours: pd.DataFrame) -> np.ndarray:
        """Frozen: rank_IF. Ablation (use_lstm): max(rank_IF, rank_LSTM), rank_IF alone without history."""
        s_if, s_lstm = self._raw(hours)
        r_if = self._rank(s_if, self.ref_if)
        if not self.use_lstm:
            return r_if
        r_lstm = np.where(np.isnan(s_lstm), -np.inf, self._rank(np.nan_to_num(s_lstm), self.ref_lstm))
        return np.maximum(r_if, r_lstm)


def conformal_p(scores: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    """p = (1 + #{calibration >= s}) / (n + 1)."""
    cal = np.sort(calibration)
    n_ge = len(cal) - np.searchsorted(cal, scores, side="left")
    return (1 + n_ge) / (len(cal) + 1)
