"""Baselines, fixed before any Zone C output was read.

All use the same ten features, the same 323 Zone A fit hours and StandardScaler, raw scores
(higher = more anomalous), and the same conformal calibration on the 119 Zone B hours.
Hyperparameters are fixed a priori; nothing is searched.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .detector import matrix

# KitNET and LSTM-AE were added after the calibration amendment (hyperparameters in configs/protocol.yaml, fixed before
# their first run); they belong to the development table and don't change the primary scorer.
NOT_RUN = {}


class _Scaled:
    def __init__(self, features, impute):
        self.features, self.impute = features, impute

    def _z(self, hours):
        return self.scaler.transform(matrix(hours, self.features, self.impute))

    def fit(self, train):
        self.scaler = StandardScaler().fit(matrix(train, self.features, self.impute))
        self._fit(self._z(train), train)
        return self


class IFRaw(_Scaled):
    """Isolation forest, sklearn defaults, raw score (no rank normalization)."""
    def __init__(self, features, impute, seed):
        super().__init__(features, impute)
        self.seed = seed

    def _fit(self, Z, _):
        self.m = IsolationForest(random_state=self.seed).fit(Z)

    def score(self, hours):
        return -self.m.score_samples(self._z(hours))


class OCSVM(_Scaled):
    def _fit(self, Z, _):
        self.m = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(Z)

    def score(self, hours):
        return -self.m.decision_function(self._z(hours))


class LOF(_Scaled):
    def _fit(self, Z, _):
        self.m = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Z)

    def score(self, hours):
        return -self.m.score_samples(self._z(hours))


class KNN(_Scaled):
    def _fit(self, Z, _):
        self.m = NearestNeighbors(n_neighbors=5).fit(Z)

    def score(self, hours):
        return self.m.kneighbors(self._z(hours))[0].mean(axis=1)


class ECOD(_Scaled):
    """ECOD (Li et al. 2022): sum over features of -log empirical tail probabilities, worse tail per feature."""
    def _fit(self, Z, _):
        self.ref = np.sort(Z, axis=0)

    def score(self, hours):
        Z, n = self._z(hours), len(self.ref)
        out = np.zeros(len(Z))
        for j in range(Z.shape[1]):
            r = self.ref[:, j]
            left = np.searchsorted(r, Z[:, j], side="right") / n
            right = (n - np.searchsorted(r, Z[:, j], side="left")) / n
            out += -np.log(np.clip(np.minimum(left, right), 1 / (n + 1), 1))
        return out


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class _DA:
    """KitNET's denoising-autoencoder unit (corruption 0): tied weights, sigmoid, SGD, running min-max on its input."""
    def __init__(self, n, hidden_ratio, lr, rng):
        self.lr, nh = lr, int(np.ceil(n * hidden_ratio))
        self.W = rng.uniform(-1.0 / n, 1.0 / n, (n, nh))
        self.hb, self.vb = np.zeros(nh), np.zeros(n)
        self.lo, self.hi = np.full(n, np.inf), np.full(n, -np.inf)

    def _norm(self, x):
        return (x - self.lo) / (self.hi - self.lo + 1e-16)

    def train(self, x):
        self.lo, self.hi = np.minimum(self.lo, x), np.maximum(self.hi, x)
        x = self._norm(x)
        h = _sigmoid(x @ self.W + self.hb)
        z = _sigmoid(h @ self.W.T + self.vb)
        e2 = x - z
        e1 = (e2 @ self.W) * h * (1 - h)
        self.W += self.lr * (np.outer(x, e1) + np.outer(e2, h))
        self.hb += self.lr * e1
        self.vb += self.lr * e2
        return np.sqrt(np.mean(e2 ** 2))

    def rmse(self, X):
        X = self._norm(X)
        Z = _sigmoid(_sigmoid(X @ self.W + self.hb) @ self.W.T + self.vb)
        return np.sqrt(np.mean((X - Z) ** 2, axis=1))


def feature_map(X, max_cluster):
    """KitNET corClust: single-linkage on 1 - correlation, dendrogram split until every cluster has <= max_cluster."""
    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.spatial.distance import squareform
    C = np.nan_to_num(np.corrcoef(X, rowvar=False))           # a constant column gets correlation 0
    D = np.clip(1.0 - C, 0.0, None)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2

    def split(node):
        if node.get_count() <= max_cluster:
            return [sorted(node.pre_order())]
        return split(node.get_left()) + split(node.get_right())
    return split(to_tree(linkage(squareform(D, checks=False), method="single")))


class KitNET(_Scaled):
    """KitNET (Mirsky et al. 2018) on the ten hourly features, trained offline for a fixed number of epochs."""
    def __init__(self, features, impute, seed, cfg):
        super().__init__(features, impute)
        self.seed, self.c = seed, cfg["kitnet"]

    def _fit(self, Z, _):
        rng = np.random.default_rng(self.seed)
        self.groups = feature_map(Z, self.c["max_cluster"])
        args = (self.c["hidden_ratio"], self.c["learning_rate"], rng)
        self.ens = [_DA(len(g), *args) for g in self.groups]
        self.out = _DA(len(self.groups), *args)
        for _ in range(self.c["epochs"]):
            for i in rng.permutation(len(Z)):
                self.out.train(np.array([da.train(Z[i, g]) for da, g in zip(self.ens, self.groups)]))

    def score(self, hours):
        Z = self._z(hours)
        return self.out.rmse(np.column_stack([da.rmse(Z[:, g]) for da, g in zip(self.ens, self.groups)]))


class LSTMAE(_Scaled):
    """LSTM sequence autoencoder over the current hour and its previous contiguous hours; unscored hours get -inf."""
    uses_holdout = True

    def __init__(self, features, impute, seed, cfg):
        super().__init__(features, impute)
        self.seed, self.c = seed, cfg["lstm_ae"]

    def _seq(self, hours, Z):
        from .detector import sequences
        xs, ys, rows = sequences(hours, Z, self.c["seq_hours"] - 1)
        if not len(rows):
            return np.zeros((0, self.c["seq_hours"], Z.shape[1])), rows
        return np.concatenate([xs, ys[:, None, :]], axis=1), rows

    def fit(self, train, holdout):
        import tensorflow as tf
        self.scaler = StandardScaler().fit(matrix(train, self.features, self.impute))
        st, _ = self._seq(train, self._z(train))
        sh, _ = self._seq(holdout, self._z(holdout))
        k, d, u = self.c["seq_hours"], st.shape[2], self.c["units"]
        tf.keras.utils.set_random_seed(self.seed)
        self.m = tf.keras.Sequential([
            tf.keras.Input((k, d)), tf.keras.layers.LSTM(u), tf.keras.layers.RepeatVector(k),
            tf.keras.layers.LSTM(u, return_sequences=True), tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(d))])
        self.m.compile(optimizer="adam", loss="mse")
        stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.c["patience"], restore_best_weights=True)
        h = self.m.fit(st, st, validation_data=(sh, sh), epochs=self.c["epochs"], batch_size=self.c["batch_size"],
                       callbacks=[stop], verbose=0)
        self.info = {"train_sequences": len(st), "holdout_sequences": len(sh), "epochs_run": len(h.history["loss"]),
                     "best_val_loss": round(float(min(h.history["val_loss"])), 4)}
        return self

    def score(self, hours):
        seq, rows = self._seq(hours, self._z(hours))
        out = np.full(len(hours), -np.inf)
        if len(rows):
            out[rows] = ((self.m.predict(seq, verbose=0) - seq) ** 2).mean(axis=(1, 2))
        return out


class RateThreshold:
    """Alerts per hour, the rule the rate-blindness lemma is about."""
    def __init__(self, *_):
        pass

    def fit(self, train):
        return self

    def score(self, hours):
        return hours["alert_frequency_per_hour"].fillna(0).values.astype(float)


class CUSUM:
    """One-sided CUSUM on standardized alerts/hour over consecutive complete hours; reset at gaps."""
    def __init__(self, *_):
        pass

    def fit(self, train):
        x = train["alert_frequency_per_hour"].fillna(0).values
        self.mu, self.sd = x.mean(), x.std() or 1.0
        return self

    def score(self, hours):
        x = (hours["alert_frequency_per_hour"].fillna(0).values - self.mu) / self.sd
        idx, s, out = hours["hour_index"].values, 0.0, []
        for i in range(len(x)):
            if i and idx[i] != idx[i - 1] + 1:
                s = 0.0
            s = max(0.0, s + x[i] - 0.5)
            out.append(s)
        return np.array(out)


BASELINES = {"IF (raw, default)": IFRaw, "OCSVM": OCSVM, "LOF": LOF, "kNN (k=5)": KNN, "ECOD": ECOD,
             "Rate threshold": RateThreshold, "CUSUM (alert rate)": CUSUM, "KitNET": KitNET, "LSTM-AE": LSTMAE}


def build(name, features, impute, seed, cfg):
    cls = BASELINES[name]
    if cls in (KitNET, LSTMAE):
        return cls(features, impute, seed, cfg)
    return cls(features, impute, seed) if cls is IFRaw else cls(features, impute)
