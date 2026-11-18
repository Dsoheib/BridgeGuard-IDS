import numpy as np
import pandas as pd

from bridgeguard.baselines import feature_map
from bridgeguard.detector import conformal_p


def test_feature_map_partitions_features_within_cluster_limit():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(200, 3))
    X = np.column_stack([base, base + 0.01 * rng.normal(size=(200, 3)), rng.normal(size=(200, 4))])
    groups = feature_map(X, 5)
    assert sorted(i for g in groups for i in g) == list(range(10))
    assert all(len(g) <= 5 for g in groups)


def test_unscored_hours_never_alarm():
    cal = np.array([-np.inf, 0.1, 0.2, 0.3])
    assert conformal_p(np.array([-np.inf]), cal)[0] == 1.0
