"""Label firewall: no stage before final evaluation may see attack labels.

Only the lock-box evaluation and run_oracle.py are allowed to read these columns.
"""
import pandas as pd

LABEL_COLUMNS = frozenset({"label", "attack_type", "is_attack", "_is_attack", "y"})


class LabelLeakError(RuntimeError):
    pass


def label_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c in LABEL_COLUMNS]


def assert_label_free(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    found = label_columns(df)
    if found:
        raise LabelLeakError(f"stage '{stage}' received label columns {found}")
    return df


def strip_labels(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=label_columns(df))
