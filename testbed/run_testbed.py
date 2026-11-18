"""Testbed pipeline after the zones are windowed (protocol/TESTBED_SCORERS.md).

  train     fit on testbed Zone A complete hours (results/testbed/selected_features.json), calibrate on Zone B
  inject    18 attacked copies of Zone C (testbed_inject config), labels only in episodes.json
  copies    merge clean Zone C with each copy's injected alerts (replace episodes drop that bed's own emergencies
            in [start, end)), window in one streamed pass
  evaluate  score clean Zone C and every copy once with the frozen scorer list
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import run_dev  # noqa: E402
from bridgeguard import inject  # noqa: E402
from bridgeguard import inject_testbed as T  # noqa: E402
from bridgeguard import windows as W  # noqa: E402
from generator import EPOCH  # noqa: E402

ZONES = Path("testbed_data/zones")
WIN = Path("testbed_data/windows")
INJ = Path("testbed_data/injected")
MODELS = Path("testbed_data/models")
FEATURES = "results/testbed/selected_features.json"
ATTACKS = ["A5-S", "A5-R", "A5-P", "A5-M", "A5-U", "A2"]


def hours(zone):
    if zone == "C":
        raise PermissionError("train/calibrate never read Zone C")
    return pd.read_csv(WIN / f"zone_{zone}_hours.csv")


def split_a(cfg):
    A = hours("A")
    cut = int(len(A) * (1 - cfg["detector"]["holdout_frac"]))
    return A, A.iloc[:cut], A.iloc[cut:]


def stage_train(cfg):
    A, _, _ = split_a(cfg)
    run_dev.stage_train(cfg, FEATURES, A, hours("B"), MODELS, "results/testbed/train_report.json")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_inject(cfg):
    c, n_beds = cfg["testbed_inject"], cfg["testbed"]["n_beds"]
    t = pd.concat([ch["timestamp"] for ch in pd.read_csv(ZONES / "zone_C_events.csv.gz", usecols=["timestamp"],
                                                          chunksize=2_000_000)], ignore_index=True)
    spans = inject.stretches(t)
    pool = T.zone_a_alerts(ZONES / "zone_A_events.csv.gz")

    def dev_file(path):
        d = pd.read_csv(path, usecols=["timestamp", "type", "topic", "heart_rate", "emergency_type", "message"])
        d = d[d["type"] == "emergency"].reset_index(drop=True)
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        return d

    a5, a2 = dev_file(cfg["inject"]["attack5_file"]), dev_file(cfg["inject"]["attack2_file"])
    INJ.mkdir(parents=True, exist_ok=True)
    record = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
              "inject_testbed_py_sha256": _sha(T.__file__), "inject_py_sha256": _sha(inject.__file__),
              "config": c, "stretches": len(spans), "files": {}, "episodes": []}
    for attack in ATTACKS:
        for k in range(c["copies"]):
            ev, eps = T.build(attack, spans, c, k, pool, a5, a2, n_beds)
            ev = ev.sort_values("timestamp", kind="stable").reset_index(drop=True)
            path = INJ / f"{attack}_copy{k}.csv.gz"
            ev.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
            record["files"][path.name] = {"sha256": _sha(path), "rows": len(ev)}
            record["episodes"] += eps
            print(f"{attack} copy {k}: {len(eps)} episodes, {len(ev)} injected alerts")
    (INJ / "episodes.json").write_text(json.dumps(record["episodes"], indent=1))
    del record["episodes"]
    record["episodes_sha256"] = _sha(INJ / "episodes.json")
    Path("lockbox/testbed_injections.json").write_text(json.dumps(record, indent=1))


def merged_chunks(injected: pd.DataFrame, removals, chunksize=1_000_000, src=None):
    """Clean zone chunks (default Zone C) with that bed's emergencies removed in replace episodes and injected rows
    merged in."""
    inj = injected.copy()
    inj["timestamp"] = pd.to_datetime(inj["timestamp"])
    i = 0
    for ch in pd.read_csv(src or ZONES / "zone_C_events.csv.gz", chunksize=chunksize):
        ch["timestamp"] = pd.to_datetime(ch["timestamp"])
        drop = np.zeros(len(ch), dtype=bool)
        for bed, s, e in removals:
            drop |= ((ch["type"] == "emergency") & ch["topic"].str.startswith(f"ward/bed{bed}/emergency/")
                     & (ch["timestamp"] >= s) & (ch["timestamp"] < e)).values
        ch = ch[~drop]
        j = int(np.searchsorted(inj["timestamp"].values, ch["timestamp"].values[-1], "right"))
        part = pd.concat([ch, inj.iloc[i:j]], ignore_index=True)
        i = j
        part["_emg"] = (part["type"] == "emergency").astype(int)
        yield part.sort_values(["timestamp", "_emg"], kind="stable").drop(columns="_emg")
    if i < len(inj):
        yield inj.iloc[i:]


def stage_copies(cfg):
    wm, sm = cfg["windowing"]["window_minutes"], cfg["windowing"]["stride_minutes"]
    man = json.loads(Path("lockbox/testbed_manifest.json").read_text())["C"]
    g = W.grid(EPOCH, pd.Timestamp(man["all_beds_occupied_until"]), wm, sm)
    episodes = pd.DataFrame(json.loads((INJ / "episodes.json").read_text()))
    report = {}
    for attack in ATTACKS:
        for k in range(cfg["testbed_inject"]["copies"]):
            name = f"{attack}_copy{k}"
            ep = episodes[(episodes["attack"] == attack) & (episodes["copy"] == k)]
            removals = [(r["bed"], pd.Timestamp(r["start"]), pd.Timestamp(r["end"]))
                        for _, r in ep.iterrows() if r["mode"] == "replace"]
            injected = pd.read_csv(INJ / f"{name}.csv.gz")
            feats, spans = W.featurize_stream(merged_chunks(injected, removals), g, wm, sm)
            feats.to_csv(WIN / f"zone_C_{name}.csv", index=False)
            report[name] = {"windows": len(feats), "stretches": len(spans), "replace_episodes": len(removals)}
            print(name, report[name])
    Path("testbed_data/windows/copies_report.json").write_text(json.dumps(report, indent=1))


def stage_evaluate(cfg):
    from bridgeguard import evaluate as E
    features = json.loads(Path(FEATURES).read_text())["selected"]
    _, train, holdout = split_a(cfg)
    clean = pd.read_csv(WIN / "zone_C_hours.csv")
    clean["window_start_time"] = pd.to_datetime(clean["window_start_time"])
    clean["window_end_time"] = pd.to_datetime(clean["window_end_time"])
    episodes = pd.DataFrame(json.loads((INJ / "episodes.json").read_text()))
    copy_hours = {f"{a}_copy{k}": E.at_starts(pd.read_csv(WIN / f"zone_C_{a}_copy{k}.csv"), clean)
                  for a in ATTACKS for k in range(cfg["testbed_inject"]["copies"])}
    run_dev.evaluate_core(cfg, features, train, holdout, hours("B"), clean, copy_hours, episodes, MODELS,
                         "results/testbed/eval_zoneC.json", "protocol/TESTBED_SCORERS.md", following_minutes=40,
                         v20_gate=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "inject", "copies", "evaluate"], required=True)
    cfg = yaml.safe_load(open("configs/protocol.yaml"))
    {"train": stage_train, "inject": stage_inject, "copies": stage_copies,
     "evaluate": stage_evaluate}[ap.parse_args().stage](cfg)


if __name__ == "__main__":
    main()
