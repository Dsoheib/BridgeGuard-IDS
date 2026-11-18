"""Per-bed study pipeline (protocol/PERBED_PROTOCOL.md). Outputs never overwrite the ward-level study.

  windows --zone A|B   one streamed pass per bed: bed windows + complete bed-hours (bed column added)
  select               frozen label-free rule on every kept bed window, 4 beds pooled, sorted by window start
  train                IF-rank detector on bed-hours with hour_index < 1861; rank reference = hour_index >= 1861;
                       calibration = all Zone B bed-hours; records the smallest p the Zone B ranks can return
  inject               Zone D: 18 copies, bed drawn first, episodes inside that bed's stretches, seed 2027
  copies               merge clean Zone D with each copy (replace episodes drop that bed's emergencies), bed-window
  evaluate             one look on Zone D: bed alarm at p <= alpha/4, ward alarm = any bed, gate G3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import run_dev  # noqa: E402
import run_testbed as R  # noqa: E402
from bridgeguard import inject  # noqa: E402
from bridgeguard import inject_testbed as T  # noqa: E402
from bridgeguard import windows as W  # noqa: E402
from bridgeguard.hours import hourly_complete  # noqa: E402
from generator import EPOCH  # noqa: E402

ZONES = Path("testbed_data/zones")
BW = Path("testbed_data/bedwindows")
RES = Path("results/perbed")
MODELS = Path("testbed_data/models_perbed")
HOLDOUT_HOUR = 1861                      # the ward-hour cut: int(2190 * 0.85)
INJ_D = Path("testbed_data/injected_D")
INJECT_SEED_D = 2027
ATTACKS = ["A5-S", "A5-R", "A5-P", "A5-M", "A5-U", "A2"]


def bed_chunks(src, bed, chunksize=1_000_000):
    """One bed's stream (vitals and emergencies) from a time-sorted zone file or chunk iterator."""
    it = pd.read_csv(src, chunksize=chunksize) if isinstance(src, (str, Path)) else src
    for ch in it:
        yield ch[ch["topic"].str.startswith(f"ward/bed{bed}/")]


def window_beds(source_factory, grid, wm, sm, n_beds):
    """Per-bed windows and complete bed-hours; hour_index anchored at the epoch for every bed."""
    wins, hrs = [], []
    for b in range(n_beds):
        f, spans = W.featurize_stream(bed_chunks(source_factory(), b), grid, wm, sm)
        if pd.to_datetime(f["window_start_time"]).min() != EPOCH:
            raise RuntimeError(f"bed {b}: first window is not at the epoch; hour_index would not align")
        h = hourly_complete(f, None, sm, wm, spans=spans)
        wins.append(f.assign(bed=b))
        hrs.append(h.assign(bed=b))
    return wins, hrs


def stage_windows(cfg, zone, allow):
    if zone != "A" and not allow:
        raise PermissionError(f"Zone {zone} is windowed only with --allow-zone")
    wm, sm = cfg["windowing"]["window_minutes"], cfg["windowing"]["stride_minutes"]
    man = json.loads(Path("lockbox/testbed_manifest_D.json" if zone == "D"
                          else "lockbox/testbed_manifest.json").read_text())[zone]
    g = W.grid(EPOCH, pd.Timestamp(man["all_beds_occupied_until"]), wm, sm)
    src = ZONES / f"zone_{zone}_events.csv.gz"
    wins, hrs = window_beds(lambda: src, g, wm, sm, cfg["testbed"]["n_beds"])
    BW.mkdir(parents=True, exist_ok=True)
    pd.concat(wins, ignore_index=True).to_csv(BW / f"zone_{zone}_windows.csv", index=False)
    pd.concat(hrs, ignore_index=True).to_csv(BW / f"zone_{zone}_hours.csv", index=False)
    print(json.dumps({"zone": zone, "windows_per_bed": [len(w) for w in wins],
                      "complete_bed_hours_per_bed": [len(h) for h in hrs]}))


def stage_select(cfg):
    w = pd.read_csv(BW / "zone_A_windows.csv")
    w = w.sort_values(["window_start_time", "bed"], kind="stable").drop(columns="bed")
    pooled = BW / "zone_A_windows_pooled_sorted.csv"
    w.to_csv(pooled, index=False)
    run_dev.stage_select(cfg, pooled, RES)


def bed_hours(zone):
    if zone not in ("A", "B"):
        raise PermissionError("design inputs are Zones A and B only")
    return pd.read_csv(BW / f"zone_{zone}_hours.csv").sort_values(["hour_index", "bed"], kind="stable")


def split_a():
    A = bed_hours("A")
    return A, A[A["hour_index"] < HOLDOUT_HOUR], A[A["hour_index"] >= HOLDOUT_HOUR]


def stage_train(cfg):
    A, fit, ref = split_a()
    B = bed_hours("B")
    run_dev.stage_train(cfg, RES / "selected_features.json", A, B, MODELS, RES / "train_report.json",
                       split=(fit, ref))
    rep = json.loads((RES / "train_report.json").read_text())
    for seed in cfg["seeds"]:
        cal = np.load(MODELS / f"seed{seed}" / "calibration_B.npy")
        top = int((cal >= cal.max()).sum())
        rep["seeds"][str(seed)]["min_p_zone_B_ranks_can_return"] = round((1 + top) / (len(cal) + 1), 5)
        rep["seeds"][str(seed)]["zone_B_bed_hours_at_top_rank"] = top
    rep["bed_alarm_threshold"] = cfg["detector"]["alpha"] / cfg["testbed"]["n_beds"]
    rep["holdout_rule"] = f"fit: hour_index < {HOLDOUT_HOUR}; rank reference: hour_index >= {HOLDOUT_HOUR}"
    rep["fit_bed_hours"], rep["rank_reference_bed_hours"] = len(fit), len(ref)
    (RES / "train_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({s: rep["seeds"][str(s)]["min_p_zone_B_ranks_can_return"] for s in cfg["seeds"]}))


def stage_inject(cfg):
    c = dict(cfg["testbed_inject"], seed=INJECT_SEED_D)
    n_beds = cfg["testbed"]["n_beds"]
    bed_t = {b: [] for b in range(n_beds)}
    for ch in pd.read_csv(ZONES / "zone_D_events.csv.gz", usecols=["timestamp", "topic"], chunksize=2_000_000):
        bed = ch["topic"].str.split("/").str[1].str[3:].astype(int).values
        for b in range(n_beds):
            bed_t[b].append(ch["timestamp"].values[bed == b])
    bed_spans = {b: inject.stretches(pd.Series(np.concatenate(v))) for b, v in bed_t.items()}
    pool = T.zone_a_alerts(ZONES / "zone_A_events.csv.gz")

    def dev_file(path):
        d = pd.read_csv(path, usecols=["timestamp", "type", "topic", "heart_rate", "emergency_type", "message"])
        d = d[d["type"] == "emergency"].reset_index(drop=True)
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        return d

    a5, a2 = dev_file(cfg["inject"]["attack5_file"]), dev_file(cfg["inject"]["attack2_file"])
    INJ_D.mkdir(parents=True, exist_ok=True)
    record = {"git_commit": R.subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
              "inject_testbed_py_sha256": R._sha(T.__file__), "config": c,
              "bed_stretches": {b: len(s) for b, s in bed_spans.items()}, "files": {}, "episodes": []}
    for attack in ATTACKS:
        for k in range(c["copies"]):
            ev, eps = T.build_bed_first(attack, bed_spans, c, k, pool, a5, a2, n_beds)
            ev = ev.sort_values("timestamp", kind="stable").reset_index(drop=True)
            path = INJ_D / f"{attack}_copy{k}.csv.gz"
            ev.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
            record["files"][path.name] = {"sha256": R._sha(path), "rows": len(ev)}
            record["episodes"] += eps
            print(f"{attack} copy {k}: {len(eps)} episodes, {len(ev)} injected alerts")
    (INJ_D / "episodes.json").write_text(json.dumps(record["episodes"], indent=1))
    del record["episodes"]
    record["episodes_sha256"] = R._sha(INJ_D / "episodes.json")
    Path("lockbox/testbed_injections_D.json").write_text(json.dumps(record, indent=1))


def stage_copies(cfg):
    wm, sm = cfg["windowing"]["window_minutes"], cfg["windowing"]["stride_minutes"]
    man = json.loads(Path("lockbox/testbed_manifest_D.json").read_text())["D"]
    g = W.grid(EPOCH, pd.Timestamp(man["all_beds_occupied_until"]), wm, sm)
    episodes = pd.DataFrame(json.loads((INJ_D / "episodes.json").read_text()))
    report = {}
    for attack in ATTACKS:
        for k in range(cfg["testbed_inject"]["copies"]):
            name = f"{attack}_copy{k}"
            ep = episodes[(episodes["attack"] == attack) & (episodes["copy"] == k)]
            removals = [(r["bed"], pd.Timestamp(r["start"]), pd.Timestamp(r["end"]))
                        for _, r in ep.iterrows() if r["mode"] == "replace"]
            injected = pd.read_csv(INJ_D / f"{name}.csv.gz")
            src = ZONES / "zone_D_events.csv.gz"
            wins, hrs = window_beds(lambda: R.merged_chunks(injected, removals, src=src), g, wm, sm,
                                    cfg["testbed"]["n_beds"])
            pd.concat(hrs, ignore_index=True).to_csv(BW / f"zone_D_{name}_hours.csv", index=False)
            report[name] = {"bed_hours_per_bed": [len(h) for h in hrs], "replace_episodes": len(removals)}
            print(name, report[name])
    (BW / "zone_D_copies_report.json").write_text(json.dumps(report, indent=1))


def at_bed_starts(copy_hours, clean):
    """Copy bed-hours at the clean grid's (bed, hour) rows, in clean order."""
    key = clean[["bed", "window_start_time", "hour_index"]]
    w = copy_hours.drop(columns=["hour_index"]).copy()
    w["window_start_time"] = pd.to_datetime(w["window_start_time"])
    w["window_end_time"] = pd.to_datetime(w["window_end_time"])
    out = key.merge(w, on=["bed", "window_start_time"], how="left")
    if out["window_end_time"].isna().any():
        raise RuntimeError("copy is missing bed-hours at clean (bed, hour) rows")
    return out


def bed_metrics(p_clean, clean, alpha, n_beds, level):
    from bridgeguard import evaluate as E
    thr = alpha / n_beds
    alarm = p_clean <= thr
    ward = pd.Series(alarm).groupby(clean["hour_index"].values).any()
    k, n = int(ward.sum()), int(len(ward))
    per_slot = {}
    for b in range(n_beds):
        m = clean["bed"].values == b
        per_slot[b] = {"bed_hours": int(m.sum()), "alarms": int(alarm[m].sum()),
                       "fpr": round(float(alarm[m].mean()), 4), "fpr_ci": E.clopper_pearson(int(alarm[m].sum()), int(m.sum()))}
    return {"bed_alarm_threshold": thr, "bed_hours": int(len(alarm)), "bed_hour_alarms": int(alarm.sum()),
            "bed_hour_fpr": round(float(alarm.mean()), 4), "per_slot": per_slot,
            "ward_hours": n, "ward_alarms": k, "ward_fpr": round(k / n, 4),
            "ward_calibration_test": E.calibration_test(k, n, alpha, level)}


def copy_bed_metrics(p, hrs, ep, thr, bed_fpr, following_min):
    from bridgeguard import evaluate as E
    s, e = hrs["window_start_time"].values, hrs["window_end_time"].values
    bed = hrs["bed"].values
    alarm = p <= thr
    attacked = np.zeros(len(hrs), bool)
    following = np.zeros(len(hrs), bool)
    detected, covered, chance = 0, 0, 0.0
    for _, r in ep.iterrows():
        st, en = np.datetime64(pd.Timestamp(r["start"])), np.datetime64(pd.Timestamp(r["end"]))
        on = (bed == r["bed"]) & (s <= en) & (e > st)
        attacked |= on
        following |= (bed == r["bed"]) & (s > en) & (s <= en + np.timedelta64(following_min, "m"))
        n_e = int(on.sum())
        covered += n_e > 0
        detected += bool(alarm[on].any())
        chance += 1 - (1 - bed_fpr) ** n_e
    following &= ~attacked
    k_tp, n_att = int((alarm & attacked).sum()), int(attacked.sum())
    n = len(ep)
    return {"episodes_detected": detected, "campaign_recall": round(detected / n, 4),
            "campaign_recall_ci": E.clopper_pearson(detected, n), "episodes_covered": covered,
            "chance_episodes": round(chance, 2), "attacked_bed_hours": n_att,
            "tpr": round(k_tp / n_att, 4) if n_att else None,
            "following_bed_hours": int(following.sum()), "alarms_in_following_bed_hours": int((alarm & following).sum()),
            "other_bed_hour_alarms": int((alarm & ~attacked & ~following).sum())}


def stage_evaluate(cfg):
    import joblib
    from bridgeguard import evaluate as E
    from bridgeguard import baselines as BL
    from bridgeguard import perbed as PB
    d, s = cfg["detector"], cfg["select"]
    alpha, n_beds = d["alpha"], cfg["testbed"]["n_beds"]
    features = json.loads((RES / "selected_features.json").read_text())["selected"]
    _, fit, ref = split_a()
    B = bed_hours("B")
    clean = pd.read_csv(BW / "zone_D_hours.csv")
    clean["window_start_time"] = pd.to_datetime(clean["window_start_time"])
    clean["window_end_time"] = pd.to_datetime(clean["window_end_time"])
    episodes = pd.DataFrame(json.loads((INJ_D / "episodes.json").read_text()))
    names = [f"{a}_copy{k}" for a in ATTACKS for k in range(cfg["testbed_inject"]["copies"])]
    copies = {c: at_bed_starts(pd.read_csv(BW / f"zone_D_{c}_hours.csv"), clean) for c in names}

    def scorers(seed):
        det = joblib.load(MODELS / f"seed{seed}" / "detector.joblib")
        yield "BridgeGuard (IF rank)", det.score, np.load(MODELS / f"seed{seed}" / "calibration_B.npy")
        for name in BL.BASELINES:
            m = PB.build(name, features, s["impute"], seed, cfg)
            m = m.fit(fit, ref) if getattr(m, "uses_holdout", False) else m.fit(fit)
            yield name, m.score, m.score(B)

    res = {"protocol": "protocol/PERBED_PROTOCOL.md", "alpha": alpha, "bed_alarm_threshold": alpha / n_beds,
           "clean_bed_hours": len(clean), "seeds": {}}
    for seed in cfg["seeds"]:
        res["seeds"][seed] = {}
        for name, score, cal in scorers(seed):
            pc = E.pvalues(score(clean), cal)
            r = {"min_attainable_p": round(float((1 + (cal >= cal.max()).sum()) / (len(cal) + 1)), 5),
                 "clean": bed_metrics(pc, clean, alpha, n_beds, d["g2_fa_test_level"]), "copies": {}}
            if not np.isfinite(cal).all():
                r["scored_share_B"] = round(float(np.isfinite(cal).mean()), 3)
            for c in names:
                attack, k = c.rsplit("_copy", 1)
                ep = episodes[(episodes["attack"] == attack) & (episodes["copy"] == int(k))]
                h = copies[c]
                r["copies"][c] = copy_bed_metrics(E.pvalues(score(h), cal), h, ep, alpha / n_beds,
                                                  r["clean"]["bed_hour_fpr"], 40)
            res["seeds"][seed][name] = r
    bg = res["seeds"][d["primary_seed"]]["BridgeGuard (IF rank)"]
    g = d["g2_copy"]
    a5s, a5r = (bg["copies"][f"{a}_copy{g}"]["episodes_detected"] for a in ("A5-S", "A5-R"))
    rejected = bg["clean"]["ward_calibration_test"]["rejected"]
    res["G3"] = {"A5-S_recall": a5s, "A5-R_recall": a5r, "ward_calibration_rejected": rejected,
                 "pass": a5s >= 24 and a5r >= 24 and not rejected}
    (RES / "eval_zoneD.json").write_text(json.dumps(res, indent=1))
    p0 = res["seeds"][d["primary_seed"]]
    print(f"seed {d['primary_seed']}, copy {g}; bed alarm p <= {alpha / n_beds}; recall = attacked-bed detections / 30")
    for name, r in p0.items():
        cl = r["clean"]
        cells = " ".join(f"{a}:{r['copies'][f'{a}_copy{g}']['episodes_detected']:>2d}"
                         f"({r['copies'][f'{a}_copy{g}']['chance_episodes']:.1f})" for a in ATTACKS)
        print(f"{name:24s} ward FA {cl['ward_alarms']:>3d}/{cl['ward_hours']} bed FPR {cl['bed_hour_fpr']:.4f}  {cells}")
    print("G3:", res["G3"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["windows", "select", "train", "inject", "copies", "evaluate"],
                    required=True)
    ap.add_argument("--zone", choices=["A", "B", "D"])
    ap.add_argument("--allow-zone", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/protocol.yaml"))
    if a.stage == "windows":
        stage_windows(cfg, a.zone, a.allow_zone)
    elif a.stage == "select":
        stage_select(cfg)
    elif a.stage == "train":
        stage_train(cfg)
    else:
        {"inject": stage_inject, "copies": stage_copies, "evaluate": stage_evaluate}[a.stage](cfg)


if __name__ == "__main__":
    main()
