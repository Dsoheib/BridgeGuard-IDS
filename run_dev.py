"""Development-run pipeline entry point. Label-free: no stage here reads attack labels.

Usage: ml_env/bin/python run_dev.py --stage split [--config configs/protocol.yaml]
"""
import argparse
import json
from functools import partial
from pathlib import Path

import pandas as pd
import yaml

from bridgeguard import inject
from bridgeguard.lockbox import git_commit, load_zone, sha256, write_zones

STAGES = ("split", "inject", "windows", "rates", "select", "train", "calibrate", "evaluate")


def stage_split(cfg):
    m = write_zones(cfg["data"]["background"], cfg["data"]["time_col"], cfg["data"]["out_dir"],
                    cfg["lockbox"]["manifest"], cfg["split"]["fractions"], cfg["split"]["purge_minutes"])
    print(json.dumps({z: {k: v[k] for k in ("rows", "start", "end")} for z, v in m["zones"].items()}, indent=1))


def _emergencies(path):
    df = pd.read_csv(path)
    return df[df["type"] == "emergency"].reset_index(drop=True)


def stage_inject(cfg):
    """Test-set construction: reads Zone C timestamps only to place episodes. Nothing is fitted here."""
    c = cfg["inject"]
    man = cfg["lockbox"]["manifest"]
    zone_a = load_zone(man, "A")
    zone_c_t = load_zone(man, "C", allow_test=True)[cfg["data"]["time_col"]]
    spans = inject.stretches(zone_c_t)
    length = pd.Timedelta(hours=c["episode_hours"])
    reference = zone_a[zone_a["type"] == "emergency"].reset_index(drop=True)   # A5-M learns from Zone A only
    a5, a2 = _emergencies(c["attack5_file"]), _emergencies(c["attack2_file"])
    makers = {
        "A5-S": inject.a5_script,
        "A5-R": partial(inject.replay, source=a5, length=length),
        "A5-P": partial(inject.poisson, mean_gap_min=c["a5p_mean_gap_min"], length=length),
        "A5-M": partial(inject.mimic, reference=reference, length=length),
        "A5-U": partial(inject.poisson, mean_gap_min=60 / c["a5u_rate_per_hour"], length=length),
        "A2": partial(inject.replay, source=a2, length=length),
    }
    out = Path(c["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    # git_commit can predate uncommitted code, so also hash the injector and config actually used
    record = {"git_commit": git_commit(), "inject_py_sha256": sha256(inject.__file__),
              "config": c, "seed": c["seed"], "stretches": len(spans), "files": {}, "episodes": []}
    for attack, make in makers.items():
        for k in range(c["copies"]):
            events, episodes = inject.build(attack, make, spans, c["episodes_per_copy"], k, c["seed"])
            path = out / f"{attack}_copy{k}.csv"
            events.to_csv(path, index=False)
            record["files"][path.name] = {"sha256": sha256(path), "rows": len(events)}
            record["episodes"] += episodes
            print(f"{attack} copy {k}: {len(episodes)} episodes, {len(events)} events")
    (out / "episodes.json").write_text(json.dumps(record["episodes"], indent=1))
    record["episodes_sha256"] = sha256(out / "episodes.json")
    del record["episodes"]
    Path("lockbox/injections.json").write_text(json.dumps(record, indent=1))


def stage_windows(cfg):
    """Featurize zones A, B, clean C and every injected C copy on one fixed grid."""
    from bridgeguard import windows as W
    man, wcfg, tcol = cfg["lockbox"]["manifest"], cfg["windowing"], cfg["data"]["time_col"]
    wm, sm = wcfg["window_minutes"], wcfg["stride_minutes"]
    zones = json.loads(Path(man).read_text())["zones"]
    out = Path(cfg["data"]["out_dir"]) / "windows"
    out.mkdir(parents=True, exist_ok=True)

    def run(name, events, zone):
        g = W.grid(pd.Timestamp(zones[zone]["start"]), pd.Timestamp(zones[zone]["end"]), wm, sm)
        f = W.featurize(events, g, wm, sm)
        f.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(f)} windows")
        return f

    for z in ("A", "B"):
        run(f"zone_{z}", load_zone(man, z), z)
    zone_c = load_zone(man, "C", allow_test=True)
    run("zone_C_clean", zone_c, "C")
    inj = Path(cfg["inject"]["out_dir"])
    episodes = pd.DataFrame(json.loads((inj / "episodes.json").read_text()))
    for f in sorted(inj.glob("*_copy*.csv")):
        attack, k = f.stem.rsplit("_copy", 1)
        ep = episodes[(episodes["attack"] == attack) & (episodes["copy"] == int(k))]
        w = run(f"zone_C_{f.stem}", pd.concat([zone_c, pd.read_csv(f)], ignore_index=True), "C")
        W.window_labels(w, ep).to_csv(out / f"labels_{f.stem}.csv", index=False)
    stage_rates(cfg, zone_c)


def stage_rates(cfg, zone_c=None):
    """Merged alerts per episode (background + injected), written to windows/episode_rates.csv."""
    from bridgeguard import windows as W
    if zone_c is None:
        zone_c = load_zone(cfg["lockbox"]["manifest"], "C", allow_test=True)
    inj = Path(cfg["inject"]["out_dir"])
    episodes = pd.DataFrame(json.loads((inj / "episodes.json").read_text()))
    rates = []
    for f in sorted(inj.glob("*_copy*.csv")):
        attack, k = f.stem.rsplit("_copy", 1)
        ep = episodes[(episodes["attack"] == attack) & (episodes["copy"] == int(k))]
        rates.append(W.merged_alerts_per_episode(zone_c, pd.read_csv(f), ep))
    pd.concat(rates).to_csv(Path(cfg["data"]["out_dir"]) / "windows" / "episode_rates.csv", index=False)


def stage_select(cfg, windows_csv=None, out_dir="results/dev"):
    """Label-free selection on Zone A windows; writes <out_dir>/selected_features.json (tracked).

    Default: the development windows. The testbed calls this with its own Zone A table and out_dir.
    """
    from bridgeguard import select as S
    s = cfg["select"]
    windows_csv = windows_csv or Path(cfg["data"]["out_dir"]) / "windows" / "zone_A.csv"
    X = S.prepare(pd.read_csv(windows_csv), s["exclude"], s["impute"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rev0, rep0 = S.select(X, s["min_nonmode_share"], s["blocks"], s["max_ks"], s["redundancy"])
    (out / "selected_features_rev0.json").write_text(json.dumps({"selected": rev0, "report": rep0}, indent=1))
    rows = (X["n_emergency"] > 0) if s.get("redundancy_on_alert_windows") else None
    kept, report = S.select(X, s["min_nonmode_share"], s["blocks"], s["max_ks"], s["redundancy"], rows)
    (out / "selected_features.json").write_text(json.dumps(
        {"selected": kept, "rules": s, "zone_A_windows": len(X), "report": report}, indent=1))
    print("selected:", kept)
    for c, r in sorted(report.items(), key=lambda kv: kv[1]["max_block_ks"]):
        print(f"  {c:28s} nonmode={r['nonmode_share']:.3f} ks={r['max_block_ks']:.3f}"
              f"{' DEGENERATE' if r['degenerate'] else ''}{' UNSTABLE' if r['unstable'] else ''}"
              f"{' redundant~' + r['redundant_with'] if 'redundant_with' in r else ''}")


def zone_hours(cfg, zone):
    """Complete hours of a zone's (uninjected) windows. Zone C is refused here: training never sees it."""
    from bridgeguard.hours import hourly_complete
    if zone == "C":
        raise PermissionError("train/calibrate never read Zone C")
    w = pd.read_csv(Path(cfg["data"]["out_dir"]) / "windows" / f"zone_{zone}.csv")
    ev = load_zone(cfg["lockbox"]["manifest"], zone)
    return hourly_complete(w, ev, cfg["windowing"]["stride_minutes"], cfg["windowing"]["window_minutes"])


def stage_train(cfg, features_json="results/dev/selected_features.json", A=None, B=None, model_root=None,
                report_path="results/dev/train_report.json", split=None):
    """Fit forest + LSTM forecaster on Zone A (minus held-out tail); calibrate on Zone B complete hours.

    Defaults: the development data. The testbed passes its own features, hours, model directory and report path.
    split: an explicit (fit, rank reference) pair, used by the per-bed study (cut by hour index, not by row).
    """
    import joblib
    import numpy as np
    from bridgeguard.detector import Detector
    d, s = cfg["detector"], cfg["select"]
    features = json.loads(Path(features_json).read_text())["selected"]
    A = zone_hours(cfg, "A") if A is None else A
    B = zone_hours(cfg, "B") if B is None else B
    model_root = Path(model_root or Path(cfg["data"]["out_dir"]) / "models")
    if split is None:
        cut = int(len(A) * (1 - d["holdout_frac"]))
        train, holdout = A.iloc[:cut], A.iloc[cut:]
    else:
        train, holdout = split
    report = {"features": features, "zone_A_complete_hours": len(A), "zone_B_complete_hours": len(B), "seeds": {}}
    for seed in cfg["seeds"]:
        det = Detector(features, s["impute"], d["history_hours"], d["lstm_units"], seed, d["use_lstm"])
        r = det.fit(train, holdout)
        cal = det.score(B)
        out = model_root / f"seed{seed}"
        if out.exists():
            for f in out.iterdir():
                f.unlink()
        out.mkdir(parents=True, exist_ok=True)
        if det.lstm is not None:
            det.lstm.save(out / "lstm.keras")
        lstm, det.lstm = det.lstm, None
        joblib.dump(det, out / "detector.joblib")
        det.lstm = lstm
        np.save(out / "calibration_B.npy", cal)
        r["calibration_hours"] = len(cal)
        r["min_attainable_p"] = 1 / (len(cal) + 1)
        r["score"] = "max(rank_IF, rank_LSTM)" if d["use_lstm"] else "rank_IF"
        report["seeds"][seed] = r
        print(f"seed {seed}: {r}")
    Path(report_path).write_text(json.dumps(report, indent=1))


def stage_evaluate(cfg):
    """Score Zone C under the frozen protocol: BridgeGuard (saved models) and the baselines."""
    from bridgeguard import evaluate as E
    from bridgeguard.hours import hourly_complete
    d = cfg["detector"]
    wdir = Path(cfg["data"]["out_dir"]) / "windows"
    features = json.loads(Path("results/dev/selected_features.json").read_text())["selected"]
    A, B = zone_hours(cfg, "A"), zone_hours(cfg, "B")
    train = A.iloc[: int(len(A) * (1 - d["holdout_frac"]))]
    holdout = A.iloc[len(train):]
    zone_c = load_zone(cfg["lockbox"]["manifest"], "C", allow_test=True)
    clean = hourly_complete(pd.read_csv(wdir / "zone_C_clean.csv"), zone_c,
                            cfg["windowing"]["stride_minutes"], cfg["windowing"]["window_minutes"])
    episodes = pd.DataFrame(json.loads((Path(cfg["inject"]["out_dir"]) / "episodes.json").read_text()))
    copies = sorted(p.stem.replace("zone_C_", "") for p in wdir.glob("zone_C_*_copy*.csv"))
    copy_hours = {c: E.at_starts(pd.read_csv(wdir / f"zone_C_{c}.csv"), clean) for c in copies}
    evaluate_core(cfg, features, train, holdout, B, clean, copy_hours, episodes,
                  Path(cfg["data"]["out_dir"]) / "models", "results/dev/eval_zoneC.json", "protocol/PROTOCOL.md")


def evaluate_core(cfg, features, train, holdout, B, clean, copy_hours, episodes, model_root, out_path, protocol,
                  following_minutes=None, v20_gate=True):
    """Score clean Zone C and every injected copy with BridgeGuard and the baselines; write results and gate G2.

    following_minutes (testbed): hours starting within that many minutes after an episode ends are reported
    separately, neither outside-episode false alarms nor detections. v20_gate: also record the original G2 verdict.
    """
    import joblib
    import numpy as np
    from bridgeguard import baselines as BL
    from bridgeguard import evaluate as E
    d, s = cfg["detector"], cfg["select"]
    alpha, model_root = d["alpha"], Path(model_root)

    def scorers(seed):
        det = joblib.load(model_root / f"seed{seed}" / "detector.joblib")
        yield "BridgeGuard (IF rank)", det.score, np.load(model_root / f"seed{seed}" / "calibration_B.npy")
        for name in BL.BASELINES:
            m = BL.build(name, features, s["impute"], seed, cfg)
            m = m.fit(train, holdout) if getattr(m, "uses_holdout", False) else m.fit(train)
            yield name, m.score, m.score(B)

    results = {"protocol": protocol, "alpha": alpha, "clean_hours": len(clean),
               "baselines_not_run": BL.NOT_RUN, "seeds": {}}
    if following_minutes is not None:
        results["following_minutes"] = following_minutes
    for seed in cfg["seeds"]:
        results["seeds"][seed] = {}
        for name, score, cal in scorers(seed):
            r = {"min_attainable_p": round(float((1 + (cal >= cal.max()).sum()) / (len(cal) + 1)), 4),
                 "clean": E.clean_metrics(E.pvalues(score(clean), cal), alpha), "copies": {}}
            if following_minutes is not None and not np.isfinite(cal).all():     # testbed rule: report scored share
                r["scored_share"] = {"B": round(float(np.isfinite(cal).mean()), 3)}
            for c, h in copy_hours.items():
                attack, k = c.rsplit("_copy", 1)
                ep = episodes[(episodes["attack"] == attack) & (episodes["copy"] == int(k))]
                fol = None if following_minutes is None else E.following_hours(h, ep, following_minutes)
                r["copies"][c] = E.copy_metrics(E.pvalues(score(h), cal), E.episode_of_hour(h, ep), len(ep), alpha,
                                                following=fol)
            results["seeds"][seed][name] = r
    bg = results["seeds"][d["primary_seed"]]["BridgeGuard (IF rank)"]
    g = d["g2_copy"]
    results["G2"] = {
        "A5-S_recall": bg["copies"][f"A5-S_copy{g}"]["episodes_detected"],
        "A5-R_recall": bg["copies"][f"A5-R_copy{g}"]["episodes_detected"],
        "clean_false_alarms": bg["clean"]["false_alarms"]}
    recall_ok = results["G2"]["A5-S_recall"] >= 24 and results["G2"]["A5-R_recall"] >= 24
    if v20_gate:
        # The original (frozen) verdict is kept as recorded; the amendment replaces only the false-alarm rule.
        results["G2"]["pass"] = recall_ok and results["G2"]["clean_false_alarms"] <= 1
    cal_test = E.calibration_test(bg["clean"]["false_alarms"], bg["clean"]["hours"], alpha, d["g2_fa_test_level"])
    results["G2_amended"] = {"recall_ok": recall_ok, "calibration_test": cal_test,
                          "pass": recall_ok and not cal_test["rejected"],
                          "tpr_vs_best_baseline": "not operationalized; reported descriptively"}
    results["calibration_test_all"] = {
        sd: {n: E.calibration_test(r["clean"]["false_alarms"], r["clean"]["hours"], alpha, d["g2_fa_test_level"])["tail_p"]
             for n, r in res.items()} for sd, res in results["seeds"].items()}
    Path(out_path).write_text(json.dumps(results, indent=1))
    p0 = results["seeds"][d["primary_seed"]]
    print(f"seed {d['primary_seed']}, copy {g}, alpha {alpha}  (recall = episodes detected / 30; TPR = attack hours)")
    attacks = ["A5-S", "A5-R", "A5-P", "A5-M", "A5-U", "A2"]
    print(f"{'scorer':24s} {'minp':>6s} {'FA/' + str(len(clean)):>7s} " + " ".join(f"{a:>13s}" for a in attacks))
    for name, r in p0.items():
        cells = []
        for a in attacks:
            m = r["copies"][f"{a}_copy{g}"]
            cells.append(f"{m['episodes_detected']:>2d}/30 {m['tpr']:.2f}")
        print(f"{name:24s} {r['min_attainable_p']:6.3f} {r['clean']['false_alarms']:>4d}    " + " ".join(f"{x:>13s}" for x in cells))
    if v20_gate:
        print("G2 (original, frozen):", results["G2"])
    print("G2 (amended):", results["G2_amended"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--stage", choices=STAGES, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.stage == "split":
        stage_split(cfg)
    elif args.stage == "inject":
        stage_inject(cfg)
    elif args.stage == "windows":
        stage_windows(cfg)
    elif args.stage == "rates":
        stage_rates(cfg)
    elif args.stage == "select":
        stage_select(cfg)
    elif args.stage == "train":
        stage_train(cfg)
    elif args.stage == "evaluate":
        stage_evaluate(cfg)
    else:
        raise NotImplementedError(f"stage '{args.stage}' is not implemented")


if __name__ == "__main__":
    main()
