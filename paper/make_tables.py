"""Generate every result table and number macro of the paper from the committed result files.

No number in the manuscript is typed by hand: tables/*.tex and numbers.tex are written here and \\input by main.tex.
Run from the repository root:  ml_env/bin/python paper/make_tables.py
"""
import json
from pathlib import Path

OUT = Path("paper")
ATTACKS = ["A5-S", "A5-R", "A5-P", "A5-M", "A5-U", "A2"]
# Display names; the result files and frozen protocol documents keep the A2/A5-* identifiers.
NAME = {"A5-S": "Steady", "A5-R": "Replay", "A5-P": "Poisson", "A5-M": "Mimic", "A5-U": "Sparse", "A2": "Flood"}
SCORERS = ["BridgeGuard (IF rank)", "IF (raw, default)", "OCSVM", "LOF", "kNN (k=5)", "ECOD", "Rate threshold",
           "CUSUM (alert rate)", "KitNET", "LSTM-AE"]
SHORT = {"BridgeGuard (IF rank)": "BridgeGuard", "IF (raw, default)": "IF (raw)", "kNN (k=5)": "kNN",
         "CUSUM (alert rate)": "CUSUM", "Rate threshold": "Rate rule"}


def load(p):
    return json.loads(Path(p).read_text())


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}\\%"


def ward_table(ev):
    s0 = ev["seeds"]["0"]
    rows = []
    for m in SCORERS:
        v = s0[m]
        f = v["clean"]["fpr"]
        cells = []
        for a in ATTACKS:
            c = v["copies"][f"{a}_copy0"]
            chance = 30 * (1 - (1 - f) ** (c["attack_hours"] / 30))
            cells.append(f"{c['episodes_detected']} ({chance:.1f})")
        rows.append(f"{SHORT.get(m, m)} & {v['min_attainable_p']:.4f} & {v['clean']['false_alarms']} & "
                    + " & ".join(cells) + r" \\")
    head = (r"\begin{tabular}{l r r " + "r" * len(ATTACKS) + "}\n\\toprule\n"
            r"Scorer & min $p$ & FA/" + str(ev["clean_hours"]) + " & " + " & ".join(NAME[a] for a in ATTACKS) + r" \\" + "\n\\midrule\n")
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def bed_table(ev):
    s0 = ev["seeds"]["0"]
    rows = []
    for m in SCORERS:
        v = s0[m]
        cells = [f"{v['copies'][f'{a}_copy0']['episodes_detected']} ({v['copies'][f'{a}_copy0']['chance_episodes']:.1f})"
                 for a in ATTACKS]
        cl = v["clean"]
        rows.append(f"{SHORT.get(m, m)} & {v['min_attainable_p']:.4f} & {cl['ward_alarms']} & {pct(cl['bed_hour_fpr'], 2)} & "
                    + " & ".join(cells) + r" \\")
    hours = s0[SCORERS[0]]["clean"]["ward_hours"]
    head = (r"\begin{tabular}{l r r r " + "r" * len(ATTACKS) + "}\n\\toprule\n"
            r"Scorer & min $p$ & Ward FA/" + str(hours) + r" & Bed FPR & " + " & ".join(NAME[a] for a in ATTACKS) + r" \\" + "\n\\midrule\n")
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def zones_table(man, man_d, zd, zones):
    rows = []
    for z in ("A", "B", "C"):
        m = man[z]
        rows.append(f"{z} & {len(zones[z]['patients'])} & {len(zones[z]['records'])} & {m['span_hours']:.1f} & "
                    f"{m['events']:,} & {m['emergencies']:,} & {m['emergencies'] / m['span_hours']:.1f}" + r" \\")
    m = man_d["D"]
    rows.append(f"D & {len(zd['patients'])} & {len(zd['records'])} & {m['span_hours']:.1f} & {m['events']:,} & "
                f"{m['emergencies']:,} & {m['emergencies'] / m['span_hours']:.1f}" + r" \\")
    return ("\\begin{tabular}{l r r r r r r}\n\\toprule\nZone & Patients & Records & Hours & Events & Alerts & "
            "Alerts/h \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def attacks_table(rates, cfg):
    names = list(rates["attacks"])
    mech = {"A5-S": "20 alerts, waits 300--600\\,s", "A5-R": "dev.\\ slow-poisoning timing, alert-anchored 2\\,h",
            "A5-P": f"Poisson, {cfg['a5p_rate_per_hour']}/h", "A5-M": "Zone A bed segment, alert-anchored 2\\,h",
            "A5-U": f"Poisson, {cfg['a5u_rate_per_hour']}/h", "A2": "dev.\\ flood timing, alert-anchored 2\\,h"}
    rows = []
    for a in ("A2", "A5-S", "A5-R", "A5-P", "A5-M", "A5-U"):
        key = next(k for k in names if k.startswith(a + " "))
        r = rates["attacks"][key]
        mode = cfg["mode"][a]
        w = r["ward_hour_if_added"] if mode == "add" else r["ward_hour_if_replaced"]
        rows.append(f"{NAME[a]} & {mode} & {mech[a]} & {r['injected_per_attacked_hour']['p50']:.0f} & "
                    f"{w['p50']:.0f} ({w['p95']:.0f})" + r" \\")
    return ("\\begin{tabular}{l l l r r}\n\\toprule\nAttack & Mode & Timing & Overlapped-hour count (median) & "
            "Ward hour, median (p95) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def spread_table(ward, bed):
    """Min-max over 3 seeds x 3 copies of episodes detected, A5-S and A5-R, both studies."""
    rows = []
    for m in SCORERS:
        cells = []
        for ev in (ward, bed):
            for a in ("A5-S", "A5-R"):
                v = [ev["seeds"][s][m]["copies"][f"{a}_copy{k}"]["episodes_detected"] for s in ev["seeds"] for k in range(3)]
                cells.append(f"{min(v)}--{max(v)}" if min(v) != max(v) else f"{min(v)}")
        rows.append(f"{SHORT.get(m, m)} & " + " & ".join(cells) + r" \\")
    return ("\\begin{tabular}{l rr rr}\n\\toprule\n & \\multicolumn{2}{c}{Ward study (Zone C)} & "
            "\\multicolumn{2}{c}{Bed study (Zone D)} \\\\\nScorer & Steady & Replay & Steady & Replay \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def features_table(ward_f, bed_f):
    nice = lambda f: f.replace("_", "\\_")
    allf = sorted(set(ward_f) | set(bed_f))
    rows = [f"\\texttt{{{nice(f)}}} & {'yes' if f in ward_f else '--'} & {'yes' if f in bed_f else '--'}" + r" \\"
            for f in allf]
    return ("\\begin{tabular}{l c c}\n\\toprule\nFeature & Ward study & Bed study \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def main():
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    ward, bed, dev = load("results/testbed/eval_zoneC.json"), load("results/perbed/eval_zoneD.json"), \
        load("results/dev/eval_zoneC.json")
    rates = load("results/testbed/zone_A_bed_rates.json")
    man, man_d = load("lockbox/testbed_manifest.json"), load("lockbox/testbed_manifest_D.json")
    zd, zones = load("lockbox/testbed_zone_D.json"), load("lockbox/testbed_zones.json")["zones"]
    (OUT / "tables" / "ward.tex").write_text(ward_table(ward))
    (OUT / "tables" / "bed.tex").write_text(bed_table(bed))
    (OUT / "tables" / "zones.tex").write_text(zones_table(man, man_d, zd, zones))
    (OUT / "tables" / "spread.tex").write_text(spread_table(ward, bed))
    wf = load("results/testbed/selected_features.json")["selected"]
    bf = load("results/perbed/selected_features.json")["selected"]
    (OUT / "tables" / "features.tex").write_text(features_table(wf, bf))
    import yaml
    (OUT / "tables" / "attacks.tex").write_text(attacks_table(rates, yaml.safe_load(open("configs/protocol.yaml"))["testbed_inject"]))

    n = ""
    bgw = ward["seeds"]["0"]["BridgeGuard (IF rank)"]
    n += macro("WardFA", bgw["clean"]["false_alarms"]) + macro("WardHours", ward["clean_hours"])
    n += macro("WardFPR", pct(bgw["clean"]["fpr"])) + macro("WardFPRlo", pct(bgw["clean"]["fpr_ci"][0]))
    n += macro("WardFPRhi", pct(bgw["clean"]["fpr_ci"][1]))
    n += macro("WardTail", f"{ward['G2_amended']['calibration_test']['tail_p']:.3f}")
    n += macro("WardReject", ward["G2_amended"]["calibration_test"]["reject_at_or_above"])
    n += macro("WardSRecall", ward["G2"]["A5-S_recall"]) + macro("WardRRecall", ward["G2"]["A5-R_recall"])
    bgb = bed["seeds"]["0"]["BridgeGuard (IF rank)"]
    cl = bgb["clean"]
    n += macro("BedWardFA", cl["ward_alarms"]) + macro("BedWardHours", cl["ward_hours"])
    n += macro("BedWardFPR", pct(cl["ward_fpr"])) + macro("BedTail", f"{cl['ward_calibration_test']['tail_p']:.3f}")
    n += macro("BedWardFPRlo", pct(cl["ward_calibration_test"]["fpr_ci"][0]))
    n += macro("BedWardFPRhi", pct(cl["ward_calibration_test"]["fpr_ci"][1]))
    n += macro("BedReject", cl["ward_calibration_test"]["reject_at_or_above"])
    n += macro("BedHours", cl["bed_hours"]) + macro("BedFPR", pct(cl["bed_hour_fpr"], 2))
    for b, x in cl["per_slot"].items():
        n += macro(f"BedSlot{'ABCD'[int(b)]}FPR", pct(x["fpr"], 2))
    n += macro("BedSRecall", bed["G3"]["A5-S_recall"]) + macro("BedRRecall", bed["G3"]["A5-R_recall"])
    c = bgb["copies"]["A5-S_copy0"]
    n += macro("BedSAttackedHours", c["attacked_bed_hours"]) + macro("BedSChance", f"{c['chance_episodes']:.1f}")
    n += macro("BedRChance", f"{bgb['copies']['A5-R_copy0']['chance_episodes']:.1f}")
    for m, key in (("KitNET", "KitNET"), ("kNN (k=5)", "kNN"), ("OCSVM", "OCSVM")):
        n += macro(f"Bed{key}R", bed["seeds"]["0"][m]["copies"]["A5-R_copy0"]["episodes_detected"])
    a5s = next(v for k, v in rates["attacks"].items() if k.startswith("A5-S "))
    n += macro("ASOverlapMean", f"{a5s['injected_per_attacked_hour']['mean']:.2f}")
    cw = ward["seeds"]["0"]["CUSUM (alert rate)"]["copies"]["A2_copy0"]
    n += macro("WardCusumOut", cw["false_alarms_outside_episodes"]) + macro("WardCusumOutOf", cw["hours_outside_episodes"])
    n += macro("WardNFeat", len(load("results/testbed/selected_features.json")["selected"]))
    n += macro("BedNFeat", len(load("results/perbed/selected_features.json")["selected"]))
    w, b_ = rates["ward_alerts_per_hour"], rates["per_bed_alerts_per_hour"]
    n += macro("WardRateMean", f"{w['mean']:.1f}") + macro("WardRateSD", f"{w['sd']:.1f}")
    n += macro("WardRateP", f"{w['p95']:.0f}") + macro("BedRateMean", f"{b_['mean']:.1f}")
    n += macro("BedRateSD", f"{b_['sd']:.1f}") + macro("BedRateP", f"{b_['p95']:.0f}")
    bgd = dev["seeds"]["0"]["BridgeGuard (IF rank)"]
    n += macro("DevFA", bgd["clean"]["false_alarms"]) + macro("DevHours", bgd["clean"]["hours"])
    for a in ATTACKS:
        n += macro("Dev" + a.replace("-", "").replace("5", "five").replace("2", "two"),  # TeX names: letters only
                   bgd["copies"][f"{a}_copy0"]["episodes_detected"])
    (OUT / "numbers.tex").write_text("% generated by make_tables.py from committed result files; do not edit\n" + n)
    print(n)


if __name__ == "__main__":
    main()
