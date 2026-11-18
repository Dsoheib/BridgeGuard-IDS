"""Generate the paper's figures from committed result files (and the Zone A event file for the rate histograms).

  F2 rates.pdf    clean Zone A ward-hour and bed-hour alert counts, with each attack's ward-hour median marked
  F3 recall.pdf   episodes detected vs chance, every scorer x attack, ward study (Zone C) and bed study (Zone D)
  F4 fpr.pdf      clean false-alarm rate with 95% intervals, every scorer, both studies
Palette: reference slots 1 (blue) and 2 (orange), validated; marker shape is the secondary encoding.
Run from the repository root:  ml_env/bin/python paper/make_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "testbed")
sys.path.insert(0, "paper")
from make_tables import ATTACKS, NAME, SCORERS, SHORT, load  # noqa: E402

OUT = Path("paper/figures")
C1, C2, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "pdf.fonttype": 42})


def fig_rates():
    import measure_bed_rates as M
    _, bed = M.zone_a_counts("testbed_data/zones/zone_A_events.csv.gz", "testbed_data/windows/zone_A_hours.csv", 4)
    ward = bed.sum(axis=1)
    rates = load("results/testbed/zone_A_bed_rates.json")
    mode = {"A5-S": "add", "A5-R": "add", "A5-P": "add", "A5-M": "replace", "A5-U": "replace"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    a.hist(ward, bins=np.arange(0, ward.max() + 2) - 0.5, color=C1, edgecolor="white", linewidth=0.3)
    a.set_xlabel("Alerts per complete ward hour (Zone A, 4 beds)")
    a.set_ylabel("Hours")
    lines = []
    for atk, m in mode.items():
        key = next(k for k in rates["attacks"] if k.startswith(atk + " "))
        med = rates["attacks"][key]["ward_hour_if_added" if m == "add" else "ward_hour_if_replaced"]["p50"]
        a.axvline(med, color=MUTED, linewidth=0.7, linestyle="--")
        lines.append(f"{NAME[atk]} {med:.0f}")
    a.axvline(np.median(ward), color=INK, linewidth=1.2)
    a.text(0.98, 0.97, f"clean median {np.median(ward):.0f} (solid)\nattacked-hour medians (dashed):\n"
           + "\n".join(lines), transform=a.transAxes, ha="right", va="top", fontsize=6.5, color=INK,
           bbox={"facecolor": "white", "edgecolor": GRID, "linewidth": 0.5, "pad": 2})
    v = bed.ravel()
    b.hist(v, bins=np.arange(0, 31) - 0.5, color=C1, edgecolor="white", linewidth=0.3)
    b.set_xlim(-1, 30)
    b.set_xlabel("Alerts per complete bed-hour (Zone A; cut at 30)")
    b.axvline(v.mean(), color=INK, linewidth=1.2)
    b.axvline(v.mean() + 20 / 2.375, color=MUTED, linewidth=0.7, linestyle="--")
    b.text(0.98, 0.97, f"bed mean {v.mean():.1f} (solid)\nbed mean + Steady episode rate\n"
           f"{v.mean() + 20 / 2.375:.1f} (dashed)", transform=b.transAxes, ha="right", va="top", fontsize=6.5,
           color=INK, bbox={"facecolor": "white", "edgecolor": GRID, "linewidth": 0.5, "pad": 2})
    fig.tight_layout()
    fig.savefig(OUT / "rates.pdf")


def fig_recall():
    ward, bed = load("results/testbed/eval_zoneC.json"), load("results/perbed/eval_zoneD.json")
    fig, axes = plt.subplots(2, len(ATTACKS), figsize=(7.2, 4.4), sharey=True, sharex=True)
    y = np.arange(len(SCORERS))
    for row, (ev, label) in enumerate(((ward, "Ward study (Zone C)"), (bed, "Bed study (Zone D)"))):
        s0 = ev["seeds"]["0"]
        for col, atk in enumerate(ATTACKS):
            ax = axes[row, col]
            rec, ch = [], []
            for m in SCORERS:
                c = s0[m]["copies"][f"{atk}_copy0"]
                rec.append(c["episodes_detected"])
                if "chance_episodes" in c:
                    ch.append(c["chance_episodes"])
                else:
                    f = s0[m]["clean"]["fpr"]
                    ch.append(30 * (1 - (1 - f) ** (c["attack_hours"] / 30)))
            ax.axvline(24, color=MUTED, linewidth=0.8, linestyle=":")
            ax.scatter(ch, y, marker="|", s=60, color=MUTED, linewidths=1.2, label="chance", zorder=2)
            ax.scatter(rec, y, marker="o", s=18, color=C1, edgecolors="white", linewidths=0.6, label="detected",
                       zorder=3)
            ax.set_xlim(-1, 31)
            ax.grid(False)
            if row == 0:
                ax.set_title(NAME[atk], fontsize=8, color=INK)
            if row == 1:
                ax.set_xlabel("episodes / 30")
        axes[row, 0].set_yticks(y)
        axes[row, 0].set_yticklabels([SHORT.get(m, m) for m in SCORERS])
        axes[row, 0].invert_yaxis()
        axes[row, 0].set_ylabel(label, color=INK)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=2, frameon=False, fontsize=7)
    fig.text(0.99, 0.005, "dotted line: gate (24/30)", ha="right", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(OUT / "recall.pdf")


def fig_fpr():
    ward, bed = load("results/testbed/eval_zoneC.json"), load("results/perbed/eval_zoneD.json")
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    x = np.arange(len(SCORERS))
    for off, ev, col, mk, lab in ((-0.15, ward, C1, "o", "Ward study, clean Zone C hours"),
                                  (0.15, bed, C2, "s", "Bed study, clean Zone D ward hours")):
        s0 = ev["seeds"]["0"]
        pts, lo, hi = [], [], []
        for m in SCORERS:
            cl = s0[m]["clean"]
            if "ward_calibration_test" in cl:
                t = cl["ward_calibration_test"]
                r, ci = t["false_alarms"] / t["hours"], t["fpr_ci"]
            else:
                r, ci = cl["fpr"], cl["fpr_ci"]
            pts.append(100 * r)
            lo.append(100 * (r - ci[0]))
            hi.append(100 * (ci[1] - r))
        ax.errorbar(x + off, pts, yerr=[lo, hi], fmt=mk, color=col, ms=4, elinewidth=1, capsize=0, label=lab)
    ax.axhline(5, color=MUTED, linewidth=0.8, linestyle="--")
    ax.text(len(SCORERS) - 0.5, 5.15, "α = 5%", color=INK, fontsize=7, ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(m, m) for m in SCORERS], rotation=30, ha="right")
    ax.set_ylabel("Clean false-alarm rate (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fpr.pdf")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_recall()
    fig_fpr()
    fig_rates()
    print("written:", sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
