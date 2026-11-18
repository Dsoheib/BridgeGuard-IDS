"""window one testbed zone in a single streamed pass (60 min, 10-min stride, one fixed grid from the epoch).

Writes testbed_data/windows/zone_<Z>.csv (all windows with >= 1 message) and zone_<Z>_hours.csv (complete hours).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from bridgeguard import windows as W  # noqa: E402
from bridgeguard.hours import hourly_complete  # noqa: E402
from generator import EPOCH  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", choices=["A", "B", "C"], required=True)
    ap.add_argument("--allow-zone", action="store_true", help="required for Zones B and C")
    args = ap.parse_args()
    if args.zone != "A" and not args.allow_zone:
        raise PermissionError(f"Zone {args.zone} is windowed only with --allow-zone")
    cfg = yaml.safe_load(open("configs/protocol.yaml"))
    wm, sm = cfg["windowing"]["window_minutes"], cfg["windowing"]["stride_minutes"]
    man = json.loads(Path("lockbox/testbed_manifest.json").read_text())[args.zone]
    g = W.grid(EPOCH, pd.Timestamp(man["all_beds_occupied_until"]), wm, sm)
    src = Path(cfg["testbed"]["out_dir"]) / f"zone_{args.zone}_events.csv.gz"
    feats, spans = W.featurize_stream(src, g, wm, sm)
    out = Path("testbed_data/windows")
    out.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out / f"zone_{args.zone}.csv", index=False)
    hours = hourly_complete(feats, None, sm, wm, spans=spans)
    hours.to_csv(out / f"zone_{args.zone}_hours.csv", index=False)
    print(json.dumps({"zone": args.zone, "grid_windows": len(g), "windows": len(feats), "complete_hours": len(hours),
                      "stretches": [[str(a), str(b)] for a, b in spans],
                      "alerts_per_hour_median_on_complete_hours": float(hours["alert_frequency_per_hour"].median())},
                     indent=1))


if __name__ == "__main__":
    main()
