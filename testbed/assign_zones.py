"""patient-disjoint testbed zones from headers only (no signal values, no labels).

The patients with a 1 Hz numerics record, in the seed-2026 draw order, are split by count: the first 120 are Zone A,
the next 40 Zone B, the last 41 Zone C (configs/protocol.yaml, testbed.split). Minute-rate records are not used; a patient
with both rates keeps only its 1 Hz records, and records are never stitched across clocks.
(Supersedes an earlier cumulative-hours cut, 114/44/43, made before any Zone B/C signal value was read.)
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from fetch_numerics import OUT  # noqa: E402


def main():
    split = yaml.safe_load(open("configs/protocol.yaml"))["testbed"]["split"]
    d = json.loads((OUT / "drawn_records.json").read_text())
    recs = [r for r in d["records"] if round(r["fs_hz"], 5) == 1.0]
    have = {r["patient"] for r in recs}
    order = [p for p in d["patients"] if p in have]
    if len(order) != sum(split.values()):
        raise ValueError(f"{len(order)} one-second patients, split sums to {sum(split.values())}")
    zones, i = {}, 0
    for z in ("A", "B", "C"):
        ps = order[i: i + split[z]]
        i += split[z]
        zones[z] = {"patients": ps, "records": [r["path"] for r in recs if r["patient"] in ps],
                    "header_hours": round(sum(r["n_samples"] for r in recs if r["patient"] in ps) / 3600, 1)}
    out = {"seed": d["seed"], "rule": "1 Hz records only; seed-2026 draw order; split by patient count", "zones": zones}
    Path("lockbox/testbed_zones.json").write_text(json.dumps(out, indent=1))
    for z, v in zones.items():
        print(z, len(v["patients"]), "patients", len(v["records"]), "records", v["header_hours"], "h")


if __name__ == "__main__":
    main()
