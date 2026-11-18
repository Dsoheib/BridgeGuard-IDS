"""Per-bed study, Zone D patient list from headers only (protocol/PERBED_PROTOCOL.md, sections 1 and 7.1).

Walk the seed-2026 shuffle from index 400; keep each patient with >= 1 one-second numerics record, in order; stop at the
first patient after which generator.schedule() (4 beds, 1 h gap after each record) gives an all-beds-occupied prefix of
>= 600 h. Only .hea files are fetched. drawn_records.json and lockbox/testbed_zones.json are not touched.
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from fetch_numerics import OUT, SEED, get, parse_header, patient_order  # noqa: E402
from generator import RECORD_GAP_S, schedule  # noqa: E402

START_INDEX, TARGET_H = 400, 600.0


def main():
    n_beds = yaml.safe_load(open("configs/protocol.yaml"))["testbed"]["n_beds"]
    taken = set(json.loads(Path("lockbox/testbed_zones.json").read_text())["zones"]["A"]["patients"])
    for z in ("B", "C"):
        taken |= set(json.loads(Path("lockbox/testbed_zones.json").read_text())["zones"][z]["patients"])
    patients, by_patient = patient_order()
    kept, recs, walked, occupied_h = [], {}, 0, 0.0
    for idx in range(START_INDEX, len(patients)):
        p = patients[idx]
        if p in taken:
            raise RuntimeError(f"{p} at index {idx} is already in a zone")
        walked += 1
        one_hz = []
        for r in by_patient[p]:
            h = parse_header(get(r + ".hea").decode())
            if round(h["fs_hz"], 5) == 1.0:
                one_hz.append({"path": r, "patient": p, "fs_hz": h["fs_hz"], "n_samples": h["n_samples"],
                               "start": h["start"], "signals": [s["name"] for s in h["signals"]]})
        if not one_hz:
            continue
        kept.append(p)
        recs[p] = one_hz
        _, bed_end = schedule(kept, recs, n_beds)
        occupied_h = (min(bed_end.values()) - RECORD_GAP_S) / 3600
        if occupied_h >= TARGET_H:
            break
    else:
        raise RuntimeError("shuffle exhausted before the target")
    records = [r for p in kept for r in recs[p]]
    (OUT / "zoneD_records.json").write_text(json.dumps({"seed": SEED, "start_index": START_INDEX,
                                                        "records": records}, indent=1))
    out = {"seed": SEED, "rule": "seed-2026 shuffle from index 400; 1 Hz patients in order; stop at >= 600 h occupied",
           "start_index": START_INDEX, "last_index": START_INDEX + walked - 1, "patients_walked": walked,
           "patients": kept, "records": [r["path"] for r in records],
           "header_hours": round(sum(r["n_samples"] for r in records) / 3600, 1),
           "header_all_beds_occupied_h": round(occupied_h, 2)}
    Path("lockbox/testbed_zone_D.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("patients", "records")} | {"patients_kept": len(kept),
                                                                                           "records_kept": len(records)},
                     indent=1))


if __name__ == "__main__":
    main()
