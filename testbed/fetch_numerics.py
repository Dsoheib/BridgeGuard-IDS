"""Fetch MIMIC-III Waveform Matched Subset *numerics* records (ODbL v1.0, open) and measure them.

Patients are drawn in a fixed seeded order from RECORDS-numerics; every numerics record of a drawn patient is kept,
so zones can be patient-disjoint. Only numerics (".../<record>n") are fetched, never waveforms or clinical data.

  python testbed/fetch_numerics.py --patients 400            # headers only
  python testbed/fetch_numerics.py --patients 400 --data     # headers + .dat
"""
import argparse
import http.client
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BASE = "https://physionet.org/files/mimic3wdb-matched/1.0/"
OUT = Path("testbed_data/mimic3_numerics")
SEED = 2026


def get(rel: str) -> bytes:
    path = OUT / rel
    if path.exists():
        return path.read_bytes()
    for attempt in range(5):
        try:
            with urllib.request.urlopen(BASE + rel, timeout=300) as r:
                data = r.read()
            break
        except (TimeoutError, OSError, http.client.HTTPException):     # incl. IncompleteRead (truncated body)
            if attempt == 4:
                raise
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def patient_order():
    recs = (OUT / "RECORDS-numerics").read_text().split()
    by_patient = {}
    for r in recs:
        by_patient.setdefault(r.split("/")[1], []).append(r)
    patients = sorted(by_patient)
    random.Random(SEED).shuffle(patients)
    return patients, by_patient


def parse_header(text: str) -> dict:
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    head = lines[0].split()
    fs = float(head[2].split("/")[0])
    signals = []
    for ln in lines[1:]:
        f = ln.split()
        gain_units = f[2]
        gain = float(gain_units.split("/")[0].split("(")[0])
        baseline = float(gain_units.split("(")[1].split(")")[0]) if "(" in gain_units else float(f[4])
        signals.append({"file": f[0], "fmt": f[1], "gain": gain, "baseline": baseline, "name": " ".join(f[8:])})
    return {"record": head[0], "fs_hz": fs, "n_samples": int(head[3]),
            "start": " ".join(head[4:6]) if len(head) > 5 else None, "signals": signals}


def read_dat(rel_dir: str, hdr: dict) -> dict:
    """Format 16 only: interleaved little-endian int16, -32768 = missing."""
    sig = hdr["signals"]
    if {s["fmt"] for s in sig} != {"16"} or len({s["file"] for s in sig}) != 1:
        raise ValueError(f"{hdr['record']}: unsupported layout")
    buf = get(f"{rel_dir}/{sig[0]['file']}")
    if len(buf) != 2 * len(sig) * hdr["n_samples"]:
        raise ValueError(f"{hdr['record']}: {len(buf)} bytes, header implies {2 * len(sig) * hdr['n_samples']}")
    raw = np.frombuffer(buf, dtype="<i2").reshape(-1, len(sig))
    out = {}
    for j, s in enumerate(sig):
        x = raw[:, j].astype(float)
        x[raw[:, j] == -32768] = np.nan
        out[s["name"]] = (x - s["baseline"]) / s["gain"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, required=True)
    ap.add_argument("--data", action="store_true")
    ap.add_argument("--rate", type=float, default=None, help="fetch .dat only for records at this header rate (Hz)")
    args = ap.parse_args()
    patients, by_patient = patient_order()
    drawn = patients[: args.patients]
    recs = [r for p in drawn for r in by_patient[p]]
    with ThreadPoolExecutor(8) as ex:
        headers = list(ex.map(lambda r: parse_header(get(r + ".hea").decode()), recs))
    if args.data:
        todo = [(r, h) for r, h in zip(recs, headers) if args.rate is None or round(h["fs_hz"], 5) == args.rate]
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda rh: read_dat(rh[0].rsplit("/", 1)[0], rh[1]), todo))
    meta = [{"path": r, "patient": r.split("/")[1], "fs_hz": h["fs_hz"], "n_samples": h["n_samples"],
             "start": h["start"], "signals": [s["name"] for s in h["signals"]]} for r, h in zip(recs, headers)]
    (OUT / "drawn_records.json").write_text(json.dumps({"seed": SEED, "patients": drawn, "records": meta}, indent=1))
    print(f"{len(drawn)} patients, {len(recs)} records")


if __name__ == "__main__":
    main()
