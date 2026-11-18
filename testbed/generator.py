"""Trace-driven generator: MIMIC-III 1 Hz numerics -> bridge event stream (rules in protocol/TESTBED_SCORERS.md).

- One message per present timestamp with whichever of HR, SpO2, RESP are present; nothing when all three are missing.
- NEWS2 score-of-3 bands, SpO2 Scale 1 (Royal College of Physicians, NEWS2, 2017), same for every patient:
  HR <= 40 or >= 131, SpO2 <= 91, RESP <= 8 or >= 25. One extreme parameter, not a total NEWS2 score.
- Emergency: fires once when a present sample enters a band; re-arms only when a later present sample is outside the band
  and >= 60 s of record time have passed since the last in-band sample (gaps count). At a record start the band is
  disarmed until an out-of-band sample is seen, so a record that starts in band does not fire.
- Beds play their patients' records back to back with >= 1 h between records; all beds share one timeline.
"""
import heapq
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fetch_numerics import OUT, get, parse_header, read_dat  # noqa: E402

ALIASES = {"%SpO2": "SpO2"}
SIGNALS = {"HR": "hr", "SpO2": "spo2", "RESP": "resp"}
BANDS = [("HR", "low", 40), ("HR", "high", 131), ("SpO2", "low", 91), ("RESP", "low", 8), ("RESP", "high", 25)]
REARM_S = 60
RECORD_GAP_S = 3600
EPOCH = pd.Timestamp("2026-01-05 00:00:00")


def load_record(path: str) -> pd.DataFrame:
    hdr = parse_header(get(path + ".hea").decode())
    x = {ALIASES.get(k, k): v for k, v in read_dat(path.rsplit("/", 1)[0], hdr).items()}
    df = pd.DataFrame({s: x.get(s, np.full(hdr["n_samples"], np.nan)) for s in SIGNALS})
    df["t"] = np.arange(hdr["n_samples"], dtype=float) / hdr["fs_hz"]
    return df


def band_entries(t: np.ndarray, v: np.ndarray, side: str, limit: float) -> np.ndarray:
    """Times at which an emergency fires for one band (t, v: present samples only, in time order)."""
    inb = v <= limit if side == "low" else v >= limit
    if not inb.any():
        return np.array([])
    edges = np.diff(np.r_[0, inb.astype(np.int8), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    fire = []
    for k, s in enumerate(starts):
        if s == 0:
            continue                                  # disarmed at record start; nothing out-of-band before
        if k == 0 or t[s - 1] >= t[ends[k - 1]] + REARM_S:
            fire.append(t[s])                         # last out-of-band sample before s re-armed the band
    return np.array(fire)


def record_events(df: pd.DataFrame, bed: int, offset_s: float) -> pd.DataFrame:
    present = df[list(SIGNALS)].notna().any(axis=1).values
    p = df[present]
    ts = EPOCH + pd.to_timedelta(offset_s + p["t"].values, unit="s")
    msgs = []
    vals = {s: p[s].values for s in SIGNALS}
    for i in range(len(p)):
        msgs.append(json.dumps({"bed": bed, **{k: round(float(vals[s][i]), 1) for s, k in SIGNALS.items()
                                                if not np.isnan(vals[s][i])}}, separators=(",", ":")))
    normal = pd.DataFrame({"timestamp": ts, "type": "normal", "topic": f"ward/bed{bed}/vitals", "message": msgs,
                           "heart_rate": p["HR"].values})
    alerts = []
    for s, side, limit in BANDS:
        ok = df[s].notna().values
        for t in band_entries(df["t"].values[ok], df[s].values[ok], side, limit):
            v = float(df[s].values[int(round(t))])
            alerts.append({"timestamp": EPOCH + pd.to_timedelta(offset_s + t, unit="s"), "type": "emergency",
                           "topic": f"ward/bed{bed}/emergency/{SIGNALS[s]}",
                           "message": json.dumps({"bed": bed, "alert": f"{SIGNALS[s]}_{side}", "value": round(v, 1)},
                                                 separators=(",", ":")),
                           "heart_rate": df["HR"].values[int(round(t))]})
    return pd.concat([normal, pd.DataFrame(alerts, columns=normal.columns)], ignore_index=True)


def schedule(patients, records_by_patient, n_beds: int):
    """Greedy: next patient goes to the bed that frees first; records in path order, >= 1 h apart."""
    beds = [(0.0, b) for b in range(n_beds)]
    plan = []
    for p in patients:
        free, b = heapq.heappop(beds)
        for r in records_by_patient[p]:
            plan.append({"bed": b, "patient": p, "path": r["path"], "offset_s": free})
            free += r["n_samples"] + RECORD_GAP_S
        heapq.heappush(beds, (free, b))
    return plan, {b: f for f, b in beds}


TS_FORMAT = "%Y-%m-%d %H:%M:%S"     # 1 Hz records and whole-second offsets: fixed width, so text order = time order
COLUMNS = ["timestamp", "type", "topic", "message", "heart_rate"]


def _gzip_writer(path: Path):
    import gzip
    import io
    return io.TextIOWrapper(gzip.GzipFile(filename="", mode="wb", fileobj=open(path, "wb"), mtime=0), newline="")


def write_bed(entries, path: Path, cut: pd.Timestamp) -> None:
    """One bed's records in time order, one record in memory at a time, rows sorted (timestamp, normal before emergency)."""
    with _gzip_writer(path) as f:
        f.write(",".join(COLUMNS) + "\n")
        for e in entries:
            if EPOCH + pd.to_timedelta(e["offset_s"], unit="s") > cut:
                break
            ev = record_events(load_record(e["path"]), e["bed"], e["offset_s"])
            ev["_emg"] = (ev["type"] == "emergency").astype(int)
            ev = ev.sort_values(["timestamp", "_emg"], kind="stable")
            ev = ev[ev["timestamp"] <= cut].drop(columns="_emg")
            ev.to_csv(f, index=False, header=False, date_format=TS_FORMAT, lineterminator="\n")


def _key(line: str):
    ts, typ = line.split(",", 2)[:2]
    return ts, typ == "emergency"


def merge_beds(parts, path: Path) -> dict:
    """Line-by-line k-way merge of the sorted bed files; ties keep bed order. Counts are taken while streaming."""
    import gzip
    readers = [gzip.open(q, "rt", newline="") for q in parts]
    for r in readers:
        r.readline()
    n = n_emg = 0
    first = last = None
    by_signal = {}
    tmp = path.with_suffix(".partial")
    with _gzip_writer(tmp) as f:
        f.write(",".join(COLUMNS) + "\n")
        for line in heapq.merge(*readers, key=_key):
            f.write(line)
            ts, emg = _key(line)
            first, last, n = first or ts, ts, n + 1
            if emg:
                n_emg += 1
                sig = line.split(",", 3)[2].rsplit("/", 1)[-1]
                by_signal[sig] = by_signal.get(sig, 0) + 1
    for r in readers:
        r.close()
    tmp.rename(path)
    return {"events": n, "emergencies": n_emg, "first": first, "last": last, "emergencies_by_signal": by_signal}


def main():
    import argparse
    import hashlib

    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", choices=["A", "B", "C", "D"], required=True)
    ap.add_argument("--allow-zone", action="store_true", help="required for Zones B, C and D")
    args = ap.parse_args()
    if args.zone != "A" and not args.allow_zone:
        raise PermissionError(f"Zone {args.zone} is generated only with --allow-zone")
    tb = yaml.safe_load(open("configs/protocol.yaml"))["testbed"]
    if tb["cut"] != "all_beds_occupied":
        raise ValueError(f"unknown cut {tb['cut']}")
    if args.zone == "D":            # per-bed study (protocol/PERBED_PROTOCOL.md): its own list and manifest
        zone = json.loads(Path("lockbox/testbed_zone_D.json").read_text())
        meta = json.loads((OUT / "zoneD_records.json").read_text())["records"]
        man = Path("lockbox/testbed_manifest_D.json")
    else:
        zone = json.loads(Path("lockbox/testbed_zones.json").read_text())["zones"][args.zone]
        meta = json.loads((OUT / "drawn_records.json").read_text())["records"]
        man = Path("lockbox/testbed_manifest.json")
    by_patient = {}
    for r in meta:
        if r["path"] in zone["records"]:
            by_patient.setdefault(r["patient"], []).append(r)
    plan, bed_end = schedule(zone["patients"], by_patient, tb["n_beds"])
    cut = EPOCH + pd.to_timedelta(min(bed_end.values()) - RECORD_GAP_S, unit="s")
    out = Path(tb["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    parts = [out / f"zone_{args.zone}_bed{b}.csv.gz" for b in range(tb["n_beds"])]
    for b, q in enumerate(parts):
        write_bed([e for e in plan if e["bed"] == b], q, cut)
    path = out / f"zone_{args.zone}_events.csv.gz"
    s = merge_beds(parts, path)
    for q in parts:
        q.unlink()
    (out / f"zone_{args.zone}_plan.json").write_text(json.dumps({"plan": plan, "all_beds_occupied_until": str(cut)},
                                                                indent=1))
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    summary = {"zone": args.zone, "n_beds": tb["n_beds"], **s, "all_beds_occupied_until": str(cut),
               "span_hours": round((pd.Timestamp(s["last"]) - pd.Timestamp(s["first"])).total_seconds() / 3600, 1),
               "bytes": path.stat().st_size, "sha256": h.hexdigest()}
    m = json.loads(man.read_text()) if man.exists() else {}
    m[args.zone] = summary
    man.write_text(json.dumps(m, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
