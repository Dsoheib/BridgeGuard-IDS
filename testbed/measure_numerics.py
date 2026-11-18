"""Measure the drawn numerics records: header rate vs. how often HR / SpO2 / RESP actually update."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fetch_numerics import OUT, get, parse_header, read_dat  # noqa: E402

SIGNALS = ["HR", "SpO2", "RESP"]
ALIASES = {"%SpO2": "SpO2"}          # minute-rate records mostly use "%SpO2"


def main(n_patients: int):
    drawn = json.loads((OUT / "drawn_records.json").read_text())
    keep = set(drawn["patients"][:n_patients])
    rows = []
    for r in drawn["records"]:
        if r["patient"] not in keep:
            continue
        hdr = parse_header(get(r["path"] + ".hea").decode())
        x = {ALIASES.get(k, k): v for k, v in read_dat(r["path"].rsplit("/", 1)[0], hdr).items()}
        row = {"path": r["path"], "fs_hz": hdr["fs_hz"], "hours": hdr["n_samples"] / hdr["fs_hz"] / 3600}
        for s in SIGNALS:
            v = x.get(s)
            if v is None:
                row[s] = None
                continue
            ok = ~np.isnan(v)
            pos = np.flatnonzero(ok)
            new_at = pos[1:][np.diff(v[ok]) != 0]          # position of each new value (not the sample before it)
            cons = np.diff(pos) == 1
            row[s] = {"present": round(float(ok.mean()), 3),
                      "median_s_between_changes": (float(np.median(np.diff(new_at))) / hdr["fs_hz"]
                                                   if len(new_at) > 1 else None),
                      "new_value_share": round(float((np.diff(v[ok])[cons] != 0).mean()), 3) if cons.any() else None}
        rows.append(row)
    out = {"patients": n_patients, "records": len(rows),
           "fs_counts": dict(collections.Counter(round(r["fs_hz"], 5) for r in rows)), "by_rate": {}}
    for fs in sorted({round(r["fs_hz"], 5) for r in rows}):
        rr = [r for r in rows if round(r["fs_hz"], 5) == fs]
        d = {"records": len(rr), "hours": round(sum(r["hours"] for r in rr), 1)}
        for s in SIGNALS:
            have = [r[s] for r in rr if r[s]]
            gaps = [h["median_s_between_changes"] for h in have if h["median_s_between_changes"]]
            d[s] = {"records_with_signal": len(have),
                    "median_present_share": round(float(np.median([h["present"] for h in have])), 3) if have else None,
                    "median_s_between_changes": round(float(np.median(gaps)), 1) if gaps else None,
                    "median_new_value_share": round(float(np.median([h["new_value_share"] for h in have
                                                                     if h["new_value_share"] is not None])), 3)}
        out["by_rate"][str(fs)] = d
    (OUT / "measurements.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]))
