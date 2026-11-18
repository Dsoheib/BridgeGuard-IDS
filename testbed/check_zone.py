"""Stream-check a generated zone file: time order, emergency-after-sample at ties, per-bed counts,
and one record reproduced exactly from the raw numerics."""
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import generator as G  # noqa: E402


def main(zone: str):
    out = Path("testbed_data/zones")
    plan = json.loads((out / f"zone_{zone}_plan.json").read_text())
    prev, per_bed, bad_order = ("", False), {}, 0
    with gzip.open(out / f"zone_{zone}_events.csv.gz", "rt", newline="") as f:
        f.readline()
        for line in f:
            k = G._key(line)
            if k < prev:
                bad_order += 1
            prev = k
            bed = line.split(",", 3)[2].split("/")[1]
            per_bed[bed] = per_bed.get(bed, 0) + 1
    # reproduce the first record of bed 0 from the raw numerics and compare with the zone file's rows for it
    e = next(x for x in plan["plan"] if x["bed"] == 0)
    want = G.record_events(G.load_record(e["path"]), 0, e["offset_s"])
    end = want["timestamp"].max()
    got = pd.read_csv(out / f"zone_{zone}_events.csv.gz", nrows=len(want) * 6, parse_dates=["timestamp"])
    got = got[(got["topic"].str.startswith("ward/bed0/")) & (got["timestamp"] <= end)]
    same = (len(got) == len(want) and
            sorted(zip(got["timestamp"], got["message"])) == sorted(zip(want["timestamp"], want["message"])))
    print(json.dumps({"order_violations": bad_order, "rows_per_bed": per_bed,
                      "first_record_reproduced": bool(same), "first_record_rows": len(want)}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
