"""Lock box: zone files written once, hashed, and Zone C readable only on explicit request."""
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from .zones import ZONES, split_events, zone_bounds


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(root=".") -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_zones(events_path, time_col, out_dir, manifest_path, fractions, purge_minutes) -> dict:
    df = pd.read_csv(events_path)
    t = pd.to_datetime(df[time_col])
    bounds = zone_bounds(t.min(), t.max(), fractions, purge_minutes)
    parts = split_events(df, time_col, bounds)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(events_path), "source_sha256": sha256(events_path),
        "git_commit": git_commit(), "fractions": list(fractions), "purge_minutes": purge_minutes,
        "zones": {},
    }
    for z in ZONES:
        path = out_dir / f"zone_{z}.csv"
        parts[z].to_csv(path, index=False)
        s, e = bounds[z]
        manifest["zones"][z] = {"file": str(path), "sha256": sha256(path), "rows": len(parts[z]),
                                "start": str(s), "end": str(e)}
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(manifest, indent=1))
    return manifest


def load_zone(manifest_path, zone: str, allow_test: bool = False) -> pd.DataFrame:
    if zone == "C" and not allow_test:
        raise PermissionError("Zone C is locked; only the final evaluation may pass allow_test=True")
    manifest = json.loads(Path(manifest_path).read_text())
    entry = manifest["zones"][zone]
    if sha256(entry["file"]) != entry["sha256"]:
        raise RuntimeError(f"zone {zone} file changed since the lock box was written: {entry['file']}")
    return pd.read_csv(entry["file"])
