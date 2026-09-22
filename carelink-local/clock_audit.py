"""Audit pump-local timestamp ambiguity without rewriting any source record.

Only aggregates leave private-data. Parsed record ordinals are not CSV line numbers.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

from archive_inventory import read_records

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "private-data"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def private_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(PRIVATE.resolve()):
        raise ValueError("Row-level outputs must be under the ignored private-data directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_archive(paths):
    """Keep all conflicting observations; equal duplicate SG rows count once later."""
    glucose = defaultdict(list)
    clocks, files, carbs = [], [], set()
    for number, path in enumerate(paths, 1):
        private_path(path)
        digest = sha256(path)
        files.append({"block": number, "sha256": digest, "bytes": path.stat().st_size})
        for ordinal, (day, record) in enumerate(read_records(path), 1):
            stamp = datetime.combine(day, datetime.strptime(record["Time"], "%H:%M:%S").time())
            sg = record.get("Sensor Glucose (mmol/L)", "").strip()
            if sg:
                glucose[stamp].append({"block": number, "ordinal": ordinal,
                                       "record": {k:v for k,v in record.items() if v.strip()}, "value": float(sg)})
            if record.get("BWZ Carb Input (grams)", "").strip():
                carbs.add(stamp)
            new = record.get("New Device Time", "").strip()
            if new:
                clocks.append({"block": number, "ordinal": ordinal, "record": record,
                               "old": stamp, "new": datetime.strptime(new, "%Y/%m/%d %H:%M:%S")})
        if sha256(path) != digest:
            raise RuntimeError("Input changed while reading")
    return glucose, clocks, sorted(carbs), files


def audit(glucose, clocks, files):
    conflicts = {t: rows for t, rows in glucose.items() if len({r["value"] for r in rows}) > 1}
    conflict_days = {t.date() for t in conflicts}
    blocked_days = set(conflict_days)
    for c in clocks:
        lo, hi = sorted((c["old"].date(), c["new"].date()))
        blocked_days.update(lo + timedelta(days=i) for i in range((hi-lo).days + 1))
    # Index is export-local: check whether conflicting rows bracket the exported
    # clock-change row, but do not mistake an export index for a device UTC clock.
    between = 0
    for t, rows in conflicts.items():
        block = rows[0]["block"]
        indices = [float(r["record"]["Index"]) for r in rows]
        between += any(c["block"] == block and min(indices) < float(c["record"]["Index"]) < max(indices)
                       for c in clocks)
    summary = {
        "status": "measured; physical ordering unresolved; full-day exclusions selected before windows",
        "files": files,
        "unique_sensor_timestamps": len(glucose),
        "repeated_sensor_rows": sum(len(r)-1 for r in glucose.values()),
        "conflicting_timestamps": len(conflicts),
        "conflict_days": len(conflict_days),
        "clock_change_rows": len(clocks),
        "clock_row_populated_fields": sorted({k for c in clocks for k,v in c["record"].items() if v.strip()}),
        "clock_new_minus_row_seconds": dict(sorted(Counter(int((c["new"]-c["old"]).total_seconds()) for c in clocks).items())),
        "conflicts_bracketing_clock_row_by_export_index": between,
        "conflict_days_with_clock_rows": len(conflict_days & {c["old"].date() for c in clocks}),
        "excluded_calendar_days": len(blocked_days),
        "sensor_timestamps_on_excluded_days": sum(t.date() in blocked_days for t in glucose),
        "rule": "Reject any full history-to-last-target interval intersecting any conflict day or any calendar day between a clock row's old and new date inclusive. No UTC or DST reconstruction. Equal SG duplicates collapse only after the audit; conflicting days are never averaged. All original records preserved.",
        "missing_ordering_evidence": "No timezone/UTC offset, monotone device event time, or documented relation of export Index and New Device Time to physical SG acquisition order. Index brackets alone do not supply that contract.",
    }
    evidence = {"blocked_days": sorted(map(str, blocked_days)),
                "conflicts": {str(t): rows for t,rows in conflicts.items()}, "clock_rows": clocks}
    return summary, evidence, blocked_days


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=PRIVATE / "carelink")
    ap.add_argument("--output", type=Path, default=ROOT / "assets/article/clock-audit.json")
    args = ap.parse_args()
    glucose, clocks, _, files = load_archive(sorted(args.input_dir.glob("*.csv")))
    summary, evidence, _ = audit(glucose, clocks, files)
    private_path(PRIVATE / "evaluation/clock-evidence.json").write_text(json.dumps(evidence, default=str, indent=2)+"\n")
    summary["code_sha256"] = sha256(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
