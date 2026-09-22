"""Read CareLink CSV tables; report coverage and field availability, not forecasts.

Inputs must stay outside the repository. Supply non-overlapping export blocks.
The output contains aggregates only; no glucose values or event timestamps.
"""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path


def read_records(path):
    header = None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if "Date" in row and "Time" in row:
                header = row
                if "Sensor Glucose (mmol/L)" not in header:
                    raise ValueError("Unsupported glucose column/units; do not silently mix schemas")
                continue
            if header is None:
                continue
            record = dict(zip(header, row))
            # Device-section labels appear between tables. Event indices in this
            # export are decimal strings (for example, 0.00000), not integers.
            try:
                float(record.get("Index", ""))
            except ValueError:
                continue
            date = record.get("Date", "").strip()
            if not date:
                continue
            # This inventory deliberately supports only the observed CSV contract.
            day = dt.datetime.strptime(date, "%Y/%m/%d").date()
            yield day, record


def inventory(paths):
    fields = collections.Counter()
    all_fields = set()
    categories = collections.defaultdict(collections.Counter)
    years = collections.defaultdict(collections.Counter)
    dates = set()
    glucose = {}
    source_days = set()
    files = []
    duplicate_sg = conflicting_sg = 0
    conflicting_days, clock_change_days = set(), set()
    for number, path in enumerate(paths, 1):
        file_dates, file_sg_dates = set(), set()
        count = 0
        for day, record in read_records(path):
            count += 1
            file_dates.add(day)
            dates.add(day)
            year = str(day.year)
            all_fields.update(record)
            if record.get("New Device Time", "").strip():
                clock_change_days.add(day)
            years[year]["event_rows"] += 1
            for key, value in record.items():
                if value.strip():
                    fields[key] += 1
            for key in ("Bolus Source", "Sensor Exception", "Suspend", "Alert"):
                value = record.get(key, "").strip()
                if value:
                    categories[key][value] += 1
            for key, label in (("BWZ Carb Input (grams)", "carb_rows"),
                               ("Sensor Exception", "sensor_exception_rows"),
                               ("Alert", "alert_rows"),
                               ("Bolus Volume Delivered (U)", "delivered_bolus_rows")):
                if record.get(key, "").strip():
                    years[year][label] += 1
            if record.get("Bolus Source") == "CLOSED_LOOP_AUTO_INSULIN":
                years[year]["automatic_aggregate_rows"] += 1
                categories["automatic_time_of_day"][record.get("Time", "").strip()] += 1
            value = record.get("Sensor Glucose (mmol/L)", "").strip()
            if value:
                stamp = dt.datetime.combine(day, dt.time.fromisoformat(record["Time"].strip()))
                years[year]["sensor_rows"] += 1
                file_sg_dates.add(day)
                if stamp in glucose:
                    duplicate_sg += 1
                    if float(glucose[stamp]) != float(value):
                        conflicting_sg += 1
                        conflicting_days.add(day)
                else:
                    glucose[stamp] = value
        overlap = source_days & file_dates
        if overlap:
            raise ValueError("Input blocks overlap in calendar dates; supply non-overlapping blocks")
        source_days.update(file_dates)
        files.append({"block": number, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "bytes": path.stat().st_size, "event_rows": count,
                      "first_record_date": str(min(file_dates)) if file_dates else None,
                      "last_record_date": str(max(file_dates)) if file_dates else None,
                      "days_with_records": len(file_dates), "days_with_sensor": len(file_sg_dates)})
    stamps = sorted(glucose)
    if not stamps:
        raise ValueError("No sensor data in observed mmol/L schema")
    first, last = stamps[0].date(), stamps[-1].date()
    sensor_days = {x.date() for x in stamps}
    # Floor into local wall-clock five-minute slots for coverage only.
    # This is not modeling preprocessing and cannot resolve timezone/DST ambiguity.
    slots = {(x.date(), (x.hour * 60 + x.minute) // 5) for x in stamps}
    for year in years:
        start, end = max(first, dt.date(int(year), 1, 1)), min(last, dt.date(int(year), 12, 31))
        n_days = max(0, (end - start).days + 1)
        years[year]["calendar_days_in_sensor_span"] = n_days
        years[year]["days_with_sensor"] = sum(x.year == int(year) for x in sensor_days)
        n_slots = sum(x[0].year == int(year) for x in slots)
        years[year]["occupied_five_minute_slots"] = n_slots
        years[year]["nominal_slot_coverage_percent"] = round(100 * n_slots / (288 * n_days), 2) if n_days else None
    gaps = [(b - a).total_seconds() / 3600 for a, b in zip(stamps, stamps[1:])]
    missing_days, longest_missing = 0, 0
    for offset in range((last - first).days + 1):
        day = first + dt.timedelta(days=offset)
        missing_days = missing_days + 1 if day not in sensor_days else 0
        longest_missing = max(longest_missing, missing_days)
    return {"status": "measured export inventory; no model run",
            "method": "Non-overlapping dated blocks; all repeated CSV headers parsed. SG deduplicated by local timestamp; duplicate conflicts counted. Coverage uses occupied local five-minute slots / 288 per calendar day, including partial boundary days. No timezone/DST correction; row counts are not distinct meals or failures.",
            "files": files, "event_rows": sum(f["event_rows"] for f in files),
            "first_sensor_date": str(first), "last_sensor_date": str(last),
            "calendar_days_in_sensor_span": (last-first).days+1,
            "days_with_sensor": len(sensor_days), "unique_sensor_timestamps": len(glucose),
            "duplicate_sensor_timestamps": duplicate_sg, "conflicting_sensor_values": conflicting_sg,
            "days_with_conflicting_sensor_timestamps": len(conflicting_days),
            "conflict_days_with_recorded_clock_changes": len(conflicting_days & clock_change_days),
            "longest_run_of_days_without_sensor": longest_missing,
            "longest_gap_between_sensor_readings_hours": round(max(gaps), 2),
            "gaps_over_24_hours": sum(g > 24 for g in gaps),
            "years": dict(years), "populated_fields": dict(fields),
            "empty_fields": sorted(all_fields - set(fields)),
            "category_counts": {k: dict(v) for k, v in categories.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(inventory(args.inputs), indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")
