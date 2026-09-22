"""Controlled glucose-context comparison on the author's CareLink archive.

This script is deliberately narrower than the historical meal harness.  It
uses the same carb-event prediction times, 5-minute glucose grid and 30/60/90/
120-minute outcomes for three regularised linear arms:

  A  glucose history, summaries, corrected 30-minute slope and time of day
  B  A plus the carb entry at issue time and prior recorded carbs
  C  B plus prior delivered boluses from the three reconciled user categories

The automatic-insulin category is excluded from every four-hour bolus feature.
It is an aggregate written at midnight, not a delivered-event timestamp.

Outer evaluation is forward in calendar weeks: each test week is trained only
on earlier weeks, with a four-week minimum training prefix. Training samples
whose two-hour target would cross the test-week boundary are purged. Test
histories may use already-observed glucose before the issue time, as they would
in a real forward forecast. The dataset and evaluation period were explored by
earlier agents, so this is an exploratory comparison rather than a pristine
holdout.

Only aggregate metrics are printed or optionally written. No row-level data is
written to the repository.
"""
from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
BOOTSTRAPS = 1000
ALPHA_GRID = np.logspace(-3.0, 3.0, 25)
HORIZONS_SLOTS = {30: 5, 60: 11, 90: 17, 120: 23}
SLOT_S = 300
HISTORY_SLOTS = 288
TARGET_SLOTS = 24
SLOPE_SLOTS = 6
MIN_TRAIN_WEEKS = 4
DB = "carelink.db"

RECONCILED_BOLUS_SOURCES = (
    "Normal BOLUS_WIZARD",
    "Normal CLOSED_LOOP_BG_CORRECTION",
    "Normal CLOSED_LOOP_BG_CORRECTION_AND_FOOD_BOLUS",
)

GLUCOSE_FEATURES = [
    "last_g", "hist_mean", "hist_std", "hist_min", "hist_max",
    "slope_30m", "tod_sin", "tod_cos",
]
CARB_FEATURES = ["grams_at_issue", "prior_carbs_4h"]
BOLUS_FEATURES = [
    "prior_bolus_4h_bolus_wizard",
    "prior_bolus_4h_bg_correction",
    "prior_bolus_4h_bg_correction_and_food",
]
MODEL_FEATURES = {
    "A": GLUCOSE_FEATURES,
    "B": GLUCOSE_FEATURES + CARB_FEATURES,
    "C": GLUCOSE_FEATURES + CARB_FEATURES + BOLUS_FEATURES,
}


def parse_ts(s: str) -> datetime:
    """Parse pump-local wall time at a fixed offset for monotone differences."""
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)


def source_feature(source: str) -> str:
    return {
        RECONCILED_BOLUS_SOURCES[0]: BOLUS_FEATURES[0],
        RECONCILED_BOLUS_SOURCES[1]: BOLUS_FEATURES[1],
        RECONCILED_BOLUS_SOURCES[2]: BOLUS_FEATURES[2],
    }[source]


def build_samples(conn: sqlite3.Connection) -> tuple[pd.DataFrame, dict[str, int]]:
    sg_rows = conn.execute(
        "SELECT ts, sg_mgdl FROM sg_website ORDER BY ts"
    ).fetchall()
    sg_epoch = np.array([int(parse_ts(r[0]).timestamp()) for r in sg_rows])
    sg_val = np.array([r[1] for r in sg_rows], dtype=float)

    bolus_rows = conn.execute(
        "SELECT ts, detail, value FROM events_website WHERE kind='bolus'"
    ).fetchall()
    carb_rows = conn.execute(
        "SELECT ts, value FROM events_website WHERE kind='carb'"
    ).fetchall()
    conn.close()

    boluses = sorted(
        (int(parse_ts(ts).timestamp()), detail, None if value is None else float(value))
        for ts, detail, value in bolus_rows
    )
    carbs = sorted(
        (parse_ts(ts), None if value is None else float(value))
        for ts, value in carb_rows
    )

    drop = {
        "carb_events_in": len(carbs),
        "another_carb_within_2h": 0,
        "missing_target": 0,
        "history_coverage": 0,
    }
    samples: list[dict[str, object]] = []

    for i, (t0, grams) in enumerate(carbs):
        e0 = int(t0.timestamp())
        if any(
            e0 < int(c_ts.timestamp()) <= e0 + 7200
            for c_ts, _ in carbs[i + 1:]
        ):
            drop["another_carb_within_2h"] += 1
            continue

        hist_vals = np.full(HISTORY_SLOTS, np.nan)
        hist_mask = np.zeros(HISTORY_SLOTS, dtype=int)
        tgt_vals = np.full(TARGET_SLOTS, np.nan)
        tgt_mask = np.zeros(TARGET_SLOTS, dtype=int)
        refs = np.concatenate([
            e0 - 86400 + SLOT_S * np.arange(HISTORY_SLOTS),
            e0 + SLOT_S * np.arange(1, TARGET_SLOTS + 1),
        ])
        for k, ref in enumerate(refs):
            j = bisect.bisect_left(sg_epoch, ref - 150)
            if j < len(sg_epoch) and sg_epoch[j] < ref + 150:
                if k < HISTORY_SLOTS:
                    hist_vals[k] = sg_val[j]
                    hist_mask[k] = 1
                else:
                    tgt_vals[k - HISTORY_SLOTS] = sg_val[j]
                    tgt_mask[k - HISTORY_SLOTS] = 1

        if tgt_mask.sum() != TARGET_SLOTS:
            drop["missing_target"] += 1
            continue
        if hist_mask[-72:].sum() < 0.8 * 72:
            drop["history_coverage"] += 1
            continue

        observed = hist_vals[hist_mask == 1]
        last_g = float(hist_vals[np.where(hist_mask == 1)[0][-1]])
        tail = np.where(hist_mask[-SLOPE_SLOTS:] == 1)[0]
        if len(tail) >= 2:
            # x is minutes, not slot number: the slope is mg/dL per minute.
            x = tail.astype(float) * (SLOT_S / 60.0)
            slope = float(np.polyfit(x, hist_vals[-SLOPE_SLOTS:][tail], 1)[0])
        else:
            slope = 0.0

        seconds = t0.hour * 3600 + t0.minute * 60 + t0.second
        prior_carbs = sum(
            c_grams or 0.0
            for c_ts, c_grams in carbs
            if e0 - 14400 < int(c_ts.timestamp()) < e0
        )
        prior_bolus = {feature: 0.0 for feature in BOLUS_FEATURES}
        for b_ep, detail, value in boluses:
            if (
                value is not None
                and e0 - 14400 < b_ep < e0
                and detail in RECONCILED_BOLUS_SOURCES
            ):
                prior_bolus[source_feature(detail)] += value

        iso = t0.isocalendar()
        week_id = f"{iso.year:04d}-W{iso.week:02d}"
        week_start = t0 - timedelta(
            days=t0.weekday(), hours=t0.hour,
            minutes=t0.minute, seconds=t0.second,
            microseconds=t0.microsecond,
        )
        row: dict[str, object] = {
            "meal_ts": t0.strftime("%Y-%m-%dT%H:%M:%S"),
            "t0_epoch": e0,
            "target_end_epoch": e0 + 7200,
            "week_id": week_id,
            "week_start_epoch": int(week_start.timestamp()),
            "grams_at_issue": float(grams or 0.0),
            "prior_carbs_4h": float(prior_carbs),
            "last_g": last_g,
            "hist_mean": float(observed.mean()),
            "hist_std": float(observed.std(ddof=1)) if len(observed) > 1 else 0.0,
            "hist_min": float(observed.min()),
            "hist_max": float(observed.max()),
            "slope_30m": slope,
            "tod_sin": float(np.sin(2 * np.pi * seconds / 86400.0)),
            "tod_cos": float(np.cos(2 * np.pi * seconds / 86400.0)),
            "tgt_vals": tgt_vals,
        }
        row.update(prior_bolus)
        samples.append(row)

    return pd.DataFrame(samples).sort_values("meal_ts").reset_index(drop=True), drop


def make_folds(df: pd.DataFrame) -> list[dict[str, object]]:
    week_order = (
        df[["week_id", "week_start_epoch"]]
        .drop_duplicates()
        .sort_values("week_start_epoch")
    )
    weeks = list(week_order["week_id"])
    folds: list[dict[str, object]] = []
    for i, test_week in enumerate(weeks):
        if i < MIN_TRAIN_WEEKS:
            continue
        test_start = int(
            week_order.loc[week_order["week_id"] == test_week, "week_start_epoch"].iloc[0]
        )
        test_idx = np.where(df["week_id"].to_numpy() == test_week)[0]
        train_idx = np.where(
            (df["week_start_epoch"].to_numpy() < test_start)
            & (df["target_end_epoch"].to_numpy() < test_start)
        )[0]
        train_weeks = sorted(df.iloc[train_idx]["week_id"].unique())
        if len(train_idx) == 0 or len(train_weeks) < MIN_TRAIN_WEEKS:
            continue
        assert all(
            int(df.iloc[j]["target_end_epoch"]) < test_start for j in train_idx
        )
        folds.append({
            "test_week": test_week,
            "test_idx": test_idx,
            "train_idx": train_idx,
            "train_weeks": train_weeks,
        })
    if not folds:
        raise RuntimeError("no forward folds survived the minimum training prefix")
    return folds


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    z = (X - mean) / scale
    y_mean = float(y.mean())
    gram = z.T @ z + alpha * np.eye(z.shape[1])
    coef = np.linalg.solve(gram, z.T @ (y - y_mean))
    return mean, scale, coef, y_mean


def ridge_predict(
    X: np.ndarray,
    fitted: tuple[np.ndarray, np.ndarray, np.ndarray, float],
) -> np.ndarray:
    mean, scale, coef, y_mean = fitted
    return ((X - mean) / scale) @ coef + y_mean


def fit_arm(
    df: pd.DataFrame,
    features: list[str],
    folds: list[dict[str, object]],
) -> tuple[dict[int, np.ndarray], dict[int, list[float]]]:
    X = df[features].to_numpy(dtype=float)
    y_hor = {
        h: np.array([float(v[slot]) for v in df["tgt_vals"]], dtype=float)
        for h, slot in HORIZONS_SLOTS.items()
    }
    pred = {h: np.full(len(df), np.nan) for h in HORIZONS_SLOTS}
    chosen: dict[int, list[float]] = {h: [] for h in HORIZONS_SLOTS}
    week_arr = df["week_id"].to_numpy()

    for fold in folds:
        train_idx = np.asarray(fold["train_idx"], dtype=int)
        test_idx = np.asarray(fold["test_idx"], dtype=int)
        train_weeks = list(fold["train_weeks"])
        for h in HORIZONS_SLOTS:
            alpha_scores = []
            for alpha in ALPHA_GRID:
                inner_scores = []
                for week in train_weeks:
                    val_idx = train_idx[week_arr[train_idx] == week]
                    inner_train = train_idx[week_arr[train_idx] != week]
                    if len(val_idx) == 0 or len(inner_train) == 0:
                        continue
                    fitted = ridge_fit(X[inner_train], y_hor[h][inner_train], float(alpha))
                    p = ridge_predict(X[val_idx], fitted)
                    inner_scores.append(float(np.mean((p - y_hor[h][val_idx]) ** 2)))
                alpha_scores.append(float(np.mean(inner_scores)))
            best_alpha = float(ALPHA_GRID[int(np.argmin(alpha_scores))])
            chosen[h].append(best_alpha)
            fitted = ridge_fit(X[train_idx], y_hor[h][train_idx], best_alpha)
            pred[h][test_idx] = ridge_predict(X[test_idx], fitted)

    evaluated_idx = np.concatenate([
        np.asarray(f["test_idx"], dtype=int) for f in folds
    ])
    for h in HORIZONS_SLOTS:
        assert not np.isnan(pred[h][evaluated_idx]).any(), (features, h)
    return pred, chosen


def bootstrap_indices(
    week_arr: np.ndarray,
    weeks: list[str],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    by_week = {week: np.where(week_arr == week)[0] for week in weeks}
    samples = []
    for _ in range(BOOTSTRAPS):
        picked = rng.choice(weeks, size=len(weeks), replace=True)
        samples.append(np.concatenate([by_week[w] for w in picked]))
    return samples


def interval(values: np.ndarray) -> tuple[float, float]:
    return tuple(float(x) for x in np.percentile(values, [2.5, 97.5]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    df, drop = build_samples(conn)
    folds = make_folds(df)
    week_arr = df["week_id"].to_numpy()
    test_idx = np.concatenate([np.asarray(f["test_idx"], dtype=int) for f in folds])
    test_weeks = [str(f["test_week"]) for f in folds]
    assert len(np.unique(test_idx)) == len(test_idx)

    preds: dict[str, dict[int, np.ndarray]] = {}
    chosen: dict[str, dict[int, list[float]]] = {}
    for arm in ("A", "B", "C"):
        preds[arm], chosen[arm] = fit_arm(df, MODEL_FEATURES[arm], folds)

    y_hor = {
        h: np.array([float(v[slot]) for v in df["tgt_vals"]], dtype=float)
        for h, slot in HORIZONS_SLOTS.items()
    }
    rng = np.random.default_rng(SEED)
    boot_idx = bootstrap_indices(week_arr[test_idx], test_weeks, rng)
    results: dict[str, object] = {
        "exploratory": True,
        "minimum_training_weeks": MIN_TRAIN_WEEKS,
        "test_weeks": test_weeks,
        "test_events": int(len(test_idx)),
        "training_weeks_per_fold": [len(f["train_weeks"]) for f in folds],
        "drops": drop,
        "features": MODEL_FEATURES,
        "bolus_sources": list(RECONCILED_BOLUS_SOURCES),
        "horizons_min": list(HORIZONS_SLOTS),
        "metrics": {},
    }

    print("context comparison (exploratory: prior agents used this evaluation period)")
    print("  slope: six 5-minute slots, 282..287; units mg/dL per minute")
    print("  prior bolus window: strict t0-4h < ts < t0")
    print("  automatic-insulin aggregates: excluded")
    print("  model family: standardised ridge; alpha grid size: %d" % len(ALPHA_GRID))
    print("  outer split: forward weeks, minimum training prefix: %d weeks" % MIN_TRAIN_WEEKS)
    print("  uncertainty: %d cluster bootstrap resamples over test weeks, seed %d" % (BOOTSTRAPS, SEED))
    print("\ncounts")
    for key, value in drop.items():
        print(f"  {key}: {value}")
    print(f"  retained events: {len(df)}")
    print(f"  forward test weeks: {len(test_weeks)} ({', '.join(test_weeks)})")
    print(f"  forward test events: {len(test_idx)}")

    header = (
        "\nhorizon | A MAE [95% CI] | B MAE [95% CI] | C MAE [95% CI] "
        "| delta B-A [95% CI] | delta C-B [95% CI]"
    )
    print(header)
    print("-" * len(header))
    for h in HORIZONS_SLOTS:
        errors = {
            arm: np.abs(preds[arm][h][test_idx] - y_hor[h][test_idx])
            for arm in ("A", "B", "C")
        }
        deltas = {
            "B-A": errors["B"] - errors["A"],
            "C-B": errors["C"] - errors["B"],
        }
        metrics = {}
        parts = []
        for arm in ("A", "B", "C"):
            boots = np.array([errors[arm][idx].mean() for idx in boot_idx])
            lo, hi = interval(boots)
            metrics[arm] = {"mae": float(errors[arm].mean()), "ci95": [lo, hi]}
            parts.append(f"{errors[arm].mean():6.2f} [{lo:6.2f},{hi:6.2f}]")
        for name in ("B-A", "C-B"):
            boots = np.array([deltas[name][idx].mean() for idx in boot_idx])
            lo, hi = interval(boots)
            metrics[name] = {"delta_mae": float(deltas[name].mean()), "ci95": [lo, hi]}
            parts.append(f"{deltas[name].mean():6.2f} [{lo:6.2f},{hi:6.2f}]")
        results["metrics"][str(h)] = metrics
        print(f"{h:>7} | " + " | ".join(parts))

    print("\nselected alpha values by arm and horizon (one per forward test week):")
    for arm in ("A", "B", "C"):
        for h in HORIZONS_SLOTS:
            values = ", ".join(f"{x:g}" for x in chosen[arm][h])
            print(f"  {arm} {h:>3}m: {values}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n")
        print("\naggregate JSON written:", out)


if __name__ == "__main__":
    main()
