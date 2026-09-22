"""Frozen, local comparison of summary Ridge and summary + OpenCGM-StateEvent.

Run smoke, build, then score. Public artifacts are aggregates only. The original
meal-event experiment is not reused as data; its checked Ridge implementation is.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import importlib.metadata
import json
import platform
import sys

import numpy as np
import onnx
import onnxruntime as ort

from clock_audit import ROOT, PRIVATE, audit, load_archive, private_path, sha256
from context_comparison import GLUCOSE_FEATURES, interval, ridge_fit, ridge_predict
from prepare_opencgm import ENCODER_SHA, HF_REV, MODEL_DIR, SOURCE_DIR, SOURCE_FILES, SOURCE_REV

OUT = ROOT / "assets/article"
WORK = PRIVATE / "evaluation"
EPOCH = datetime(1970, 1, 1)  # numerical origin for naive pump-local times, not UTC
HORIZONS = [30, 60, 90, 120]
ALPHAS = np.logspace(-3, 6, 19)
BOOTSTRAPS = 2000
SEED = 42


def seconds(t):
    return int((t-EPOCH).total_seconds())


def protocol():
    text = (ROOT / "TANDEM.md").read_text()
    return text.split("### Frozen protocol BEGIN\n", 1)[1].split("### Frozen protocol END", 1)[0]


def fingerprint():
    return hashlib.sha256(protocol().encode()).hexdigest()


def environment():
    return {"python": sys.version, "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ["numpy", "pandas", "onnx", "onnxruntime"]}}


def write_public(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def session():
    path = MODEL_DIR / "glucofm_encoder.onnx"
    if sha256(path) != ENCODER_SHA:
        raise ValueError("Encoder hash mismatch")
    provenance = json.loads((OUT / "opencgm-provenance.json").read_text())
    assert provenance["hf_revision"] == HF_REV and provenance["source_revision"] == SOURCE_REV
    assert provenance["source_files"] == {p: sha256(SOURCE_DIR / p) for p in SOURCE_FILES}
    onnx.checker.check_model(str(path), full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    result = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    assert [(x.name, x.type, x.shape) for x in result.get_inputs()] == [
        ("values", "tensor(float)", ["B", 288]), ("mask", "tensor(float)", ["B", 288]),
        ("circadian_start", "tensor(int64)", ["B"])]
    assert [(x.name, x.type, x.shape) for x in result.get_outputs()] == [
        ("embedding", "tensor(float)", ["B", 128])]
    return result


def encode(sess, values, mask, circadian, batch=64):
    assert values.dtype == mask.dtype == np.float32 and circadian.dtype == np.int64
    assert values.shape == mask.shape and values.shape[1] == 288
    assert np.isfinite(values).all() and np.isin(mask, [0, 1]).all()
    assert ((circadian >= 0) & (circadian < 288)).all()
    out = []
    for lo in range(0, len(values), batch):
        out.append(sess.run(["embedding"], {"values": values[lo:lo+batch],
                   "mask": mask[lo:lo+batch], "circadian_start": circadian[lo:lo+batch]})[0])
    result = np.concatenate(out)
    assert result.shape == (len(values), 128) and np.isfinite(result).all()
    return result


def smoke():
    sess = session()
    rng = np.random.default_rng(SEED)
    x = np.stack([np.full(288, 110), np.linspace(70, 230, 288),
                  130 + 25*np.sin(np.arange(288)/15), np.zeros(288),
                  rng.uniform(65, 250, 288)]).astype(np.float32)
    m = np.ones_like(x)
    m[2, 50:110] = 0  # an entire empty hour patch and a long hole
    m[3] = 0
    m[4] = (np.arange(288) % 3 == 0).astype(np.float32)
    x[m == 0] = 0
    circ = np.array([0, 72, 144, 216, 287], dtype=np.int64)
    z = encode(sess, x, m, circ)
    assert np.array_equal(z, encode(sess, x, m, circ))
    one = encode(sess, x, m, circ, batch=1)
    np.testing.assert_allclose(z, one, atol=1e-4, rtol=1e-5)
    altered = x.copy()
    altered[m == 0] = 999.0  # finite fill; NaN*0 is not a supported mask contract
    fill = encode(sess, altered, m, circ)
    np.testing.assert_allclose(z, fill, atol=1e-4, rtol=1e-5)
    assert not np.allclose(z[0], z[1]), "Encoder is insensitive to distinct observed input"
    result = {"status": "passed on synthetic input before private inference", "encoder_sha256": ENCODER_SHA,
              "protocol_sha256": fingerprint(), "code_sha256": sha256(__file__),
              "environment": environment(), "providers": sess.get_providers(),
              "checks": ["ONNX full graph check", "exact input/output contract", "constant/ramp/sinusoid/empty/sparse finite",
                         "repeat deterministic", "batch-size invariance", "masked finite-fill invariance",
                         "observed-input sensitivity"],
              "batch_max_abs_difference": float(np.max(np.abs(z-one))),
              "masked_fill_max_abs_difference": float(np.max(np.abs(z-fill)))}
    write_public("opencgm-smoke.json", result)
    print(json.dumps(result, indent=2))


def grid(times, values, origin, length, lower, upper):
    """Source NEAREST contract, vectorized. Physical input support is explicit."""
    lo, hi = np.searchsorted(times, [lower, upper])
    idx = np.floor((times[lo:hi]-origin)/300 + .5).astype(int)
    inside = (idx >= 0) & (idx < length)
    idx, vals = idx[inside], values[lo:hi][inside]
    count = np.bincount(idx, minlength=length)
    sums = np.bincount(idx, weights=vals, minlength=length)
    observed = count > 0
    out = np.zeros(length, dtype=np.float32)
    out[observed] = sums[observed]/count[observed]
    return out, observed.astype(np.float32)


def clock_overlap(start, end, blocked):
    # Dates whose [midnight, next midnight) intersects half-open support.
    first, last = (EPOCH+timedelta(seconds=int(start))).date(), (EPOCH+timedelta(seconds=int(end)-1)).date()
    return any(first+timedelta(days=i) in blocked for i in range((last-first).days+1))


def summary_features(values, mask, issue):
    observed = values[mask == 1].astype(float)
    tail = np.where(mask[-6:] == 1)[0]
    slope = float(np.polyfit(tail.astype(float)*5, values[-6:][tail], 1)[0]) if len(tail) >= 2 else 0.0
    angle = 2*np.pi*(issue % 86400)/86400
    return [values[-1], observed.mean(), observed.std(ddof=1), observed.min(), observed.max(),
            slope, np.sin(angle), np.cos(angle)]


def split_masks(issue):
    b25, b26, b07 = [seconds(datetime.fromisoformat(d)) for d in ["2025-01-01", "2026-01-01", "2026-07-01"]]
    start, end = issue-86400, issue+7350
    return {"development_train": end < b25,
            "validation": (start >= b25) & (end < b26),
            "final_train": end < b26,
            "exploratory": (start >= b26) & (end < b07),
            "later": start >= b07}


def code_hashes():
    return {p: sha256(ROOT / "carelink-local" / p) for p in
            ["archive_inventory.py", "clock_audit.py", "prepare_opencgm.py", "context_comparison.py", "opencgm_comparison.py"]}


def build():
    checked = json.loads((OUT / "opencgm-smoke.json").read_text())
    assert checked["encoder_sha256"] == ENCODER_SHA and checked["protocol_sha256"] == fingerprint()
    paths = sorted((PRIVATE / "carelink").glob("*.csv"))
    assert len(paths) == 15, "Changed input population needs an explicit protocol amendment"
    glucose, clocks, _, files = load_archive(paths)
    clock_summary, _, blocked = audit(glucose, clocks, files)
    assert files == json.loads((OUT / "clock-audit.json").read_text())["files"]
    ordered = sorted(t for t in glucose if t.date() not in blocked)
    assert all(len({r["value"] for r in glucose[t]}) == 1 for t in ordered)
    times = np.array([seconds(t) for t in ordered], dtype=np.int64)
    raw = np.array([glucose[t][0]["value"]*18.0182 for t in ordered], dtype=float)
    assert np.isfinite(raw).all() and (raw > 0).all() and (np.diff(times) > 0).all()
    del glucose
    first = seconds(datetime.combine(ordered[0].date(), datetime.min.time()))
    last = int(times[-1])
    counts = Counter()
    inputs, masks, ys, summaries, stamps, circadian = [], [], [], [], [], []
    for issue in range(first, last+1, 7200):
        counts["candidate_issue_times"] += 1
        start, end = issue-86400, issue+7350
        if start < times[0] or end > times[-1]:
            counts["archive_boundary"] += 1
            continue
        if clock_overlap(start, end, blocked):
            counts["clock_support_overlap"] += 1
            continue
        lo, hi = np.searchsorted(times, [start, issue])
        gap = np.diff(np.concatenate(([start], times[lo:hi], [issue]))).max()
        if gap > 3600:
            counts["history_gap_over_60min"] += 1
            continue
        v, m = grid(times, raw, start, 288, start, issue)
        if m.sum() < .8*288 or m[-36:].sum() < .8*36 or m[-1] != 1:
            counts["history_coverage_or_last_bin"] += 1
            continue
        target, observed = grid(times, raw, issue+300, 24, issue+150, end)
        if observed.sum() != 24:
            counts["missing_target_bins"] += 1
            continue
        inputs.append(v)
        masks.append(m)
        ys.append(target[np.array(HORIZONS)//5-1])
        summaries.append(summary_features(v, m, issue))
        stamps.append(issue)
        circadian.append((start % 86400)//300)
    v, m = np.asarray(inputs, dtype=np.float32), np.asarray(masks, dtype=np.float32)
    y, x = np.asarray(ys, dtype=float), np.asarray(summaries, dtype=float)
    issue, circ = np.asarray(stamps, dtype=np.int64), np.asarray(circadian, dtype=np.int64)
    assert len(issue) > 0 and (np.diff(issue) > 0).all()
    assert np.isfinite(x).all() and np.isfinite(y).all()
    sess = session()
    z = encode(sess, v, m, circ)
    # Test private batch invariance only on training windows, before any scores.
    subset = np.where(split_masks(issue)["development_train"])[0][:8]
    np.testing.assert_allclose(z[subset], encode(sess, v[subset], m[subset], circ[subset], batch=1), atol=1e-4, rtol=1e-5)
    path = private_path(WORK / "windows.npz")
    np.savez_compressed(path, values=v, mask=m, summaries=x, embedding=z,
                        target=y, issue=issue, circadian=circ)
    counts["eligible_windows"] = len(issue)
    splits = split_masks(issue)
    manifest = {"status": "private inference passed; no forecast comparison scored at build time",
                "protocol_sha256": fingerprint(), "encoder_sha256": ENCODER_SHA,
                "input_files": files, "code_sha256": code_hashes(),
                "private_windows_sha256": sha256(path), "environment": environment(),
                "counts": dict(counts), "split_counts": {k:int(s.sum()) for k,s in splits.items()},
                "excluded_clock_days": clock_summary["excluded_calendar_days"],
                "checks": ["all input file hashes unchanged during read", "conflicts excluded before binning",
                           "unique increasing local issue identities", "finite summaries/targets/embeddings",
                           "CPU-only inference", "private training-window batch invariance"],
                "embedding_dimensions": int(z.shape[1]), "minimum_history_observed_bins": int(m.sum(axis=1).min())}
    private_path(WORK / "frozen-protocol.txt").write_text(protocol())
    write_public("opencgm-build.json", manifest)
    print(json.dumps({k: manifest[k] for k in ["status", "counts", "split_counts", "checks"]}, indent=2))


def tune_and_predict(x, y, splits):
    train, val, final = [splits[k] for k in ["development_train", "validation", "final_train"]]
    test = splits["exploratory"] | splits["later"]
    assert all(s.sum() > 0 for s in [train, val, final, test])
    prediction = np.full_like(y, np.nan)
    alphas, val_mae = [], []
    for h in range(y.shape[1]):
        scores = []
        for alpha in ALPHAS:
            fitted = ridge_fit(x[train], y[train, h], float(alpha))
            scores.append(np.abs(ridge_predict(x[val], fitted)-y[val, h]).mean())
        best = int(np.argmin(scores))
        alphas.append(float(ALPHAS[best]))
        val_mae.append(float(scores[best]))
        fitted = ridge_fit(x[final], y[final, h], alphas[-1])
        prediction[test, h] = ridge_predict(x[test], fitted)
    assert np.isfinite(prediction[test]).all()
    return prediction, alphas, val_mae


def metrics(issue, target, predictions, selection):
    stamps = issue[selection]
    week = np.array([(EPOCH+timedelta(seconds=int(t))).date().isocalendar()[:2] for t in stamps])
    weeks, group = np.unique(week, axis=0, return_inverse=True)
    n = np.bincount(group)
    # Multinomial week multiplicities are equivalent to retaining all rows from
    # each resampled week. The same draws apply to all arms and horizons.
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(weeks), size=(BOOTSTRAPS, len(weeks)))
    multiplicity = np.stack([np.bincount(row, minlength=len(weeks)) for row in draws])
    denom = multiplicity @ n

    def summarize(error, delta=False):
        sums = np.bincount(group, weights=error, minlength=len(weeks))
        boots = (multiplicity @ sums)/denom
        return {"delta_mae" if delta else "mae": float(error.mean()), "ci95": list(interval(boots))}

    result = {"events": int(selection.sum()), "weeks": len(weeks), "metrics": {}}
    for j, horizon in enumerate(HORIZONS):
        errors = {arm: np.abs(pred[selection, j]-target[selection, j]) for arm,pred in predictions.items()}
        row = {arm: summarize(error) for arm,error in errors.items()}
        for a,b in [("augmented", "summary"), ("summary", "persistence"), ("augmented", "persistence")]:
            row[f"{a}-{b}"] = summarize(errors[a]-errors[b], delta=True)
        result["metrics"][str(horizon)] = row
    return result


def score():
    manifest = json.loads((OUT / "opencgm-build.json").read_text())
    path = WORK / "windows.npz"
    assert manifest["protocol_sha256"] == fingerprint()
    assert manifest["code_sha256"] == code_hashes(), "Code changed after building windows"
    assert manifest["private_windows_sha256"] == sha256(path)
    assert manifest["encoder_sha256"] == sha256(MODEL_DIR / "glucofm_encoder.onnx") == ENCODER_SHA
    assert [p["sha256"] for p in manifest["input_files"]] == [sha256(p) for p in sorted((PRIVATE / "carelink").glob("*.csv"))]
    with np.load(path, allow_pickle=False) as data:
        x, z, y, issue = [data[k] for k in ["summaries", "embedding", "target", "issue"]]
    splits = split_masks(issue)
    for left,right in [("development_train", "validation"), ("final_train", "exploratory"), ("exploratory", "later")]:
        assert (issue[splits[left]]+7350).max() < (issue[splits[right]]-86400).min()
    assert not (splits["final_train"] & (splits["exploratory"] | splits["later"])).any()
    preds = {"persistence": np.repeat(x[:, :1], len(HORIZONS), axis=1)}
    selected, validation = {}, {}
    for arm,features in [("summary", x), ("augmented", np.column_stack([x,z]))]:
        preds[arm], selected[arm], validation[arm] = tune_and_predict(features, y, splits)
    test = splits["exploratory"] | splits["later"]
    np.savez_compressed(private_path(WORK / "predictions.npz"), issue=issue[test], target=y[test],
                        later=splits["later"][test], **{arm:p[test] for arm,p in preds.items()})
    result = {"status": "measured retrospective single-person matched comparison",
              "model": "OpenCGM-StateEvent, seed 17 epoch 40; community reconstruction, not Google's GlucoFM",
              "encoder_sha256": ENCODER_SHA, "hf_revision": HF_REV, "source_revision": SOURCE_REV,
              "protocol_sha256": fingerprint(), "code_sha256": code_hashes(),
              "build_manifest_sha256": sha256(OUT / "opencgm-build.json"), "environment": environment(),
              "summary_features": GLUCOSE_FEATURES, "embedding_dimensions": z.shape[1],
              "horizons_min": HORIZONS, "alpha_grid": ALPHAS.tolist(), "selected_alphas": selected,
              "selected_validation_mae": validation, "counts": manifest["counts"], "split_counts": manifest["split_counts"],
              "primary_outcome": "later-test 60min augmented-minus-summary MAE; negative favors embeddings",
              "units": "mg/dL", "bootstrap": {"unit": "ISO week", "resamples": BOOTSTRAPS,
                    "seed": SEED, "paired": True, "estimand": "event-weighted MAE, conditional on fitted models"},
              "checks": ["frozen protocol/code/input/model/window hashes match", "full-support split purging",
                         "same test identity array for all arms/horizons", "finite matched predictions",
                         "training-only standardization and fitting; validation-only alpha selection"],
              "exploratory": metrics(issue, y, preds, splits["exploratory"]),
              "later": metrics(issue, y, preds, splits["later"])}
    write_public("opencgm-results.json", result)
    for name in ["later", "exploratory"]:
        section = result[name]
        print(f"{name}: {section['events']} windows, {section['weeks']} weeks; mg/dL")
        for h,row in section["metrics"].items():
            diff = row["augmented-summary"]
            print(f"{h:>3}m persistence={row['persistence']['mae']:.3f} summary={row['summary']['mae']:.3f} "
                  f"augmented={row['augmented']['mae']:.3f} delta={diff['delta_mae']:.3f} CI={diff['ci95']}")
    print("selected alphas", selected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["smoke", "build", "score"])
    args = parser.parse_args()
    {"smoke": smoke, "build": build, "score": score}[args.stage]()
