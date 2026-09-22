"""Independently check saved inputs, matched predictions and reported uncertainty.

No fitting, protocol changes or private-value printing. Source-grid checks use
the revision-pinned upstream implementation, rather than our vectorized grid.
"""
import hashlib
import json
import sys
from datetime import datetime, timedelta

import numpy as np

from clock_audit import audit, load_archive, sha256
from opencgm_comparison import EPOCH, HORIZONS, OUT, PRIVATE, ROOT, WORK, code_hashes, fingerprint, split_masks
from prepare_opencgm import ENCODER_SHA, MODEL_DIR, SOURCE_DIR

sys.path.insert(0, str(SOURCE_DIR / "src"))
from opencgm_stateevent.data.grid import build_window


def main():
    result = json.loads((OUT / "opencgm-results.json").read_text())
    build = json.loads((OUT / "opencgm-build.json").read_text())
    assert result["build_manifest_sha256"] == sha256(OUT / "opencgm-build.json")
    assert result["code_sha256"] == code_hashes() == build["code_sha256"]
    assert result["protocol_sha256"] == build["protocol_sha256"] == fingerprint()
    assert hashlib.sha256((WORK / "frozen-protocol.txt").read_bytes()).hexdigest() == fingerprint()
    assert sha256(MODEL_DIR / "glucofm_encoder.onnx") == ENCODER_SHA
    assert sha256(SOURCE_DIR / "web/public/models/encoder.onnx") == ENCODER_SHA
    assert sha256(WORK / "windows.npz") == build["private_windows_sha256"]
    archive, clocks, _, files = load_archive(sorted((PRIVATE / "carelink").glob("*.csv")))
    assert files == build["input_files"]
    # Also check against the inventory produced before this assignment.
    inventory = json.loads((OUT / "archive-inventory.json").read_text())
    assert sorted(f["sha256"] for f in files) == sorted(f["sha256"] for f in inventory["files"])
    _, _, blocked = audit(archive, clocks, files)
    with np.load(WORK / "windows.npz", allow_pickle=False) as data:
        v, m, y, issue, x = [data[k] for k in ["values", "mask", "target", "issue", "summaries"]]
    assert v.shape == m.shape == (build["counts"]["eligible_windows"], 288)
    splits = split_masks(issue)
    assert {k: int(s.sum()) for k,s in splits.items()} == build["split_counts"]
    # Calendar-day intersection is checked independently for every saved window.
    for t in issue:
        start = EPOCH+timedelta(seconds=int(t)-86400)
        end = EPOCH+timedelta(seconds=int(t)+7350)
        assert not any(start < datetime.combine(day+timedelta(days=1), datetime.min.time()) and
                       end > datetime.combine(day, datetime.min.time()) for day in blocked)
    rng = np.random.default_rng(911)
    checked = []
    # Stratify across all four chronological populations, not just early training.
    for key in ["development_train", "validation", "exploratory", "later"]:
        checked.extend(rng.choice(np.flatnonzero(splits[key]), 16, replace=False))
    times = sorted(archive)
    local_seconds = np.array([int((t-EPOCH).total_seconds()) for t in times])
    raw = np.array([archive[t][0]["value"]*18.0182 for t in times])
    for i in checked:
        start = EPOCH+timedelta(seconds=int(issue[i])-86400)
        lo, hi = np.searchsorted(local_seconds, [issue[i]-86400, issue[i]])
        window = build_window(times[lo:hi], list(raw[lo:hi]), start)
        np.testing.assert_array_equal(window.values, v[i])
        np.testing.assert_array_equal(window.mask, m[i])
        for j,h in enumerate(HORIZONS):
            ref = issue[i]+60*h
            lo,hi = np.searchsorted(local_seconds, [ref-150, ref+150])
            assert hi > lo
            # Targets were deliberately stored on the float32 observation grid.
            assert y[i,j] == np.float32(raw[lo:hi].mean())
    del archive
    with np.load(WORK / "predictions.npz", allow_pickle=False) as predictions:
        test = splits["exploratory"] | splits["later"]
        np.testing.assert_array_equal(predictions["issue"], issue[test])
        np.testing.assert_array_equal(predictions["target"], y[test])
        np.testing.assert_array_equal(predictions["later"], splits["later"][test])
        np.testing.assert_array_equal(predictions["persistence"], np.repeat(x[test, :1], 4, axis=1))
        for slice_name in ["exploratory", "later"]:
            selected = predictions["later"] if slice_name == "later" else ~predictions["later"]
            stamps = predictions["issue"][selected]
            weeks = [(EPOCH+timedelta(seconds=int(t))).isocalendar()[:2] for t in stamps]
            unique = sorted(set(weeks))
            groups = [np.array([i for i,w in enumerate(weeks) if w == week]) for week in unique]
            arms = ["persistence", "summary", "augmented"]
            errors = np.stack([np.abs(predictions[a][selected]-predictions["target"][selected]) for a in arms], axis=-1)
            assert np.isfinite(errors).all()
            draws = np.random.default_rng(42).integers(0, len(unique), size=(2000, len(unique)))
            bootstrap = np.array([errors[np.concatenate([groups[i] for i in row])].mean(axis=0) for row in draws])
            summary = result[slice_name]
            assert summary["events"] == len(stamps) and summary["weeks"] == len(unique)
            for j,h in enumerate(HORIZONS):
                metrics = summary["metrics"][str(h)]
                for a,arm in enumerate(arms):
                    np.testing.assert_allclose(metrics[arm]["mae"], errors[:,j,a].mean(), atol=1e-10, rtol=0)
                    np.testing.assert_allclose(metrics[arm]["ci95"], np.percentile(bootstrap[:,j,a], [2.5,97.5]), atol=1e-10, rtol=0)
                for a,b in [(2,1), (1,0), (2,0)]:
                    entry = metrics[f"{arms[a]}-{arms[b]}"]
                    np.testing.assert_allclose(entry["delta_mae"], (errors[:,j,a]-errors[:,j,b]).mean(), atol=1e-10, rtol=0)
                    np.testing.assert_allclose(entry["ci95"], np.percentile(bootstrap[:,j,a]-bootstrap[:,j,b], [2.5,97.5]), atol=1e-10, rtol=0)
    assert "private-data/" in (ROOT / ".gitignore").read_text().splitlines()
    checks = {"status": "passed", "verification_code_sha256": sha256(__file__),
              "results_sha256": sha256(OUT / "opencgm-results.json"),
              "test_code_sha256": sha256(ROOT / "carelink-local/test_opencgm_comparison.py"),
              "checks": ["all 15 CSV hashes match pre-assignment inventory",
                         "protocol, code, model and window hashes match",
                         "GitHub source-checkout ONNX is byte-identical to Hugging Face download",
                         "all saved windows exclude all clock-ambiguity days",
                         "64 stratified private histories match pinned upstream grid exactly",
                         "256 private target values match independent observed-bin means",
                         "saved prediction identities, targets and persistence match input windows",
                         "all reported MAEs, paired differences and 95% intervals match explicit-row week bootstrap",
                         "root private-data ignore rule present"],
              "unused_boundary_windows": int((~(splits["final_train"] | splits["exploratory"] | splits["later"])).sum())}
    (OUT / "opencgm-checks.json").write_text(json.dumps(checks, indent=2)+"\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
