"""Run the original scientific checks with public source and synthetic inputs.

The adapter changes only public-download locations and the smoke-report output
sink. The frozen encoder, preprocessing, Ridge and uncertainty functions are
the same byte-identified files used in the measured personal evaluation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import contextlib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import unittest
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/opencgm"
OUTPUT = ROOT / ".demo-output"


def fetch_verified(entry, *, cache=CACHE, offline=False):
    destination = (cache / entry["path"]).resolve()
    if not destination.is_relative_to(cache.resolve()):
        raise ValueError("Download path escapes the public cache")
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Cached file hash mismatch: {entry['path']}")
        return destination
    if offline:
        raise FileNotFoundError(f"Offline cache miss: {entry['path']}")
    if not entry["url"].startswith("https://"):
        raise ValueError("Public downloads must use HTTPS")
    with urllib.request.urlopen(entry["url"], timeout=120) as response:
        content = response.read()
    if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
        raise ValueError(f"Downloaded file identity mismatch: {entry['path']}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+".partial")
    temporary.write_bytes(content)
    temporary.replace(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if sys.flags.optimize:
        raise SystemExit("Assertions must be enabled")
    for line in (ROOT / "carelink-local/requirements-opencgm.txt").read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            package, expected = line.split("==")
            actual = importlib.metadata.version(package)
            if actual != expected:
                raise RuntimeError(f"Runtime differs from pin: {package} {actual} != {expected}")
    lock = json.loads((ROOT / "demo-lock.json").read_text())
    print(f"Verifying {len(lock['files'])} revision-pinned public files...", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda entry: fetch_verified(entry, offline=args.offline), lock["files"]))
    sys.path.insert(0, str(ROOT / "carelink-local"))
    import prepare_opencgm
    prepare_opencgm.MODEL_DIR = CACHE / "model"
    prepare_opencgm.SOURCE_DIR = CACHE / "source"
    import opencgm_comparison as comparison

    provenance = json.loads((ROOT / "assets/article/opencgm-provenance.json").read_text())
    for name, expected in provenance["files"].items():
        assert comparison.sha256(prepare_opencgm.MODEL_DIR / name) == expected["sha256"]
    assert comparison.sha256(CACHE / "source/web/public/models/encoder.onnx") == comparison.ENCODER_SHA
    smoke_result = {}

    def save_synthetic_report(name, result):
        assert name == "opencgm-smoke.json"
        result["status"] = "passed public synthetic-only inference; no personal forecast scores reproduced"
        smoke_result.update(result)

    comparison.write_public = save_synthetic_report
    with contextlib.redirect_stdout(io.StringIO()):
        comparison.smoke()
    assert smoke_result["providers"] == ["CPUExecutionProvider"]
    scientific = unittest.defaultTestLoader.discover(str(ROOT / "carelink-local"), pattern="test_opencgm_comparison.py")
    integrity = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_public_demo.py", top_level_dir=str(ROOT))
    scientific_result = unittest.TextTestRunner(verbosity=2).run(scientific)
    integrity_result = unittest.TextTestRunner(verbosity=2).run(integrity)
    if not scientific_result.wasSuccessful() or scientific_result.testsRun != 7 or not integrity_result.wasSuccessful() or integrity_result.testsRun != 4:
        raise SystemExit("Public verification failed or did not run the expected tests")
    OUTPUT.mkdir(exist_ok=True)
    report = {"status": "passed", "private_data_used": False, "personal_scores_reproduced": False,
              "checkpoint": provenance["model"], "encoder_sha256": comparison.ENCODER_SHA,
              "hf_revision": comparison.HF_REV, "source_revision": comparison.SOURCE_REV,
              "public_download_files_verified": len(lock["files"]),
              "source_encoder_matches_hf_encoder": True,
              "scientific_contract_tests_passed": scientific_result.testsRun,
              "download_integrity_tests_passed": integrity_result.testsRun,
              "synthetic_inference": smoke_result}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print("\nPASS — frozen encoder on synthetic glucose, 7 scientific contract tests, 4 download-integrity tests.")
    print("CPU only; no CareLink inputs, personal embeddings or personal predictions used.")
    print("Report: .demo-output/report.json")


if __name__ == "__main__":
    main()
