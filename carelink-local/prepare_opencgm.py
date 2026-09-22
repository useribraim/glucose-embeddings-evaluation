"""Fetch public, revision-pinned encoder artifacts; record hashes and source provenance."""
from pathlib import Path
import json
import subprocess
import urllib.request

from clock_audit import ROOT, PRIVATE, sha256

HF_REV = "6c53e572c2dafcb0b030f2e1aa4f3a2a8d7f7ca3"
SOURCE_REV = "bcf8ac2dfca00f8fb3c7728fb2a3a2bd2803bebf"
ENCODER_SHA = "b1349deffd15ab62a5a98d7c7c4a7e143bcf1dee7ad745b1e05533ff54f34768"
MODEL_DIR = PRIVATE / "models/opencgm-stateevent"
SOURCE_DIR = PRIVATE / "model-source/opencgm"
SOURCE_FILES = [
    "scripts/export_encoder_onnx.py", "src/opencgm_stateevent/model/encoder.py",
    "src/opencgm_stateevent/model/causal_gaussian.py", "src/opencgm_stateevent/model/reference.py",
    "src/opencgm_stateevent/model/statistics.py", "src/opencgm_stateevent/model/stream_embedder.py",
    "src/opencgm_stateevent/data/grid.py", "src/opencgm_stateevent/data/timestamps.py",
    "src/opencgm_stateevent/eval/ppgr.py", "manifests/sources/registry.yaml",
]


def main():
    if not SOURCE_DIR.exists():
        SOURCE_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/sfourdrinier/opencgm.git", str(SOURCE_DIR)], check=True)
        subprocess.run(["git", "-C", str(SOURCE_DIR), "checkout", "--detach", SOURCE_REV], check=True)
    head = subprocess.check_output(["git", "-C", str(SOURCE_DIR), "rev-parse", "HEAD"], text=True).strip()
    if head != SOURCE_REV:
        raise ValueError("Source checkout differs from pinned revision; preserve and investigate")
    if subprocess.check_output(["git", "-C", str(SOURCE_DIR), "status", "--porcelain", "--untracked-files=no"], text=True):
        raise ValueError("Source tracked files changed")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ["README.md", "glucofm_encoder.onnx.meta.json", "glucofm_encoder.onnx", "LICENSE-WEIGHTS", "NOTICE", "CITATION.cff"]:
        url = f"https://huggingface.co/sfourdrinier/opencgm-stateevent/resolve/{HF_REV}/{name}"
        dest = MODEL_DIR / name
        if not dest.exists():
            with urllib.request.urlopen(url, timeout=120) as response:
                content = response.read()
            dest.write_bytes(content)
        files[name] = {"url": url, "sha256": sha256(dest), "bytes": dest.stat().st_size}
    meta = json.loads((MODEL_DIR / "glucofm_encoder.onnx.meta.json").read_text())
    assert files["glucofm_encoder.onnx"]["sha256"] == meta["sha256"] == ENCODER_SHA
    assert files["glucofm_encoder.onnx"]["bytes"] == meta["size_bytes"] == 1991782
    manifest = {"status": "measured file identity; training history is maintainer-reported, not independently reproduced",
                "model": "OpenCGM-StateEvent community reconstruction; not Google's GlucoFM",
                "hf_revision": HF_REV, "source_revision": SOURCE_REV,
                "source_url": "https://github.com/sfourdrinier/opencgm", "files": files,
                "source_files": {p: sha256(SOURCE_DIR / p) for p in SOURCE_FILES}, "metadata": meta}
    out = ROOT / "assets/article/opencgm-provenance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Verified pinned encoder: {ENCODER_SHA}; {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
