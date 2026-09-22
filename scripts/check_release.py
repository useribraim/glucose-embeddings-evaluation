"""Verify the allowlisted public snapshot and the frozen evidence identities."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / "release-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = ROOT / name
        assert not path.is_symlink() and path.resolve().is_relative_to(ROOT.resolve()), name
        assert digest(path) == expected, f"Public release file changed: {name}"
    result = json.loads((ROOT / "assets/article/opencgm-results.json").read_text())
    for name, expected in result["code_sha256"].items():
        assert digest(ROOT / "carelink-local" / name) == expected, name
    checks = json.loads((ROOT / "assets/article/opencgm-checks.json").read_text())
    assert digest(ROOT / "assets/article/opencgm-results.json") == checks["results_sha256"]
    assert digest(ROOT / "carelink-local/test_opencgm_comparison.py") == checks["test_code_sha256"]
    protocol = (ROOT / "TANDEM.md").read_text().split("### Frozen protocol BEGIN\n", 1)[1].split("### Frozen protocol END", 1)[0]
    assert hashlib.sha256(protocol.encode()).hexdigest() == result["protocol_sha256"]
    if (ROOT / ".git").exists():
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
        allowed = set(manifest["files"]) | {"release-manifest.json"}
        assert set(filter(None, tracked)) <= allowed, "Unexpected tracked file outside publication allowlist"
    print(f"Public snapshot verified: {len(manifest['files'])} allowlisted files; measured code/results/protocol hashes unchanged.")


if __name__ == "__main__":
    main()
