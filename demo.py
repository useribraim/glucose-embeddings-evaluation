#!/usr/bin/env python3
"""One command: install the pinned runtime, then verify the public CPU demo."""
from pathlib import Path
import argparse
import hashlib
import os
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use an already-installed environment and verified download cache")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 14):
        raise SystemExit("Use Python 3.14: python3.14 demo.py (the recorded evaluation used Python 3.14.3).")
    if sys.flags.optimize or os.environ.get("PYTHONOPTIMIZE"):
        raise SystemExit("Run without -O/PYTHONOPTIMIZE: the frozen scientific code uses assertions.")
    subprocess.run([sys.executable, str(ROOT / "scripts/check_release.py")], check=True, cwd=ROOT)
    environment = ROOT / ".venv/demo"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    requirements = ROOT / "carelink-local/requirements-opencgm.txt"
    requirement_hash = hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker = environment / "requirements.sha256"
    installed = python.exists() and marker.exists() and marker.read_text().strip() == requirement_hash
    if not installed:
        if args.offline:
            raise SystemExit("Offline mode needs one successful online run first.")
        if not python.exists():
            print("Creating the isolated demo environment...", flush=True)
            venv.EnvBuilder(with_pip=True).create(environment)
        subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirements)], check=True)
        marker.write_text(requirement_hash+"\n")
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    command = [str(python), str(ROOT / "scripts/synthetic_demo.py")]
    if args.offline:
        command.append("--offline")
    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
