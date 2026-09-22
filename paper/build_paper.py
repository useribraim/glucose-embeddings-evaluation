"""Build the checkpoint preprint from ARTICLE.md and public aggregates only.

Requires matplotlib, Pandoc 3.11 and Tectonic 0.17.0 (or compatible versions).
No experiment imports, private files, model fitting or network inference.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
ASSETS = ROOT / "assets/article"
BUILD = PAPER / "build"
HORIZONS = [30, 60, 90, 120]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed(x):
    return f"{x:+.2f}".replace("-", "−")


def validate_manuscript(article, results):
    """Catch transcription drift in all eight main result rows before rendering."""
    for key, label in [("later", "Later"), ("exploratory", "Exploratory")]:
        for h in HORIZONS:
            row = results[key]["metrics"][str(h)]
            delta = row["augmented-summary"]
            expected = (f"| {label} | {h} min | {row['persistence']['mae']:.2f} | "
                        f"{row['summary']['mae']:.2f} | {row['augmented']['mae']:.2f} | "
                        f"{signed(delta['delta_mae'])} [{signed(delta['ci95'][0])}, {signed(delta['ci95'][1])}] |")
            if expected not in article:
                raise ValueError(f"Manuscript result row differs from measured aggregate: {label}, {h}min")
    remaining = results["counts"]["candidate_issue_times"]
    for label, key in [("Incomplete archive-boundary support", "archive_boundary"),
                       ("Clock-ambiguity support overlap", "clock_support_overlap"),
                       ("History gap longer than 60 minutes", "history_gap_over_60min"),
                       ("History coverage or final-bin failure", "history_coverage_or_last_bin"),
                       ("Missing future target bins", "missing_target_bins")]:
        removed = results["counts"][key]
        remaining -= removed
        assert f"| {label} | {removed:,} | {remaining:,} |" in article
    assert remaining == results["counts"]["eligible_windows"]
    for key in ["encoder_sha256", "protocol_sha256", "source_revision", "hf_revision"]:
        assert results[key] in article
    protocol = (ROOT / "TANDEM.md").read_text().split("### Frozen protocol BEGIN\n", 1)[1].split("### Frozen protocol END", 1)[0]
    assert hashlib.sha256(protocol.encode()).hexdigest() == results["protocol_sha256"]
    for filename in ["opencgm-build.json", "opencgm-smoke.json"]:
        assert json.loads((ASSETS / filename).read_text())["protocol_sha256"] == results["protocol_sha256"]
    checks = json.loads((ASSETS / "opencgm-checks.json").read_text())
    assert checks["results_sha256"] == digest(ASSETS / "opencgm-results.json")


def draw_figure(results):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "pdf.fonttype": 42,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=.105, right=.975, top=.76, bottom=.23, wspace=.22)
    for ax, key, title, color, marker in [(axes[0], "later", "Later retrospective test", "#245B78", "o"),
                                         (axes[1], "exploratory", "Exploratory test", "#665C51", "s")]:
        metrics = [results[key]["metrics"][str(h)]["augmented-summary"] for h in HORIZONS]
        points = np.array([m["delta_mae"] for m in metrics])
        lo = np.array([m["ci95"][0] for m in metrics])
        hi = np.array([m["ci95"][1] for m in metrics])
        ax.axvline(0, color="#667078", linestyle="--", linewidth=.8, zorder=1)
        ax.errorbar(points, np.arange(4), xerr=[points-lo, hi-points], fmt=marker,
                    color=color, markersize=5, linewidth=1.4, capsize=3, zorder=3)
        ax.set_title(f"{title}\n{results[key]['events']:,} windows · {results[key]['weeks']} weeks",
                     loc="left", fontsize=9.5, pad=13, linespacing=1.6)
        ax.set_yticks(np.arange(4), [f"{h} min" for h in HORIZONS])
        ax.set_ylim(3.5, -.5)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xticks([-1, -.5, 0, .5, 1])
        ax.grid(axis="y", color="#E8ECEF", linewidth=.6)
        ax.tick_params(axis="y", length=0, pad=7)
        ax.spines["bottom"].set_color("#C4C9CC")
        ax.set_xlabel("Augmented − summary MAE (mg/dL)", labelpad=8)
    axes[0].set_ylabel("Forecast horizon", labelpad=10)
    fig.text(.52, .028, "Negative favors embeddings; positive favors summaries alone.", ha="center", fontsize=8.5)
    (PAPER / "figures").mkdir(exist_ok=True)
    fig.savefig(ASSETS / "opencgm-comparison.png", dpi=220)
    fig.savefig(PAPER / "figures/opencgm-comparison.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def executable(name, supplied):
    local = ROOT / ".venv/paper-tools/bin" / name
    path = supplied or shutil.which(name) or (str(local) if local.exists() else None)
    if path is None:
        raise RuntimeError(f"Install {name} or pass --{name} /path/to/{name}")
    return str(Path(path).resolve())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pandoc")
    ap.add_argument("--tectonic")
    args = ap.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    article = (ROOT / "ARTICLE.md").read_text()
    results = json.loads((ASSETS / "opencgm-results.json").read_text())
    validate_manuscript(article, results)
    draw_figure(results)
    head, rest = article.split("## Abstract\n", 1)
    abstract, body = rest.split("## 1. Introduction\n", 1)
    title = head.splitlines()[0].removeprefix("# ")
    subtitle = re.search(r"^\*\*(.+)\*\*$", head, re.M).group(1)
    author, status, date = re.search(r"^\*([^*]+)\*$", head, re.M).group(1).split(" · ")
    assert status == "Working preprint"
    meta = {"title": title, "subtitle": subtitle, "author": author, "date": date,
            "abstract": abstract.strip()}
    (BUILD / "metadata.json").write_text(json.dumps(meta))
    body = "## 1. Introduction\n"+body
    body = body.replace("(assets/article/opencgm-comparison.png)", "(figures/opencgm-comparison.pdf){width=100%}")
    (BUILD / "body.md").write_text(body)
    pandoc, tectonic = executable("pandoc", args.pandoc), executable("tectonic", args.tectonic)
    subprocess.run([pandoc, str(BUILD / "body.md"), "--from=markdown", "--to=latex", "--standalone",
                    "--shift-heading-level-by=-1", "--syntax-highlighting=none", "--wrap=auto",
                    f"--template={PAPER / 'template.tex'}", f"--metadata-file={BUILD / 'metadata.json'}",
                    "--output", str(PAPER / "main.tex")], check=True, cwd=PAPER)
    # Text minus must survive both XeTeX and pdfLaTeX; math-source minus is ASCII.
    tex = (PAPER / "main.tex").read_text().replace("−", r"\ensuremath{-}")
    # Long inline paths and hashes must wrap rather than run beyond the margin.
    # Braced array shapes are left alone; plain code identifiers need only the
    # underscore escape reversed for nolinkurl's literal-argument parser.
    tex = re.sub(r"\\texttt\{([^{}]{21,})\}",
                 lambda m: r"{\ttfamily\nolinkurl{"+m[1].replace(r"\_", "_")+"}}", tex)
    (PAPER / "main.tex").write_text(tex)
    env = dict(os.environ, SOURCE_DATE_EPOCH="1789948800")
    completed = subprocess.run([tectonic, "--keep-logs", "--outdir", str(BUILD), "main.tex"],
                               check=False, cwd=PAPER, env=env, text=True, capture_output=True)
    (BUILD / "compile-output.txt").write_text(completed.stdout+completed.stderr)
    messages = (completed.stdout+completed.stderr).splitlines()
    if completed.returncode:
        print("\n".join(messages[-35:]))
    else:
        print("\n".join(line for line in messages if "warning:" in line or "error:" in line))
    completed.check_returncode()
    shutil.copyfile(BUILD / "main.pdf", PAPER / "glucose-embeddings-preprint.pdf")
    # A self-contained typesetting source archive, with an explicit two-file allowlist.
    with tarfile.open(PAPER / "glucose-embeddings-source.tar.gz", "w:gz") as archive:
        for filename in ["main.tex", "figures/opencgm-comparison.pdf"]:
            archive.add(PAPER / filename, arcname=filename)
    generated = [PAPER / "main.tex", PAPER / "glucose-embeddings-preprint.pdf",
                 PAPER / "glucose-embeddings-source.tar.gz", PAPER / "figures/opencgm-comparison.pdf",
                 ASSETS / "opencgm-comparison.png"]
    manifest = {"status": "preprint draft built from measured aggregates; not submitted",
                "command": "carelink-local/.venv/bin/python paper/build_paper.py",
                "inputs_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                    [ROOT / "ARTICLE.md", ASSETS / "opencgm-results.json", PAPER / "template.tex", Path(__file__)]},
                "outputs_sha256": {str(p.relative_to(ROOT)): digest(p) for p in generated},
                "protocol_sha256": results["protocol_sha256"],
                "tools": {"pandoc": subprocess.check_output([pandoc, "--version"], text=True).splitlines()[0],
                          "tectonic": subprocess.check_output([tectonic, "--version"], text=True).strip(),
                          "matplotlib": importlib.metadata.version("matplotlib")},
                "checks": ["all eight result rows match measured JSON", "eligibility-flow arithmetic matches JSON",
                           "checkpoint/revision/protocol identities present", "frozen protocol unchanged",
                           "results match independently verified artifact hash", "TeX compiled successfully"],
                "private_data_read": False}
    (PAPER / "build-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("Built paper/glucose-embeddings-preprint.pdf and paper/glucose-embeddings-source.tar.gz")


if __name__ == "__main__":
    main()
