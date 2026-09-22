# Working preprint

- [PDF](glucose-embeddings-preprint.pdf): eight-page checkpoint paper.
- [Editable manuscript](../ARTICLE.md): canonical text.
- [LaTeX source](main.tex) and [self-contained source archive](glucose-embeddings-source.tar.gz).
- [Build fingerprints](build-manifest.json) and [layout/source verification](validation.json).

The preprint reports the completed personal comparison. Its primary 60-minute
MAE difference is −0.30 mg/dL [−0.87, +0.34]. It is a working preprint, not an
arXiv submission or accepted publication.

The PDF and vector figure can be rebuilt from the public aggregates alone. With
matplotlib 3.11.2, NumPy, Pandoc 3.11 and Tectonic 0.17.0 installed, run from the
repository root:

```sh
python3 paper/build_paper.py
```

Explicit compiler paths are supported with `--pandoc /path/to/pandoc` and
`--tectonic /path/to/tectonic`. After extracting the source archive, `tectonic
main.tex` compiles the standalone paper. Model inference is unnecessary for paper
rendering. Full forecast-score replication requires the author's private inputs.
