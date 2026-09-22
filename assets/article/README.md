# Measured evidence

The primary later-test 60-minute augmented-minus-summary MAE difference is
−0.30 mg/dL [−0.87, +0.34]. These files preserve the completed personal evaluation;
the public demo writes its new synthetic report under `.demo-output/`.

| Artifact | Evidence |
|---|---|
| [archive-inventory.json](archive-inventory.json) | Source coverage, counts, midnight aggregates and file fingerprints |
| [clock-audit.json](clock-audit.json) | Conflicting timestamps and full-day exclusion rule |
| [opencgm-provenance.json](opencgm-provenance.json) | Pinned checkpoint/source identities and input contract |
| [opencgm-smoke.json](opencgm-smoke.json) | Original pre-inference synthetic checks on the author's machine |
| [opencgm-build.json](opencgm-build.json) | Eligibility, splits, code/input hashes and private inference checks |
| [opencgm-results.json](opencgm-results.json) | All models/horizons, selected alphas and paired uncertainty |
| [opencgm-checks.json](opencgm-checks.json) | Independent verification against saved private predictions |
| [opencgm-comparison.png](opencgm-comparison.png) | Paired differences rendered from the measured results |

The exact protocol is in [TANDEM.md](../../TANDEM.md). Appendix B of the
[manuscript](../../ARTICLE.md) gives the private-input reproduction commands.
The public command `python3 demo.py` verifies synthetic inference and the seven
original scientific contracts. It does not reproduce personal forecast scores.

All personal result files and the analysis scripts match their recorded hashes.
Raw exports, individual windows, embeddings and predictions are absent from this
publication snapshot. The earlier exploratory meal experiment is documented in
the author's working archive and is not pooled with the current comparison.
