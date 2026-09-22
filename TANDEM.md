# Frozen evaluation protocol

Recorded locally before private encoder output and comparison scores were examined.
This publication preserves the original protocol section byte-for-byte.
The completed results and public-demo instructions are linked from README.md.

### Frozen protocol BEGIN

1. **Model identity.** OpenCGM-StateEvent, an independent community reconstruction,
   not Google's GlucoFM. Hugging Face revision
   `6c53e572c2dafcb0b030f2e1aa4f3a2a8d7f7ca3`; corresponding source revision
   `bcf8ac2dfca00f8fb3c7728fb2a3a2bd2803bebf`. ONNX SHA-256
   `b1349deffd15ab62a5a98d7c7c4a7e143bcf1dee7ad745b1e05533ff54f34768`.
   The card reports seed 17, epoch 40, `raw_statistics=true`,
   `zero_empty_patches=false`; its five-seed results are not results for this file.
2. **Inputs and clock rule.** Use all 15 immutable CSVs in `private-data/carelink/`.
   Hash each input. Audit conflicting values before deduplication. Reject a window
   if its full support `[issue−24h, issue+122.5min)` intersects any conflict day or
   any day between a recorded clock change's old/new dates, inclusive. Equal
   timestamp/value duplicates can collapse. Never average conflicting timestamp
   values. No invented UTC ordering. Clock audit measured 24 conflicts on 2 days
   and 17 excluded calendar days; preserve raw records and private audit evidence.
3. **Forecast population.** Issue every two hours on the pump-local wall clock,
   starting at midnight. This changes the earlier meal-event estimand: all eligible
   times, without selection on subsequent meals. No carbs or insulin features.
   All automatic-insulin aggregates remain excluded. The same event identities,
   history, targets and splits serve all three arms and every horizon.
4. **Grid and eligibility.** Oldest-first history `[issue−24h, issue)`, 288 positions.
   Convert observed mmol/L to mg/dL with the existing `18.0182` factor. Use source
   `build_window` semantics: nearest index `floor(offset/300 + 0.5)`, mean readings
   within a bin, reject indices outside 0..287. Zero fill; mask 1 only for observed
   bins; no interpolation, clipping or external per-window normalization. Require
   ≥80% coverage over 24h and the last 3h, an observed last bin, and no raw history
   gap >60min (including history edges). Targets use mean observed glucose in
   disjoint half-open ±2.5min bins around issue+5,...,+120min. Require all 24 bins
   so missingness cannot change the population across horizons. Evaluate glucose
   at 30, 60, 90 and 120min; label tolerances explicitly, not as exact-time readings.
5. **Encoder contract.** ONNX inputs `values`, `mask`: float32 `[B,288]`;
   `circadian_start`: int64 `[B]`, `floor((60*start.hour+start.minute)/5)`.
   The graph performs observed-only instance normalization (epsilon 1e-6, population
   variance, scale floor 1e-4), causal filtering, raw-mg/dL patch statistics, and
   unweighted mean pooling of 24 online context tokens. Output float32 `[B,128]`.
   Use the released ONNX unchanged, CPU inference only. Synthetic verification
   precedes private inference; finite masked fill, sparse/empty/constant inputs,
   deterministic and batch-size invariance checks are required. No probe heads.
6. **Splits and leakage.** Development training: full target support before
   2025-01-01, starting at the first eligible 2023 window. Validation: full history
   on/after 2025-01-01 and full target support before 2026-01-01. After selecting
   hyperparameters, refit on every eligible sample whose full target support is
   before 2026-01-01. Freeze that fit for all test predictions. Exploratory test:
   full history on/after 2026-01-01 and full target support before 2026-07-01.
   Primary later test: full history on/after 2026-07-01 through available September
   data. Purging whole support prevents shared readings at split boundaries.
   The March–June 2026 period has previous forecast examination; report the entire
   January–June slice as exploratory. July–September has coverage/export inspection
   but no recorded prior forecast comparison; call it a later retrospective test,
   never a prospective/pristine holdout. No test-driven changes to features or tuning.
7. **Arms and tuning.** Persistence: last observed history bin. Summary Ridge:
   the existing eight features `last_g`, history mean/sample-SD/min/max,
   last-six-bin least-squares slope (mg/dL/min), issue-time sine/cosine.
   Augmented Ridge: those exact eight plus all 128 frozen embedding dimensions.
   Reuse `context_comparison.ridge_fit/ridge_predict`. Standardize columns with
   training-only mean/population SD; no PCA or learned target transformation.
   Independent alpha selection per arm/horizon from 19 log-spaced values 1e-3..1e6,
   minimum event-weighted validation MAE, smaller alpha on ties. Same 19 candidates
   and validation samples for each Ridge arm. Refit scaling on final training only.
8. **Outcomes and uncertainty.** Primary: later-test paired difference in 60min MAE,
   augmented minus summary Ridge, in mg/dL (negative favors embeddings). Report
   all four horizons, all three arm MAEs, augmented−summary and each Ridge−persistence
   difference, for both exploratory and later slices. Use 2,000 paired bootstrap
   resamples of ISO calendar weeks, seed 42, retaining all events in a sampled week;
   event-weighted estimates and percentile 95% intervals. Same draws for all arms
   and horizons within a slice. Intervals condition on the fitted models and do not
   capture retraining uncertainty or guarantee removal of inter-week dependence.
   Secondary horizons are descriptive, without multiplicity-adjusted claims.
9. **Evidence and privacy.** Save inputs, features, embeddings, targets, predictions,
   clock details and sample identities only under ignored `private-data/evaluation/`.
   Public outputs contain aggregate counts, metrics, selected alphas, code/input/model
   hashes, environment and meaningful checks; no per-day metrics or traces. Record
   the SHA-256 of this protocol section with outputs before scoring. Report losses
   as prominently as wins. Attribution is only to the pinned community checkpoint;
   forecast gains do not establish causal or physiological understanding.

### Frozen protocol END
