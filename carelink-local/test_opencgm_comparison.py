"""Scientific-contract tests, with synthetic inputs only."""
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np

import opencgm_comparison as comparison
from clock_audit import audit, private_path
from prepare_opencgm import SOURCE_DIR

sys.path.insert(0, str(SOURCE_DIR / "src"))
from opencgm_stateevent.data.grid import build_window


class ComparisonChecks(unittest.TestCase):
    def test_grid_matches_pinned_source_and_excludes_future(self):
        start = datetime(2024, 1, 2, 4, 0)
        origin = comparison.seconds(start)
        # Include exact half-bin ties, duplicate bins, missing bins, the final
        # out-of-range rounded position and a value exactly at issue time.
        offsets = np.array([-1, 0, 149, 150, 301, 3599, 86400-151, 86400-150, 86400])
        times = origin+offsets
        values = np.arange(len(times), dtype=float)*10+90
        actual, mask = comparison.grid(times, values, origin, 288, origin, origin+86400)
        reference = build_window([start+timedelta(seconds=int(t)) for t in offsets], list(values), start)
        np.testing.assert_array_equal(actual, reference.values)
        np.testing.assert_array_equal(mask, reference.mask)
        changed = values.copy()
        changed[-1] = 1e9
        future, _ = comparison.grid(times, changed, origin, 288, origin, origin+86400)
        np.testing.assert_array_equal(actual, future)
        self.assertEqual(actual[0], 105)  # average offsets 0 and 149 only
        self.assertEqual(actual[1], 125)  # 150 tie rounds up; 301 joins it

    def test_target_tolerance_is_half_open_and_nonoverlapping(self):
        t = np.array([149, 150, 449, 450, 7349, 7350])
        values = np.array([999, 100, 120, 200, 300, 999])
        y,m = comparison.grid(t, values, 300, 24, 150, 7350)
        self.assertEqual(y[0], 110)
        self.assertEqual(y[1], 200)
        self.assertEqual(y[-1], 300)
        self.assertEqual(m.sum(), 3)

    def test_clock_day_excludes_crossing_support(self):
        blocked = {datetime(2024, 1, 2).date()}
        begin = comparison.seconds(datetime(2024, 1, 2))
        self.assertFalse(comparison.clock_overlap(begin-86400, begin, blocked))
        self.assertTrue(comparison.clock_overlap(begin-86400, begin+1, blocked))
        self.assertTrue(comparison.clock_overlap(begin+100, begin+86401, blocked))
        self.assertFalse(comparison.clock_overlap(begin+86400, begin+90000, blocked))
        t = datetime(2024, 1, 2, 2)
        rows = [{"block": 1, "value": v, "record": {"Index": str(i)}} for i,v in enumerate([6., 7.])]
        report, evidence, days = audit({t: rows}, [], [])
        self.assertEqual(report["conflicting_timestamps"], 1)
        self.assertEqual(days, blocked)
        self.assertEqual(len(evidence["conflicts"][str(t)]), 2)

    def test_split_support_purge(self):
        t = np.arange(comparison.seconds(datetime(2024, 12, 1)), comparison.seconds(datetime(2026, 9, 1)), 7200)
        splits = comparison.split_masks(t)
        for a,b in [("development_train", "validation"), ("final_train", "exploratory"), ("exploratory", "later")]:
            self.assertLess((t[splits[a]]+7350).max(), (t[splits[b]]-86400).min())
            self.assertFalse((splits[a] & splits[b]).any())

    def test_scaling_and_fit_never_see_validation_or_test_as_training(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(90, 8))
        x[:, 0] = np.arange(90)  # identify exact rows passed to the fitted scaler
        y = rng.normal(size=(90, 4))
        idx = np.arange(90)
        splits = {"development_train": idx < 30, "validation": (idx >= 30) & (idx < 50),
                  "final_train": idx < 50, "exploratory": (idx >= 50) & (idx < 70), "later": idx >= 70}
        original = comparison.ridge_fit
        fitted_rows = []

        def checked_fit(features, target, alpha):
            ids = features[:, 0].astype(int)
            self.assertTrue(np.array_equal(ids, np.arange(30)) or np.array_equal(ids, np.arange(50)))
            fitted = original(features, target, alpha)
            np.testing.assert_allclose(fitted[0], x[ids].mean(axis=0))
            np.testing.assert_allclose(fitted[1], x[ids].std(axis=0))
            fitted_rows.append(len(ids))
            return fitted

        with patch.object(comparison, "ridge_fit", checked_fit):
            pred, alpha, _ = comparison.tune_and_predict(x, y, splits)
        changed_y = y.copy()
        changed_y[idx >= 50] = 1e12
        pred2, alpha2, _ = comparison.tune_and_predict(x, changed_y, splits)
        np.testing.assert_array_equal(pred[idx >= 50], pred2[idx >= 50])
        self.assertEqual(alpha, alpha2)
        self.assertEqual(fitted_rows.count(30), 19*4)
        self.assertEqual(fitted_rows.count(50), 4)

    def test_paired_week_bootstrap_matches_explicit_row_resampling(self):
        # Unequal week sizes distinguish event weighting from mean-week weighting.
        base = comparison.seconds(datetime(2026, 7, 6))
        t = base+np.array([0, 7200, 604800, 2*604800, 2*604800+7200])
        y = np.zeros((5, 4))
        error = np.array([1, 3, 8, 2, 4])
        p = np.repeat(error[:, None], 4, axis=1)
        result = comparison.metrics(t, y, {"summary": p, "augmented": p, "persistence": p+5}, np.ones(5, dtype=bool))
        row = result["metrics"]["60"]
        self.assertEqual(row["augmented-summary"], {"delta_mae": 0.0, "ci95": [0.0, 0.0]})
        draws = np.random.default_rng(comparison.SEED).integers(0, 3, size=(comparison.BOOTSTRAPS, 3))
        groups = [np.array([0, 1]), np.array([2]), np.array([3, 4])]
        explicit = [error[np.concatenate([groups[i] for i in row])].mean() for row in draws]
        np.testing.assert_allclose(row["summary"]["ci95"], np.percentile(explicit, [2.5, 97.5]))
        self.assertEqual(row["summary-persistence"]["delta_mae"], -5)

    def test_private_output_guard(self):
        with self.assertRaises(ValueError):
            private_path(comparison.ROOT / "assets/article/raw-predictions.npz")


if __name__ == "__main__":
    unittest.main(verbosity=2)
