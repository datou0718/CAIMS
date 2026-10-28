"""DASS selection and LAST tie handling regressions."""
import contextlib
import io
import unittest
import numpy as np
from caims import dass, last, orca
from caims.selection import CalibrationResult


class DASSFallbackTests(unittest.TestCase):
    def test_unstable_fallback_works_without_flag_and_at_rank_boundary(self):
        with contextlib.redirect_stderr(io.StringIO()):
            pair = dass.select_parameters(
                np.zeros((4, 3), dtype=np.uint8), bits=3, array_width=3,
                k=1, n_smp=2, device="cpu",
            )
        self.assertFalse(pair.stable)
        self.assertEqual(pair.t_select, 1)
        self.assertEqual(pair.c, 3.0)

    def test_stable_candidate_is_preferred_over_closer_unstable_candidate(self):
        stable = CalibrationResult(-10, 8, 9.5, True)
        unstable = CalibrationResult(-20, 7, 10.0, False)
        self.assertEqual(
            dass.best_pair([unstable, stable], 10, 0.1), stable
        )

    def test_closest_unstable_candidate_is_used_if_none_are_stable(self):
        closest = CalibrationResult(-10, 8, 9.5, False)
        other = CalibrationResult(-20, 7, 12.0, False)
        self.assertEqual(
            dass.best_pair([other, closest], 10, 0.1), closest
        )

    def test_cam_still_returns_all_cutoff_ties(self):
        documents = np.zeros((4, 3), dtype=np.float32)
        with contextlib.redirect_stderr(io.StringIO()):
            results, _ = last.select_candidates(
                documents, documents[:1], 0.0, 3, 1, "cpu",
                query_batch_size=1, document_batch_size=2,
            )
        np.testing.assert_array_equal(results[0], np.arange(4))


class PaperSelectionTests(unittest.TestCase):
    def test_last_matches_direct_distances_with_partial_partition_and_self_exclusion(self):
        rng = np.random.default_rng(21)
        candidates = rng.integers(0, 8, size=(19, 7)).astype(np.float32)
        self_indices = np.array([0, 8, 18])
        queries = candidates[self_indices]
        t_filter, array_width, t_select = -30.0, 3, 4
        scores = sum(
            (-np.square(queries[:, None, start:start + array_width]
                        - candidates[None, :, start:start + array_width]).sum(axis=2)
             >= t_filter).astype(np.int64)
            for start in range(0, candidates.shape[1], array_width)
        )
        scores[np.arange(len(queries)), self_indices] = -1
        with contextlib.redirect_stderr(io.StringIO()):
            result, cutoff_counts = last.select_candidates(
                candidates, queries, t_filter, array_width, t_select, 'cpu',
                query_batch_size=2, document_batch_size=5, self_indices=self_indices,
            )
        for row, actual in enumerate(result):
            cutoff = np.sort(scores[row])[-t_select]
            np.testing.assert_array_equal(actual, np.flatnonzero(scores[row] >= cutoff))
        self.assertEqual(sum(cutoff_counts), len(queries))

    def test_orca_preserves_filter_and_reports_actual_tie_inclusive_count(self):
        candidates = np.zeros((8, 7), dtype=np.uint8)
        queries = candidates[:3]
        with contextlib.redirect_stderr(io.StringIO()):
            selected = orca.adapt_t_select(
                candidates, queries, t_filter=-10, array_width=3, k=2, device='cpu',
            )
            result, _ = last.select_candidates(
                candidates, queries, selected.t_filter, 3, selected.t_select, 'cpu',
            )
        self.assertEqual(selected.t_filter, -10)
        self.assertFalse(selected.stable)
        self.assertEqual(selected.c, float(np.mean([len(row) for row in result])))
        self.assertTrue(all(len(row) >= selected.t_select for row in result))
