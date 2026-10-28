"""Regression checks for IVF index ownership and missing retrieval neighbors."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import faiss
import numpy as np
from caims import baselines, metrics


class IVFRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.cache_patch = mock.patch.multiple(
            baselines, _IVF_CACHE_KEY=None, _IVF_CACHE_INDEX=None,
            IVF_NLIST=1, IVF_NPROBE=1,
        )
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)
        rng = np.random.default_rng(42)
        self.documents = rng.normal(size=(80, 4)).astype(np.float32)
        self.documents /= np.linalg.norm(self.documents, axis=1, keepdims=True)
        self.query = self.documents[:1].copy()

    def test_equal_shaped_corpora_without_paths_do_not_share_an_index(self):
        first = baselines.ivf_search(self.documents, self.query, 1)
        second = baselines.ivf_search(self.documents[::-1].copy(), self.query, 1)
        self.assertEqual(first, [[0]])
        self.assertEqual(second, [[79]])

    def test_modified_array_without_path_does_not_reuse_old_index(self):
        self.assertEqual(baselines.ivf_search(self.documents, self.query, 1), [[0]])
        self.documents[:] = self.documents[::-1].copy()
        self.assertEqual(baselines.ivf_search(self.documents, self.query, 1), [[79]])

    def test_saved_index_can_still_be_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "index.faiss"
            with contextlib.redirect_stdout(io.StringIO()):
                first = baselines.ivf_search(self.documents, self.query, 1, path)
            self.assertTrue(path.exists())
            with mock.patch.object(faiss, "read_index", side_effect=AssertionError("unnecessary reload")):
                second = baselines.ivf_search(self.documents, self.query, 1, path)
            self.assertEqual(first, second)

    def test_incomplete_ivf_results_do_not_contain_padding(self):
        with mock.patch.object(baselines, "IVF_NLIST", 2):
            results = baselines.ivf_search(self.documents, self.query, 80)
        self.assertGreater(len(results[0]), 0)
        self.assertLess(len(results[0]), 80)
        self.assertTrue(all(0 <= index < 80 for index in results[0]))
        self.assertEqual(results[0][0], 0)


class MissingNeighborEvaluationTests(unittest.TestCase):
    def test_beir_padding_is_a_miss_and_valid_neighbors_still_count(self):
        rows = [[-1], [-1, 1], [0, -1]]
        query_ids = ["q1", "q2", "q3"]
        qrels = {query_id: {"answer": 1} for query_id in query_ids}
        score = metrics.hit_rate_at_k(
            rows, query_ids, np.array([b"unrelated", b"answer"]), qrels
        )
        self.assertAlmostEqual(score, 1 / 3)

    def test_dpr_padding_is_ignored_before_passage_loading_and_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_dir = root / "dataset" / "nq-large"
            dataset_dir.mkdir(parents=True)
            query_ids = ["q1", "q2", "q3"]
            records = [{"text": "unrelated passage"}, {"text": "the answer is Paris"}]
            (dataset_dir / "corpus.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            (dataset_dir / "answers.jsonl").write_text(
                "".join(json.dumps({"_id": query_id, "answers": ["Paris"]}) + "\n"
                        for query_id in query_ids), encoding="utf-8"
            )
            with mock.patch.multiple(
                metrics, DATASET_DIR=root / "dataset", DATABASE_DIR=root / "database"
            ), contextlib.redirect_stderr(io.StringIO()):
                score = metrics.answer_hit_rate_at_k(
                    [[-1], [-1, 1], [0, -1]], query_ids, "nq-large", 2
                )
                self.assertAlmostEqual(score, 1 / 3)
                self.assertEqual(
                    metrics.answer_hit_rate_at_k([[-1]], ["q1"], "nq-large", 2), 0.0
                )
