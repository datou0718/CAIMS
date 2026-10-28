"""Exercise the retained data preparation and core CLI with local fixtures."""

import contextlib
import csv
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from caims import paths, datasets, embeddings


class DataPreparationTests(unittest.TestCase):
    def test_dpr_conversion_preserves_answers_corpus_links_and_qrels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qa = root / "qa.tsv"
            with qa.open("w", encoding="utf-8", newline="") as destination:
                writer = csv.writer(destination, delimiter="\t")
                writer.writerow(["Capital of France?", "['Paris']"])
                writer.writerow(["Missing answer?", "['Atlantis']"])
            records = datasets.read_qa(qa, compressed=False, prefix="nq-large")
            records_by_dataset = {"nq-large": records}
            datasets.write_queries(root / "nq-large", records)
            corpus = datasets.ensure_corpus_links(root, records_by_dataset)
            source = root / "passages.tsv.gz"
            with gzip.open(source, "wt", encoding="utf-8", newline="") as destination:
                writer = csv.writer(destination, delimiter="\t")
                writer.writerow(["id", "text", "title"])
                writer.writerow(["1", "Paris is the capital of France.", "France"])
                writer.writerow(["2", "A separate passage.", "Other"])
            automaton, owners = datasets.build_answer_automaton(records_by_dataset)
            with contextlib.redirect_stderr(io.StringIO()):
                positives = datasets.convert_corpus(source, corpus, automaton, owners, records_by_dataset)
            datasets.write_qrels(root, records_by_dataset, positives)
            self.assertEqual(positives, {"nq-large": {records[0]["_id"]: "1"}})
            self.assertEqual((root / "nq-large" / "corpus.jsonl").resolve(), corpus)
            self.assertEqual(len(corpus.read_text().splitlines()), 2)
            self.assertIn("\t1\t0", (root / "nq-large/qrels/test.tsv").read_text())
            answers = [json.loads(line) for line in (root / "nq-large/answers.jsonl").read_text().splitlines()]
            self.assertEqual(answers[1]["answers"], ["Atlantis"])
            self.assertEqual(datasets.ensure_dataset("nq-large", root), root / "nq-large")

    def test_streaming_encoding_preserves_row_order_and_normalization(self):
        class LocalEncoder:
            def get_sentence_embedding_dimension(self):
                return 3

            def encode(self, texts, **options):
                if not options["normalize_embeddings"]:
                    raise AssertionError("Embeddings must be normalized")
                rows = np.array([[len(text), sum(map(ord, text)) % 17, 1] for text in texts], dtype=np.float32)
                return rows / np.linalg.norm(rows, axis=1, keepdims=True)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "dataset" / "scifact"
            data.mkdir(parents=True)
            corpus = [{"_id": str(i), "title": "Title", "text": f"Passage {i}"} for i in range(5)]
            queries = [{"_id": "q", "text": "Question?"}]
            for filename, rows in [("corpus.jsonl", corpus), ("queries.jsonl", queries)]:
                (data / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))
            model = LocalEncoder()
            with mock.patch.object(paths, "EMBEDDING_DIR", root / "embedding"), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                embeddings.encode_corpus(model, "BGE", "scifact", data, root / "dataset", "cpu", 2, 2, None)
                embeddings.encode_queries(model, "BGE", "scifact", data, "cpu", 2, None)
                directory = paths.document_embedding_dir("BGE", "scifact")
                np.testing.assert_array_equal(np.load(directory / "doc_ids.npy"), np.array([str(i).encode() for i in range(5)]))
                vectors = np.load(directory / "doc_emb.npy")
                self.assertEqual(vectors.shape, (5, 3))
                np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
                np.testing.assert_array_equal(np.load(directory / "qry_ids.npy"), ["q"])
                self.assertEqual(np.load(directory / "qry_emb.npy").shape, (1, 3))


class CoreCommandTests(unittest.TestCase):
    def test_quantize_dass_orca_and_last_without_experiment_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "embedding" / "BGE" / "scifact"
            directory.mkdir(parents=True)
            rng = np.random.default_rng(7)
            documents = rng.normal(size=(12, 6)).astype(np.float32)
            queries = rng.normal(size=(4, 6)).astype(np.float32)
            np.save(directory / "doc_emb.npy", documents)
            np.save(directory / "qry_emb.npy", queries)
            np.save(directory / "doc_ids.npy", np.array([f"d{i}".encode() for i in range(12)]))
            np.save(directory / "qry_ids.npy", np.array([f"q{i}" for i in range(4)]))
            environment = os.environ.copy()
            environment.update(CAIMS_DATA_ROOT=str(root), PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
            dass_path = root / "dass.json"
            config_path = root / "configuration.json"
            output_path = root / "matches.jsonl"
            common = ["--encoder", "BGE", "--dataset", "scifact"]
            commands = [
                ["quantize", *common],
                ["dass", *common, "--array-width", "3", "--k", "2", "--n-smp", "5",
                 "--seed", "17", "--device", "cpu",
                 "--query-batch-size", "2", "--document-batch-size", "5", "--output", str(dass_path)],
                ["orca", "--config", str(dass_path), "--n-online", "2", "--seed", "17",
                 "--device", "cpu", "--query-batch-size", "2", "--document-batch-size", "5",
                 "--output", str(config_path)],
                ["last", "--config", str(config_path), "--device", "cpu",
                 "--query-batch-size", "2", "--document-batch-size", "5", "--output", str(output_path)],
            ]
            for arguments in commands:
                completed = subprocess.run(
                    [sys.executable, "-m", "caims", *arguments], env=environment,
                    cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            configuration = json.loads(config_path.read_text())
            self.assertEqual(len(configuration["pseudo_query_indices"]), 5)
            self.assertEqual(len(configuration["online_query_indices"]), 2)
            self.assertIsNotNone(configuration["orca"])
            self.assertEqual(configuration["t_filter"], configuration["dass"]["t_filter"])
            quantized_docs = np.load(root / "database/BGE/scifact/3-bit.npy").astype(np.float32)
            quantized_queries = np.load(root / "database/BGE/scifact/3-bit_qry.npy").astype(np.float32)
            votes = sum(
                (-((quantized_queries[:, None, start:start + 3] - quantized_docs[None, :, start:start + 3]) ** 2).sum(axis=2)
                 >= configuration["t_filter"]).astype(np.int64)
                for start in range(0, 6, 3)
            )
            rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            counts = []
            for i, row in enumerate(rows):
                cutoff = np.sort(votes[i])[-configuration["t_select"]]
                expected = [f"d{j}" for j in np.flatnonzero(votes[i] >= cutoff)]
                self.assertEqual(row, {"query_id": f"q{i}", "document_ids": expected})
                counts.append(len(expected))
            self.assertAlmostEqual(
                configuration["c"],
                float(np.mean(np.array(counts)[configuration["online_query_indices"]])),
            )


if __name__ == "__main__":
    unittest.main()
