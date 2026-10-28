"""Reusable BEIR and DPR accuracy metrics, without experiment orchestration."""

import csv
import json
import os
import unicodedata
from typing import Dict, List

import numpy as np
import regex
from tqdm import tqdm

from .paths import DATABASE_DIR, DATASET_DIR, corpus_name


def load_qrels(dataset: str) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    path = DATASET_DIR / dataset / "qrels" / "test.tsv"
    with open(path, encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return qrels


TOKEN_PATTERN = regex.compile(
    r"([\p{L}\p{N}\p{M}]+)|([^\p{Z}\p{C}])",
    flags=regex.IGNORECASE | regex.UNICODE | regex.MULTILINE,
)
TOKEN_SEPARATOR = "\x1f"
ANSWER_EVALUATION_DATASETS = {"nq-large", "tqa"}


def normalize_answer_tokens(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    tokens = [match.group() for match in TOKEN_PATTERN.finditer(normalized)]
    return TOKEN_SEPARATOR + TOKEN_SEPARATOR.join(tokens) + TOKEN_SEPARATOR if tokens else ""


def load_answers(dataset: str) -> Dict[str, List[str]]:
    answers = {}
    path = DATASET_DIR / dataset / "answers.jsonl"
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            answers[record["_id"]] = [
                pattern
                for answer in record["answers"]
                if (pattern := normalize_answer_tokens(answer))
            ]
    return answers


def load_corpus_offsets(dataset: str, document_count: int) -> np.ndarray:
    index_dir = DATABASE_DIR / "corpus-index"
    index_dir.mkdir(parents=True, exist_ok=True)
    path = index_dir / f"{corpus_name(dataset)}-offsets.npy"
    if path.exists():
        offsets = np.load(path, mmap_mode="r")
        if offsets.shape != (document_count,) or offsets.dtype != np.uint64:
            raise ValueError(f"Stale or invalid corpus offset index: {path}")
        return offsets

    temporary = index_dir / f"{path.name}.tmp"
    offsets = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.uint64, shape=(document_count,)
    )
    corpus_path = DATASET_DIR / dataset / "corpus.jsonl"
    byte_offset = 0
    count = 0
    with corpus_path.open("rb") as source:
        for count, line in enumerate(
            tqdm(source, total=document_count, desc="Indexing corpus offsets"), start=1
        ):
            if count > document_count:
                raise ValueError("Corpus contains more documents than embeddings")
            offsets[count - 1] = byte_offset
            byte_offset += len(line)
    if count != document_count:
        raise ValueError(f"Corpus has {count} documents; embeddings have {document_count}")
    offsets.flush()
    del offsets
    os.replace(temporary, path)
    return np.load(path, mmap_mode="r")


def load_retrieved_passages(
    dataset: str,
    top_k_indices: List[List[int]],
    document_count: int,
) -> Dict[int, str]:
    indices = sorted({int(index) for row in top_k_indices for index in row if index != -1})
    if indices and (indices[0] < 0 or indices[-1] >= document_count):
        raise IndexError("Retrieved document index is outside the corpus")
    if not indices:
        return {}
    offsets = load_corpus_offsets(dataset, document_count)
    corpus_path = DATASET_DIR / dataset / "corpus.jsonl"
    passages = {}
    with corpus_path.open("rb") as source:
        for index in tqdm(indices, desc="Loading retrieved passages"):
            source.seek(int(offsets[index]))
            passages[index] = json.loads(source.readline())["text"]
    return passages


def answer_hit_rate_at_k(
    top_k_indices: List[List[int]],
    qry_ids: List[str],
    dataset: str,
    document_count: int,
) -> float:
    answers = load_answers(dataset)
    passages = load_retrieved_passages(dataset, top_k_indices, document_count)
    normalized_passages = {
        index: normalize_answer_tokens(text) for index, text in passages.items()
    }
    hits = []
    for query_id, indices in zip(qry_ids, top_k_indices):
        patterns = answers.get(query_id, [])
        if not patterns:
            continue
        hit = any(
            pattern in normalized_passages[int(index)]
            for index in indices
            if index != -1
            for pattern in patterns
        )
        hits.append(float(hit))
    return sum(hits) / len(hits) if hits else 0.0


def hit_rate_at_k(
    top_k_indices: List[List[int]],
    qry_ids: List[str],
    doc_ids: List[str],
    qry_rels: Dict[str, Dict[str, int]],
) -> float:

    individual_hits = []

    for i, q_id in enumerate(qry_ids):
        rel_docs = {doc_id for doc_id, rel in qry_rels.get(q_id, {}).items() if rel > 0}

        if not rel_docs:
            continue

        top_k_doc_ids = {
            doc_ids[j].decode("utf-8") for j in top_k_indices[i] if j != -1
        }
        hit = bool(rel_docs & top_k_doc_ids)
        individual_hits.append(float(hit))

    return sum(individual_hits) / len(individual_hits) if individual_hits else 0.0
