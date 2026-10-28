#!/usr/bin/env python3
"""Download BEIR datasets and prepare shared DPR corpus/query data."""

import ast
import csv
import gzip
import json
import os
import shutil
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path

from .paths import DATASET_DIR, DATASETS

import ahocorasick
import regex
from tqdm import tqdm

CORPUS_URL = "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
NQ_URL = "https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv"
TQA_URL = "https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-test.qa.csv.gz"
TOKEN_PATTERN = regex.compile(
    r"([\p{L}\p{N}\p{M}]+)|([^\p{Z}\p{C}])",
    flags=regex.IGNORECASE | regex.UNICODE | regex.MULTILINE,
)
SEPARATOR = "\x1f"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        print(f"Using existing {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    print(f"Downloading {url} -> {destination}")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=16 * 1024 * 1024)
    os.replace(temporary, destination)


def normalize_tokens(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    tokens = [match.group() for match in TOKEN_PATTERN.finditer(normalized)]
    return SEPARATOR + SEPARATOR.join(tokens) + SEPARATOR if tokens else ""


def read_qa(path: Path, compressed: bool, prefix: str):
    opener = gzip.open if compressed else open
    records = []
    with opener(path, "rt", encoding="utf-8", newline="") as source:
        for index, row in enumerate(csv.reader(source, delimiter="\t")):
            if len(row) != 2:
                raise ValueError(f"Malformed QA row {index + 1} in {path}")
            answers = ast.literal_eval(row[1].strip())
            records.append(
                {
                    "_id": f"{prefix}-test-{index:05d}",
                    "text": row[0],
                    "answers": answers,
                }
            )
    return records


def write_queries(dataset_dir: Path, records) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with (dataset_dir / "queries.jsonl").open("w", encoding="utf-8") as output:
        for record in records:
            output.write(
                json.dumps(
                    {"_id": record["_id"], "text": record["text"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (dataset_dir / "answers.jsonl").open("w", encoding="utf-8") as output:
        for record in records:
            output.write(
                json.dumps(
                    {"_id": record["_id"], "answers": record["answers"]},
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_answer_automaton(datasets):
    owners = defaultdict(set)
    for dataset_name, records in datasets.items():
        for record in records:
            for answer in record["answers"]:
                pattern = normalize_tokens(answer)
                if pattern:
                    owners[pattern].add((dataset_name, record["_id"]))

    automaton = ahocorasick.Automaton()
    for pattern in owners:
        automaton.add_word(pattern, pattern)
    automaton.make_automaton()
    return automaton, owners


def ensure_corpus_links(output_root: Path, dataset_names) -> Path:
    shared_dir = output_root / "dpr-wikipedia"
    shared_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = shared_dir / "corpus.jsonl"
    for dataset_name in dataset_names:
        dataset_dir = output_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        link = dataset_dir / "corpus.jsonl"
        if os.path.lexists(link):
            if link.is_symlink() and link.resolve() == corpus_path.resolve():
                continue
            raise FileExistsError(f"Refusing to replace existing {link}")
        link.symlink_to(Path("..") / "dpr-wikipedia" / "corpus.jsonl")
    return corpus_path


def convert_corpus(source_path: Path, corpus_path: Path, automaton, owners, datasets):
    positives = {name: {} for name in datasets}
    temporary = corpus_path.with_suffix(".jsonl.tmp")
    with gzip.open(source_path, "rt", encoding="utf-8", newline="") as source, temporary.open(
        "w", encoding="utf-8"
    ) as output:
        reader = csv.DictReader(source, delimiter="\t")
        for row in tqdm(reader, total=21_015_324, desc="Converting DPR Wikipedia"):
            passage_id = row["id"]
            text = row["text"]
            output.write(
                json.dumps(
                    {"_id": passage_id, "title": row["title"], "text": text},
                    ensure_ascii=False,
                )
                + "\n"
            )

            normalized = normalize_tokens(text)
            if normalized:
                for _, pattern in automaton.iter(normalized):
                    for dataset_name, query_id in owners[pattern]:
                        positives[dataset_name].setdefault(query_id, passage_id)
    os.replace(temporary, corpus_path)
    return positives


def write_qrels(output_root: Path, datasets, positives) -> None:
    for dataset_name, records in datasets.items():
        qrels_dir = output_root / dataset_name / "qrels"
        qrels_dir.mkdir(parents=True, exist_ok=True)
        with (qrels_dir / "test.tsv").open("w", encoding="utf-8") as output:
            output.write("query-id\tcorpus-id\tscore\n")
            for record in records:
                passage_id = positives[dataset_name].get(record["_id"])
                if passage_id is not None:
                    output.write(f"{record['_id']}\t{passage_id}\t1\n")
                else:
                    # BEIR filters queries missing from qrels. A score-zero row keeps
                    # the official query without declaring the placeholder relevant.
                    output.write(f"{record['_id']}\t1\t0\n")


def download_dpr(output_root: Path = DATASET_DIR, source_dir: Path = None) -> None:
    """Prepare the shared DPR Wikipedia corpus and both NQ/TQA query sets."""
    source_dir = source_dir or output_root / "dpr-source"
    corpus_source = source_dir / "psgs_w100.tsv.gz"
    nq_source = source_dir / "nq-test.qa.csv"
    tqa_source = source_dir / "trivia-test.qa.csv.gz"
    download(CORPUS_URL, corpus_source)
    download(NQ_URL, nq_source)
    download(TQA_URL, tqa_source)

    datasets = {
        "nq-large": read_qa(nq_source, compressed=False, prefix="nq-large"),
        "tqa": read_qa(tqa_source, compressed=True, prefix="tqa"),
    }
    for name, records in datasets.items():
        write_queries(output_root / name, records)

    corpus_path = ensure_corpus_links(output_root, datasets)
    automaton, owners = build_answer_automaton(datasets)
    if corpus_path.exists():
        raise FileExistsError(
            f"{corpus_path} already exists; remove it to rebuild qrels consistently"
        )
    positives = convert_corpus(corpus_source, corpus_path, automaton, owners, datasets)
    write_qrels(output_root, datasets, positives)

    manifest = {
        "corpus": str(corpus_path),
        "passages": 21_015_324,
        "datasets": {
            name: {
                "queries": len(records),
                "queries_with_answer_match": len(positives[name]),
                "evaluation": "Use answers.jsonl for DPR/BPR token-sequence Hit Rate",
            }
            for name, records in datasets.items()
        },
        "sources": {"corpus": CORPUS_URL, "nq-large": NQ_URL, "tqa": TQA_URL},
    }
    with (output_root / "dpr-wikipedia" / "manifest.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(manifest, indent=2))



def ensure_dataset(dataset: str, dataset_root: Path = DATASET_DIR) -> Path:
    """Download a supported BEIR dataset or prepare the shared DPR datasets."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    path = dataset_root / dataset
    required = [path / "corpus.jsonl", path / "queries.jsonl"]
    if dataset in {"nq-large", "tqa"}:
        required.append(path / "answers.jsonl")
    else:
        required.append(path / "qrels" / "test.tsv")
    if all(file.exists() for file in required):
        return path
    if dataset in {"nq-large", "tqa"}:
        download_dpr(dataset_root)
        return path
    from beir import util as beir_util

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    return Path(beir_util.download_and_unzip(url, str(dataset_root)))
