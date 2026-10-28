"""Data and vector artifact paths, including the shared DPR corpus."""

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("CAIMS_DATA_ROOT", "data")).expanduser().resolve()
DATASET_DIR = DATA_ROOT / "dataset"
EMBEDDING_DIR = DATA_ROOT / "embedding"
DATABASE_DIR = DATA_ROOT / "database"

SHARED_CORPORA = {"nq-large": "dpr-wikipedia", "tqa": "dpr-wikipedia"}


def corpus_name(dataset: str) -> str:
    return SHARED_CORPORA.get(dataset, dataset)


def document_embedding_dir(encoder: str, dataset: str) -> Path:
    return EMBEDDING_DIR / encoder / corpus_name(dataset)


def query_embedding_dir(encoder: str, dataset: str) -> Path:
    return EMBEDDING_DIR / encoder / dataset


def document_database_dir(encoder: str, dataset: str) -> Path:
    return DATABASE_DIR / encoder / corpus_name(dataset)


def query_database_dir(encoder: str, dataset: str) -> Path:
    return DATABASE_DIR / encoder / dataset

DATASETS = ["scifact", "scidocs", "quora", "nq", "nq-large", "tqa"]
