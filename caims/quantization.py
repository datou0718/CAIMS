"""Per-vector Quantile Quantization and vector loading (Section III-E)."""

import math
import os

import numpy as np
from tqdm import tqdm

from .paths import (
    document_database_dir, document_embedding_dir,
    query_database_dir, query_embedding_dir,
)
from .runtime import iter_slices


def load_vectors(
    encoder: str,
    dataset: str,
    bits: int = 3,
):
    """Return document/query vectors and their IDs; labels are not required."""
    doc_embedding_path = document_embedding_dir(encoder, dataset)
    qry_embedding_path = query_embedding_dir(encoder, dataset)
    doc_ids = np.load(doc_embedding_path / "doc_ids.npy", mmap_mode="r")
    qry_ids = np.load(qry_embedding_path / "qry_ids.npy", mmap_mode="r").tolist()

    if bits > 0:
        doc_db_dir = document_database_dir(encoder, dataset)
        qry_db_dir = query_database_dir(encoder, dataset)
        doc_db_dir.mkdir(parents=True, exist_ok=True)
        qry_db_dir.mkdir(parents=True, exist_ok=True)
        db_path = doc_db_dir / f"{bits}-bit.npy"
        qry_path = qry_db_dir / f"{bits}-bit_qry.npy"

        if not os.path.exists(db_path):
            source = np.load(doc_embedding_path / "doc_emb.npy", mmap_mode="r")
            quantile_quantize_to_file(source, bits, db_path)
        if not os.path.exists(qry_path):
            source = np.load(qry_embedding_path / "qry_emb.npy", mmap_mode="r")
            quantile_quantize_to_file(source, bits, qry_path)

        doc_emb = np.load(db_path, mmap_mode="r")
        qry_emb = np.load(qry_path, mmap_mode="r")
    else:
        doc_emb = np.load(doc_embedding_path / "doc_emb.npy", mmap_mode="r")
        qry_emb = np.load(qry_embedding_path / "qry_emb.npy", mmap_mode="r")

    qry_emb = np.array(qry_emb, dtype=np.float32, copy=True)
    return doc_emb, qry_emb, doc_ids, qry_ids


def quantile_quantize_to_file(
    X: np.ndarray,
    bits: int,
    output_path: str,
    batch_size: int = 65536,
) -> None:
    """Quantize a potentially memory-mapped matrix without a full output copy."""
    temporary_path = f"{output_path}.tmp"
    output = np.lib.format.open_memmap(
        temporary_path, mode="w+", dtype=np.uint8, shape=X.shape
    )
    quantiles = np.linspace(0, 100, 2**bits + 1)[1:-1]
    for start, end in tqdm(
        iter_slices(len(X), batch_size),
        total=math.ceil(len(X) / batch_size),
        desc="Quantizing embedding blocks",
    ):
        block = np.asarray(X[start:end], dtype=np.float32)
        thresholds = np.percentile(block, quantiles, axis=1).T
        quantized = np.zeros(block.shape, dtype=np.uint8)
        for level in range(thresholds.shape[1]):
            quantized += block >= thresholds[:, level : level + 1]
        output[start:end] = quantized
    output.flush()
    del output
    os.replace(temporary_path, output_path)
