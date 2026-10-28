"""Stream normalized BGE/MiniLM embeddings to disk."""

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from .paths import DATASET_DIR, document_embedding_dir, query_embedding_dir
from .runtime import select_device
from .datasets import ensure_dataset

ENCODERS = {
    "BGE": "BAAI/bge-base-en-v1.5",
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
}
ENCODER_REVISIONS = {
    "BGE": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
}
DOCUMENT_ID_BYTES = 64


def encode_batch(model, texts, device, batch_size, pool):
    if pool is not None:
        result = model.encode(
            texts,
            pool=pool,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    else:
        result = model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True, device=device,
            normalize_embeddings=True, show_progress_bar=False,
        )
    return result.astype(np.float32, copy=False)


def read_queries(path: Path) -> Tuple[List[str], List[str]]:
    ids, texts = [], []
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            ids.append(str(record["_id"]))
            texts.append(record["text"])
    return ids, texts


def corpus_size(dataset: str, corpus_path: Path, dataset_root: Path) -> int:
    if dataset in {"nq-large", "tqa"}:
        manifest = dataset_root / "dpr-wikipedia" / "manifest.json"
        if manifest.exists():
            with manifest.open(encoding="utf-8") as source:
                return int(json.load(source)["passages"])
    print(f"Counting passages in {corpus_path}...")
    with corpus_path.open("rb") as source:
        return sum(1 for _ in source)


def encode_corpus(
    model, encoder_type, dataset, data_dir, dataset_root, device,
    batch_size, chunk_size, pool,
):
    output_dir = document_embedding_dir(encoder_type, dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / "doc_emb.npy"
    ids_path = output_dir / "doc_ids.npy"
    if embedding_path.exists() and ids_path.exists():
        print(f"Using existing document artifacts in {output_dir}")
        return
    if embedding_path.exists() or ids_path.exists():
        raise RuntimeError(f"Incomplete document artifacts in {output_dir}")

    corpus_path = data_dir / "corpus.jsonl"
    count = corpus_size(dataset, corpus_path, dataset_root)
    dimension = model.get_sentence_embedding_dimension()
    embedding_tmp = Path(f"{embedding_path}.tmp")
    ids_tmp = Path(f"{ids_path}.tmp")
    embeddings = np.lib.format.open_memmap(
        embedding_tmp, mode="w+", dtype=np.float32, shape=(count, dimension)
    )
    document_ids = np.lib.format.open_memmap(
        ids_tmp, mode="w+", dtype=f"S{DOCUMENT_ID_BYTES}", shape=(count,)
    )
    offset, texts, ids = 0, [], []
    progress = tqdm(total=count, desc=f"Encoding {output_dir.name} corpus")

    def flush():
        nonlocal offset, texts, ids
        if not texts:
            return
        end = offset + len(texts)
        if end > count:
            raise ValueError("Corpus has more passages than expected")
        embeddings[offset:end] = encode_batch(model, texts, device, batch_size, pool)
        document_ids[offset:end] = ids
        progress.update(len(texts))
        offset, texts, ids = end, [], []

    try:
        with corpus_path.open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                encoded_id = str(record["_id"]).encode("utf-8")
                if len(encoded_id) > DOCUMENT_ID_BYTES:
                    raise ValueError(f"Document ID exceeds {DOCUMENT_ID_BYTES} bytes: {record['_id']}")
                ids.append(encoded_id)
                texts.append(f"{record.get('title') or ''} {record.get('text') or ''}")
                if len(texts) == chunk_size:
                    flush()
        flush()
        if offset != count:
            raise ValueError(f"Expected {count} passages but encoded {offset}")
        embeddings.flush()
        document_ids.flush()
    finally:
        progress.close()
        del embeddings
        del document_ids
    os.replace(embedding_tmp, embedding_path)
    os.replace(ids_tmp, ids_path)


def encode_queries(model, encoder_type, dataset, data_dir, device, batch_size, pool):
    output_dir = query_embedding_dir(encoder_type, dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    query_ids, query_texts = read_queries(data_dir / "queries.jsonl")
    np.save(output_dir / "qry_emb.npy", encode_batch(model, query_texts, device, batch_size, pool))
    np.save(output_dir / "qry_ids.npy", np.asarray(query_ids, dtype=str))
    print(f"Saved {len(query_ids)} query embeddings in {output_dir}")


def encode_datasets(
    encoder_type, datasets, batch_size=256, corpus_chunk_size=16_384, devices=None
):
    from sentence_transformers import SentenceTransformer

    if batch_size <= 0 or corpus_chunk_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if devices is None:
        selected = select_device(max_memory_used_percent=50, multiple=True, exclude_cpu=False)
        devices = selected if isinstance(selected, list) else [selected]
    if not devices or devices == [None]:
        raise RuntimeError("No encoding device is available")
    multi_gpu = len(devices) > 1 and all(x.startswith("cuda:") for x in devices)
    model = SentenceTransformer(
        ENCODERS[encoder_type], device="cpu" if multi_gpu else devices[0],
        revision=ENCODER_REVISIONS.get(encoder_type),
    )
    pool = model.start_multi_process_pool(devices) if multi_gpu else None
    print(f"Encoding {datasets} with {ENCODERS[encoder_type]} on {devices}")
    try:
        dataset_root = DATASET_DIR
        for dataset in datasets:
            print(f"\n=== {encoder_type}: {dataset} ===")
            data_dir = ensure_dataset(dataset, dataset_root)
            encode_corpus(
                model, encoder_type, dataset, data_dir, dataset_root, devices[0],
                batch_size, corpus_chunk_size, pool,
            )
            encode_queries(
                model, encoder_type, dataset, data_dir, devices[0], batch_size, pool
            )
    finally:
        if pool is not None:
            model.stop_multi_process_pool(pool)


def encode_dataset(encoder_type, dataset, batch_size=256, corpus_chunk_size=16_384, devices=None):
    encode_datasets(
        encoder_type, [dataset], batch_size, corpus_chunk_size, devices
    )
