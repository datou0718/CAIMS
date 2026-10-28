"""CPU brute-force and IVF retrieval baselines (Section IV)."""
import os
from typing import List
import faiss
import numpy as np
import torch

_IVF_CACHE_KEY = None
_IVF_CACHE_INDEX = None
IVF_NLIST = 4096
IVF_NPROBE = 256


@torch.no_grad()
def brute_force_search(
    cand_vec: np.ndarray,
    qry_vec: np.ndarray,
    k: int,
    metric: str = "inner_product",
    self_indices: np.ndarray = None,
) -> List[List[int]]:
    """Naive CPU exact search using one full query-document score matrix."""
    Nd = len(cand_vec)
    effective_k = min(k, Nd - (1 if self_indices is not None else 0))
    if effective_k <= 0:
        return [[] for _ in range(len(qry_vec))]

    documents = torch.from_numpy(np.array(cand_vec, dtype=np.float32, copy=True))
    queries = torch.from_numpy(np.array(qry_vec, dtype=np.float32, copy=True))
    similarities = torch.matmul(queries, documents.T)
    if metric == "negative_l2":
        similarities.mul_(2.0)
        similarities.sub_(torch.sum(queries * queries, dim=1, keepdim=True))
        similarities.sub_(torch.sum(documents * documents, dim=1).unsqueeze(0))
    elif metric != "inner_product":
        raise ValueError(f"Unknown metric: {metric}")

    if self_indices is not None:
        rows = torch.arange(len(queries))
        columns = torch.as_tensor(self_indices, dtype=torch.int64)
        similarities[rows, columns] = -float("inf")

    return torch.topk(similarities, k=effective_k, dim=1).indices.tolist()


def ivf_search(
    cand_vec: np.ndarray,
    qry_vec: np.ndarray,
    k: int,
    index_path=None,
) -> List[List[int]]:
    """Load or build a persistent CPU IVF index without application batching."""
    global _IVF_CACHE_KEY, _IVF_CACHE_INDEX

    Nd, dim = cand_vec.shape
    k = min(k, Nd)
    nlist = min(IVF_NLIST, Nd)
    nprobe = min(IVF_NPROBE, nlist)
    # Unpersisted indexes have no stable corpus identity. Rebuild them even for
    # equal-shaped arrays, which may contain different or modified documents.
    cache_key = (str(index_path.resolve()), Nd, dim, nlist) if index_path is not None else None

    if cache_key is not None and _IVF_CACHE_KEY == cache_key:
        index = _IVF_CACHE_INDEX
    else:
        if index_path is not None and index_path.exists():
            print(f"Loading FAISS index from {index_path}")
            index = faiss.read_index(str(index_path))
            if index.d != dim or index.ntotal != Nd or index.nlist != nlist:
                raise ValueError(f"Stored FAISS index is incompatible: {index_path}")
        else:
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(
                quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
            train_size = min(Nd, max(100_000, min(1_000_000, 40 * nlist)))
            train_indices = np.linspace(0, Nd - 1, train_size, dtype=np.int64)
            training = np.asarray(cand_vec[train_indices], dtype=np.float32)
            index.train(training)
            index.add(np.asarray(cand_vec, dtype=np.float32))
            if index_path is not None:
                index_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = f"{index_path}.tmp"
                faiss.write_index(index, temporary_path)
                os.replace(temporary_path, index_path)
                print(f"Saved FAISS index to {index_path}")

        _IVF_CACHE_KEY = cache_key
        _IVF_CACHE_INDEX = index if cache_key is not None else None

    index.nprobe = nprobe
    _, retrieved_indices = index.search(
        np.asarray(qry_vec, dtype=np.float32), k
    )
    # IVF pads incomplete result rows with -1; these are not document IDs.
    return [row[row != -1].astype(np.int64).tolist() for row in retrieved_indices]
