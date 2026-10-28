"""Local Aggregation and Selective Transfer: functional software selection.

Histograms find the same global score cutoff as LAST's descending-score
broadcasts, and a second pass collects every candidate at or above that cutoff.
This implements the selection semantics, not NAND peripherals or timing/energy.
"""
import math
from typing import List, Tuple
import numpy as np
import torch
from tqdm import tqdm
from .cam import aggregate_matches
from .runtime import iter_slices, numpy_rows_to_device


@torch.no_grad()
def select_candidates(
    cand_vec: np.ndarray,
    qry_vec: np.ndarray,
    t_filter: float,
    array_width: int,
    t_select: int,
    device: str,
    query_batch_size: int = 64,
    document_batch_size: int = 65536,
    self_indices: np.ndarray = None,
) -> Tuple[List[List[int]], List[int]]:
    """Find the T_select-th score and return all candidates tied at its cutoff."""
    Nd, dim = cand_vec.shape
    Nq = len(qry_vec)
    partition_count = math.ceil(dim / array_width)
    effective_t_select = min(t_select, Nd - (1 if self_indices is not None else 0))
    if effective_t_select <= 0:
        return [[] for _ in range(Nq)], [0] * (partition_count + 1)

    queries = torch.from_numpy(np.asarray(qry_vec, dtype=np.float32)).to(device)
    histograms = torch.zeros((Nq, partition_count + 1), dtype=torch.int64, device=device)

    for ds, de in tqdm(
        iter_slices(Nd, document_batch_size),
        total=math.ceil(Nd / document_batch_size),
        desc="LAST score histogram",
    ):
        documents = numpy_rows_to_device(cand_vec, ds, de, device)
        for qs, qe in iter_slices(Nq, query_batch_size):
            scores = aggregate_matches(queries[qs:qe], documents, array_width, t_filter).long()
            weights = torch.ones_like(scores, dtype=torch.int64)
            if self_indices is not None:
                ids = self_indices[qs:qe]
                rows = np.flatnonzero((ids >= ds) & (ids < de))
                if len(rows):
                    row_tensor = torch.as_tensor(rows, device=device)
                    col_tensor = torch.as_tensor(ids[rows] - ds, device=device)
                    weights[row_tensor, col_tensor] = 0
            histograms[qs:qe].scatter_add_(1, scores, weights)

    descending = torch.flip(histograms, dims=[1]).cumsum(dim=1)
    positions = (descending >= effective_t_select).long().argmax(dim=1)
    cutoffs = partition_count - positions
    cutoff_distribution = torch.bincount(
        cutoffs, minlength=partition_count + 1
    ).cpu().tolist()

    result_chunks = [[] for _ in range(Nq)]
    for ds, de in tqdm(
        iter_slices(Nd, document_batch_size),
        total=math.ceil(Nd / document_batch_size),
        desc="LAST candidate collection",
    ):
        documents = numpy_rows_to_device(cand_vec, ds, de, device)
        for qs, qe in iter_slices(Nq, query_batch_size):
            scores = aggregate_matches(queries[qs:qe], documents, array_width, t_filter)
            matches = scores >= cutoffs[qs:qe].unsqueeze(1)
            if self_indices is not None:
                ids = self_indices[qs:qe]
                rows = np.flatnonzero((ids >= ds) & (ids < de))
                if len(rows):
                    row_tensor = torch.as_tensor(rows, device=device)
                    col_tensor = torch.as_tensor(ids[rows] - ds, device=device)
                    matches[row_tensor, col_tensor] = False
            for row in range(qe - qs):
                local = torch.nonzero(matches[row], as_tuple=True)[0].cpu().numpy()
                if len(local):
                    result_chunks[qs + row].append(local.astype(np.int64) + ds)

    results = [
        np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
        for chunks in result_chunks
    ]
    return results, cutoff_distribution
