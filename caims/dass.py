"""Dataset-Assisted Scalable Search: offline parameter selection (Algorithm 1)."""
import math
from typing import List
import numpy as np
import torch
from tqdm import tqdm
from .cam import aggregate_matches_for_filters
from .runtime import select_device, iter_slices, numpy_rows_to_device
from .selection import CalibrationResult, mean_result_count, refine_t_select


def sample_pseudo_query_indices(document_count: int, n_smp: int = 1000, seed: int = 42) -> np.ndarray:
    """Sample corpus rows without computing unused exact nearest neighbors."""
    if document_count < 2:
        raise ValueError("DASS needs at least two documents for leave-one-out calibration")
    if n_smp <= 0:
        raise ValueError("n_smp must be positive")
    return np.random.default_rng(seed).choice(
        document_count, size=min(n_smp, document_count), replace=False
    )


def candidate_t_filters(bits: int, array_width: int, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("t_filter count must be positive")
    maximum_segment_distance = (2 << (bits - 1)) ** 2 * array_width
    return np.unique(
        np.rint(
            np.linspace(
                -maximum_segment_distance * 0.035,
                -maximum_segment_distance * 0.07,
                count,
            )
        )
    ).astype(np.float32)


@torch.no_grad()
def build_pseudo_query_histograms(
    cand_vec: np.ndarray,
    query_indices: np.ndarray,
    t_filters: np.ndarray,
    array_width: int,
    device: str,
    query_batch_size: int = 64,
    document_batch_size: int = 65536,
) -> np.ndarray:
    document_count, dimension = cand_vec.shape
    query_count = len(query_indices)
    partition_count = math.ceil(dimension / array_width)
    filter_tensor = torch.as_tensor(t_filters, dtype=torch.float32, device=device)
    queries = torch.from_numpy(
        np.array(cand_vec[query_indices], dtype=np.float32, copy=True)
    ).to(device)
    histograms = torch.zeros(
        (len(t_filters), query_count, partition_count + 1),
        dtype=torch.int64,
        device=device,
    )

    for document_start, document_end in tqdm(
        iter_slices(document_count, document_batch_size),
        total=math.ceil(document_count / document_batch_size),
        desc="Building CAM vote histograms",
    ):
        documents = numpy_rows_to_device(
            cand_vec, document_start, document_end, device
        )
        for query_start, query_end in iter_slices(query_count, query_batch_size):
            scores = aggregate_matches_for_filters(
                queries[query_start:query_end],
                documents,
                array_width,
                filter_tensor,
            ).long()
            histograms[:, query_start:query_end].scatter_add_(
                2, scores, torch.ones_like(scores, dtype=torch.int64)
            )

    # Every sampled candidate matches itself in all partitions. Calibration
    # is leave-one-out, so remove that one entry from the maximum-vote bin.
    histograms[:, :, partition_count] -= 1
    return histograms.cpu().numpy()


def evaluate_pairs(
    histograms: np.ndarray,
    t_filters: np.ndarray,
    k: int,
    max_t_select: int,
    delta: float,
) -> List[CalibrationResult]:
    """Keep every t_filter's candidate so an unstable fallback always exists."""
    descending_counts = np.flip(histograms, axis=2).cumsum(axis=2)
    results = []

    for filter_index, t_filter in enumerate(t_filters):
        cumulative = descending_counts[filter_index]
        t_select, neighbor_averages, stable = refine_t_select(
            cumulative, k, max_t_select, delta
        )
        results.append(
            CalibrationResult(
                t_filter=float(t_filter),
                t_select=t_select,
                c=neighbor_averages[t_select],
                stable=stable,
            )
        )
    return results


def best_pair(
    results: List[CalibrationResult],
    k: int,
    delta: float,
):
    """Prefer stable pairs, falling back to the closest unstable candidate."""
    if not results:
        raise RuntimeError("No t_filter candidates were evaluated.")

    stable_results = [result for result in results if result.stable]
    candidates = stable_results or results
    return min(
        candidates,
        key=lambda result: (
            abs(result.c - k),
            result.t_select,
            result.t_filter,
        ),
    )


def select_parameters(
    cand_vec: np.ndarray,
    bits: int = 3,
    array_width: int = 24,
    k: int = 100,
    n_smp: int = 1000,
    device: str = None,
    delta: float = 0.10,
    filter_count: int = 8,
    query_batch_size: int = 64,
    document_batch_size: int = 65536,
    seed: int = 42,
    query_indices: np.ndarray = None,
) -> CalibrationResult:
    """Calibrate DASS; prefer stable pairs and always permit an unstable fallback.

    Pass existing pseudo-query indices to reuse a previous sample. Otherwise a
    sample is drawn directly from the corpus using ``seed``.
    """
    if bits <= 0 or array_width <= 0 or k <= 0 or len(cand_vec) < 2:
        raise ValueError("DASS needs positive bits/array_width/k and at least two documents")
    if query_indices is None:
        query_indices = sample_pseudo_query_indices(len(cand_vec), n_smp, seed)
    else:
        query_indices = np.asarray(query_indices, dtype=np.int64)
        if (query_indices.ndim != 1 or not len(query_indices)
                or np.any(query_indices < 0) or np.any(query_indices >= len(cand_vec))):
            raise ValueError("Pseudo-query indices must be nonempty valid corpus row indices")
    device = device or select_device(max_memory_used_percent=99)
    t_filters = candidate_t_filters(bits, array_width, filter_count)
    histograms = build_pseudo_query_histograms(
        cand_vec, query_indices, t_filters, array_width, device,
        query_batch_size, document_batch_size,
    )
    results = evaluate_pairs(
        histograms, t_filters, k, min(len(cand_vec) - 1, k), delta
    )
    return best_pair(results, k, delta)
