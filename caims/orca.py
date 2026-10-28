"""Online Result Count Adaptation: update T_select at fixed T_filter (Algorithm 2)."""
import math
import numpy as np
import torch
from tqdm import tqdm
from .cam import aggregate_matches
from .runtime import select_device, iter_slices, numpy_rows_to_device
from .selection import CalibrationResult, refine_t_select


@torch.no_grad()
def build_query_histograms(
    cand_vec: np.ndarray,
    qry_vec: np.ndarray,
    t_filter: float,
    array_width: int,
    device: str,
    query_batch_size: int,
    document_batch_size: int,
    description: str,
) -> np.ndarray:
    document_count, dimension = cand_vec.shape
    partition_count = math.ceil(dimension / array_width)
    query_tensor = torch.from_numpy(
        np.array(qry_vec, dtype=np.float32, copy=True)
    ).to(device)
    histograms = torch.zeros(
        (len(qry_vec), partition_count + 1), dtype=torch.int64, device=device
    )
    for document_start, document_end in tqdm(
        iter_slices(document_count, document_batch_size),
        total=math.ceil(document_count / document_batch_size),
        desc=description,
    ):
        documents = numpy_rows_to_device(
            cand_vec, document_start, document_end, device
        )
        for query_start, query_end in iter_slices(
            len(qry_vec), query_batch_size
        ):
            scores = aggregate_matches(
                query_tensor[query_start:query_end], documents, array_width, t_filter
            ).long()
            histograms[query_start:query_end].scatter_add_(
                1, scores, torch.ones_like(scores, dtype=torch.int64)
            )
    return histograms.cpu().numpy()


def adapt_t_select(
    cand_vec: np.ndarray,
    qry_vec: np.ndarray,
    t_filter: float,
    array_width: int = 24,
    k: int = 100,
    device: str = None,
    delta: float = 0.10,
    query_batch_size: int = 64,
    document_batch_size: int = 65536,
) -> CalibrationResult:
    """Adapt the nominal rank from query result counts while keeping t_filter fixed."""
    if array_width <= 0 or k <= 0 or not len(cand_vec) or not len(qry_vec):
        raise ValueError("ORCA needs positive array_width/k and nonempty documents/qry_vec")
    device = device or select_device(max_memory_used_percent=99)
    histograms = build_query_histograms(
        cand_vec, qry_vec, t_filter, array_width, device,
        query_batch_size, document_batch_size, "Online calibration histogram",
    )
    t_select, neighbors, stable = refine_t_select(
        np.flip(histograms, axis=1).cumsum(axis=1),
        k, min(len(cand_vec), k), delta,
    )
    return CalibrationResult(float(t_filter), t_select, neighbors[t_select], stable)
