"""Uniform T_filter matching and dimension-wise score aggregation (Section III-B)."""
import torch


@torch.no_grad()
def aggregate_matches(
    qry_vec: torch.Tensor,
    cand_vec: torch.Tensor,
    array_width: int,
    t_filter: float,
) -> torch.Tensor:
    """Compute CAM segment scores for one query/document block."""
    if array_width <= 0:
        raise ValueError("array_width must be positive")
    dim = qry_vec.shape[1]
    scores = torch.zeros(
        (qry_vec.shape[0], cand_vec.shape[0]),
        dtype=torch.int16,
        device=qry_vec.device,
    )
    for start in range(0, dim, array_width):
        end = min(start + array_width, dim)
        query_segment = qry_vec[:, start:end]
        document_segment = cand_vec[:, start:end]
        query_norm = torch.sum(query_segment * query_segment, dim=1, keepdim=True)
        document_norm = torch.sum(
            document_segment * document_segment, dim=1
        ).unsqueeze(0)
        dot = torch.matmul(query_segment, document_segment.T)
        similarity = -(query_norm + document_norm - 2.0 * dot)
        scores += similarity.ge(t_filter)
    return scores


@torch.no_grad()
def aggregate_matches_for_filters(
    qry_vec: torch.Tensor,
    cand_vec: torch.Tensor,
    array_width: int,
    t_filters: torch.Tensor,
) -> torch.Tensor:
    """Compute vote matrices for several t_filters in one similarity pass."""
    if array_width <= 0:
        raise ValueError("array_width must be positive")
    scores = torch.zeros(
        (len(t_filters), qry_vec.shape[0], cand_vec.shape[0]),
        dtype=torch.int16,
        device=qry_vec.device,
    )
    dim = qry_vec.shape[1]
    filter_view = t_filters.to(qry_vec.device).view(-1, 1, 1)
    for start in range(0, dim, array_width):
        end = min(start + array_width, dim)
        query_segment = qry_vec[:, start:end]
        document_segment = cand_vec[:, start:end]
        query_norm = torch.sum(query_segment * query_segment, dim=1, keepdim=True)
        document_norm = torch.sum(
            document_segment * document_segment, dim=1
        ).unsqueeze(0)
        dot = torch.matmul(query_segment, document_segment.T)
        similarity = -(query_norm + document_norm - 2.0 * dot)
        scores += similarity.unsqueeze(0).ge(filter_view)
    return scores
