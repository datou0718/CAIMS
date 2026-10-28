"""Shared proportional updates and NeighborRefine for T_select (Algorithms 1–2)."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    """Selected T_filter/T_select and mean result count c on calibration queries."""

    t_filter: float
    t_select: int
    c: float
    stable: bool = True


def mean_result_count(cumulative: np.ndarray, t_select: int) -> float:
    positions = np.argmax(cumulative >= t_select, axis=1)
    counts = cumulative[np.arange(len(cumulative)), positions]
    return float(np.mean(counts))


def refine_t_select(
    cumulative: np.ndarray,
    k: int,
    max_t_select: int,
    delta: float,
    max_rounds: int = 20,
):
    """Update T_select proportionally, then apply NeighborRefine and stability checks."""
    current_t_select = min(max(k, 1), max_t_select)
    evaluated = {}

    for _ in range(max_rounds):
        observed = evaluated.setdefault(
            current_t_select, mean_result_count(cumulative, current_t_select)
        )
        if observed <= 0:
            break
        next_t_select = int(round(current_t_select * k / observed))
        next_t_select = min(max(next_t_select, 1), max_t_select)
        if next_t_select == current_t_select or next_t_select in evaluated:
            break
        current_t_select = next_t_select

    def ranking(rank):
        return abs(evaluated[rank] - k), rank

    closest_t_select = min(evaluated, key=ranking)
    while True:
        candidates = []
        for rank in (closest_t_select - 1, closest_t_select, closest_t_select + 1):
            if 1 <= rank <= max_t_select:
                evaluated.setdefault(rank, mean_result_count(cumulative, rank))
                candidates.append(rank)
        next_closest = min(candidates, key=ranking)
        if next_closest == closest_t_select:
            break
        closest_t_select = next_closest

    lower = k * (1.0 - delta)
    upper = k * (1.0 + delta)
    initial_closest = closest_t_select
    initial_neighbors = None
    visited = set()

    while closest_t_select not in visited:
        visited.add(closest_t_select)
        neighbor_averages = {}
        for rank in (closest_t_select - 1, closest_t_select, closest_t_select + 1):
            if 1 <= rank <= max_t_select:
                neighbor_averages[rank] = evaluated.setdefault(
                    rank, mean_result_count(cumulative, rank)
                )
            else:
                neighbor_averages[rank] = None
        if closest_t_select == initial_closest:
            initial_neighbors = neighbor_averages

        values = list(neighbor_averages.values())
        stable = all(value is not None for value in values) and all(
            lower <= value <= upper for value in values
        )
        if stable:
            return closest_t_select, neighbor_averages, True

        previous = neighbor_averages[closest_t_select - 1]
        following = neighbor_averages[closest_t_select + 1]
        if previous is None or previous < lower:
            next_t_select = closest_t_select + 1
        elif following is None or following > upper:
            next_t_select = closest_t_select - 1
        else:
            break
        if not 1 <= next_t_select <= max_t_select:
            break
        closest_t_select = next_t_select

    return initial_closest, initial_neighbors, False
