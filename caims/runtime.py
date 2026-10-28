"""Batching, device transfer, and compute-device selection."""
from typing import Iterator, Tuple, Union, List
import numpy as np
import torch


def iter_slices(length: int, batch_size: int) -> Iterator[Tuple[int, int]]:
    """Yield bounded half-open slices without allocating an index array."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, length, batch_size):
        yield start, min(start + batch_size, length)


def numpy_rows_to_device(
    array: np.ndarray,
    start: int,
    end: int,
    device: str,
) -> torch.Tensor:
    """Convert only one NumPy/memmap row block to float32 on the device."""
    block = np.array(array[start:end], dtype=np.float32, copy=True)
    return torch.from_numpy(block).to(device)


def select_device(
    max_memory_used_percent: int = 50,
    multiple: bool = False,
    exclude_cpu: bool = False,
) -> Union[str, List[str], None]:
    """Select logical CUDA devices, respecting CUDA_VISIBLE_DEVICES remapping."""
    if torch.cuda.is_available():
        candidates = []
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            used_percent = (1.0 - free / max(total, 1)) * 100.0
            if used_percent < max_memory_used_percent:
                candidates.append((used_percent, f"cuda:{index}"))
        candidates.sort(key=lambda item: item[0])
        devices = [device for _, device in candidates]
        if devices:
            return devices if multiple else devices[0]

    if torch.backends.mps.is_available():
        return ["mps"] if multiple else "mps"
    if not exclude_cpu:
        return ["cpu"] if multiple else "cpu"
    return None
