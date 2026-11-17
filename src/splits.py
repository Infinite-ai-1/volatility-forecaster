from collections.abc import Iterator

import numpy as np


def walk_forward_splits(
    n_rows: int, min_train: int, val_size: int, gap: int, step: int | None = None
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    step = val_size if step is None else step
    if min_train < 1 or val_size < 1 or gap < 0 or step < val_size:
        raise ValueError(
            "Positive windows, a non-negative gap and non-overlapping "
            "validation are required"
        )
    val_start = min_train + gap
    while val_start + val_size <= n_rows:
        yield np.arange(val_start - gap), np.arange(val_start, val_start + val_size)
        val_start += step


def holdout_indices(
    n_rows: int, test_size: int, gap: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if test_size < 1 or gap < 0 or n_rows <= test_size + gap:
        raise ValueError("Insufficient rows for development, purge and test")
    test_start = n_rows - test_size
    return (
        np.arange(test_start - gap),
        np.arange(test_start - gap, test_start),
        np.arange(test_start, n_rows),
    )
