"""Minimax strategy: minimise the size of the LARGEST pattern group (the worst case)."""
import numpy as np

from .matrix_strategy import MatrixStrategy


def worst_case_sizes(counts: np.ndarray) -> np.ndarray:
    return counts.max(axis=1)


class MinimaxStrategy(MatrixStrategy):
    name = "minimax"

    def score(self, counts, k):
        return -worst_case_sizes(counts).astype(float)   # smaller worst case = higher score
