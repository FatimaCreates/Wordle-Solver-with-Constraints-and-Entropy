"""Entropy strategy: pick the guess with the highest expected information (bits)."""
import numpy as np

from .matrix_strategy import MatrixStrategy


def entropy_scores(counts: np.ndarray, k: int) -> np.ndarray:
    """H = -sum p*log2(p) over the pattern distribution of the k candidates (one value per guess)."""
    p = counts / k
    logs = np.log2(np.maximum(counts, 1) / k)       # empty patterns contribute 0
    return -(p * logs).sum(axis=1)


class EntropyStrategy(MatrixStrategy):
    name = "entropy"

    def score(self, counts, k):
        return entropy_scores(counts, k)
