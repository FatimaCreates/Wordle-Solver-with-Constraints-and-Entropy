"""Shared logic for strategies that score guesses with the cached pattern matrix."""
from typing import List

import numpy as np

from .patterns import PatternMatrix
from .strategies import Strategy


class MatrixStrategy(Strategy):
    def __init__(self, pm: PatternMatrix):
        self.pm = pm
        self._memo = {}          # candidate list -> chosen guess (many games share early states)

    def score(self, counts: np.ndarray, k: int) -> np.ndarray:
        """Return one score per guess; HIGHER is better. Implemented by subclasses."""
        raise NotImplementedError

    def choose_guess(self, candidates: List[str], pool: List[str] = None) -> str:
        k = len(candidates)
        if k <= 2:
            return candidates[0]            # 1 left: certain. 2 left: a 50% shot is optimal
        key = tuple(candidates)
        if key in self._memo:
            return self._memo[key]

        counts = self.pm.pattern_counts(candidates)
        scores = self.score(counts, k)
        tied = scores >= scores.max() - 1e-9

        is_candidate = np.zeros(len(self.pm.guesses), dtype=bool)        # prefer guesses that could win now
        is_candidate[[self.pm.guess_index[w] for w in candidates]] = True
        preferred = tied & is_candidate
        pick = np.flatnonzero(preferred if preferred.any() else tied)[0]  # first = alphabetical

        self._memo[key] = self.pm.guesses[pick]
        return self._memo[key]
