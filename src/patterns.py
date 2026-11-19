"""Pattern matrix: the Wordle pattern of EVERY valid guess against EVERY answer.

matrix[g, a] = pattern_to_int(get_feedback(guesses[g], answers[a]))  (0..242)
Built once with NumPy (vectorised version of the two-pass rule in game.py),
then cached to data/processed/ so strategies never recompute it.
"""
from pathlib import Path
from typing import List

import numpy as np

from .game import WORD_LEN

CACHE = Path(__file__).resolve().parent.parent / "data" / "processed" / "pattern_matrix.npy"
N_PATTERNS = 3 ** WORD_LEN  # 243


def _encode(words: List[str]) -> np.ndarray:
    return np.array([[ord(c) - 97 for c in w] for w in words], dtype=np.int16)


def build_matrix(guesses: List[str], answers: List[str], chunk: int = 256) -> np.ndarray:
    G, A = _encode(guesses), _encode(answers)
    n_g, n_a = len(guesses), len(answers)

    ans_count = np.zeros((n_a, 26), dtype=np.int16)        # copies of each letter in each answer
    for i in range(WORD_LEN):
        ans_count[np.arange(n_a), A[:, i]] += 1

    weights = 3 ** np.arange(WORD_LEN - 1, -1, -1)         # position 0 is the most significant digit
    matrix = np.empty((n_g, n_a), dtype=np.uint8)

    for s in range(0, n_g, chunk):
        Gc = G[s:s + chunk]                                # (c, 5)
        c = len(Gc)
        green = Gc[:, None, :] == A[None, :, :]            # (c, a, 5)
        same = Gc[:, :, None] == Gc[:, None, :]            # same[c, i, j]: guess letters i and j equal
        green_i16 = green.astype(np.int16)
        yellow = np.zeros((c, n_a, WORD_LEN), dtype=bool)

        for i in range(WORD_LEN):
            same_i = same[:, i, :].astype(np.int16)                      # (c, 5)
            green_same = np.einsum("caj,cj->ca", green_i16, same_i)      # greens of this letter
            available = ans_count[:, Gc[:, i]].T                         # (c, a) copies in answer
            earlier = np.zeros((c, n_a), dtype=np.int16)                 # yellows already given to this letter
            for j in range(i):
                earlier += yellow[:, :, j] * same_i[:, j][:, None]
            yellow[:, :, i] = (~green[:, :, i]) & ((available - green_same - earlier) > 0)

        digits = 2 * green.astype(np.int16) + yellow.astype(np.int16)    # 0 gray, 1 yellow, 2 green
        matrix[s:s + c] = (digits * weights).sum(axis=2).astype(np.uint8)
    return matrix


class PatternMatrix:
    def __init__(self, guesses: List[str], answers: List[str], matrix: np.ndarray):
        self.guesses, self.answers, self.matrix = guesses, answers, matrix
        self.guess_index = {w: i for i, w in enumerate(guesses)}
        self.answer_index = {w: i for i, w in enumerate(answers)}
        self._offsets = (np.arange(len(guesses), dtype=np.int64) * N_PATTERNS)[:, None]

    @classmethod
    def load_or_build(cls, guesses, answers, path: Path = CACHE):
        if path.exists():
            matrix = np.load(path)
            if matrix.shape == (len(guesses), len(answers)):
                return cls(guesses, answers, matrix)
        matrix = build_matrix(guesses, answers)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix)
        return cls(guesses, answers, matrix)

    def pattern_counts(self, candidate_words: List[str]) -> np.ndarray:
        """counts[g, p] = how many of the candidates would give pattern p if guess g is played."""
        cols = [self.answer_index[w] for w in candidate_words]
        sub = self.matrix[:, cols].astype(np.int64)                      # (guesses, k)
        flat = (sub + self._offsets).ravel()
        return np.bincount(flat, minlength=len(self.guesses) * N_PATTERNS).reshape(-1, N_PATTERNS)
