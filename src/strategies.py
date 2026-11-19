"""Guessing strategies (agent policies). All share choose_guess(candidates, pool)."""
import random
from abc import ABC, abstractmethod
from collections import Counter
from typing import List


class Strategy(ABC):
    name = "strategy"

    @abstractmethod
    def choose_guess(self, candidates: List[str], pool: List[str]) -> str:
        """Pick the next guess. `candidates` = words still possible, `pool` = all legal guesses."""


class RandomStrategy(Strategy):
    """Baseline 1: a random word from the remaining candidates."""
    name = "random"

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def choose_guess(self, candidates, pool):
        return self.rng.choice(candidates)


class FrequencyStrategy(Strategy):
    """Baseline 2: the candidate whose DISTINCT letters have the highest total frequency.

    The frequency table is built once from the answer list (how many answer
    words contain each letter), so repeated letters in a guess add nothing.
    """
    name = "frequency"

    def __init__(self, answers: List[str]):
        self.freq = Counter()
        for w in answers:
            self.freq.update(set(w))

    def score(self, word: str) -> int:
        return sum(self.freq[ch] for ch in set(word))

    def choose_guess(self, candidates, pool):
        # max() keeps the first best word, and candidates are in list order -> deterministic
        return max(candidates, key=self.score)