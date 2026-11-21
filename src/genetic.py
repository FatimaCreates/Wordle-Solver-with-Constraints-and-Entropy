"""Genetic algorithm (evolutionary search) for a strong opening guess.

Individual = one guess word.  Fitness = entropy (bits) of its pattern distribution over all answers.
Selection = tournament, crossover = take each letter from one of two parents, mutation = change one letter.
A child that is not a real word is repaired to the most similar word in the guess list.
"""
import numpy as np

from .entropy import entropy_scores
from .game import WORD_LEN
from .patterns import N_PATTERNS


class GeneticOpener:
    def __init__(self, pm, pop: int = 40, generations: int = 25, mutation: float = 0.3, seed: int = 42):
        self.pm, self.pop, self.generations, self.mutation = pm, pop, generations, mutation
        self.rng = np.random.default_rng(seed)
        self.letters = np.array([[ord(c) - 97 for c in w] for w in pm.guesses])
        self.k = len(pm.answers)
        self._fit = {}                                        # word index -> entropy (also counts evaluations)

    @property
    def evaluations(self) -> int:
        return len(self._fit)

    def fitness(self, i: int) -> float:
        if i not in self._fit:
            counts = np.bincount(self.pm.matrix[i], minlength=N_PATTERNS)
            self._fit[i] = float(entropy_scores(counts[None, :], self.k)[0])
        return self._fit[i]

    def _repair(self, letters: np.ndarray) -> int:
        match = (self.letters == letters).sum(axis=1)
        return int(self.rng.choice(np.flatnonzero(match == match.max())))

    def _pick(self, pop) -> int:                              # tournament of 3
        return max(self.rng.choice(pop, 3), key=self.fitness)

    def run(self):
        """Returns (best word, its entropy, best entropy per generation)."""
        pop = [int(i) for i in self.rng.integers(0, len(self.pm.guesses), self.pop)]
        history = []
        for _ in range(self.generations):
            pop.sort(key=self.fitness, reverse=True)
            history.append(self.fitness(pop[0]))
            nxt = pop[:2]                                     # elitism: keep the two best
            while len(nxt) < self.pop:
                a, b = self.letters[self._pick(pop)], self.letters[self._pick(pop)]
                child = np.where(self.rng.random(WORD_LEN) < 0.5, a, b)
                if self.rng.random() < self.mutation:
                    child[self.rng.integers(WORD_LEN)] = self.rng.integers(26)
                nxt.append(self._repair(child))
            pop = nxt
        best = max(pop, key=self.fitness)
        history.append(self.fitness(best))
        return self.pm.guesses[best], self.fitness(best), history
