"""Genetic algorithm (evolutionary search) for the best TWO-guess opening.

Why a pair and not one word?  A single word is a search space of 12,972 points,
so `argsort` over the entropy column solves it exactly and instantly and a GA
there is only a demonstration.  The pair space has 12,972 * 12,971 / 2 =
84,129,906 points: large enough that the exact answer is expensive, and small
enough that an evolutionary search is realistic.  An evolved pair is also a
*feature*: a fixed two-guess opener that a game can actually play.

Genome      (g1, g2): two distinct indices into the guess list.
Fitness     joint entropy, in bits, of the 243 x 243 pattern distribution the
            pair produces over all 2,315 answers.  Equivalently: how much the
            pair narrows the answer list, on average, after two guesses.
Selection   tournament of 3, on distinct genomes.
Crossover   uniform at the gene level: each gene comes from either parent.
Mutation    move ONE gene: usually to a legal word sharing four of its five
            letters (a local move), sometimes to a uniformly random legal word.
Elitism     the best genomes always survive.
Immigrants  a few fresh random genomes enter every generation.  Without them the
            population collapses onto one basin within a dozen generations (the
            measured symptom: ~160 unique genomes in 1,600 children).

Measured on this dataset (see run_genetic.py): one word tops out at 5.886 bits
(soare).  The GA reaches 9.5838 bits (liane + crost) / 4.43 answers left after two
guesses, against a 9.6329-bit ceiling found by an exhaustive partner sweep for
soare.  It does that with ~4,000 objective evaluations out of 84,129,906 possible
pairs.  A good pair is NOT two good words: the best partner for soare is `clint`,
whose own entropy is only 5.0 bits - exactly the complementarity a GA can find and
a ranking cannot.

The encoding is the whole genome, so every child is automatically two real,
distinct words: there is no "repair" step that would silently replace an
invalid child with an approximation.  Fitness calls are memoised, so
`evaluations` is the honest number of objective evaluations (compare: 12,972
for a single-word brute force, 84 million for a pair brute force).
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .entropy import EntropyStrategy
from .game import WORD_LEN
from .patterns import N_PATTERNS
from .strategies import Strategy

Pair = Tuple[int, int]


def pair_counts(pm, pair: Sequence[str]) -> np.ndarray:
    """Pattern-pair histogram over all answers: counts[p1 * 243 + p2]."""
    a, b = sorted(pm.guess_index[w] for w in pair)
    p1 = pm.matrix[a].astype(np.int32)
    p2 = pm.matrix[b].astype(np.int32)
    return np.bincount(p1 * N_PATTERNS + p2, minlength=N_PATTERNS * N_PATTERNS)


def pair_metrics(pm, pair: Sequence[str]) -> Dict[str, float]:
    """Entropy (bits), expected and worst-case answers left after the pair."""
    counts = pair_counts(pm, pair)
    k = len(pm.answers)
    nz = counts[counts > 0].astype(np.float64)
    p = nz / k
    return {"entropy": float(-(p * np.log2(p)).sum()),
            "avg_left": float((nz ** 2).sum() / k),
            "worst_group": int(nz.max()),
            "patterns": int(len(nz))}


class GeneticOpener:
    """Evolve (word1, word2): the opening pair with the highest joint entropy."""

    def __init__(self, pm, pop: int = 60, generations: int = 150,
                 mutation: float = 0.6, elite: int = 2, immigrants: int = 10,
                 seed: int = 42):
        if pop < 8:
            raise ValueError("population must be at least 8")
        self.pm = pm
        self.pop = int(pop)
        self.generations = int(generations)
        self.mutation = float(mutation)
        self.elite = int(elite)
        self.immigrants = int(immigrants)
        self.rng = np.random.default_rng(seed)
        self.n = len(pm.guesses)
        self.k = len(pm.answers)
        self.letters = np.array([[ord(c) - 97 for c in w] for w in pm.guesses], dtype=np.int16)
        self._fit: Dict[Pair, float] = {}
        self.history: List[float] = []          # best fitness per generation
        self._seed_pool = None                  # good single words, used only for initialisation

    # ---------- objective ----------
    @property
    def evaluations(self) -> int:
        return len(self._fit)

    def fitness(self, pair: Pair) -> float:
        a, b = (int(pair[0]), int(pair[1]))
        key: Pair = (a, b) if a < b else (b, a)
        if key not in self._fit:
            counts = pair_counts(self.pm, (self.pm.guesses[key[0]], self.pm.guesses[key[1]]))
            nz = counts[counts > 0].astype(np.float64)
            p = nz / self.k
            self._fit[key] = float(-(p * np.log2(p)).sum())
        return self._fit[key]

    # ---------- operators ----------
    def _tournament(self, pop: List[Pair]) -> Pair:
        picks = self.rng.choice(len(pop), size=min(3, len(pop)), replace=False)
        return max((pop[int(i)] for i in picks), key=self.fitness)

    def _crossover(self, p: Pair, q: Pair) -> Pair:
        g1 = int(p[0]) if self.rng.random() < 0.5 else int(q[0])
        g2 = int(p[1]) if self.rng.random() < 0.5 else int(q[1])
        if g1 == g2:                                  # keep the genome valid
            g2 = self._near(g2, g1)
        return (g1, g2)

    def _near(self, g: int, forbidden: int) -> int:
        """A legal word sharing four of g's five letters, else any other legal word."""
        close = np.flatnonzero((self.letters == self.letters[g]).sum(axis=1) >= WORD_LEN - 1)
        close = close[(close != g) & (close != forbidden)]
        if len(close):
            return int(self.rng.choice(close))
        for _ in range(100):
            j = int(self.rng.integers(self.n))
            if j != g and j != forbidden:
                return j
        return (g + 1) % self.n

    def _random_gene(self, forbidden: int) -> int:
        for _ in range(100):
            j = int(self.rng.integers(self.n))
            if j != forbidden:
                return j
        return (forbidden + 1) % self.n

    def _mutate(self, pair: Pair) -> Pair:
        """One gene moves: 60% locally (4 shared letters), 40% to a random word."""
        a, b = int(pair[0]), int(pair[1])
        first = self.rng.random() < 0.5
        old, other = (a, b) if first else (b, a)
        new = (self._near(old, other) if self.rng.random() < 0.6
               else self._random_gene(other))
        return (new, b) if first else (a, new)

    # ---------- main loop ----------
    def _seeds(self) -> np.ndarray:
        """The 200 highest single-word entropies: a starting point, not an answer.

        A good PAIR is not two good words (that is measured in run_genetic.py); but
        starting half the population on strong words lets the GA spend its budget
        on the complementary second word, which is where the gain is.
        """
        if self._seed_pool is None:
            counts = self.pm.pattern_counts(self.pm.answers)
            nz = counts[counts > 0].astype(np.float64)
            rows = np.repeat(np.arange(len(self.pm.guesses)), (counts > 0).sum(axis=1))
            p = nz / self.k
            bits = np.bincount(rows, weights=-(p * np.log2(p)), minlength=len(self.pm.guesses))
            self._seed_pool = np.argsort(-bits)[:200]
        return self._seed_pool

    def run(self) -> Tuple[Tuple[str, str], float, List[float]]:
        """Returns (best pair of words, its joint entropy, best fitness per generation)."""
        seeds = self._seeds()
        pop: List[Pair] = []
        while len(pop) < self.pop:                     # valid, distinct initial genomes
            if self.rng.random() < 0.6:                # half-informed start
                a = int(self.rng.choice(seeds))
            else:
                a = int(self.rng.integers(self.n))
            b = int(self.rng.integers(self.n))
            if a != b:
                pop.append((a, b))

        for _ in range(self.generations):
            pop.sort(key=self.fitness, reverse=True)
            self.history.append(self.fitness(pop[0]))
            nxt: List[Pair] = pop[:self.elite]           # elitism
            while len(nxt) < self.pop - self.immigrants:
                child = self._crossover(self._tournament(pop), self._tournament(pop))
                if self.rng.random() < self.mutation:
                    child = self._mutate(child)
                nxt.append(child)
            while len(nxt) < self.pop:                   # immigrants keep the search diverse
                a = int(self.rng.integers(self.n))
                nxt.append((a, self._random_gene(a)))
            pop = nxt

        pop.sort(key=self.fitness, reverse=True)
        self.history.append(self.fitness(pop[0]))
        best = pop[0]
        words = (self.pm.guesses[best[0]], self.pm.guesses[best[1]])
        return words, self.fitness(best), list(self.history)


class GeneticPairStrategy(Strategy):
    """Play the evolved pair first, then fall back to the entropy strategy.

    The pair is a fixed, clue-independent opening: two guesses that together
    split the answers as evenly as possible.  After that the belief state is
    small enough for the entropy strategy to finish the game.
    """
    name = "genetic"

    def __init__(self, pm, opener: Sequence[str], fallback: Strategy = None):
        self.pm = pm
        self.opener = tuple(opener)
        self.fallback = fallback or EntropyStrategy(pm)
        self._used = 0

    def choose_guess(self, candidates: List[str], pool: List[str] = None) -> str:
        if len(candidates) == len(self.pm.answers):      # a fresh game
            self._used = 0
        if len(candidates) <= 1:
            return candidates[0]
        if self._used < len(self.opener):
            word = self.opener[self._used]
            self._used += 1
            return word
        return self.fallback.choose_guess(candidates, pool)
