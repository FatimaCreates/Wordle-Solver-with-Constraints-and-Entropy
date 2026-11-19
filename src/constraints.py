"""Constraint-based candidate filtering (the CSP part of the project).

CSP view: 5 letter positions are the variables, the word list is the domain.
Every (guess, pattern) pair adds constraints:
  - fixed letters        (green)
  - forbidden positions  (yellow and gray: the letter is not at that spot)
  - minimum letter count (green + yellow copies of a letter)
  - maximum letter count (a gray copy of a letter caps it at its green+yellow count)
CandidateSet keeps the words that satisfy every constraint so far and counts
the checks it performs (reported in the evaluation).
"""
from collections import Counter
from typing import Dict, List, Set

from .game import WORD_LEN, Pattern


class Constraints:
    """All constraints learned so far from the (guess, pattern) history."""

    def __init__(self):
        self.fixed: Dict[int, str] = {}                 # position -> letter
        self.forbidden: Dict[int, Set[str]] = {i: set() for i in range(WORD_LEN)}
        self.min_count: Dict[str, int] = {}
        self.max_count: Dict[str, int] = {}

    def add(self, guess: str, pattern: Pattern) -> None:
        colored = Counter()   # green + yellow copies of each letter in this guess
        has_gray = set()      # letters that got at least one gray in this guess

        for i, (ch, p) in enumerate(zip(guess, pattern)):
            if p == 2:
                self.fixed[i] = ch
                colored[ch] += 1
            elif p == 1:
                self.forbidden[i].add(ch)
                colored[ch] += 1
            else:
                self.forbidden[i].add(ch)   # a gray letter is not at this spot either
                has_gray.add(ch)

        for ch, n in colored.items():
            self.min_count[ch] = max(self.min_count.get(ch, 0), n)
        for ch in has_gray:
            n = colored.get(ch, 0)          # gray => exactly n copies in the answer
            self.max_count[ch] = min(self.max_count.get(ch, WORD_LEN), n)

    def allows(self, word: str) -> bool:
        for i, ch in self.fixed.items():
            if word[i] != ch:
                return False
        for i, banned in self.forbidden.items():
            if word[i] in banned:
                return False
        counts = Counter(word)
        for ch, n in self.min_count.items():
            if counts[ch] < n:
                return False
        for ch, n in self.max_count.items():
            if counts[ch] > n:
                return False
        return True


class CandidateSet:
    """Belief state: the words still consistent with every clue so far."""

    def __init__(self, words: List[str]):
        self.candidates: List[str] = list(words)
        self.history: List[tuple] = []        # (guess, pattern)
        self.constraints = Constraints()
        self.checks = 0                       # total words tested against constraints
        self.sizes: List[int] = [len(self.candidates)]  # candidates left after each turn

    def update(self, guess: str, pattern: Pattern) -> None:
        """Record a clue, add its constraints and filter the candidates."""
        self.history.append((guess, pattern))
        self.constraints.add(guess, pattern)
        self.filter()

    def filter(self) -> None:
        self.checks += len(self.candidates)
        self.candidates = [w for w in self.candidates if self.constraints.allows(w)]
        self.sizes.append(len(self.candidates))

    @property
    def remaining(self) -> int:
        return len(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)