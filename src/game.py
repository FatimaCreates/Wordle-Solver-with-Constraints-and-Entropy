"""Wordle environment: feedback rules and the WordleGame class.

Pattern encoding (per letter): 0 = gray, 1 = yellow, 2 = green.
A pattern is a tuple of 5 ints, e.g. (0, 1, 0, 2, 0).
Every module (constraints, strategies, experiments, UI) must use get_feedback()
so the duplicate-letter rules live in exactly one place.
"""
from typing import Tuple

Pattern = Tuple[int, ...]
WORD_LEN = 5
MAX_GUESSES = 6
ALL_GREEN: Pattern = (2,) * WORD_LEN


def get_feedback(guess: str, answer: str) -> Pattern:
    """Return the Wordle pattern for `guess` against `answer`.

    Two-pass algorithm, correct for repeated letters:
      Pass 1: mark greens, and count the answer letters NOT used by a green.
      Pass 2: for each non-green guess letter, mark yellow if an unused copy
              of that letter remains in the answer (and use it up), else gray.
    """
    if len(guess) != WORD_LEN or len(answer) != WORD_LEN:
        raise ValueError("guess and answer must both have 5 letters")

    result = [0] * WORD_LEN
    unused = {}  # answer letters not matched by a green

    for i in range(WORD_LEN):
        if guess[i] == answer[i]:
            result[i] = 2
        else:
            unused[answer[i]] = unused.get(answer[i], 0) + 1

    for i in range(WORD_LEN):
        if result[i] == 0 and unused.get(guess[i], 0) > 0:
            result[i] = 1
            unused[guess[i]] -= 1

    return tuple(result)


def pattern_to_int(pattern: Pattern) -> int:
    """Encode a pattern as a base-3 integer in 0..242 (for the pattern matrix)."""
    value = 0
    for p in pattern:
        value = value * 3 + p
    return value


def int_to_pattern(value: int) -> Pattern:
    """Inverse of pattern_to_int."""
    digits = []
    for _ in range(WORD_LEN):
        digits.append(value % 3)
        value //= 3
    return tuple(reversed(digits))


def pattern_to_emoji(pattern: Pattern) -> str:
    return "".join({0: "⬛", 1: "🟨", 2: "🟩"}[p] for p in pattern)


class WordleGame:
    """The environment. Holds the secret word and returns patterns."""

    def __init__(self, answer: str, valid_guesses=None):
        self.answer = answer.lower()
        self.valid_guesses = set(valid_guesses) if valid_guesses is not None else None
        self.history = []  # list of (guess, pattern)

    def guess(self, word: str) -> Pattern:
        word = word.lower()
        if self.valid_guesses is not None and word not in self.valid_guesses:
            raise ValueError(f"'{word}' is not in the valid guess list")
        if self.is_over():
            raise RuntimeError("game is already over")
        pattern = get_feedback(word, self.answer)
        self.history.append((word, pattern))
        return pattern

    def is_solved(self) -> bool:
        return bool(self.history) and self.history[-1][1] == ALL_GREEN

    def is_over(self) -> bool:
        return self.is_solved() or len(self.history) >= MAX_GUESSES

    @property
    def guesses_used(self) -> int:
        return len(self.history)
