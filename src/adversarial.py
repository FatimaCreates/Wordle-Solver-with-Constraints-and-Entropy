"""Adversarial search (Absurdle): the computer has NO fixed secret word.

After every guess the ADVERSARY picks the colour pattern that keeps the most words alive
(the worst case for the player). The player's side is solved with minimax + alpha-beta:

    player node (MIN)     choose a guess that minimises the words left
    adversary node (MAX)  choose the pattern that maximises the words left
    leaf (depth 0)        value = number of words still possible (0 if the guess was right)

Big word sets use the entropy strategy (opening). The alpha-beta search takes over when few words are left.
"""
from typing import List, Optional

import numpy as np

from .entropy import EntropyStrategy
from .game import ALL_GREEN, MAX_GUESSES, WORD_LEN, int_to_pattern, pattern_to_int

GREEN = pattern_to_int(ALL_GREEN)
INF = 10 ** 9


class AlphaBetaSolver:
    def __init__(self, pm, depth: int = 2, width: int = 8, max_words: int = 40):
        self.pm, self.depth, self.width, self.max_words = pm, depth, width, max_words
        self.fallback = EntropyStrategy(pm)
        self.nodes = 0                                    # search nodes visited (for the report)

    def _groups(self, g: int, cand: np.ndarray):
        """Split the candidate answers by the pattern guess g would give (biggest group first)."""
        pats = self.pm.matrix[g, cand]
        groups = [(int(p), cand[pats == p]) for p in np.unique(pats)]
        return sorted(groups, key=lambda t: -len(t[1]))

    def _moves(self, cand: np.ndarray) -> List[int]:
        """The few most promising guesses: smallest one-step worst case, candidates preferred."""
        if len(cand) == 0:
            return []
        words = [self.pm.answers[i] for i in cand]
        worst = self.pm.pattern_counts(words).max(axis=1)
        is_cand = np.zeros(len(self.pm.guesses), dtype=bool)
        is_cand[[self.pm.guess_index[w] for w in words]] = True
        top = [int(i) for i in np.lexsort((~is_cand, worst))[:self.width]]
        return top

    def _value(self, cand, depth, alpha, beta) -> int:             # player node: MIN
        self.nodes += 1
        if depth == 0 or len(cand) == 1:
            return len(cand)
        best = INF
        for g in self._moves(cand):
            best = min(best, self._adv(g, cand, depth, alpha, beta))
            beta = min(beta, best)
            if best <= alpha:                                       # the adversary above would never allow this
                break
        return best

    def _adv(self, g, cand, depth, alpha, beta) -> int:           # adversary node: MAX
        best = -INF
        for p, group in self._groups(g, cand):
            best = max(best, 0 if p == GREEN else self._value(group, depth - 1, alpha, beta))
            alpha = max(alpha, best)
            if best >= beta:                                        # the player above would never allow this
                break
        return best

    def choose_guess(self, candidates: List[str], pool: List[str] = None) -> str:
        k = len(candidates)
        if k == 0:
            raise ValueError("no candidates left: the clues so far are inconsistent")
        if k <= 2:
            return candidates[0]
        if k > self.max_words:
            return self.fallback.choose_guess(candidates, pool)
        cand = np.array([self.pm.answer_index[w] for w in candidates])
        best, pick = INF, None
        for g in self._moves(cand):
            v = self._adv(g, cand, self.depth, -INF, best)
            if v < best:
                best, pick = v, g
        return self.pm.guesses[pick]


class AdversarySession:
    """Same interface the app uses for RaceSession, but the secret is never fixed.

    The player still gets the normal Wordle allowance (`MAX_GUESSES` = 6), because
    Absurdle is played here under the same rules as the rest of the project;
    `max_guesses` is a parameter so an experiment can look at other rules.
    """
    adversarial = True
    max_guesses = MAX_GUESSES

    def __init__(self, answers: List[str], pool: List[str], pm,
                 solver: Optional[AlphaBetaSolver] = None, max_guesses: int = MAX_GUESSES):
        self.answers, self.pool, self.pm = answers, set(pool), pm
        self.max_guesses = int(max_guesses)
        self._cand = np.arange(len(answers))
        self._history = []
        self._solver = solver or AlphaBetaSolver(pm)

    def player_guess(self, word: str):
        word = word.strip().lower()
        if self.is_over():
            raise RuntimeError("the game is over")
        if word not in self.pool:
            raise ValueError(f"'{word}' is not a valid guess")
        g = self.pm.guess_index[word]
        pats = self.pm.matrix[g, self._cand]
        # adversary: biggest group; on a tie the pattern with fewer colours (least help); never green if avoidable
        p = max(np.unique(pats), key=lambda q: (int((pats == q).sum()), -sum(int_to_pattern(int(q)))))
        self._cand = self._cand[pats == p]
        pattern = int_to_pattern(int(p))
        self._history.append((word, pattern))
        return pattern

    def hint(self) -> str:
        return self._solver.choose_guess(self.candidates)

    @property
    def candidates(self) -> List[str]:
        return [self.answers[i] for i in self._cand]

    @property
    def remaining(self) -> int:
        return len(self._cand)

    @property
    def player_history(self):
        return list(self._history)

    @property
    def ai_history(self):
        return []

    @property
    def secret(self) -> str:
        return self.answers[self._cand[0]]           # the word the adversary was left with

    def is_over(self) -> bool:
        return bool(self._history and self._history[-1][1] == ALL_GREEN) or len(self._history) >= self.max_guesses


def play_vs_adversary(chooser, answers, pool, pm, limit: int = MAX_GUESSES):
    """Let an AI strategy play the adversary. Returns (guesses used, solved?, guesses)."""
    s = AdversarySession(answers, pool, pm, max_guesses=limit)
    while not s.is_over():
        s.player_guess(chooser.choose_guess(s.candidates, list(pool)))
    return len(s.player_history), s.player_history[-1][1] == ALL_GREEN, [w for w, _ in s.player_history]
