"""RaceSession: the player and an AI try to guess the SAME hidden word.

Turn-based use from a UI (one round):
    session.player_guess("crane")     # player's move
    if not session.is_over(): session.ai_step()   # AI's move
The winner is whoever solves first in the order the moves were made
(so if both solve in the same round, the player wins because they moved first).
"""
import random
from typing import List, Optional, Tuple

from .constraints import CandidateSet
from .entropy import EntropyStrategy
from .game import ALL_GREEN, MAX_GUESSES, WordleGame, Pattern
from .minimax import MinimaxStrategy
from .patterns import PatternMatrix
from .strategies import FrequencyStrategy, RandomStrategy, Strategy

STRATEGY_NAMES = ("random", "frequency", "entropy", "minimax")


def make_strategy(name: str, answers: List[str], pool: List[str],
                  pm: Optional[PatternMatrix] = None, seed: int = 42) -> Strategy:
    if name == "random":
        return RandomStrategy(seed)
    if name == "frequency":
        return FrequencyStrategy(answers)
    if name in ("entropy", "minimax"):
        pm = pm or PatternMatrix.load_or_build(pool, answers)
        return EntropyStrategy(pm) if name == "entropy" else MinimaxStrategy(pm)
    raise ValueError(f"unknown strategy '{name}', choose from {STRATEGY_NAMES}")


class RaceSession:
    def __init__(self, answers: List[str], pool: List[str], strategy_name: str = "entropy",
                 seed: int = 42, secret: Optional[str] = None,
                 pm: Optional[PatternMatrix] = None):
        self.answers, self.pool = answers, pool
        self.strategy_name = strategy_name
        self.secret = secret.lower() if secret else random.Random(seed).choice(answers)

        # Separate strategy objects, so asking for a hint never changes the AI's own moves.
        self._ai_strategy = make_strategy(strategy_name, answers, pool, pm, seed)
        self._hint_strategy = make_strategy(strategy_name, answers, pool, pm, seed)

        self._player_game = WordleGame(self.secret, valid_guesses=pool)
        self._ai_game = WordleGame(self.secret)
        self._player_belief = CandidateSet(answers)   # what the player's clues allow
        self._ai_belief = CandidateSet(answers)       # what the AI's clues allow
        self._first_solver: Optional[str] = None

    # ---- moves ----
    def player_guess(self, word: str) -> Pattern:
        """Play the player's guess. Raises ValueError for an invalid word, RuntimeError if finished."""
        if self.is_over() or self._player_game.is_over():
            raise RuntimeError("the player cannot guess any more")
        word = word.strip().lower()
        pattern = self._player_game.guess(word)       # raises ValueError if not a valid word
        self._player_belief.update(word, pattern)
        if pattern == ALL_GREEN and self._first_solver is None:
            self._first_solver = "player"
        return pattern

    def ai_step(self) -> Tuple[str, Pattern]:
        """Let the AI make its next guess. Returns (guess, pattern)."""
        if self.is_over() or self._ai_game.is_over():
            raise RuntimeError("the AI cannot guess any more")
        guess = self._ai_strategy.choose_guess(self._ai_belief.candidates, self.pool)
        pattern = self._ai_game.guess(guess)
        self._ai_belief.update(guess, pattern)
        if pattern == ALL_GREEN and self._first_solver is None:
            self._first_solver = "ai"
        return guess, pattern

    def hint(self) -> str:
        """The guess the AI would make if it had only the PLAYER's clues so far."""
        return self._hint_strategy.choose_guess(self._player_belief.candidates, self.pool)

    # ---- state ----
    @property
    def player_history(self) -> List[Tuple[str, Pattern]]:
        return list(self._player_game.history)

    @property
    def ai_history(self) -> List[Tuple[str, Pattern]]:
        return list(self._ai_game.history)

    @property
    def player_sizes(self) -> List[int]:
        """Candidates left before any guess and after each of the player's guesses."""
        return list(self._player_belief.sizes)

    @property
    def ai_sizes(self) -> List[int]:
        """Candidates left before any guess and after each of the AI's guesses."""
        return list(self._ai_belief.sizes)

    @property
    def player_remaining(self) -> int:
        """How many answers are still possible given the player's clues."""
        return self._player_belief.remaining

    @property
    def ai_remaining(self) -> int:
        return self._ai_belief.remaining

    def is_over(self) -> bool:
        if self._first_solver is not None:
            return True
        return self._player_game.guesses_used >= MAX_GUESSES and self._ai_game.guesses_used >= MAX_GUESSES

    @property
    def winner(self) -> Optional[str]:
        """None while running; then "player", "ai", or "draw" (nobody solved in 6)."""
        if not self.is_over():
            return None
        return self._first_solver or "draw"
