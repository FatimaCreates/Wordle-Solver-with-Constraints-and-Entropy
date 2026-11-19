"""SolverAgent: a ConstraintSolver (CandidateSet) + a Strategy playing one game."""
from typing import List

from .constraints import CandidateSet
from .game import WordleGame
from .strategies import Strategy


class SolverAgent:
    def __init__(self, strategy: Strategy, answers: List[str], pool: List[str]):
        self.strategy = strategy
        self.answers = answers
        self.pool = pool
        self.last_sizes: List[int] = []   # candidates left after each guess (last game)
        self.last_checks = 0

    def play(self, game: WordleGame) -> List[str]:
        """Play one game to the end; return the list of guesses made."""
        belief = CandidateSet(self.answers)
        guesses = []
        while not game.is_over():
            guess = self.strategy.choose_guess(belief.candidates, self.pool)
            pattern = game.guess(guess)
            guesses.append(guess)
            belief.update(guess, pattern)
        self.last_sizes = belief.sizes
        self.last_checks = belief.checks
        return guesses