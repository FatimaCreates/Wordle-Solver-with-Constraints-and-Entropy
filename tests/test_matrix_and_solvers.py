import random

import numpy as np

from src.agent import SolverAgent
from src.entropy import EntropyStrategy, entropy_scores
from src.game import WordleGame, get_feedback, pattern_to_int
from src.minimax import MinimaxStrategy, worst_case_sizes
from src.patterns import PatternMatrix
from src.wordlist import load_answers, load_guesses

A, G = load_answers(), load_guesses()
PM = PatternMatrix.load_or_build(G, A)


def test_matrix_matches_get_feedback():
    rng = random.Random(42)
    for _ in range(5000):
        g, a = rng.randrange(len(G)), rng.randrange(len(A))
        assert PM.matrix[g, a] == pattern_to_int(get_feedback(G[g], A[a]))


def test_matrix_duplicate_letter_cases():
    for guess, answer in [("speed", "abide"), ("geese", "those"), ("robot", "floor"),
                          ("eerie", "eagle"), ("llama", "mamma")]:
        if guess in PM.guess_index and answer in PM.answer_index:
            assert PM.matrix[PM.guess_index[guess], PM.answer_index[answer]] == \
                pattern_to_int(get_feedback(guess, answer))


def test_pattern_counts_sum_to_k():
    cands = A[:100]
    counts = PM.pattern_counts(cands)
    assert counts.shape == (len(G), 243)
    assert (counts.sum(axis=1) == 100).all()


def test_entropy_bounds():
    counts = PM.pattern_counts(A)
    H = entropy_scores(counts, len(A))
    assert (H >= 0).all() and H.max() <= np.log2(243) + 1e-9
    # splitting k equal candidates perfectly into k groups gives log2(k) bits
    c = np.zeros((1, 243)); c[0, :4] = 1
    assert abs(entropy_scores(c, 4)[0] - 2.0) < 1e-9


def test_minimax_worst_case():
    c = np.zeros((1, 243)); c[0, 0] = 5; c[0, 1] = 2
    assert worst_case_sizes(c)[0] == 5


def test_strategies_finish_a_game_and_win():
    for strat in (EntropyStrategy(PM), MinimaxStrategy(PM)):
        agent = SolverAgent(strat, A, G)
        game = WordleGame("wound")
        guesses = agent.play(game)
        assert game.is_solved() and guesses[-1] == "wound" and len(guesses) <= 6


def test_one_or_two_candidates():
    s = EntropyStrategy(PM)
    assert s.choose_guess(["crane"], G) == "crane"
    assert s.choose_guess(["crane", "slate"], G) in ("crane", "slate")
