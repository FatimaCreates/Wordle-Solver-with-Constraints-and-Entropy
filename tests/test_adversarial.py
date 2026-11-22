import pytest

from src.adversarial import AdversarySession, AlphaBetaSolver, play_vs_adversary
from src.game import ALL_GREEN
from src.genetic import GeneticOpener
from src.patterns import PatternMatrix
from src.race import make_strategy
from src.wordlist import load_answers, load_guesses

A, G = load_answers(), load_guesses()
PM = PatternMatrix.load_or_build(G, A)


def test_adversary_keeps_the_biggest_group():
    s = AdversarySession(A, G, PM)
    s.player_guess("crane")
    assert 0 < s.remaining < len(A)
    assert all(w in A for w in s.candidates)


def test_adversary_is_consistent_with_its_answers():
    s = AdversarySession(A, G, PM)
    pat = s.player_guess("soare")
    from src.game import get_feedback
    assert all(get_feedback("soare", w) == pat for w in s.candidates)      # every kept word fits the colours shown


def test_invalid_word_rejected():
    with pytest.raises(ValueError):
        AdversarySession(A, G, PM).player_guess("zzzzz")


def test_alphabeta_solves_the_adversary_game():
    n, ok, _ = play_vs_adversary(AlphaBetaSolver(PM), A, G, PM)
    assert ok and n <= 6


def test_alphabeta_not_worse_than_random():
    ab, _, _ = play_vs_adversary(AlphaBetaSolver(PM), A, G, PM)
    rnd, _, _ = play_vs_adversary(make_strategy("random", A, G, PM), A, G, PM)
    assert ab <= rnd


def test_alphabeta_wins_when_one_word_left():
    assert AlphaBetaSolver(PM).choose_guess(["wound"]) == "wound"


def test_genetic_returns_a_valid_pair_with_high_joint_entropy():
    words, h, hist = GeneticOpener(PM, pop=12, generations=8, seed=1).run()
    assert len(set(words)) == 2 and all(w in G for w in words)     # two distinct real words
    assert h > 8.0                                                 # a single word tops out at 5.9 bits
    assert hist[-1] >= hist[0]                                     # elitism: never gets worse
