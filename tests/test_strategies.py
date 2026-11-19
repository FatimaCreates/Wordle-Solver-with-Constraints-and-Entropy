from src.agent import SolverAgent
from src.game import WordleGame
from src.strategies import RandomStrategy, FrequencyStrategy
from src.wordlist import load_answers, load_guesses

A, P = load_answers(), load_guesses()


def test_random_guess_is_a_candidate_and_seeded():
    cands = ["crane", "slate", "pound"]
    assert RandomStrategy(42).choose_guess(cands, P) == RandomStrategy(42).choose_guess(cands, P)
    assert RandomStrategy(1).choose_guess(cands, P) in cands


def test_frequency_ignores_repeated_letters():
    s = FrequencyStrategy(A)
    assert s.score("aaaaa") == s.freq["a"]
    assert s.score("eerie") == s.freq["e"] + s.freq["r"] + s.freq["i"]


def test_frequency_is_deterministic():
    s = FrequencyStrategy(A)
    assert s.choose_guess(A, P) == s.choose_guess(A, P)


def test_agent_plays_and_ends():
    for strat in (RandomStrategy(42), FrequencyStrategy(A)):
        agent = SolverAgent(strat, A, P)
        game = WordleGame("wound")
        guesses = agent.play(game)
        assert game.is_over() and 1 <= len(guesses) <= 6
        assert agent.last_sizes[0] == 2315
        if game.is_solved():
            assert guesses[-1] == "wound"