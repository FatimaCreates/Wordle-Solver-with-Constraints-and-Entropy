import pytest

from src.game import ALL_GREEN
from src.patterns import PatternMatrix
from src.race import RaceSession
from src.wordlist import load_answers, load_guesses

A, G = load_answers(), load_guesses()
PM = PatternMatrix.load_or_build(G, A)


def make(**kw):
    return RaceSession(A, G, pm=PM, **kw)


def test_secret_is_seeded_and_can_be_forced():
    assert make(seed=42).secret == make(seed=42).secret
    assert make(secret="WOUND").secret == "wound"


def test_player_wins_by_guessing_the_word():
    s = make(secret="wound")
    assert not s.is_over() and s.winner is None
    assert s.player_guess("crane") != ALL_GREEN
    assert s.player_guess("wound") == ALL_GREEN
    assert s.is_over() and s.winner == "player"
    assert len(s.player_history) == 2


def test_invalid_word_raises_and_is_not_counted():
    s = make(secret="wound")
    with pytest.raises(ValueError):
        s.player_guess("zzzzz")
    assert s.player_history == []


def test_ai_solves_and_wins_if_player_does_not():
    s = make(secret="wound", strategy_name="entropy")
    for _ in range(6):
        if s.is_over():
            break
        s.player_guess("slate")
        if not s.is_over():
            s.ai_step()
    assert s.winner == "ai"
    assert s.ai_history[-1][0] == "wound"


def test_same_round_tie_goes_to_player():
    s = make(secret="wound")
    s.player_guess("wound")
    assert s.winner == "player"
    with pytest.raises(RuntimeError):
        s.ai_step()


def test_draw_when_nobody_solves():
    s = make(secret="wound", strategy_name="random", seed=3)
    s._ai_game.history = [("slate", (0,) * 5)] * 5      # AI has one guess left
    for _ in range(6):
        s.player_guess("slate")
    assert not s.is_over()                               # AI still has a move
    s._ai_game.history.append(("slate", (0,) * 5))
    assert s.is_over() and s.winner == "draw"


def test_hint_uses_player_clues_and_does_not_change_ai():
    a, b = make(secret="wound", seed=1), make(secret="wound", seed=1)
    a.player_guess("crane")
    hint = a.hint()
    assert hint in G
    assert a.ai_step() == b.ai_step()                              # hint did not disturb the AI


def test_remaining_counts_shrink():
    s = make(secret="wound")
    assert s.player_remaining == 2315
    s.player_guess("crane")
    assert s.player_remaining < 2315