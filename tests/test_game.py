import pytest
from src.game import (get_feedback, pattern_to_int, int_to_pattern,
                      WordleGame, ALL_GREEN)


@pytest.mark.parametrize("guess,answer,expected", [
    ("crane", "crane", (2, 2, 2, 2, 2)),          # exact match
    ("zzzzz", "crane", (0, 0, 0, 0, 0)),          # nothing matches
    ("nacre", "crane", (1, 1, 1, 1, 2)),          # misplaced letters, last is green
    ("speed", "abide", (0, 0, 1, 0, 1)),          # 2 E's in guess, 1 in answer
    ("geese", "those", (0, 0, 0, 2, 2)),          # greens use up the E's first
    ("robot", "floor", (1, 1, 0, 2, 0)),          # repeated O
    ("aaaaa", "llama", (0, 0, 2, 0, 2)),          # 5 copies in guess, 2 in answer
    ("llama", "aaaaa", (0, 0, 2, 0, 2)),
    ("eerie", "eagle", (2, 0, 0, 0, 2)),          # extra E's become gray
])
def test_known_examples(guess, answer, expected):
    assert get_feedback(guess, answer) == expected


def test_answer_against_itself_is_all_green():
    for w in ["crane", "llama", "eerie", "mamma"]:
        assert get_feedback(w, w) == ALL_GREEN


def test_pattern_int_roundtrip_covers_all_243():
    seen = set()
    for v in range(243):
        p = int_to_pattern(v)
        assert pattern_to_int(p) == v
        seen.add(p)
    assert len(seen) == 243


def test_bad_length_raises():
    with pytest.raises(ValueError):
        get_feedback("cranes", "crane")


def test_game_flow_win_and_loss():
    g = WordleGame("crane")
    g.guess("slate")
    assert not g.is_over()
    g.guess("crane")
    assert g.is_solved() and g.guesses_used == 2

    g = WordleGame("crane")
    for _ in range(6):
        g.guess("slate")
    assert g.is_over() and not g.is_solved()
