from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.race import RaceSession
from src.wordlist import load_answers, load_guesses

APP = str(Path(__file__).resolve().parent.parent / "app.py")
A, G = load_answers(), load_guesses()


def page_text(at):
    return " ".join(m.value for m in at.markdown)


def click(at, key):
    at.button(key=key).click()
    return at.run()


def start_game(at=None):
    at = at or AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    at.button[0].click()                       # the only button on the start page
    at = at.run()
    assert not at.exception
    return at


def force_secret(at, secret="wound", strategy="entropy"):
    pm = at.session_state["session"]._ai_strategy.pm
    at.session_state["session"] = RaceSession(A, G, strategy_name=strategy, secret=secret, pm=pm)
    return at.run()


def guess(at, word):
    for ch in word:
        at = click(at, f"k_{ch}")
    return click(at, "k_enter")


def test_start_page_then_game():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    text = page_text(at)
    assert "WORDLE" in text and "How to play" in text
    assert "Artificial Intelligence" not in text            # course name removed
    assert len(at.button) == 1                               # only the start button, no keyboard yet
    at = start_game(at)
    assert at.button(key="k_q") is not None and "Words still possible" not in page_text(at)


def test_home_button_returns_to_start():
    at = click(start_game(), "home")
    assert "How to play" in page_text(at) and len(at.button) == 1


def test_typing_and_backspace():
    at = start_game()
    for ch in "crx":
        at = click(at, f"k_{ch}")
    assert at.session_state["current"] == "crx"
    at = click(at, "k_back")
    assert at.session_state["current"] == "cr"
    for ch in "anex":                                        # a 5-letter limit
        at = click(at, f"k_{ch}")
    assert at.session_state["current"] == "crane"


def test_play_win_flow():
    at = force_secret(start_game())
    at = guess(at, "crane")
    assert not at.exception and len(at.session_state["session"].player_history) == 1
    assert at.session_state["current"] == ""
    at = guess(at, "wound")
    assert "Solved in 2 guesses" in page_text(at)


def test_not_enough_letters_and_invalid_word():
    at = force_secret(start_game())
    at = click(click(at, "k_a"), "k_enter")
    assert "Not enough letters" in page_text(at)
    at = guess(click(at, "k_back"), "zzzzz")
    assert "not in the word list" in page_text(at)
    assert at.session_state["session"].player_history == []
    assert at.session_state["current"] == "zzzzz"            # letters kept so they can be fixed


def test_hint_button():
    at = force_secret(start_game())
    at = click(at, "hint_btn")
    assert "The AI would guess" in page_text(at)


def test_new_button_resets():
    at = force_secret(start_game())
    at = guess(at, "crane")
    at = click(at, "new_btn")
    assert at.session_state["session"].player_history == []


def test_key_colours_follow_clues():
    at = force_secret(start_game())
    at = guess(at, "crane")                                  # vs wound: n, e... n is yellow, others red/green
    css = page_text(at)
    assert ".st-key-k_c button" in css and ".st-key-k_n button" in css


def test_race_mode_ai_moves_after_player():
    at = start_game()
    at.radio[0].set_value("Race vs AI")
    at = force_secret(at.run())
    at = guess(at, "crane")
    s = at.session_state["session"]
    assert len(s.player_history) == 1 and len(s.ai_history) == 1


def test_race_player_runs_out_of_guesses_ai_finishes():
    at = start_game()
    at.radio[0].set_value("Race vs AI")
    at = force_secret(at.run())
    for _ in range(6):
        at = guess(at, "slate")
        if at.session_state["session"].is_over():
            break
    assert at.session_state["session"].is_over()
    assert not at.exception
