from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.race import RaceSession
from src.wordlist import load_answers, load_guesses

APP = str(Path(__file__).resolve().parent.parent / "app.py")
A, G = load_answers(), load_guesses()


def button(at, text):
    return [b for b in at.button if text in b.label][0]


def page_text(at):
    return " ".join(m.value for m in at.markdown)


def start_game(at=None):
    at = at or AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    button(at, "Start Playing").click()
    at = at.run()
    assert not at.exception
    return at


def force_secret(at, secret="wound", strategy="entropy"):
    pm = at.session_state["session"]._ai_strategy.pm
    at.session_state["session"] = RaceSession(A, G, strategy_name=strategy, secret=secret, pm=pm)
    return at.run()


def guess(at, word):
    at.text_input[0].set_value(word)
    button(at, "Guess").click()
    return at.run()


def test_start_page_then_game():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "WORDLE" in page_text(at) and "How to play" in page_text(at)
    assert not at.text_input                       # no board / input on the start page
    at = start_game(at)
    assert at.text_input and "Words still possible" in page_text(at)


def test_home_button_returns_to_start():
    at = start_game()
    button(at, "🏠").click()
    at = at.run()
    assert "How to play" in page_text(at) and not at.text_input


def test_play_win_flow():
    at = force_secret(start_game())
    at = guess(at, "crane")
    assert not at.exception and len(at.session_state["session"].player_history) == 1
    at = guess(at, "wound")
    assert "Solved in 2 guesses" in page_text(at)


def test_invalid_word_warns_and_is_not_counted():
    at = force_secret(start_game())
    at = guess(at, "zzzzz")
    assert "not in the word list" in page_text(at)
    assert at.session_state["session"].player_history == []
    at = guess(at, "abc")
    assert "5-letter word" in page_text(at)


def test_hint_button():
    at = force_secret(start_game())
    button(at, "Hint").click()
    at = at.run()
    assert "The AI would guess" in page_text(at)


def test_new_button_resets():
    at = force_secret(start_game())
    at = guess(at, "crane")
    button(at, "New").click()
    at = at.run()
    assert at.session_state["session"].player_history == []


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
