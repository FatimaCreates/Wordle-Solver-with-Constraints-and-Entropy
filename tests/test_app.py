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


def test_hint_fills_one_letter_and_stops_after_three():
    at = force_secret(start_game())                     # secret: wound
    at = click(at, "hint_btn")
    assert at.session_state["current"] == "w"
    assert at.session_state["hints_used"] == 1 and "in spot 1 is 'W'" in page_text(at)
    for n in (2, 3):
        at = click(at, "hint_btn")
        assert at.session_state["hints_used"] == n
    assert at.session_state["current"] == "wou"
    assert at.button(key="hint_btn").disabled                # the third hint is the last one
    assert "Hint 3/3" in at.button(key="hint_btn").label


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


# ---------- fun extras ----------
def test_win_shows_praise_confetti_once_and_share_grid():
    at = force_secret(start_game())
    at = guess(at, "crane")
    assert 'class="confetti"' not in page_text(at)
    at = guess(at, "wound")
    text = page_text(at)
    assert "Magnificent" in text and "Solved in 2 guesses" in text
    assert 'class="confetti"' in text                                   # confetti on the winning run
    assert any("Wordle Solver 2/6" in c.value for c in at.code)
    assert "🟩🟩🟩🟩🟩" in at.code[0].value
    at = at.run()                                               # any later rerun: no new confetti
    assert 'class="confetti"' not in page_text(at)


def test_stats_count_each_game_once():
    at = force_secret(start_game())
    at = guess(at, "wound")                                     # win in 1
    at = at.run()
    s = at.session_state["stats"]
    assert s["played"] == 1 and s["wins"] == 1 and s["streak"] == 1 and s["dist"][1] == 1
    at = force_secret(click(at, "new_btn"))
    for _ in range(6):
        at = guess(at, "slate")
    s = at.session_state["stats"]
    assert s["played"] == 2 and s["wins"] == 1 and s["streak"] == 0 and s["dist"]["fail"] == 1
    assert "Played" in page_text(at)


def test_wrong_word_shakes_the_row():
    at = force_secret(start_game())
    at = guess(at, "zzzzz")
    assert "row shake" in page_text(at)
    at = click(at, "k_back")
    assert "row shake" not in page_text(at)


def test_ai_note_in_race_mode_hides_letters():
    at = start_game()
    at.radio[0].set_value("Race vs AI")
    at = force_secret(at.run(), secret="wound")
    assert "words left" not in page_text(at)                    # AI has not moved yet
    at = guess(at, "crane")
    text = page_text(at)
    assert "words left" in text and "→" in text
    ai_word = at.session_state["session"].ai_history[0][0]
    assert f">{ai_word[0]}</div>" not in text                  # AI tiles stay blank while the race runs
