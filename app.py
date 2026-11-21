"""Streamlit UI: a start page, then Wordle you can play or race against the AI (with a hint button).

Run from the project folder:   streamlit run app.py
Colours: lavender (#B1A9DA) + pale lemon (#FFEDA8), gingham-and-stars background.
Tiles:   green = right letter, right spot | yellow = right letter, wrong spot | red = not in the word
"""
import base64
import random
from pathlib import Path

import streamlit as st

from src.game import ALL_GREEN, MAX_GUESSES
from src.patterns import PatternMatrix
from src.race import STRATEGY_NAMES, RaceSession
from src.wordlist import load_answers, load_guesses

st.set_page_config(page_title="Wordle Solver", page_icon="⭐", layout="centered")

ROOT = Path(__file__).resolve().parent
PLAY, RACE = "Play", "Race vs AI"
CLASS = {0: "r", 1: "y", 2: "g"}      # pattern value -> CSS class (red / yellow / green)

CSS = """
<style>
/* ---------- page: gingham + stars background, content in a soft card ---------- */
.stApp { background: #FFF6D6 url("__BG__") repeat; background-size: 360px auto; }
header[data-testid="stHeader"], #MainMenu, footer { display: none !important; }
.block-container, [data-testid="stMainBlockContainer"] {
  max-width: 640px !important; margin: 0.8rem auto 0 auto !important;
  padding: 1rem 1.4rem 1.4rem 1.4rem !important;
  background: rgba(255, 251, 230, 0.93); border: 4px solid #fff; border-radius: 26px;
  box-shadow: 0 8px 30px rgba(90, 70, 160, 0.25); }
[data-testid="stVerticalBlock"] { gap: 0.3rem; }
.pad { padding-bottom: 1.1rem; }

/* ---------- text ---------- */
.title { text-align:center; font-weight:800; color:#3B3560; letter-spacing:.06em; font-size:2rem; margin:0; }
.title span { background:#FFEDA8; padding:0 .4rem; border-radius:8px; }
.title.big { font-size:2.7rem; margin-top:.4rem; }
.subtitle { text-align:center; color:#6c6591; margin:.2rem 0 .8rem 0; }
.stars { text-align:center; font-size:1.6rem; letter-spacing:.4rem; }
.how { background:#EFEBFA; border-radius:18px; padding:.8rem 1rem; margin:.8rem auto 1.1rem auto; max-width:420px; }
.how h4 { margin:0 0 .5rem 0; color:#3B3560; text-align:center; }
.how-row { display:flex; align-items:center; gap:.7rem; margin:.35rem 0; color:#3B3560; }
.tip { text-align:center; color:#6c6591; font-size:.85rem; margin-top:.5rem; }

/* ---------- board ---------- */
.board { margin:.1rem auto .2rem auto; }
.row { display:flex; gap:5px; justify-content:center; margin-bottom:5px; }
.tile { width:50px; height:50px; display:flex; align-items:center; justify-content:center;
        font-size:1.6rem; font-weight:800; color:#fff; text-transform:uppercase;
        border:2px solid #B1A9DA; border-radius:7px; background:rgba(255,255,255,.55); box-sizing:border-box; }
.small .tile { width:38px; height:38px; font-size:1.2rem; border-radius:6px; }
.tile.typed { background:#fff; border-color:#3B3560; color:#3B3560; }
.tile.mini { width:34px; height:34px; font-size:1.1rem; flex:none; }
.tile.g { background:#6AAA64; border-color:#6AAA64; }
.tile.y { background:#E5B22A; border-color:#E5B22A; }
.tile.r { background:#E4575B; border-color:#E4575B; }
.tile.pop { animation:flip .5s ease both; }
@keyframes flip { 0% { transform:rotateX(0); } 50% { transform:rotateX(90deg); } 100% { transform:rotateX(0); } }
.label { text-align:center; font-weight:700; color:#3B3560; font-size:.9rem; margin-bottom:.2rem; }

/* ---------- status line (one fixed-height slot, so nothing jumps) ---------- */
.status { height:34px; display:flex; align-items:center; justify-content:center; text-align:center;
          border-radius:12px; font-size:.9rem; font-weight:600; color:#3B3560; padding:0 .6rem; }
.status.plain { color:#7a739f; font-weight:500; }
.status.ok   { background:#DDF2D9; }
.status.err  { background:#FBDADB; }
.status.info { background:#FFEDA8; }
.status.warn { background:#FFE3C2; }

/* ---------- keyboard (real buttons, so the keys can be clicked) ---------- */
.st-key-kb [data-testid="stHorizontalBlock"] { flex-wrap:nowrap !important; gap:.25rem !important; }
.st-key-kb [data-testid="stColumn"] { min-width:0 !important; }
.st-key-kb [data-testid="stVerticalBlock"] { gap:.3rem; }
.st-key-kb .stButton button { height:42px; min-height:42px; padding:0; width:100%; border-radius:8px;
                    background:#B1A9DA; color:#3B3560; border:none; box-shadow:0 3px 0 #8f86c4; }
.st-key-kb .stButton button p { font-size:.95rem; font-weight:800; margin:0; }
.st-key-k_enter button p { font-size:.62rem; }
.st-key-kb .stButton button:hover { transform:translateY(-1px); color:#3B3560; border:none; }
.st-key-kb .stButton button:active { transform:translateY(2px); box-shadow:0 1px 0 #8f86c4; }
.st-key-kb .stButton button:disabled { opacity:.55; }

/* ---------- cute buttons ---------- */
.stButton button, .stFormSubmitButton button {
  border-radius:999px; font-weight:800; color:#3B3560; background:#B1A9DA;
  border:2px solid #fff; box-shadow:0 3px 0 #8f86c4; transition:transform .12s, box-shadow .12s; }
.stButton button:hover, .stFormSubmitButton button:hover { transform:translateY(-2px); color:#3B3560; border-color:#fff; }
.stButton button:active, .stFormSubmitButton button:active { transform:translateY(2px); box-shadow:0 1px 0 #8f86c4; }
.stFormSubmitButton button[kind="primaryFormSubmit"], [data-testid="stBaseButton-primaryFormSubmit"] {
  background:#FFEDA8; box-shadow:0 3px 0 #e0c24f; }
.st-key-start_btn button {
  background:linear-gradient(135deg, #FFEDA8, #FFD86B); font-size:1.3rem; padding:.7rem 1rem;
  border:3px solid #fff; box-shadow:0 6px 0 #B1A9DA, 0 12px 20px rgba(80,60,150,.28); }
.st-key-start_btn { padding-bottom:.5rem; }
.st-key-start_btn button:hover { transform:translateY(-4px) scale(1.04); }
.st-key-start_btn button:active { transform:translateY(4px); box-shadow:0 2px 0 #B1A9DA; }
[data-testid="stRadio"] label p, [data-testid="stRadio"] label div { font-weight:700; color:#3B3560; }
</style>
"""


@st.cache_resource(show_spinner="Loading word lists and pattern matrix...")
def load_data():
    answers, pool = load_answers(), load_guesses()
    return answers, pool, PatternMatrix.load_or_build(pool, answers)


@st.cache_resource
def background_uri() -> str:
    path = ROOT / "assets" / "bg_tile.jpg"
    if not path.exists():
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


# ---------- HTML helpers ----------
def board_html(history, current="", show_letters=True, animate_last=False, small=False, rows=MAX_GUESSES):
    out = [f'<div class="pad"><div class="board{" small" if small else ""}">']
    for r in range(rows):
        out.append('<div class="row">')
        if r < len(history):
            word, pattern = history[r]
            pop = animate_last and r == len(history) - 1
            for i, (ch, p) in enumerate(zip(word, pattern)):
                style = f' style="animation-delay:{i * 0.12:.2f}s"' if pop else ""
                letter = ch if show_letters else ""
                out.append(f'<div class="tile {CLASS[p]}{" pop" if pop else ""}"{style}>{letter}</div>')
        elif r == len(history) and current:                      # the row being typed
            for i in range(5):
                ch = current[i] if i < len(current) else ""
                out.append(f'<div class="tile{" typed" if ch else ""}">{ch}</div>')
        else:
            out.append('<div class="tile"></div>' * 5)
        out.append("</div>")
    out.append("</div></div>")
    return "".join(out)


KEY_COLORS = {0: ("#E4575B", "#b83b3f"), 1: ("#E5B22A", "#b8891a"), 2: ("#6AAA64", "#4d8548")}


def key_status_css(history):
    """CSS that colours each on-screen key by the best clue seen for its letter (green > yellow > red)."""
    status = {}
    for word, pattern in history:
        for ch, p in zip(word, pattern):
            status[ch] = max(status[ch], p) if ch in status else p
    rules = []
    for ch, p in status.items():
        bg, shadow = KEY_COLORS[p]
        rules.append(f".st-key-k_{ch} button {{ background:{bg} !important; color:#fff !important; "
                     f"box-shadow:0 3px 0 {shadow} !important; }}"
                     f".st-key-k_{ch} button p {{ color:#fff !important; }}")
    return "<style>" + "".join(rules) + "</style>"


def status_html(kind, text, tip="", extra_css=""):
    tip_attr = f' title="{tip}"' if tip else ""
    body = f'<div class="status {kind}"{tip_attr}>{text}</div>' if text else '<div class="status"></div>'
    return f'{extra_css}<div class="pad">{body}</div>'


# ---------- state and callbacks ----------
answers, pool, pm = load_data()
ss = st.session_state

HINT_TIP = ("A hint is not always a possible answer. Sometimes it is chosen only because it splits "
            "the remaining words as well as possible.")


def player_done(session):
    hist = session.player_history
    return bool(hist) and (hist[-1][1] == ALL_GREEN or len(hist) >= MAX_GUESSES)


def is_over():
    return ss.session.is_over() if ss.mode == RACE else player_done(ss.session)


def new_game():
    strategy = ss.strategy if ss.mode == RACE else "entropy"
    ss.session = RaceSession(answers, pool, strategy_name=strategy,
                             seed=random.randrange(10 ** 9), pm=pm)
    ss.hint, ss.msg, ss.hints_used, ss.current = None, None, 0, ""


def start():
    ss.page = "game"
    new_game()


def go_home():
    ss.page = "start"


def set_mode():
    ss.mode = ss.mode_w
    new_game()


def set_strategy():
    ss.strategy = ss.strategy_w
    new_game()


def press_key(ch):
    ss.msg = None
    if not is_over() and len(ss.current) < 5:
        ss.current += ch


def press_back():
    ss.msg = None
    ss.current = ss.current[:-1]


def press_enter():
    session, race = ss.session, ss.mode == RACE
    ss.msg = None
    if is_over():
        return
    word = ss.current
    if len(word) < 5:
        ss.msg = ("warn", "Not enough letters.")
        return
    try:
        session.player_guess(word)
    except ValueError:
        ss.msg = ("warn", f"'{word.upper()}' is not in the word list.")   # keep the letters so they can be fixed
        return
    except RuntimeError:
        return
    ss.current, ss.hint = "", None
    if race:
        if not session.is_over():
            session.ai_step()
        while player_done(session) and not session.is_over():          # player is out of guesses
            session.ai_step()


def give_hint():
    ss.msg = None
    if not is_over():
        ss.hint = ss.session.hint()
        ss.hints_used += 1


if "page" not in ss:
    ss.page, ss.mode, ss.strategy = "start", PLAY, "entropy"
    ss.session, ss.hint, ss.msg, ss.hints_used, ss.current = None, None, None, 0, ""

st.markdown(CSS.replace("__BG__", background_uri()), unsafe_allow_html=True)


# ---------- start page ----------
def start_page():
    st.markdown(
        '<div class="stars">⭐ ✨ ⭐</div>'
        '<div class="title big">WORDLE <span>SOLVER</span></div>'
        '<div class="subtitle">Guess the hidden word in 6 tries, or race an AI that thinks in entropy.</div>'
        '<div class="how"><h4>How to play</h4>'
        '<div class="how-row"><div class="tile mini g">C</div><div>Right letter, right spot</div></div>'
        '<div class="how-row"><div class="tile mini y">R</div><div>Right letter, wrong spot</div></div>'
        '<div class="how-row"><div class="tile mini r">E</div><div>Letter is not in the word</div></div>'
        '</div><div class="pad"><div class="tip">Tap the keys under the board, then press ENTER.</div></div>',
        unsafe_allow_html=True)
    middle = st.columns([1, 3, 1])[1]
    with middle:
        with st.container(key="start_btn"):
            st.button("⭐ Start Playing ⭐", on_click=start, use_container_width=True)


# ---------- on-screen keyboard ----------
def keyboard(over):
    with st.container(key="kb"):
        row1 = st.columns(10)
        for col, ch in zip(row1, "qwertyuiop"):
            col.button(ch.upper(), key=f"k_{ch}", on_click=press_key, args=(ch,), disabled=over,
                       use_container_width=True)
        row2 = st.columns([0.5] + [1] * 9 + [0.5])
        for col, ch in zip(row2[1:10], "asdfghjkl"):
            col.button(ch.upper(), key=f"k_{ch}", on_click=press_key, args=(ch,), disabled=over,
                       use_container_width=True)
        row3 = st.columns([1.6] + [1] * 7 + [1.6])
        row3[0].button("ENTER", key="k_enter", on_click=press_enter, disabled=over, use_container_width=True)
        for col, ch in zip(row3[1:8], "zxcvbnm"):
            col.button(ch.upper(), key=f"k_{ch}", on_click=press_key, args=(ch,), disabled=over,
                       use_container_width=True)
        row3[8].button("⌫", key="k_back", on_click=press_back, disabled=over, use_container_width=True)


# ---------- game page ----------
def game_page():
    session, race, over = ss.session, ss.mode == RACE, is_over()

    c_home, c_mode, c_ai = st.columns([1, 3.2, 2.2], vertical_alignment="center")
    c_home.button("🏠", on_click=go_home, key="home", help="Back to the start page")
    c_mode.radio("Mode", [PLAY, RACE], index=[PLAY, RACE].index(ss.mode), key="mode_w", horizontal=True,
                 on_change=set_mode, label_visibility="collapsed")
    if race:
        c_ai.selectbox("AI strategy", STRATEGY_NAMES, index=STRATEGY_NAMES.index(ss.strategy), key="strategy_w",
                       on_change=set_strategy, format_func=str.capitalize, label_visibility="collapsed")

    player_hist, ai_hist = session.player_history, session.ai_history
    if race:
        left, right = st.columns(2)
        with left:
            st.markdown('<div class="label">You</div>'
                        + board_html(player_hist, current=ss.current, animate_last=True, small=True),
                        unsafe_allow_html=True)
        with right:
            st.markdown(f'<div class="label">AI ({session.strategy_name})</div>'
                        + board_html(ai_hist, show_letters=session.is_over(), small=True),
                        unsafe_allow_html=True)
    else:
        st.markdown(board_html(player_hist, current=ss.current, animate_last=True), unsafe_allow_html=True)

    # one status line: result > warning > hint (empty otherwise). The key colours ride along in the same block.
    keys_css = key_status_css(player_hist)
    if over:
        secret = session.secret.upper()
        if race:
            kind, text = {"player": ("ok", f"🎉 You win the race! The word was {secret}."),
                          "ai": ("err", f"🤖 The AI wins this one. The word was {secret}."),
                          "draw": ("info", f"🤝 Draw, nobody solved it. The word was {secret}.")}[session.winner]
        elif player_hist and player_hist[-1][1] == ALL_GREEN:
            kind, text = "ok", f"🎉 Solved in {len(player_hist)} guess{'es' if len(player_hist) > 1 else ''}!"
        else:
            kind, text = "err", f"Out of guesses. The word was {secret}."
        st.markdown(status_html(kind, text, extra_css=keys_css), unsafe_allow_html=True)
    elif ss.msg:
        st.markdown(status_html(ss.msg[0], ss.msg[1], extra_css=keys_css), unsafe_allow_html=True)
    elif ss.hint:
        st.markdown(status_html("info", f"💡 The AI would guess {ss.hint.upper()} here", HINT_TIP, keys_css),
                    unsafe_allow_html=True)
    else:
        st.markdown(status_html("", "", extra_css=keys_css), unsafe_allow_html=True)

    keyboard(over)

    _, c_hint, c_new, _ = st.columns([1, 1.6, 1.6, 1])
    c_hint.button("💡 Hint", key="hint_btn", on_click=give_hint, disabled=over, use_container_width=True)
    c_new.button("🔄 New", key="new_btn", on_click=new_game, use_container_width=True)


if ss.page == "start":
    start_page()
else:
    game_page()