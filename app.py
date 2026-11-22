"""Streamlit UI: a start page, then Wordle you can play or race against the AI (with a hint button).

Run from the project folder:   streamlit run app.py
Colours: lavender (#B1A9DA) + pale lemon (#FFEDA8), gingham-and-stars background.
Tiles:   green = right letter, right spot | yellow = right letter, wrong spot | red = not in the word
"""
import base64
import random
from pathlib import Path

import streamlit as st

from src.adversarial import AdversarySession
from src.game import ALL_GREEN, MAX_GUESSES
from src.patterns import PatternMatrix
from src.race import STRATEGY_NAMES, RaceSession
from src.wordlist import load_answers, load_guesses

st.set_page_config(page_title="Wordle Solver", page_icon="⭐", layout="centered")

ROOT = Path(__file__).resolve().parent
PLAY, RACE, ABSURDLE = "Play", "Race vs AI", "Absurdle"
MODES = [PLAY, RACE, ABSURDLE]
CLASS = {0: "r", 1: "y", 2: "g"}      # pattern value -> CSS class (red / yellow / green)
MAX_HINTS = 3                           # per game
DRAFT, BLANK = 5, " "                   # a guess row is 5 slots; BLANK holds a letter's place open

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
.tile.hint { background:#DDF2D9; border-color:#6AAA64; color:#3B3560; }   /* a letter a hint filled in */
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
.st-key-hint_btn button:disabled, .st-key-hint_btn button:disabled:hover {
  background:#DCD8E8 !important; color:#9A93B8 !important; border-color:#fff !important;
  box-shadow:0 3px 0 #C3BED4 !important; transform:none !important; }
.st-key-start_btn { padding-bottom:.5rem; }
.st-key-start_btn button:hover { transform:translateY(-4px) scale(1.04); }
.st-key-start_btn button:active { transform:translateY(4px); box-shadow:0 2px 0 #B1A9DA; }
[data-testid="stRadio"] label p, [data-testid="stRadio"] label div { font-weight:700; color:#3B3560; }

/* ---------- fun extras ---------- */
.row.shake { animation:shake .4s; }
@keyframes shake { 0%,100% { transform:translateX(0); } 20% { transform:translateX(-7px); } 40% { transform:translateX(7px); }
                   60% { transform:translateX(-5px); } 80% { transform:translateX(5px); } }
.tile.win { animation:flip .5s ease both, bounce .6s ease both; }
@keyframes bounce { 0%,100% { transform:translateY(0); } 40% { transform:translateY(-14px); } 60% { transform:translateY(-6px); } }
.confetti { position:fixed; inset:0; pointer-events:none; overflow:hidden; z-index:9999; }
.confetti span { position:absolute; top:-50px; font-size:1.7rem; animation:fall 3s linear forwards; }
@keyframes fall { to { transform:translateY(115vh) rotate(420deg); } }
.bubble { background:#EFEBFA; border-radius:12px; padding:.35rem .6rem; font-size:.78rem; color:#3B3560;
          text-align:center; margin-top:-.4rem; }
.stats { display:flex; justify-content:space-around; text-align:center; margin-bottom:.7rem; color:#3B3560; }
.stat { font-size:.75rem; color:#6c6591; }
.stat b { display:block; font-size:1.5rem; color:#3B3560; }
.bar-row { display:flex; align-items:center; gap:.4rem; margin:.15rem 0; font-size:.8rem; color:#3B3560; }
.bar-row span { width:1rem; text-align:center; font-weight:700; }
.bar { color:#fff; text-align:right; padding:0 .4rem; border-radius:4px; font-weight:700; min-width:1.6rem; }
[data-testid="stPopover"] button { border-radius:999px; font-weight:800; background:#B1A9DA; color:#3B3560;
                                   border:2px solid #fff; box-shadow:0 3px 0 #8f86c4; }
/* ---------- 3D: block tiles, flip-reveal hero, floating mascots ---------- */
.row { perspective:600px; }
.tile { box-shadow:0 4px 0 rgba(90,70,160,.28), 0 7px 10px rgba(60,40,120,.18); }
.tile.g { box-shadow:0 4px 0 #4d8548, 0 7px 10px rgba(40,90,40,.3); }
.tile.y { box-shadow:0 4px 0 #b8891a, 0 7px 10px rgba(130,90,10,.3); }
.tile.r { box-shadow:0 4px 0 #b83b3f, 0 7px 10px rgba(130,30,35,.3); }
.tile.typed { box-shadow:0 4px 0 #3B3560, 0 7px 10px rgba(60,40,120,.25); }
.small .tile { box-shadow:0 3px 0 rgba(90,70,160,.28); }
.small .tile.g { box-shadow:0 3px 0 #4d8548; } .small .tile.y { box-shadow:0 3px 0 #b8891a; }
.small .tile.r { box-shadow:0 3px 0 #b83b3f; }

.sr { position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); }
.hero { display:flex; gap:7px; justify-content:center; margin:.6rem 0 .2rem 0; perspective:700px; }
.hero .face { width:46px; height:52px; display:flex; align-items:center; justify-content:center;
              font-size:1.9rem; font-weight:800; color:#fff; border-radius:9px; box-sizing:border-box;
              transform-style:preserve-3d; background:#fff; color:#3B3560; border:2px solid #B1A9DA;
              animation:heroflip .9s cubic-bezier(.3,1.3,.5,1) both, bob 3s ease-in-out infinite; }
.hero .face.g { --c:#6AAA64; --d:#4d8548; } .hero .face.y { --c:#E5B22A; --d:#b8891a; }
.hero .face.r { --c:#E4575B; --d:#b83b3f; }
@keyframes heroflip {
  0%   { transform:rotateX(0); background:#fff; color:#3B3560; box-shadow:0 4px 0 #B1A9DA; }
  49%  { transform:rotateX(90deg); background:#fff; color:#3B3560; }
  50%  { transform:rotateX(90deg); background:var(--c); color:#fff; border-color:var(--c); }
  100% { transform:rotateX(0); background:var(--c); color:#fff; border-color:var(--c); box-shadow:0 5px 0 var(--d), 0 10px 14px rgba(60,40,120,.25); } }
@keyframes bob { 0%,100% { translate:0 0; } 50% { translate:0 -7px; } }
.hero-sub { text-align:center; font-weight:800; letter-spacing:.35em; color:#3B3560; margin:.5rem 0 0 0; }
.hero-sub span { background:#FFEDA8; padding:0 .4rem; border-radius:8px; }
.mascot { display:block; margin:.9rem auto 0 auto; width:170px; max-width:60%; animation:float 3.6s ease-in-out infinite;
          filter:drop-shadow(0 8px 6px rgba(60,40,120,.25)); }
.robot-slot { height:64px; margin-bottom:.1rem; }   /* fixed slot: the robot sits in one, the other is empty */
.robot-slot .robot { display:block; width:64px; height:64px; margin:0 auto; object-fit:contain;
         animation:float 2.8s ease-in-out infinite; filter:drop-shadow(0 5px 4px rgba(60,40,120,.25)); }
@media (max-width: 480px) { .robot-slot { display:none; } }   /* columns stack on phones */
@keyframes float { 0%,100% { transform:translateY(0) rotate(-1.5deg); } 50% { transform:translateY(-9px) rotate(1.5deg); } }
@media (prefers-reduced-motion: reduce) { .hero .face, .mascot, .robot { animation:none; }
  .hero .face { background:var(--c); color:#fff; border-color:var(--c); } }
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


@st.cache_resource
def image_uri(name: str) -> str:
    path = ROOT / "assets" / name
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def mascot_html(name, css_class, alt):
    uri = image_uri(name)
    return f'<img class="{css_class}" src="{uri}" alt="{alt}">' if uri else ""


# ---------- HTML helpers ----------
def board_html(history, current="", show_letters=True, animate_last=False, small=False,
               shake=False, win_last=False, rows=MAX_GUESSES, hinted=None):
    out = [f'<div class="pad"><div class="board{" small" if small else ""}">']
    for r in range(rows):
        typing = r == len(history) and bool(current)
        out.append(f'<div class="row{" shake" if typing and shake else ""}">')
        if r < len(history):
            word, pattern = history[r]
            pop = animate_last and r == len(history) - 1
            for i, (ch, p) in enumerate(zip(word, pattern)):
                cls, style = f"tile {CLASS[p]}", ""
                if pop:
                    cls += " pop"
                    if win_last:                                    # flip, then bounce
                        cls += " win"
                        style = f' style="animation-delay:{i * 0.12:.2f}s,{0.7 + i * 0.1:.2f}s"'
                    else:
                        style = f' style="animation-delay:{i * 0.12:.2f}s"'
                out.append(f'<div class="{cls}"{style}>{ch if show_letters else ""}</div>')
        elif typing:                                                # the row being typed
            for i in range(5):
                ch = current[i] if i < len(current) else ""
                if ch.strip():
                    out.append(f'<div class="tile typed{" hint" if i in (hinted or {}) else ""}">{ch}</div>')
                else:
                    out.append('<div class="tile"></div>')
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


# ---------- fun extras: praise, confetti, AI note, share grid, stats ----------
PRAISE = {1: "Genius! ⭐", 2: "Magnificent! ✨", 3: "Impressive! 🎉", 4: "Splendid! 💜",
          5: "Great! 💛", 6: "Phew, just in time! 😅"}
AI_STYLE = {"random": ("🎲 Random", "picked any word that still fits"),
            "frequency": ("🔤 Frequency", "went for the most common letters"),
            "minimax": ("🛡️ Minimax", "played safe against the worst case"),
            "entropy": ("🧠 Entropy", "picked the most informative word")}
SHARE_EMOJI = {0: "🟥", 1: "🟨", 2: "🟩"}


def solved(hist):
    return bool(hist) and hist[-1][1] == ALL_GREEN


def confetti_html():
    rnd = random.Random()
    bits = [f'<span style="left:{rnd.randint(0, 96)}%;animation-delay:{rnd.random() * 1.2:.2f}s;'
            f'animation-duration:{2.4 + rnd.random() * 1.6:.2f}s">{rnd.choice(["⭐", "✨", "💜", "💛", "🌟"])}</span>'
            for _ in range(26)]
    return '<div class="confetti">' + "".join(bits) + "</div>"


def ai_note_html(session):
    """What the AI just did, in numbers (never shows its letters)."""
    sizes = session.ai_sizes
    if len(sizes) < 2:
        return ""
    name, how = AI_STYLE[session.strategy_name]
    return f'<div class="bubble"><b>{name}</b> · {sizes[-2]} → {sizes[-1]} words left<br><i>{how}</i></div>'


def absurdle_note_html(session):
    """How many words the adversary is still holding on to."""
    if not getattr(session, "adversarial", False) or not session.player_history or session.is_over():
        return ""
    return (f'<div class="bubble">🧱 Absurdle still has <b>{session.remaining:,}</b> words '
            f'that fit your clues</div>')


def share_text(session, race):
    hist = session.player_history
    ok = solved(hist)
    title = f"Wordle Solver {len(hist) if ok else 'X'}/6"
    lines = [title + (f" vs {AI_STYLE[session.strategy_name][0]} AI" if race else "")]
    if race:
        lines.append({"player": "🏆 I won the race!", "ai": "🤖 The AI won", "draw": "🤝 Draw"}[session.winner])
    lines.append("")
    lines += ["".join(SHARE_EMOJI[p] for p in pattern) for _, pattern in hist]
    return "\n".join(lines)


def new_stats():
    return {"played": 0, "wins": 0, "streak": 0, "best": 0,
            "dist": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, "fail": 0}}


def stats_html(s):
    pct = round(100 * s["wins"] / s["played"]) if s["played"] else 0
    top = max(s["dist"].values()) or 1
    bars = []
    for k in (1, 2, 3, 4, 5, 6, "fail"):
        v = s["dist"][k]
        width = max(8, round(100 * v / top)) if v else 8
        color = "#E4575B" if k == "fail" else "#6AAA64"
        bars.append(f'<div class="bar-row"><span>{"✗" if k == "fail" else k}</span>'
                    f'<div class="bar" style="width:{width}%;background:{color}">{v}</div></div>')
    return ('<div class="stats">'
            f'<div class="stat"><b>{s["played"]}</b>Played</div>'
            f'<div class="stat"><b>{pct}%</b>Won</div>'
            f'<div class="stat"><b>{s["streak"]}</b>Streak</div>'
            f'<div class="stat"><b>{s["best"]}</b>Best</div></div>'
            '<div class="label">Guesses per win</div>' + "".join(bars))


# ---------- state and callbacks ----------
answers, pool, pm = load_data()
ss = st.session_state

HINT_TIP = (f"{MAX_HINTS} hints per game. Each one fills a single letter into its correct box; "
            "the rest of the word is still up to you.")


def player_done(session):
    hist = session.player_history
    return bool(hist) and (hist[-1][1] == ALL_GREEN or len(hist) >= MAX_GUESSES)


def is_over():
    """In Play the player's own board decides; race and Absurdle have their own end state."""
    return player_done(ss.session) if ss.mode == PLAY else ss.session.is_over()


def new_game():
    if ss.mode == ABSURDLE:
        ss.session = AdversarySession(answers, pool, pm)      # no secret word: the adversary decides
    else:
        strategy = ss.strategy if ss.mode == RACE else "entropy"
        ss.session = RaceSession(answers, pool, strategy_name=strategy,
                                 seed=random.randrange(10 ** 9), pm=pm)
    ss.hint, ss.msg, ss.hints_used, ss.current, ss.hinted = None, None, 0, "", {}
    ss.celebrated, ss.counted = False, False


def record_result():
    """Add a finished game to the session stats (once per game)."""
    if ss.counted or not is_over():
        return
    ss.counted = True
    stats, hist = ss.stats, ss.session.player_history
    stats["played"] += 1
    if solved(hist):
        stats["wins"] += 1
        stats["streak"] += 1
        stats["best"] = max(stats["best"], stats["streak"])
        stats["dist"][len(hist)] += 1
    else:
        stats["streak"] = 0
        stats["dist"]["fail"] += 1


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


def draft_slots():
    """The row being typed as 5 slots; hint letters sit in their own slot, gaps stay BLANK."""
    return list(ss.current.ljust(DRAFT)[:DRAFT])


def set_draft(slots):
    """Save the slots as text. Inner blanks keep the later letters in place; trailing ones are dropped."""
    ss.current = "".join(slots).rstrip()


def press_key(ch):
    ss.msg = None
    if is_over():
        return
    slots = draft_slots()
    for i in range(DRAFT):                             # fill the leftmost free slot
        if not slots[i].strip():
            slots[i] = ch
            break
    set_draft(slots)


def press_back():
    ss.msg = None
    slots = draft_slots()
    for i in range(DRAFT - 1, -1, -1):                 # a hint letter stays: it was paid for
        if slots[i].strip() and i not in ss.hinted:
            slots[i] = BLANK
            break
    set_draft(slots)


def press_enter():
    session, race = ss.session, ss.mode == RACE
    ss.msg = None
    if is_over():
        return
    word = ss.current.replace(BLANK, "")
    if BLANK in ss.current or len(word) < DRAFT:
        ss.msg = ("warn", "Not enough letters.")
        return
    try:
        session.player_guess(word)
    except ValueError:
        ss.msg = ("warn", f"'{word.upper()}' is not in the wordle dictionary.")   # keep the letters so they can be fixed
        return
    except RuntimeError:
        return
    ss.current, ss.hint, ss.hinted = "", None, {}
    if race:
        if not session.is_over():
            session.ai_step()
        while player_done(session) and not session.is_over():          # player is out of guesses
            session.ai_step()
    record_result()


def give_hint():
    """Spend one hint to lock the next wrong letter into its true position (Play and Race only)."""
    ss.msg = None
    if is_over() or ss.hints_used >= MAX_HINTS or ss.mode == ABSURDLE:
        return
    secret = ss.session.secret
    slots = draft_slots()
    pos = next((i for i in range(DRAFT) if slots[i] != secret[i]), None)
    if pos is None:                                   # the row already holds every letter
        ss.msg = ("ok", "Every letter is already in the right spot — just press Enter!")
        return
    slots[pos] = secret[pos]
    set_draft(slots)
    ss.hinted[pos] = secret[pos]
    ss.hints_used += 1
    ss.hint = f"Hint: the letter in spot {pos + 1} is '{secret[pos].upper()}' — filled in for you."


if "page" not in ss:
    ss.page, ss.mode, ss.strategy = "start", PLAY, "entropy"
    ss.session, ss.hint, ss.msg, ss.hints_used, ss.current = None, None, None, 0, ""
    ss.hinted, ss.celebrated, ss.counted, ss.stats = {}, False, False, new_stats()

st.markdown(CSS.replace("__BG__", background_uri()), unsafe_allow_html=True)


# ---------- start page ----------
HERO = [("W", "g"), ("O", "y"), ("R", "r"), ("D", "g"), ("L", "y"), ("E", "r")]


def hero_html():
    """WORDLE in six 3D tiles that flip one by one, then float."""
    faces = "".join(f'<div class="face {c}" aria-hidden="true" style="animation-delay:{0.25 + i * 0.22:.2f}s,'
                    f'{1.6 + i * 0.2:.2f}s">{ch}</div>' for i, (ch, c) in enumerate(HERO))
    return ('<div class="sr">WORDLE SOLVER</div>' f'<div class="hero">{faces}</div>'
            '<div class="hero-sub"><span>SOLVER</span></div>')


def start_page():
    st.markdown(
        '<div class="stars">⭐ ✨ ⭐</div>'
        + hero_html() + mascot_html("owl.png", "mascot", "Wizard owl mascot") +
        '<div class="subtitle">Guess the hidden word in 6 tries, or race an AI that thinks in entropy.</div>'
        '<div class="how"><h4>How to play</h4>'
        '<div class="how-row"><div class="tile mini g">C</div><div>Right letter, right spot</div></div>'
        '<div class="how-row"><div class="tile mini y">R</div><div>Right letter, wrong spot</div></div>'
        '<div class="how-row"><div class="tile mini r">E</div><div>Letter is not in the word</div></div>'
        '</div><div class="pad"></div>',
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
    player_hist, ai_hist = session.player_history, session.ai_history
    won = over and solved(player_hist)
    celebrate = won and not ss.celebrated          # confetti only on the run that first shows the win
    if celebrate:
        ss.celebrated = True
    shake = bool(ss.msg) and ss.msg[0] == "warn"

    c_home, c_stats, c_mode, c_ai = st.columns([0.9, 0.9, 3.0, 2.1], vertical_alignment="center")
    c_home.button("🏠", on_click=go_home, key="home", help="Back to the start page")
    with c_stats.popover("📊", help="Your stats (this visit)"):
        st.markdown(stats_html(ss.stats), unsafe_allow_html=True)
    c_mode.radio("Mode", MODES, index=MODES.index(ss.mode), key="mode_w", horizontal=True,
                 on_change=set_mode, label_visibility="collapsed")
    if race:
        c_ai.selectbox("AI strategy", STRATEGY_NAMES, index=STRATEGY_NAMES.index(ss.strategy), key="strategy_w",
                       on_change=set_strategy, format_func=str.capitalize, label_visibility="collapsed")

    if race:
        robot = mascot_html("robot.png", "robot", "AI robot")
        empty = '<div class="robot-slot"></div>'      # as tall as the robot, so both labels line up
        left, right = st.columns(2)
        with left:
            st.markdown(empty + '<div class="label">You</div>'
                        + board_html(player_hist, current=ss.current, animate_last=True, small=True,
                                     shake=shake, win_last=won, hinted=ss.hinted), unsafe_allow_html=True)
        with right:
            st.markdown(f'<div class="robot-slot">{robot}</div>'
                        + f'<div class="label">AI ({session.strategy_name})</div>'
                        + board_html(ai_hist, show_letters=session.is_over(), small=True)
                        + ai_note_html(session), unsafe_allow_html=True)
    else:
        st.markdown(board_html(player_hist, current=ss.current, animate_last=True, shake=shake, win_last=won,
                               hinted=ss.hinted), unsafe_allow_html=True)
        if ss.mode == ABSURDLE:
            st.markdown(absurdle_note_html(session), unsafe_allow_html=True)

    # one status line: result > warning > hint (empty otherwise). Key colours and confetti ride along.
    extra = key_status_css(player_hist) + (confetti_html() if celebrate else "")
    if over:
        secret = session.secret.upper()
        if ss.mode == ABSURDLE:
            if won:
                kind, text = ("ok", f"🎉 You beat Absurdle in {len(player_hist)} "
                                     f"guess{'es' if len(player_hist) > 1 else ''}! It settled on {secret}.")
            else:
                left = session.remaining
                shown = ", ".join(w.upper() for w in session.candidates[:4])
                more = " …" if left > 4 else ""
                kind, text = ("err", f"🧱 Absurdle survived all {MAX_GUESSES} guesses — "
                                     f"{left} words still fit ({shown}{more}).")
        elif race:
            kind, text = {"player": ("ok", f"🎉 You win the race! The word was {secret}."),
                          "ai": ("err", f"🤖 The AI wins this one. The word was {secret}."),
                          "draw": ("info", f"🤝 Draw, nobody solved it. The word was {secret}.")}[session.winner]
        elif won:
            n = len(player_hist)
            kind, text = "ok", f"🎉 {PRAISE[n]} Solved in {n} guess{'es' if n > 1 else ''}!"
        else:
            kind, text = "err", f"Out of guesses. The word was {secret}."
        st.markdown(status_html(kind, text, extra_css=extra), unsafe_allow_html=True)
    elif ss.msg:
        st.markdown(status_html(ss.msg[0], ss.msg[1], extra_css=extra), unsafe_allow_html=True)
    elif ss.hint:
        st.markdown(status_html("info", f"💡 {ss.hint}", HINT_TIP, extra), unsafe_allow_html=True)
    else:
        st.markdown(status_html("", "", extra_css=extra), unsafe_allow_html=True)

    keyboard(over)

    # Hints need a fixed secret word, which Absurdle does not have.
    hint_off = over or ss.hints_used >= MAX_HINTS or ss.mode == ABSURDLE
    hint_help = (f"Reveals one letter in its correct spot ({MAX_HINTS} per game)." if ss.mode != ABSURDLE
                 else "Absurdle has no fixed word, so there is no letter to reveal.")
    _, c_hint, c_new, _ = st.columns([1, 1.6, 1.6, 1])
    c_hint.button(f"💡 Hint {ss.hints_used}/{MAX_HINTS}", key="hint_btn", on_click=give_hint,
                  disabled=hint_off, use_container_width=True, help=hint_help)
    c_new.button("🔄 New", key="new_btn", on_click=new_game, use_container_width=True)

    if over:
        st.caption("📋 Copy your result to share it:")
        st.code(share_text(session, race), language=None)


if ss.page == "start":
    start_page()
else:
    game_page()
