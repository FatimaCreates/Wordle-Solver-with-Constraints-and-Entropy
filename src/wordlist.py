"""Load the Wordle word lists (Umama can extend this with Pandas letter analysis)."""
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_answers():
    """2,315 answers. Each line looks like 'Jun 19 2021 Day 0 CIGAR': the word is the last token."""
    with open(RAW / "wordle-answers-future.txt") as f:
        return [line.split()[-1].lower() for line in f if line.strip()]


def load_guesses():
    """12,972 valid guesses (a superset of the answers), one word per line."""
    with open(RAW / "wordle-words.txt") as f:
        return [line.strip().lower() for line in f if line.strip()]
