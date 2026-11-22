"""Load the Wordle word lists from data/raw/.

Both files come from https://github.com/boompig/wordle-py (data-raw/):
    wordle-answers-future.txt   2,315 possible answers
    wordle-words.txt           12,972 accepted guesses (a superset of the answers)
`encoding="utf-8"` is explicit so the files load the same on Windows and Linux.
"""
from pathlib import Path
from typing import List

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_answers() -> List[str]:
    """2,315 answers. Each line looks like 'Jun 19 2021 Day 0 CIGAR': the word is the last token."""
    with open(RAW / "wordle-answers-future.txt", encoding="utf-8") as f:
        return [line.split()[-1].strip().lower() for line in f if line.strip()]


def load_guesses() -> List[str]:
    """12,972 valid guesses (a superset of the answers), one word per line."""
    with open(RAW / "wordle-words.txt", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]
