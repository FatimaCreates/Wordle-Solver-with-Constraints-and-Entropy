"""Absurdle: every AI strategy against the adversary, under the normal 6-guess rule.

    python run_absurdle.py          -> results/absurdle.csv

Absurdle has no fixed secret word.  After each guess the adversary shows the
colour pattern that keeps the LARGEST group of words alive, so it can never be
caught out; the player still gets only MAX_GUESSES (6) guesses, like Wordle.
Because the adversary survives almost anything, the interesting numbers are the
solve rate and how many words it still had left at the end.

`--sample N` plays N random answers (seeded) instead of all 2,315, which keeps
the run short; the same sample is used for every strategy.
"""
import argparse
import random
import time
from pathlib import Path

import pandas as pd

from src.adversarial import AdversarySession
from src.game import MAX_GUESSES
from src.patterns import PatternMatrix
from src.race import make_strategy
from src.wordlist import load_answers, load_guesses

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
STRATEGIES = ("random", "frequency", "minimax", "entropy")


def play(chooser, answers, pool, pm, limit=MAX_GUESSES):
    """One Absurdle game with a given chooser. Returns the finished session."""
    s = AdversarySession(answers, pool, pm, max_guesses=limit)
    while not s.is_over():
        s.player_guess(chooser.choose_guess(s.candidates, list(pool)))
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=300, help="answers to play (0 = all of them)")
    parser.add_argument("--limit", type=int, default=MAX_GUESSES, help="guess limit per game")
    args = parser.parse_args()

    answers, pool = load_answers(), load_guesses()
    pm = PatternMatrix.load_or_build(pool, answers)
    secrets = answers if args.sample <= 0 else random.Random(0).sample(answers, args.sample)

    rows = []
    for name in STRATEGIES:
        chooser = make_strategy(name, answers, pool, pm)
        t = time.time()
        solved, used, left = 0, [], []
        for secret in secrets:
            s = play(chooser, answers, pool, pm, args.limit)
            left.append(s.remaining)
            if s.player_history and s.player_history[-1][1] == (2, 2, 2, 2, 2):
                solved += 1
                used.append(len(s.player_history))
        rows.append({"strategy": name, "games": len(secrets), "limit": args.limit,
                     "solved": solved, "solve_rate_pct": round(100 * solved / len(secrets), 2),
                     "avg_guesses_when_solved": round(sum(used) / len(used), 2) if used else None,
                     "avg_words_left": round(sum(left) / len(left), 1),
                     "min_words_left": min(left),
                     "seconds": round(time.time() - t, 1)})

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "absurdle.csv", index=False)
    print(f"AI vs Absurdle ({len(secrets)} answers, {args.limit}-guess limit)\n")
    print(table.to_string(index=False))
    print(f"\nSaved {OUT / 'absurdle.csv'}")


if __name__ == "__main__":
    main()
