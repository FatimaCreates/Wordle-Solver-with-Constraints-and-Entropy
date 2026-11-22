"""Run the full experiment and save the results table and charts.

    python experiment.py

Every strategy plays every one of the 2,315 answers (random: 20 repeated runs, seeds 42..61).
Global seed 42. Outputs go to results/:
    per_game.csv            one row per game
    summary.csv             the results table used in the report
    candidates_per_turn.csv average candidates left after each guess
    first_guess_table.csv   top-10 opening guesses by entropy and by minimax
    *.png                   the four charts (see charts.py)
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent import SolverAgent
from src.entropy import EntropyStrategy, entropy_scores
from src.game import MAX_GUESSES, WordleGame
from src.minimax import MinimaxStrategy, worst_case_sizes
from src.patterns import PatternMatrix
from src.strategies import FrequencyStrategy, RandomStrategy
from src.wordlist import load_answers, load_guesses

SEED = 42
RESULTS = Path(__file__).resolve().parent / "results"
ORDER = ["random", "frequency", "minimax", "entropy"]


def play_all(strategy, secrets, answers, pool, run=0):
    """Play every secret once. Returns (rows, sizes) where sizes[i] = candidates left after each guess."""
    agent = SolverAgent(strategy, answers, pool)
    rows, sizes = [], []
    for secret in secrets:
        game = WordleGame(secret)
        start = time.perf_counter()
        guesses = agent.play(game)
        seconds = time.perf_counter() - start
        rows.append({"strategy": strategy.name, "run": run, "answer": secret,
                     "n_guesses": game.guesses_used, "solved": game.is_solved(),
                     "seconds": seconds, "first_guess": guesses[0], "checks": agent.last_checks})
        sizes.append(list(agent.last_sizes))
    return rows, sizes


def candidates_per_turn(strategy_name, sizes):
    """Average candidates after each guess (turn 0 = before any guess). A finished game keeps its last size."""
    padded = np.array([s + [s[-1]] * (MAX_GUESSES + 1 - len(s)) for s in sizes], dtype=float)
    return pd.DataFrame({"strategy": strategy_name, "turn": range(MAX_GUESSES + 1),
                         "avg_candidates": padded.mean(axis=0)})


def first_guess_table(answers, pool, pm):
    counts = pm.pattern_counts(answers)
    bits = entropy_scores(counts, len(answers))
    worst = worst_case_sizes(counts)
    rows = []
    for rank, g in enumerate(np.argsort(-bits, kind="stable")[:10], 1):
        rows.append({"criterion": "entropy", "rank": rank, "word": pool[g],
                     "bits": round(float(bits[g]), 3), "worst_group": int(worst[g])})
    for rank, g in enumerate(np.lexsort((-bits, worst))[:10], 1):
        rows.append({"criterion": "minimax", "rank": rank, "word": pool[g],
                     "bits": round(float(bits[g]), 3), "worst_group": int(worst[g])})
    return pd.DataFrame(rows)


def summarise(per_game, cand_turn):
    out = []
    for name in ORDER:
        g = per_game[per_game.strategy == name]
        per_run = g.groupby("run")["n_guesses"].mean()          # losses cost the full 6 guesses
        ct = cand_turn[cand_turn.strategy == name].set_index("turn")["avg_candidates"]
        out.append({
            "strategy": name,
            "runs": g.run.nunique(),
            "games": len(g),
            "avg_guesses": round(float(g.n_guesses.mean()), 3),
            "avg_guesses_solved": round(float(g.loc[g.solved, "n_guesses"].mean()), 3)
                                  if g.solved.any() else float("nan"),
            "std_across_runs": round(float(per_run.std(ddof=0)), 3) if g.run.nunique() > 1 else 0.0,
            "max_guesses": int(g.n_guesses.max()),
            "solve_rate_pct": round(100 * float(g.solved.mean()), 2),
            "avg_candidates_after_g1": round(float(ct[1]), 1),
            "avg_candidates_after_g2": round(float(ct[2]), 1),
            "avg_runtime_ms": round(1000 * float(g.seconds.mean()), 2),
            "first_guess": g.first_guess.iloc[0],
        })
    return pd.DataFrame(out)


def run_experiment(out_dir=RESULTS, secrets=None, random_runs=20, charts=True):
    random.seed(SEED)
    np.random.seed(SEED)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    answers, pool = load_answers(), load_guesses()
    pm = PatternMatrix.load_or_build(pool, answers)
    secrets = list(secrets) if secrets is not None else answers

    rows, cand_frames = [], []
    jobs = [(RandomStrategy(SEED + r), r) for r in range(random_runs)]
    jobs += [(FrequencyStrategy(answers), 0), (MinimaxStrategy(pm), 0), (EntropyStrategy(pm), 0)]

    random_sizes = []
    for strategy, run in jobs:
        t = time.time()
        r, sizes = play_all(strategy, secrets, answers, pool, run)
        rows += r
        if strategy.name == "random":
            random_sizes += sizes
            if run == random_runs - 1:
                cand_frames.append(candidates_per_turn("random", random_sizes))
        else:
            cand_frames.append(candidates_per_turn(strategy.name, sizes))
        solved = [x["n_guesses"] for x in r if x["solved"]]
        print(f"{strategy.name:<10} run {run:<2} avg {np.mean(solved):.3f}  ({time.time() - t:.1f}s)")

    per_game = pd.DataFrame(rows)
    cand_turn = pd.concat(cand_frames, ignore_index=True)
    summary = summarise(per_game, cand_turn)

    per_game.to_csv(out_dir / "per_game.csv", index=False)
    cand_turn.to_csv(out_dir / "candidates_per_turn.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    first_guess_table(answers, pool, pm).to_csv(out_dir / "first_guess_table.csv", index=False)

    print("\n", summary.to_string(index=False))
    if charts:
        from charts import make_all
        make_all(out_dir)
        print(f"\nCharts and CSV files saved in {out_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20, help="repeated runs of the random strategy")
    args = parser.parse_args()
    run_experiment(random_runs=args.runs)
