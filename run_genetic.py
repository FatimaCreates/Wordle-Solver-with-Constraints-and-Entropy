"""Genetic algorithm experiment: evolve the best two-guess opening.

    python run_genetic.py   -> results/genetic.csv, results/genetic_summary.csv,
                               results/genetic_convergence.png

1. Brute force over all 12,972 single words is computed once as a reference.
2. The GA runs on 10 seeds over the 84-million-point pair space.
3. The best evolved pair is played against all 2,315 answers as a real strategy
   (pair first, entropy afterwards) and compared with the entropy strategy alone.
4. The convergence curve of the best seed is plotted.
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.agent import SolverAgent
from src.entropy import entropy_scores
from src.game import MAX_GUESSES, WordleGame
from src.genetic import GeneticOpener, GeneticPairStrategy, pair_metrics
from src.patterns import PatternMatrix
from src.wordlist import load_answers, load_guesses

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
SEEDS = 10


def baseline_entropy_avg() -> float:
    """Average guesses of the entropy strategy alone, if the main experiment has run."""
    path = OUT / "summary.csv"
    if not path.exists():
        return float("nan")
    s = pd.read_csv(path).set_index("strategy")
    return float(s.loc["entropy", "avg_guesses"]) if "entropy" in s.index else float("nan")


def main():
    answers, guesses = load_answers(), load_guesses()
    pm = PatternMatrix.load_or_build(guesses, answers)

    counts = pm.pattern_counts(answers)
    single = entropy_scores(counts, len(answers))
    order = np.argsort(-single)
    first = guesses[order[0]]
    print(f"Brute force, 1 word:  {first} at {single[order[0]]:.4f} bits "
          f"({len(guesses):,} evaluations)")

    # ceiling for reference: the best possible partner for that word, over every guess
    best_partner, best_entropy = None, 0.0
    for w in guesses:
        if w == first:
            continue
        e = pair_metrics(pm, (first, w))["entropy"]
        if e > best_entropy:
            best_partner, best_entropy = w, e
    ceiling = pair_metrics(pm, (first, best_partner))
    print(f"Exhaustive partner sweep: {first} + {best_partner} at {best_entropy:.4f} bits "
          f"({ceiling['avg_left']:.2f} answers left) - the bar the GA is measured against")

    rows, histories = [], {}
    start = time.time()
    for seed in range(SEEDS):
        ga = GeneticOpener(pm, seed=seed)
        words, h, hist = ga.run()
        m = pair_metrics(pm, words)
        rows.append({"seed": seed, "word_1": words[0], "word_2": words[1],
                     "entropy_1": round(float(single[pm.guess_index[words[0]]]), 4),
                     "entropy_2": round(float(single[pm.guess_index[words[1]]]), 4),
                     "pair_entropy": round(h, 4),
                     "max_pair_entropy": round(float(np.log2(len(answers))), 4),
                     "avg_left_after_2": round(m["avg_left"], 2),
                     "worst_group": m["worst_group"],
                     "evaluations": ga.evaluations,
                     "generations": ga.generations})
        histories.setdefault(round(h, 4), hist)
    gen = pd.DataFrame(rows)
    gen.to_csv(OUT / "genetic.csv", index=False)
    print(f"\nGA over {SEEDS} seeds ({time.time() - start:.1f}s, "
          f"space = {len(guesses) * (len(guesses) - 1) // 2:,} pairs):")
    print(gen.to_string(index=False))

    best = gen.sort_values(["pair_entropy", "avg_left_after_2"], ascending=[False, True]).iloc[0]
    pair = (best.word_1, best.word_2)
    metrics = pair_metrics(pm, pair)

    # ---- the evolved opener as a real strategy over all 2,315 answers ----
    print(f"\nPlaying all {len(answers)} answers with the evolved opener {pair} ...")
    agent = SolverAgent(GeneticPairStrategy(pm, pair), answers, guesses)
    t = time.time()
    solved, used = 0, []
    for secret in answers:
        game = WordleGame(secret)
        agent.play(game)
        if game.is_solved():
            solved += 1
            used.append(game.guesses_used)
    seconds = time.time() - t
    summary = pd.DataFrame([{
        "opener": f"{pair[0]} + {pair[1]}",
        "pair_entropy": round(metrics["entropy"], 4),
        "avg_left_after_2": round(metrics["avg_left"], 2),
        "games": len(answers),
        "solve_rate_pct": round(100 * solved / len(answers), 2),
        "avg_guesses": round(float(np.mean(used)), 3) if used else float("nan"),
        "max_guesses": max(used) if used else None,
        "guess_limit": MAX_GUESSES,
        "entropy_only_avg_guesses": baseline_entropy_avg(),
        "ceiling_pair": f"{first} + {best_partner}",
        "ceiling_pair_entropy": round(best_entropy, 4),
        "pct_of_ceiling": round(100 * metrics["entropy"] / best_entropy, 2),
        "avg_runtime_ms": round(1000 * seconds / len(answers), 2),
    }])
    summary.to_csv(OUT / "genetic_summary.csv", index=False)
    print(summary.to_string(index=False))

    # ---- convergence chart ----
    curve = histories[round(float(best.pair_entropy), 4)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(curve)), curve, marker="o", color="#6AAA64", lw=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best joint entropy (bits)")
    ax.set_title(f"GA convergence for the evolving opener (seed {int(best.seed)})", fontweight="bold")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "genetic_convergence.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"\nSaved genetic.csv, genetic_summary.csv and genetic_convergence.png in {OUT}")


if __name__ == "__main__":
    main()
