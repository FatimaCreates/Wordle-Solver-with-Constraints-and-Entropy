"""Two extra experiments:  python run_adversarial_genetic.py
 1. every strategy plays the adversary (Absurdle)   -> results/adversarial.csv
 2. a genetic algorithm searches for the best opening word -> results/genetic.csv
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.adversarial import AlphaBetaSolver, play_vs_adversary
from src.entropy import entropy_scores
from src.genetic import GeneticOpener
from src.patterns import PatternMatrix
from src.race import make_strategy
from src.wordlist import load_answers, load_guesses

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
A, G = load_answers(), load_guesses()
PM = PatternMatrix.load_or_build(G, A)

rows = []
for name in ("random", "frequency", "minimax", "entropy", "alphabeta"):
    chooser = AlphaBetaSolver(PM) if name == "alphabeta" else make_strategy(name, A, G, PM)
    t = time.time()
    n, ok, words = play_vs_adversary(chooser, A, G, PM)
    rows.append({"strategy": name, "guesses": n, "solved": ok, "seconds": round(time.time() - t, 2),
                 "words": " ".join(words)})
adv = pd.DataFrame(rows)
adv.to_csv(OUT / "adversarial.csv", index=False)
print("AI vs adversary (Absurdle):\n", adv.drop(columns="words").to_string(index=False))

counts = PM.pattern_counts(A)
all_h = entropy_scores(counts, len(A))
order = np.argsort(-all_h)
rank = {int(i): r + 1 for r, i in enumerate(order)}
print(f"\nBrute force: best opening {G[order[0]]} ({all_h[order[0]]:.4f} bits), {len(G)} evaluations")
rows = []
for seed in range(10):
    ga = GeneticOpener(PM, seed=seed)
    word, h, _ = ga.run()
    rows.append({"seed": seed, "best_word": word, "entropy": round(h, 4), "rank_of_12972": rank[PM.guess_index[word]],
                 "evaluations": ga.evaluations})
gen = pd.DataFrame(rows)
gen.to_csv(OUT / "genetic.csv", index=False)
print("\nGenetic algorithm (10 seeds):\n", gen.to_string(index=False))
