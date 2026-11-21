"""Genetic algorithm searches for the best opening word:  python run_genetic.py   -> results/genetic.csv"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.entropy import entropy_scores
from src.genetic import GeneticOpener
from src.patterns import PatternMatrix
from src.wordlist import load_answers, load_guesses

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
A, G = load_answers(), load_guesses()
PM = PatternMatrix.load_or_build(G, A)

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
