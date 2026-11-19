"""Ranked table of the best opening guesses (entropy and minimax) over all 2,315 answers."""
import numpy as np

from src.entropy import entropy_scores
from src.minimax import worst_case_sizes
from src.patterns import PatternMatrix
from src.wordlist import load_answers, load_guesses

answers, guesses = load_answers(), load_guesses()
pm = PatternMatrix.load_or_build(guesses, answers)
counts = pm.pattern_counts(answers)
H = entropy_scores(counts, len(answers))
worst = worst_case_sizes(counts)
n_patterns = (counts > 0).sum(axis=1)
expected_left = (counts ** 2).sum(axis=1) / len(answers)

print("Top 10 first guesses by ENTROPY")
print(f"{'rank':<5}{'word':<8}{'bits':>7}{'worst group':>13}{'avg left':>10}{'patterns':>10}")
for r, g in enumerate(np.argsort(-H, kind="stable")[:10], 1):
    print(f"{r:<5}{guesses[g]:<8}{H[g]:>7.3f}{worst[g]:>13}{expected_left[g]:>10.1f}{n_patterns[g]:>10}")

print("\nTop 10 first guesses by MINIMAX (smallest worst-case group)")
print(f"{'rank':<5}{'word':<8}{'worst group':>12}{'bits':>7}{'avg left':>10}")
for r, g in enumerate(np.lexsort((-H, worst))[:10], 1):
    print(f"{r:<5}{guesses[g]:<8}{worst[g]:>12}{H[g]:>7.3f}{expected_left[g]:>10.1f}")
