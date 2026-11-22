"""Search experiment: BFS / DFS / UCS / DLS / greedy best-first / A* on one graph.

    python run_search.py            -> results/search.csv, results/search_nodes.png

Bounded puzzle: 80 possible answers, 12 allowed probe words. Every one of the
80 hidden words is solved six times, once per algorithm, and the node counts,
probe counts and guess paths are written to a CSV.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.patterns import PatternMatrix
from src.search import METHODS, benchmark
from src.wordlist import load_answers, load_guesses

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

LABEL = {"bfs": "BFS", "dfs": "DFS", "ucs": "UCS", "dls": "DLS(3)",
         "greedy": "Greedy best-first", "astar": "A*"}


def main():
    answers, pool = load_answers(), load_guesses()
    pm = PatternMatrix.load_or_build(pool, answers)

    pool, hidden = answers[:12], answers[40:120]          # 12 legal probes, 80 possible answers
    rows = benchmark(pm, pool=pool, answers=hidden, max_nodes=60000)
    df = pd.DataFrame(rows)
    df["guesses_to_win"] = df.depth.fillna(0) + 1          # + the guess that types the identified word
    optimal = df[df.method == "bfs"].set_index("target").depth
    df["optimal"] = df.apply(lambda r: bool(r.found) and r.depth == optimal.get(r.target), axis=1)
    df.to_csv(OUT / "search.csv", index=False)

    table = (df.groupby("method")
               .agg(solved=("found", "sum"), avg_expanded=("expanded", "mean"),
                    max_expanded=("expanded", "max"), avg_probes=("depth", "mean"),
                    worst_probes=("depth", "max"), avg_guesses=("guesses_to_win", "mean"))
               .reindex(list(METHODS)).reset_index())
    table.to_csv(OUT / "search_summary.csv", index=False)

    table = table.merge(df.groupby("method").optimal.mean().mul(100).round(1)
                          .rename("optimal_pct").reset_index(), on="method")

    print(f"Bounded puzzle: {len(hidden)} possible answers, {len(pool)} probes; "
          f"every word solved by every algorithm\n")
    print("(avg_probes = guesses needed to identify the word; avg_guesses adds the final typing guess)\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"bfs": "#B1A9DA", "dfs": "#E4575B", "ucs": "#8f86c4", "dls": "#F2C94C",
              "greedy": "#E5B22A", "astar": "#6AAA64"}
    bars = ax.bar(table.method, table.avg_expanded, color=[colors[m] for m in table.method])
    for b, v in zip(bars, table.avg_expanded):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_ylabel("Average nodes expanded per word")
    ax.set_title("Search effort: uninformed vs informed", fontweight="bold")
    ax.set_xticks(range(len(table)), [LABEL[m] for m in table.method], rotation=15)
    fig.tight_layout()
    fig.savefig(OUT / "search_nodes.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"\nSaved {OUT / 'search.csv'} and {OUT / 'search_nodes.png'}")


if __name__ == "__main__":
    main()
