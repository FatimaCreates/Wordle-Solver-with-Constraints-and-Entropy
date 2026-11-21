"""The four charts for the report. Reads the CSV files written by experiment.py.

    python charts.py        # redraw the charts from results/ without rerunning the experiment
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
ORDER = ["random", "frequency", "minimax", "entropy"]
COLORS = {"random": "#B1A9DA", "frequency": "#F2C94C", "minimax": "#7F73BE", "entropy": "#6AAA64"}
INK = "#3B3560"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def histogram(per_game, summary, out):
    """Chart 1: how many guesses each strategy needs (share of games, one panel per strategy)."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    labels = ["1", "2", "3", "4", "5", "6", "Fail"]
    for ax, name in zip(axes, ORDER):
        g = per_game[per_game.strategy == name]
        shares = [100 * ((g.solved) & (g.n_guesses == k)).sum() / len(g) for k in range(1, 7)]
        shares.append(100 * (~g.solved).sum() / len(g))
        bars = ax.bar(labels, shares, color=COLORS[name])
        bars[-1].set_color("#E4575B")
        avg = summary.loc[summary.strategy == name, "avg_guesses"].iloc[0]
        ax.set_title(f"{name.capitalize()}  (avg {avg:.2f})", fontweight="bold")
        ax.set_xlabel("Guesses needed")
        for b, v in zip(bars, shares):
            if v >= 1:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.0f}", ha="center", fontsize=8)
    axes[0].set_ylabel("Share of games (%)")
    fig.suptitle("Distribution of guesses needed over all 2,315 answers", fontweight="bold")
    _save(fig, out)


def average_bar(summary, out):
    """Chart 2: average guesses per strategy (random shows its spread over 20 runs)."""
    s = summary.set_index("strategy").loc[ORDER]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(ORDER, s.avg_guesses, color=[COLORS[n] for n in ORDER],
                  yerr=s.std_across_runs.where(s.std_across_runs > 0, 0), capsize=5)
    for b, v in zip(bars, s.avg_guesses):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.3f}", ha="center", fontweight="bold")
    ax.set_ylim(0, max(s.avg_guesses) + 0.7)
    ax.set_ylabel("Average guesses per word")
    ax.set_title("Average guesses (lower is better)", fontweight="bold")
    ax.set_xticks(range(len(ORDER)), [n.capitalize() for n in ORDER])
    _save(fig, out)


def candidates_line(cand_turn, out):
    """Chart 3: average candidates left after each guess."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name in ORDER:
        c = cand_turn[cand_turn.strategy == name]
        ax.plot(c.turn, c.avg_candidates, marker="o", lw=2, color=COLORS[name], label=name.capitalize())
    ax.set_yscale("log")
    ax.set_xlabel("Number of guesses made")
    ax.set_ylabel("Average candidates left (log scale)")
    ax.set_title("How fast each strategy narrows the answers", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, which="both")
    _save(fig, out)


def runtime_bar(summary, out):
    """Chart 4: average run time per game."""
    s = summary.set_index("strategy").loc[ORDER]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(ORDER, s.avg_runtime_ms, color=[COLORS[n] for n in ORDER])
    top = max(s.avg_runtime_ms)
    for b, v in zip(bars, s.avg_runtime_ms):
        ax.text(b.get_x() + b.get_width() / 2, v + top * 0.02, f"{v:.1f} ms", ha="center", fontweight="bold")
    ax.set_ylim(0, top * 1.15)
    ax.set_ylabel("Average run time per game (ms)")
    ax.set_title("Run time per game", fontweight="bold")
    ax.set_xticks(range(len(ORDER)), [n.capitalize() for n in ORDER])
    _save(fig, out)


def make_all(results_dir=RESULTS):
    results_dir = Path(results_dir)
    per_game = pd.read_csv(results_dir / "per_game.csv")
    summary = pd.read_csv(results_dir / "summary.csv")
    cand_turn = pd.read_csv(results_dir / "candidates_per_turn.csv")
    histogram(per_game, summary, results_dir / "chart1_histogram_guesses.png")
    average_bar(summary, results_dir / "chart2_average_guesses.png")
    candidates_line(cand_turn, results_dir / "chart3_candidates_per_turn.png")
    runtime_bar(summary, results_dir / "chart4_runtime.png")


if __name__ == "__main__":
    make_all()
