import pandas as pd

from experiment import run_experiment
from src.wordlist import load_answers


def test_small_experiment_writes_csvs_and_charts(tmp_path):
    secrets = load_answers()[:40]
    summary = run_experiment(out_dir=tmp_path, secrets=secrets, random_runs=2)
    assert list(summary.strategy) == ["random", "frequency", "minimax", "entropy"]
    assert (summary.solve_rate_pct > 80).all()
    for name in ["per_game.csv", "summary.csv", "candidates_per_turn.csv", "first_guess_table.csv",
                 "chart1_histogram_guesses.png", "chart2_average_guesses.png",
                 "chart3_candidates_per_turn.png", "chart4_runtime.png"]:
        assert (tmp_path / name).exists(), name
    per_game = pd.read_csv(tmp_path / "per_game.csv")
    assert len(per_game) == 40 * (2 + 3)                      # 2 random runs + 3 other strategies
    cand = pd.read_csv(tmp_path / "candidates_per_turn.csv")
    assert (cand[cand.turn == 0].avg_candidates == 2315).all()
