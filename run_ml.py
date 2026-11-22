"""Supervised learning on the experiment data.

    python run_ml.py        -> results/ml/ml_metrics.csv + figures

Reads results/per_game.csv (written by experiment.py) and trains:
  * a classifier for "will this game be solved?"
  * a regressor for "how many guesses will a solved game take?"
See src/ml.py for the features, the preprocessing and the metrics.
"""
from src.ml import run

if __name__ == "__main__":
    run()
