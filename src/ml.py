"""Supervised learning pipeline over the recorded games (results/per_game.csv).

The experiment logs one row per game: strategy, answer, guesses used, whether it
was solved, run time and how many constraint checks the solver performed.  That
is a real labelled dataset, so the supervised-learning part of the course maps
onto it directly:

    task A  classification - will this configuration solve the word?  (solved)
    task B  regression     - how many guesses will a solved game take?  (n_guesses)

Preprocessing (scikit-learn Pipeline, fitted on the training fold only):
    strategy          -> one-hot encoded (categorical)
    checks            -> median imputation + standardisation (numeric)
    fg_distinct       -> number of distinct letters in the opening guess
    fg_vowels         -> number of vowels in the opening guess

Models: logistic regression and a random forest for task A, linear regression
and a random forest for task B.  Reported: a held-out train/test split, 5-fold
cross-validation on the training part, a confusion matrix, a ROC curve with
AUC, feature importance and the usual error metrics.

Known leak, kept on purpose and documented in the README: `checks` is counted
over the whole game, and a lost game always runs all six guesses, so it partly
encodes the label (`solved`).  `strategy` and the opening-word shape are known
before the game ends; `checks` is not.  The reported AUC is therefore
optimistic, not an estimate of predicting the outcome from information
available up front.  `answer` and `run` are never used as features.
"""
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "ml"
SEED = 42
VOWELS = set("aeiou")


def load_games(path: Path = RESULTS / "per_game.csv") -> pd.DataFrame:
    """Read the per-game log and add the opening-guess features."""
    df = pd.read_csv(path)
    df["fg_distinct"] = df["first_guess"].astype(str).map(lambda w: len(set(w)))
    df["fg_vowels"] = df["first_guess"].astype(str).map(lambda w: sum(ch in VOWELS for ch in w))
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Only columns that are known before/while the game is played are used."""
    return df[["strategy", "checks", "fg_distinct", "fg_vowels"]].copy()


def preprocessor(dense: bool = False) -> ColumnTransformer:
    """Categorical one-hot + numeric impute/scale, combined into one transformer."""
    return ColumnTransformer([
        ("strategy", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense), ["strategy"]),
        ("numbers", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), ["checks", "fg_distinct", "fg_vowels"]),
    ])


def classifier(estimator, dense: bool) -> Pipeline:
    return Pipeline([("prepare", preprocessor(dense)), ("model", estimator)])


def evaluate_classifier(X_tr, X_te, y_tr, y_te, name: str, model: Pipeline
                        ) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Fit, predict, and report accuracy / precision / recall / F1 / ROC-AUC + 5-fold CV AUC."""
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_auc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc")
    cm = confusion_matrix(y_te, pred)
    return {"model": name, "task": "classification", "target": "solved",
            "accuracy": round(accuracy_score(y_te, pred), 4),
            "precision": round(precision_score(y_te, pred, zero_division=0), 4),
            "recall": round(recall_score(y_te, pred, zero_division=0), 4),
            "f1": round(f1_score(y_te, pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_te, proba), 4),
            "cv_roc_auc_mean": round(float(cv_auc.mean()), 4),
            "cv_roc_auc_std": round(float(cv_auc.std()), 4),
            "train_rows": len(X_tr), "test_rows": len(X_te)}, cm, proba


def evaluate_regressor(X_tr, X_te, y_tr, y_te, name: str, model: Pipeline
                       ) -> Tuple[Dict, np.ndarray]:
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_mae = -cross_val_score(model, X_tr, y_tr, cv=cv, scoring="neg_mean_absolute_error")
    return {"model": name, "task": "regression", "target": "n_guesses",
            "mae": round(mean_absolute_error(y_te, pred), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_te, pred))), 4),
            "r2": round(r2_score(y_te, pred), 4),
            "cv_mae_mean": round(float(cv_mae.mean()), 4),
            "cv_mae_std": round(float(cv_mae.std()), 4),
            "train_rows": len(X_tr), "test_rows": len(X_te)}, pred


def run(out_dir: Path = OUT) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"axes.edgecolor": "#3B3560", "text.color": "#3B3560",
                         "axes.labelcolor": "#3B3560", "xtick.color": "#3B3560",
                         "ytick.color": "#3B3560"})

    df = load_games()
    X = build_features(df)
    print(f"{len(df):,} games, {df.strategy.nunique()} strategies, "
          f"{df.solved.mean():.1%} solved, features = {list(X.columns)}")

    # ---------- exploratory view ----------
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="strategy", y="n_guesses", ax=ax, hue="strategy", legend=False)
    ax.set_title("Guesses needed per strategy (the label the model must learn)", fontweight="bold")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_guesses_by_strategy.png", dpi=200, facecolor="white")
    plt.close(fig)

    # ---------- task A: classification ----------
    y = df["solved"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    models = {
        "logistic_regression": classifier(LogisticRegression(max_iter=1000), dense=False),
        "random_forest": classifier(RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                           random_state=SEED, n_jobs=-1), dense=True),
    }

    rows, cms, curves = [], {}, {}
    for name, model in models.items():
        metrics, cm, proba = evaluate_classifier(X_tr, X_te, y_tr, y_te, name, model)
        rows.append(metrics)
        cms[name] = cm
        curves[name] = roc_curve(y_te, proba)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False, ax=ax,
                    xticklabels=["not solved", "solved"], yticklabels=["not solved", "solved"])
        ax.set_title(name.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion matrices on the 25% held-out test set", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=200, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for name, (fpr, tpr, _) in curves.items():
        auc = next(r["roc_auc"] for r in rows if r["model"] == name)
        ax.plot(fpr, tpr, lw=2, label=f"{name.replace('_', ' ').title()} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves - predicting whether a game is solved", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", dpi=200, facecolor="white")
    plt.close(fig)

    # ---------- task B: regression on solved games ----------
    solved = df[df.solved].copy()
    Xr, yr = build_features(solved), solved["n_guesses"].astype(float)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.25, random_state=SEED)
    regressors = {
        "linear_regression": Pipeline([("prepare", preprocessor(dense=False)),
                                       ("model", LinearRegression())]),
        "random_forest_regressor": Pipeline([("prepare", preprocessor(dense=True)),
                                             ("model", RandomForestRegressor(
                                                 n_estimators=200, min_samples_leaf=5,
                                                 random_state=SEED, n_jobs=-1))]),
    }
    for name, model in regressors.items():
        metrics, pred = evaluate_regressor(Xr_tr, Xr_te, yr_tr, yr_te, name, model)
        rows.append(metrics)
        if name == "random_forest_regressor":
            fig, ax = plt.subplots(figsize=(4.8, 4.4))
            ax.hexbin(yr_te, pred, gridsize=30, cmap="Purples", mincnt=1)
            ax.plot([yr_te.min(), yr_te.max()], [yr_te.min(), yr_te.max()], "--", color="#E4575B")
            ax.set_xlabel("Actual guesses")
            ax.set_ylabel("Predicted guesses")
            ax.set_title(f"Random forest regression (R2 {metrics['r2']:.3f})", fontweight="bold")
            fig.tight_layout()
            fig.savefig(out_dir / "regression_actual_vs_predicted.png", dpi=200, facecolor="white")
            plt.close(fig)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "ml_metrics.csv", index=False)

    # ---------- feature importance / coefficients ----------
    names = list(models["random_forest"].named_steps["prepare"].get_feature_names_out())
    importance = models["random_forest"].named_steps["model"].feature_importances_
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    order = np.argsort(importance)
    ax.barh([names[i] for i in order], importance[order], color="#B1A9DA")
    ax.set_title("What predicts 'solved' (random forest importance)", fontweight="bold")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=200, facecolor="white")
    plt.close(fig)

    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("Task A: predict `solved` on results/per_game.csv\n")
        f.write(models["random_forest"].named_steps["prepare"].get_feature_names_out().__str__())
        f.write("\n\n")
        f.write(metrics_df[metrics_df.task == "classification"].to_string(index=False))
        f.write("\n\nConfusion matrix (random forest, rows = actual):\n")
        f.write(str(cms["random_forest"]))
        f.write("\n\nTask B: predict `n_guesses` on solved games\n")
        f.write(metrics_df[metrics_df.task == "regression"].to_string(index=False))
        f.write("\n\nFeature importances (classifier):\n")
        for n, v in sorted(zip(names, importance), key=lambda t: -t[1]):
            f.write(f"  {n:<32} {v:.4f}\n")

    print(metrics_df.to_string(index=False))
    print(f"\nSaved metrics and figures in {out_dir}")
    return metrics_df


if __name__ == "__main__":
    run()
