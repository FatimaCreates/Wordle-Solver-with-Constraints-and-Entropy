# Wordle Solver with Constraints and Entropy

An AI lab project (AL-2002) that plays Wordle — and studies *how* to play it well.

It is not a single script that guesses a word: it is a small research harness. The
2,315 real Wordle answers are played by several agents (random, letter-frequency,
minimax, entropy, an evolved genetic opener), the results are logged to CSV, and the
logs are used for supervised learning. The project also contains a **playable web
app** where you can guess a word yourself, race an AI, or play **Absurdle** against an
adversary that never commits to a secret word.

Everything is plain Python + NumPy/Pandas/Matplotlib/Seaborn/scikit-learn + Streamlit.

---

## Quick start

Requires Python 3.10+.

```bash
# 1. dependencies
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

# 2. the game (opens in your browser)
streamlit run app.py
```

The first launch builds a 12,972 × 2,315 pattern matrix (about 20-30 s) and caches it
to `data/processed/pattern_matrix.npy`; every later launch is instant.

### Running the experiments

Run these from the project root. Each writes into `results/`.

| Command | What it does | Output |
|---|---|---|
| `python experiment.py` | all four baseline strategies play all 2,315 answers (random: 20 seeds) | `results/summary.csv`, `per_game.csv`, 4 charts |
| `python charts.py` | redraw the charts from the existing CSVs | `results/chart*.png` |
| `python run_genetic.py` | genetic algorithm evolves a two-guess opener, then plays it | `genetic.csv`, `genetic_summary.csv`, `genetic_convergence.png` |
| `python run_search.py` | BFS / DFS / UCS / DLS / greedy best-first / A\* on one belief-state graph | `search.csv`, `search_summary.csv`, `search_nodes.png` |
| `python run_ml.py` | supervised learning on `per_game.csv` | `results/ml/` (metrics + figures) |
| `python run_absurdle.py` | every strategy against the adversary under the 6-guess rule | `absurdle.csv` |
| `python first_guess_table.py` | ranked opening guesses (entropy and minimax) | prints a table |
| `python run_baselines.py` | a quick one-pass sanity check of the four strategies | prints a table |

`python experiment.py --runs 5` uses fewer random repeats (faster).

---

## How it works

```
data/raw/*.txt                 word lists (2,315 answers, 12,972 allowed guesses)
      │
      ▼
src/patterns.py                pattern matrix: the colour pattern of EVERY guess
      │                        against EVERY answer (243 = 3^5 patterns), built
      │                        with NumPy and cached; this is the project's CPT
      ▼
src/constraints.py             belief state: which words are still possible
      │                        (green = fixed letter, yellow/gray = forbidden
      │                        position, min/max letter counts)
      ▼
src/strategies.py  src/entropy.py  src/minimax.py  src/genetic.py
      │                        agents that choose the next guess from the belief state
      ▼
src/game.py  src/agent.py      environment + the loop that plays a game
src/race.py  src/adversarial.py  the two-player modes
      │
      ▼
app.py  (Streamlit)            play / race / Absurdle
experiment.py + run_*.py       batch experiments → results/*.csv
src/ml.py                      supervised learning on the logs
```

`src/game.py` is the single source of truth for Wordle's duplicate-letter rule, so the
matrix, the solvers and the UI can never disagree about what a yellow tile means.

---

## The three ways to play

* **Play** — normal Wordle, 6 guesses, with a hint button (3 hints per game).
* **Race vs AI** — you and an AI get the same secret word. Pick the AI's strategy
  (random / frequency / minimax / entropy) from the dropdown. The AI's letters stay
  hidden until the game ends.
* **Absurdle** — there is **no secret word**. After each guess the adversary keeps the
  largest group of words consistent with the colours it just showed, so it can always
  wriggle out of a trap. Same 6-guess limit as normal Wordle. A live counter shows how
  many words it is still holding.

---

## Results (all reproduced by the scripts above)

**Baselines** — each strategy plays all 2,315 answers (`results/summary.csv`):

| strategy | avg guesses | avg when solved | solved |
|---|---|---|---|
| random | 4.07 | 4.03 | 98.1% |
| frequency | 3.79 | 3.77 | 98.8% |
| minimax | 3.57 | 3.57 | 100% |
| entropy | **3.46** | **3.46** | **100%** |

"avg guesses" counts a lost game as the full 6 guesses, so it is the honest number;
"avg when solved" is the average over successful games only (`summary.csv` has both,
as `avg_guesses` and `avg_guesses_solved`).

Entropy opens with `soare` and needs 3.46 guesses on average; that is the number the
rest of the project is measured against.

**Genetic algorithm** (`results/genetic.csv`) — evolving a *pair* of opening guesses
(a space of 84,129,906 pairs) instead of a single word:

* best evolved opener `liane + crost`: **9.58 bits**, 4.43 answers left after two guesses,
  which is 99.5% of the 9.63-bit ceiling obtained by an exhaustive partner sweep;
* about 4,000 evaluations out of 84 million — four orders of magnitude fewer;
* played as a strategy it solves **100%** of the answers in 3.61 guesses on average.

The interesting finding is that a good pair is *not* two good words: the best partner
for `soare` is `clint`, a word with only 5.0 bits on its own, because it covers the
letters `soare` does not. Ranking single words can never find that; the GA does.

**Search** (`results/search.csv`) — six algorithms on the same belief-state graph
(80 possible answers, 12 allowed probes):

| algorithm | nodes expanded | optimal solutions |
|---|---|---|
| BFS / UCS | 113 | 100% |
| DFS / DLS(3) | 620 / 610 | 97.5% |
| greedy best-first | 91 | 100% |
| A\* (admissible h) | **62** | **100%** |

A\* expands about half the nodes BFS needs and stays optimal; DFS is cheapest per node
but wastes five times the effort and occasionally returns a longer solution.

**Supervised learning** (`results/ml/`) — 53,245 logged games, target `solved`:

* logistic regression: 98.3% accuracy, ROC-AUC 0.66 (5-fold CV 0.65 ± 0.01);
* random forest: 98.3% accuracy, ROC-AUC 0.62, CV 0.62 ± 0.02;
* regression on `n_guesses` for solved games: random forest MAE 0.63, R² 0.26.

Accuracy is a headline, not a result: predicting "solved" for every single game already
scores 98.1%. More importantly, one of the four features leaks the label. `strategy`
and the opening-word shape (`fg_distinct`, `fg_vowels`) are known before the first
guess, but `checks` is the number of constraint checks the solver performed over the
**whole** game, and a lost game always uses all six guesses. `checks` therefore partly
encodes the outcome, which is why the AUC is around 0.66 rather than ~0.5: it measures
how much of the result is visible in a post-game number, not how well the result can be
predicted up front. This leak is left in and documented instead of being patched; a
clean version would drop `checks`, keep only pre-game features, and split by `answer`
rather than by row (the same answer appears in several runs). The figures include the
boxplot of guesses per strategy, both confusion matrices, the ROC curves, the
predicted-vs-actual scatter and the feature importances.

**Absurdle** (`results/absurdle.csv`, 300 games, 6-guess limit): entropy and minimax
solve 100% (5.0 guesses on average), frequency also 100%, random only 54% — the
adversary punishes guesses that do not split the remaining set evenly.

---

## AI topics implemented (for the report)

| Course topic | Where | Status |
|---|---|---|
| Environment / agent model (PEAS-style) | `src/game.py`, `src/agent.py` | full |
| Constraint satisfaction (CSP) | `src/constraints.py` — fixed letters, forbidden positions, letter counts | full |
| Uninformed search | `src/search.py` — BFS, DFS, UCS, DLS | full |
| Informed search | `src/search.py` — greedy best-first, A\* with an admissible heuristic | full |
| Local search | the GA's local-move operator and elitist exploitation (`src/genetic.py`) | partial |
| Genetic algorithms | `src/genetic.py` — genome, fitness, tournament, crossover, mutation, elitism, immigrants, convergence curve | full |
| Probability / information theory | `src/patterns.py` + `src/entropy.py` — pattern CPT, entropy in bits, Bayesian belief update by filtering | full |
| Supervised learning | `src/ml.py` — preprocessing, train/test split, 5-fold CV, confusion matrix, ROC/AUC, regression | full |
| Evaluation methodology | `experiment.py`, all `run_*.py` | full |

Not implemented: unsupervised learning, an explicit Bayesian-network graph, and
reinforcement learning. The adversary in Absurdle is a minimax search, not RL.

---

## Project structure

```
app.py                 Streamlit UI (Play / Race / Absurdle)
experiment.py          the main experiment: all strategies on all answers
charts.py              the four result charts
src/
  game.py              Wordle rules, feedback function, WordleGame
  patterns.py          pattern matrix (guess × answer), cached
  constraints.py       Constraints + CandidateSet (the CSP belief state)
  strategies.py        Strategy interface, Random, Frequency
  matrix_strategy.py   shared scoring helper for matrix-based strategies
  entropy.py           entropy strategy + entropy in bits
  minimax.py           minimax strategy (smallest worst-case group)
  genetic.py           genetic algorithm for a two-guess opener + its strategy
  search.py            BFS / DFS / UCS / DLS / greedy / A* over belief states
  adversarial.py       Absurdle adversary + alpha-beta hint solver
  race.py              RaceSession (player vs AI on the same word)
  ml.py                supervised learning pipeline
  agent.py             SolverAgent: belief state + strategy, plays one game
  wordlist.py          word-list loading
run_genetic.py  run_search.py  run_ml.py  run_absurdle.py  run_baselines.py
first_guess_table.py
tests/                 pytest suite (solver, constraints, UI, experiments)
data/raw/              word lists        data/processed/   cached matrix
results/               CSVs and figures produced by the scripts
```

## Data

Word lists come from <https://github.com/boompig/wordle-py> (`data-raw/`): 2,315
official answers and 12,972 accepted guesses. Used for non-commercial educational
purposes.

## Tests

```bash
python -m pytest -q
```

The suite covers the feedback rule, the constraint filter, the pattern matrix, every
solver, the race and Absurdle sessions, the experiment output and the Streamlit app.
Two assertions in `tests/test_app.py` still expect older UI wording
(`"not in the word list"`, `"The AI would guess"`); they fail and are scheduled for
cleanup.

## Notes and limitations

* `soare` is the best *single* opening word by entropy, but the best *pair* starts with
  a different word (`liane` in the last run). The GA maximises information, not the
  final score, which is why its fixed double opener is 0.14 guesses behind the fully
  adaptive entropy strategy (3.61 vs 3.46) while being far stronger on the first two
  guesses (4.4 vs 62 words left after one).
* Absurdle is winnable within 6 guesses by a good solver because the adversary must keep
  the *largest* consistent group every turn, which can trap it as the pool shrinks.
* The alpha-beta solver in `src/adversarial.py` is the Absurdle search; a hint is not a
  suggested word but one true letter in its true box (3 per game), because a whole-word
  suggestion is effectively the answer and can waste red letters. Hints are off in
  Absurdle, which has no fixed word to reveal.
* The supervised pipeline is not leak-free: `checks` is counted over the finished game,
  so it partly encodes `solved` (see the supervised-learning section). `answer` and
  `run` are never used as features.
