"""Step 3 check: run random and frequency baselines on every answer."""
import time
from statistics import mean

from src.agent import SolverAgent
from src.game import WordleGame
from src.strategies import RandomStrategy, FrequencyStrategy
from src.wordlist import load_answers, load_guesses

SEED = 42


def run_all(strategy, answers, pool):
    """Play every answer once. Returns a list of (n_guesses, solved)."""
    agent = SolverAgent(strategy, answers, pool)
    out = []
    for secret in answers:
        game = WordleGame(secret)
        agent.play(game)
        out.append((game.guesses_used, game.is_solved()))
    return out


def summarise(name, results, seconds):
    solved = [n for n, ok in results if ok]
    print(f"{name:<10} avg guesses {mean(solved):.3f} | max {max(n for n, _ in results)} | "
          f"solve rate {len(solved) / len(results):.1%} | {seconds:.1f}s")


if __name__ == "__main__":
    answers, pool = load_answers(), load_guesses()

    t = time.time()
    res = run_all(FrequencyStrategy(answers), answers, pool)
    summarise("frequency", res, time.time() - t)

    t = time.time()
    res = run_all(RandomStrategy(SEED), answers, pool)
    summarise("random", res, time.time() - t)