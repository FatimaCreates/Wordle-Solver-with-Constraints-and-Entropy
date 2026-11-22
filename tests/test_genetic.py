from src.genetic import GeneticOpener
from src.patterns import PatternMatrix
from src.wordlist import load_answers, load_guesses

A, G = load_answers(), load_guesses()
PM = PatternMatrix.load_or_build(G, A)


def test_genetic_returns_a_valid_pair_with_high_joint_entropy():
    words, h, hist = GeneticOpener(PM, pop=12, generations=8, seed=1).run()
    assert len(words) == 2 and words[0] != words[1]                # two distinct real words
    assert all(w in G for w in words)
    assert h > 8.0                                                  # a single word tops out at 5.9 bits
    assert hist[-1] >= hist[0]                                      # elitism: never gets worse
