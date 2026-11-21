from src.genetic import GeneticOpener
from src.patterns import PatternMatrix
from src.wordlist import load_answers, load_guesses

A, G = load_answers(), load_guesses()
PM = PatternMatrix.load_or_build(G, A)


def test_genetic_returns_a_real_word_with_good_entropy():
    word, h, hist = GeneticOpener(PM, seed=1).run()
    assert word in G and h > 5.0                                   # random words are far lower
    assert hist[-1] >= hist[0]                                     # elitism: never gets worse
