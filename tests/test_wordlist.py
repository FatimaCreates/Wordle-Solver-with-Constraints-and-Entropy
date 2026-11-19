from src.wordlist import load_answers, load_guesses


def test_sizes_and_format():
    a, g = load_answers(), load_guesses()
    assert len(a) == 2315 and len(g) == 12972
    assert all(len(w) == 5 and w.isalpha() and w.islower() for w in a + g)
    assert set(a) <= set(g)
