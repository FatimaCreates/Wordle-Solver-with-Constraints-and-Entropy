import random
from src.constraints import CandidateSet
from src.game import get_feedback
from src.wordlist import load_answers

ANSWERS = load_answers()


def test_constraint_filter_matches_feedback_filter():
    """Constraint filtering must keep exactly the words whose feedback equals the pattern."""
    rng = random.Random(42)
    for _ in range(150):
        guess, answer = rng.choice(ANSWERS), rng.choice(ANSWERS)
        pattern = get_feedback(guess, answer)
        cs = CandidateSet(ANSWERS)
        cs.update(guess, pattern)
        expected = [w for w in ANSWERS if get_feedback(guess, w) == pattern]
        assert cs.candidates == expected


def test_answer_always_survives_a_whole_game():
    rng = random.Random(42)
    for answer in rng.sample(ANSWERS, 100):
        cs = CandidateSet(ANSWERS)
        for _ in range(6):
            guess = rng.choice(ANSWERS)          # arbitrary (even bad) guesses
            cs.update(guess, get_feedback(guess, answer))
            assert answer in cs.candidates
            if cs.remaining == 1:
                break


def test_duplicate_letter_case():
    # speed vs abide -> (0,0,1,0,1): exactly one E, not at position 2; D present, not at 4
    cs = CandidateSet(ANSWERS)
    cs.update("speed", get_feedback("speed", "abide"))
    assert "abide" in cs.candidates
    assert all(w.count("e") == 1 for w in cs.candidates)
    assert all(w[2] != "e" for w in cs.candidates)


def test_counters_and_sizes():
    cs = CandidateSet(ANSWERS)
    assert cs.remaining == 2315 and cs.sizes == [2315]
    cs.update("crane", get_feedback("crane", "slate"))
    assert cs.remaining < 2315
    assert cs.checks == 2315
    assert cs.sizes == [2315, cs.remaining]
    assert len(cs) == cs.remaining