from flapp.equity import reward_equity_index


def test_perfect_separation_gives_correlation_near_one():
    # clients 0,1 are free-riders (low score), 2,3 are honest (high score)
    client_scores = {"0": -0.05, "1": -0.04, "2": 0.03, "3": 0.04}
    result = reward_equity_index(client_scores, num_free_riders=2)
    assert result > 0.9


def test_inverted_scores_give_negative_correlation():
    # free-riders (0,1) score HIGHER than honest clients (2,3) — a bad mechanism
    client_scores = {"0": 0.05, "1": 0.04, "2": -0.03, "3": -0.04}
    result = reward_equity_index(client_scores, num_free_riders=2)
    assert result < -0.9


def test_no_free_riders_returns_none():
    client_scores = {"0": 0.01, "1": 0.02, "2": 0.03}
    assert reward_equity_index(client_scores, num_free_riders=0) is None


def test_identical_scores_returns_none():
    client_scores = {"0": 0.5, "1": 0.5, "2": 0.5}
    assert reward_equity_index(client_scores, num_free_riders=1) is None


def test_empty_scores_returns_none():
    assert reward_equity_index({}, num_free_riders=2) is None
