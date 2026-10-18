import pytest

from flapp.fairness import normalized_shares, reference_shares, squared_distance


def test_reference_gives_zero_to_free_riders_and_data_share_to_honest_clients():
    reference = reference_shares({0: 500, 1: 100, 2: 300}, num_free_riders=1)
    assert reference == {0: 0.0, 1: 0.25, 2: 0.75}


def test_normalized_shares_sum_to_one():
    assert normalized_shares({0: 1.0, 1: 3.0}) == {0: 0.25, 1: 0.75}
    assert normalized_shares({0: 0.0, 1: 0.0}) == {0: 0.0, 1: 0.0}


def test_distance_is_zero_when_rewards_match_the_reference():
    reference = {0: 0.0, 1: 0.25, 2: 0.75}
    assert squared_distance(reference, reference) == 0.0
    assert squared_distance({0: 1.0, 1: 0.0, 2: 0.0}, reference) == pytest.approx(1.0 + 0.0625 + 0.5625)
