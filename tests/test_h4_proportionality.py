"""Testes do scripts/h4_proportionality.py."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import h4_proportionality as h4  # noqa: E402


def test_mean_scores_skips_round_zero_and_missing_clients():
    run = {
        "rounds": [
            {"round": 0, "client_scores": {"0": 99.0}},
            {"round": 1, "client_scores": {"0": 0.0, "1": 0.4}},
            {"round": 2, "client_scores": {"1": 0.2}},  # cliente 0 barrado: sem escore
        ]
    }
    assert h4.mean_scores(run) == {0: 0.0, 1: 0.30000000000000004}


def test_spearman_greater_detects_increasing_relation():
    rho, p_value = h4.spearman_greater([1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60])
    assert rho == 1.0
    assert p_value < 0.01


def test_spearman_greater_without_variation_gives_p_one():
    rho, p_value = h4.spearman_greater([1, 1, 1], [1, 2, 3])
    assert math.isnan(rho)
    assert p_value == 1.0


def test_paired_t_of_identical_values_is_zero():
    assert h4.paired_t({42: 0.1, 43: 0.2}, {42: 0.1, 43: 0.2}) == (0.0, 1.0)
