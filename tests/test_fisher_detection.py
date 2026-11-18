"""Testes do scripts/fisher_detection.py: tabela 2x2 da deteccao e o teste de Fisher."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fisher_detection as fd  # noqa: E402


def make_run(flagged_per_round):
    # A rodada 0 e a avaliacao inicial e nao conta (sr.training_rounds pula ela).
    rounds = [{"round": 0, "detected_free_riders": []}]
    for number, flagged in enumerate(flagged_per_round, start=1):
        rounds.append({"round": number, "detected_free_riders": flagged})
    return {"rounds": rounds}


def test_contingency_counts_each_client_in_each_round():
    run = make_run([[0, 1, 5], [0]])
    table = fd.contingency([run], range(2), 6)
    # Rodada 1: VP=2 (0 e 1), FP=1 (5), VN=3. Rodada 2: VP=1, FN=1, VN=4.
    assert table == [[3, 1], [1, 7]]


def test_perfect_detection_is_significant():
    runs = [make_run([[0, 1, 2]] * 49) for _ in range(5)]
    odds_ratio, p_value = fd.fisher_one_sided(fd.contingency(runs, range(3), 10))
    assert odds_ratio == float("inf")
    assert p_value < 1e-10


def test_by_client_counts_one_permanent_decision_per_seed():
    # Exclusao permanente: barrado na rodada 1 e em todas as outras conta uma vez so.
    runs = [make_run([[2]] * 49), make_run([[]] * 49)]
    table = fd.contingency_by_client(runs, range(3), 10)
    assert table == [[1, 5], [0, 14]]
    _, p_value = fd.fisher_one_sided(table)
    assert p_value > 0.2  # uma unica deteccao nao e evidencia


def test_no_flags_gives_p_one():
    runs = [make_run([[]] * 10)]
    odds_ratio, p_value = fd.fisher_one_sided(fd.contingency(runs, range(3), 10))
    assert p_value == 1.0
    assert odds_ratio != odds_ratio  # NaN
