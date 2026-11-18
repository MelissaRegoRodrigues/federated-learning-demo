"""Testes do scripts/analyze_comparison.py: o cenario carregado precisa trocar o alpha usado na particao."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import analyze_comparison as ac  # noqa: E402
import summarize_results as sr  # noqa: E402


def test_use_scenario_sets_alpha_folder_and_free_riders():
    ac.use_scenario("0.1", "fr50")
    assert sr.SETTINGS["alpha"] == 0.1
    assert sr.SETTINGS["results_dir"] == Path("results") / "a01_fr50"
    assert sr.FREE_RIDER_IDS == set(range(5))

    ac.use_scenario("0.5", "silent")
    assert sr.SETTINGS["alpha"] == 0.5
    assert sr.FREE_RIDER_IDS == set(range(3))


def test_load_uses_alpha_of_the_scenario(monkeypatch):
    # load_runs e falso aqui: so queremos ver qual alpha estava ativo quando cada pasta foi lida.
    seen = []

    def fake_load_runs(folder, strategy):
        seen.append(sr.SETTINGS["alpha"])
        return []

    monkeypatch.setattr(sr, "load_runs", fake_load_runs)
    ac.load("0.5", "silent")
    ac.load("0.1", "silent")
    assert seen[:4] == [0.5] * 4
    assert seen[4:] == [0.1] * 4
