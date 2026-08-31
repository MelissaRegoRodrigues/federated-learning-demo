"""Every strategy writes results through this one function, so the JSON
schema is guaranteed identical across FedAvg/Shapley/Stackelberg — which
is exactly what lets compare_runs.py stay strategy-agnostic."""
import json
from pathlib import Path
from typing import Any


def log_round(
    results_path: Path,
    round_num: int,
    accuracy: float | None,
    num_clients: int,
    num_free_riders: int,
    strategy_name: str,
    extra: dict[str, Any],
) -> None:
    if results_path.exists():
        data = json.loads(results_path.read_text())
    else:
        data = {
            "strategy": strategy_name,
            "num_clients": num_clients,
            "num_free_riders": num_free_riders,
            "rounds": [],
        }
    row: dict[str, Any] = {"round": round_num, "accuracy": accuracy}
    row.update(extra)
    data["rounds"].append(row)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(data, indent=2))
