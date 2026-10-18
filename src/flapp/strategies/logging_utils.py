import json


def log_round(results_path, round_num, metrics, num_clients, num_free_riders, strategy_name, extra=None):
    if results_path.exists():
        data = json.loads(results_path.read_text())
    else:
        data = {
            "strategy": strategy_name,
            "num_clients": num_clients,
            "num_free_riders": num_free_riders,
            "rounds": [],
        }

    row = {"round": round_num}
    row.update(metrics)
    if extra:
        row.update(extra)
    data["rounds"].append(row)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(data, indent=2))
