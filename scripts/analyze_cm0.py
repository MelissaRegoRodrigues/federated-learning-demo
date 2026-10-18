import argparse
import json
from pathlib import Path


def load_run(path):
    with open(path) as file:
        return json.load(file)


def compatibility_ranges(run):
    num_free_riders = run["num_free_riders"]
    free_rider_values = []
    honest_values = []
    for row in run["rounds"][1:]:
        for client_id, value in row.get("client_scores", {}).items():
            if int(client_id) < num_free_riders:
                free_rider_values.append(value)
            else:
                honest_values.append(value)
    return free_rider_values, honest_values


def main():
    parser = argparse.ArgumentParser(description="Interval of Cm0 values that give the same Stackelberg run.")
    parser.add_argument("result_paths", nargs="+", type=Path)
    args = parser.parse_args()

    all_free_riders = []
    all_honest = []
    for path in args.result_paths:
        run = load_run(path)
        free_rider_values, honest_values = compatibility_ranges(run)
        all_free_riders.extend(free_rider_values)
        all_honest.extend(honest_values)
        lowest_honest = min(honest_values) if honest_values else float("nan")
        highest_free_rider = max(free_rider_values) if free_rider_values else float("nan")
        print(f"{path.name}: maior Cm de free-rider {highest_free_rider:.4f}   menor Cm de honesto {lowest_honest:.4f}")

    print()
    if all_free_riders:
        lower = max(all_free_riders)
    else:
        lower = 0.0
    upper = min(all_honest)
    print(f"Qualquer Cm0 em ({lower:.4f}, {upper:.4f}] exclui exatamente os mesmos clientes em todas as execuções.")


if __name__ == "__main__":
    main()
