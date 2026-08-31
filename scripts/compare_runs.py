"""Reads the three strategies' result JSONs (produced independently by
three separate `python scripts/run_experiment.py --strategy ...` runs)
and plots them together. No Flower import here — this mirrors how
Flower-based simulation research is actually done: independent runs,
compared offline."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flapp.equity import reward_equity_index


def load_results(path: Path) -> dict:
    return json.loads(path.read_text())


def plot_convergence(results_by_strategy: dict[str, dict], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for strategy_name, data in results_by_strategy.items():
        rounds = [r["round"] for r in data["rounds"]]
        accuracies = [r["accuracy"] for r in data["rounds"]]
        plt.plot(rounds, accuracies, marker="o", label=strategy_name)
    plt.xlabel("Round")
    plt.ylabel("Global accuracy (centralized test set)")
    plt.title("FedAvg vs. Shapley-Weighted vs. Stackelberg")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_detection_rate(results_by_strategy: dict[str, dict], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for strategy_name, data in results_by_strategy.items():
        if strategy_name == "fedavg":
            continue  # FedAvg has no free-rider detection mechanism
        num_free_riders = data["num_free_riders"]
        rounds = [r["round"] for r in data["rounds"]]
        detected_counts = [len(r.get("detected_free_riders", [])) for r in data["rounds"]]
        rate = [c / num_free_riders if num_free_riders else 0 for c in detected_counts]
        plt.plot(rounds, rate, marker="o", label=strategy_name)
    plt.xlabel("Round")
    plt.ylabel("Free-rider detection rate")
    plt.ylim(0, 1.1)
    plt.title("Free-rider detection rate by strategy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_equity_index(results_by_strategy: dict[str, dict], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for strategy_name, data in results_by_strategy.items():
        if strategy_name == "fedavg":
            continue  # FedAvg has no per-client scores to evaluate equity of
        num_free_riders = data["num_free_riders"]
        rounds, indices = [], []
        for r in data["rounds"]:
            if "client_scores" not in r:
                continue
            index = reward_equity_index(r["client_scores"], num_free_riders)
            if index is not None:
                rounds.append(r["round"])
                indices.append(index)
        if rounds:
            plt.plot(rounds, indices, marker="o", label=strategy_name)
    plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.xlabel("Round")
    plt.ylabel("Reward equity index (point-biserial correlation)")
    plt.ylim(-1.1, 1.1)
    plt.title("Does each strategy's score rank honest clients above free-riders?")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare strategy result JSONs.")
    parser.add_argument("result_paths", nargs=3, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    results_by_strategy = {}
    for path in args.result_paths:
        data = load_results(path)
        results_by_strategy[data["strategy"]] = data

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_convergence(results_by_strategy, args.output_dir / "convergence.png")
    plot_detection_rate(results_by_strategy, args.output_dir / "detection_rate.png")
    plot_equity_index(results_by_strategy, args.output_dir / "equity_index.png")
    print(
        f"Wrote {args.output_dir / 'convergence.png'}, "
        f"{args.output_dir / 'detection_rate.png'}, and "
        f"{args.output_dir / 'equity_index.png'}"
    )


if __name__ == "__main__":
    main()
