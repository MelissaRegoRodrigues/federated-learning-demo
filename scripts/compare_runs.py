import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flapp.aggregate import aggregate_cumulative_reward_distance, aggregate_detection_rate, aggregate_metric
from flapp.fairness import reference_shares
from flapp.task import load_partition_dataset

COLORS = {
    "fedavg": "#ff7f0e",
    "shapley": "#2ca02c",
    "stackelberg": "#1f77b4",
    "shapley (V=0.2)": "#98df8a",
    "fedavg (controle)": "#d62728",
    "shapley (V=0)": "#9467bd",
}


def load_results(path):
    with open(path) as file:
        return json.load(file)


def seed_from_path(path):
    match = re.search(r"seed(\d+)", path.name)
    return int(match.group(1))


def reference_for_run(run, alpha, dataset):
    num_clients = run["num_clients"]
    sizes = {}
    for client_id in range(num_clients):
        sizes[client_id] = len(load_partition_dataset(client_id, num_clients, alpha, run["seed"], dataset))
    return reference_shares(sizes, run["num_free_riders"])


def label_for(path, strategy):
    folder = path.parent.name
    if folder == "control":
        return f"{strategy} (controle)"
    if folder == "shapley_v02":
        return "shapley (V=0.2)"
    if folder == "shapley_v0":
        return "shapley (V=0)"
    if folder == "fr0":
        return f"{strategy} (sem free-riders)"
    return strategy


def group_by_label(result_paths):
    runs_by_label = {}
    for path in result_paths:
        data = load_results(path)
        data["seed"] = seed_from_path(path)
        label = label_for(path, data["strategy"])
        if label not in runs_by_label:
            runs_by_label[label] = []
        runs_by_label[label].append(data)
    return runs_by_label


def plot_band(axis, rounds, means, stds, label, color):
    lower = []
    upper = []
    for mean, std in zip(means, stds):
        lower.append(mean - std)
        upper.append(mean + std)
    axis.plot(rounds, means, marker="o", label=label, color=color)
    axis.fill_between(rounds, lower, upper, color=color, alpha=0.2)


def plot_each_seed(axis, runs, metric_name, label, color):
    first_line = True
    for run in runs:
        rounds = [row["round"] for row in run["rounds"]]
        values = [row[metric_name] for row in run["rounds"]]
        if first_line:
            axis.plot(rounds, values, color=color, alpha=0.6, label=label + " (cada semente)")
            first_line = False
        else:
            axis.plot(rounds, values, color=color, alpha=0.6)


def plot_metric(axis, runs_by_label, metric_name):
    for label, runs in runs_by_label.items():
        color = COLORS.get(label)
        if label.startswith("fedavg"):
            plot_each_seed(axis, runs, metric_name, label, color)
            continue
        rounds, means, stds = aggregate_metric(runs, metric_name)
        if rounds:
            plot_band(axis, rounds, means, stds, label, color)


def plot_convergence(runs_by_label, output_path):
    plt.figure(figsize=(10, 6))
    axis = plt.gca()
    plot_metric(axis, runs_by_label, "accuracy")
    plt.xlabel("Round")
    plt.ylabel("Global accuracy (centralized test set)")
    plt.ylim(0, 1.05)
    plt.title("Accuracy per round")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_detection_rate(runs_by_label, output_path):
    plt.figure(figsize=(10, 6))
    axis = plt.gca()
    for label, runs in runs_by_label.items():
        if label.startswith("fedavg"):
            continue
        rounds, means, stds = aggregate_detection_rate(runs)
        if rounds:
            plot_band(axis, rounds, means, stds, label, COLORS.get(label))
    plt.xlabel("Round")
    plt.ylabel("Free-rider detection rate (recall: real free-riders flagged / total)")
    plt.ylim(0, 1.1)
    plt.title("Free-rider detection rate by strategy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_classification_metrics(runs_by_label, output_path):
    metric_names = ["accuracy", "precision", "recall", "f1"]
    figure, axes = plt.subplots(1, len(metric_names), figsize=(20, 5), sharey=True)
    for axis, metric_name in zip(axes, metric_names):
        plot_metric(axis, runs_by_label, metric_name)
        axis.set_xlabel("Round")
        axis.set_title(metric_name.capitalize())
        axis.set_ylim(0, 1.05)
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[0].set_ylabel("Macro-averaged score (centralized test set)")
    figure.suptitle("Accuracy / precision / recall / F1 by strategy")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_reward_distance(runs_by_label, output_path, alpha, dataset):
    plt.figure(figsize=(10, 6))
    axis = plt.gca()
    for label, runs in runs_by_label.items():
        if label.startswith("fedavg"):
            continue
        references = [reference_for_run(run, alpha, dataset) for run in runs]
        rounds, means, stds = aggregate_cumulative_reward_distance(runs, references)
        if rounds:
            plot_band(axis, rounds, means, stds, label, COLORS.get(label))
    plt.xlabel("Round")
    plt.ylabel("Squared distance to the reference reward vector")
    plt.title("Cumulative reward vs. reference (data share for honest clients, 0 for free-riders)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare strategy result JSONs across seeds.")
    parser.add_argument("result_paths", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dataset", choices=["mnist", "fashion-mnist"], default="mnist")
    args = parser.parse_args()

    runs_by_label = group_by_label(args.result_paths)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_convergence(runs_by_label, args.output_dir / "convergence.png")
    plot_detection_rate(runs_by_label, args.output_dir / "detection_rate.png")
    plot_classification_metrics(runs_by_label, args.output_dir / "classification_metrics.png")
    plot_reward_distance(runs_by_label, args.output_dir / "reward_distance.png", args.alpha, args.dataset)
    print(f"Gráficos salvos em {args.output_dir}")


if __name__ == "__main__":
    main()
