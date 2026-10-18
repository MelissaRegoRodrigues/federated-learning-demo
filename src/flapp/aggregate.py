import warnings

import numpy as np

from flapp.fairness import normalized_shares, squared_distance


def results_filename(strategy, seed):
    return f"{strategy}_seed{seed}.json"


def mean_std_across_runs(values_per_run):
    if len(values_per_run) == 0:
        return [], [], []

    common_rounds = set(values_per_run[0])
    for values in values_per_run[1:]:
        common_rounds = common_rounds & set(values)

    rounds = sorted(common_rounds)
    means = []
    stds = []
    for round_number in rounds:
        round_values = []
        for values in values_per_run:
            round_values.append(values[round_number])
        means.append(float(np.mean(round_values)))
        stds.append(float(np.std(round_values, ddof=1)))
    return rounds, means, stds


def mean_std_by_round(results_list, extract):
    values_per_run = []
    for run in results_list:
        values = {}
        for row in run["rounds"]:
            value = extract(row, run)
            if value is not None:
                values[row["round"]] = value

        if len(values) == 0:
            warnings.warn(
                "mean_std_by_round: one run produced no valid values for the "
                "requested metric — it will be excluded from the aggregate.",
                stacklevel=3,
            )
            continue
        values_per_run.append(values)

    return mean_std_across_runs(values_per_run)


def aggregate_metric(results_list, metric_name):
    def extract(row, run):
        return row.get(metric_name)

    return mean_std_by_round(results_list, extract)


def aggregate_detection_rate(results_list):
    def extract(row, run):
        num_free_riders = run["num_free_riders"]
        if not num_free_riders:
            return None
        real_free_riders = set(range(num_free_riders))
        flagged = set(row.get("detected_free_riders", []))
        caught = flagged & real_free_riders
        return len(caught) / num_free_riders

    return mean_std_by_round(results_list, extract)


def reward_shares(row):
    rewards = {}
    for client_id, value in row["metadata"]["rewards"].items():
        rewards[int(client_id)] = value
    return normalized_shares(rewards)


def cumulative_reward_distance_by_round(run, reference):
    totals = {}
    distance_by_round = {}
    for row in run["rounds"]:
        if "metadata" not in row or "rewards" not in row["metadata"]:
            continue
        for client_id, value in row["metadata"]["rewards"].items():
            key = int(client_id)
            if key not in totals:
                totals[key] = 0.0
            totals[key] += value
        distance_by_round[row["round"]] = squared_distance(normalized_shares(totals), reference)
    return distance_by_round


def aggregate_cumulative_reward_distance(results_list, references):
    values_per_run = []
    for run, reference in zip(results_list, references):
        values = cumulative_reward_distance_by_round(run, reference)
        if len(values) > 0:
            values_per_run.append(values)
    return mean_std_across_runs(values_per_run)
