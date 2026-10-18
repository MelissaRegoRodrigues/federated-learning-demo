import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flapp.fairness import normalized_shares, reference_shares, squared_distance
from flapp.task import load_partition_dataset

RESULTS_DIR = Path("results")
ZERO_FREE_RIDERS_DIR = RESULTS_DIR / "fr0"
STRATEGIES = ["fedavg", "shapley", "stackelberg"]
MECHANISMS = ["shapley", "stackelberg"]
SEEDS = [42, 43, 44, 45, 46]
NUM_CLIENTS = 10
NUM_FREE_RIDERS = 3
ALPHA = 0.5
FREE_RIDER_IDS = set(range(NUM_FREE_RIDERS))
HONEST_IDS = list(range(NUM_FREE_RIDERS, NUM_CLIENTS))


def load_runs(folder, strategy):
    runs = []
    for seed in SEEDS:
        path = folder / f"{strategy}_seed{seed}.json"
        if path.exists():
            with open(path) as file:
                run = json.load(file)
            run["seed"] = seed
            runs.append(run)
    return runs


def training_rounds(run):
    return run["rounds"][1:]


def final_values(runs, metric):
    return [run["rounds"][-1][metric] for run in runs]


def is_nan(value):
    return value != value


def mean_and_std(values):
    values = [value for value in values if not is_nan(value)]
    if len(values) == 0:
        return "-"
    if len(values) > 1:
        std = np.std(values, ddof=1)
    else:
        std = 0.0
    return f"{np.mean(values):.3f} ± {std:.3f}"


def print_title(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def partition_sizes(seed):
    sizes = {}
    for client_id in range(NUM_CLIENTS):
        sizes[client_id] = len(load_partition_dataset(client_id, NUM_CLIENTS, ALPHA, seed))
    return sizes


def divergence_round(run):
    for row in run["rounds"]:
        if is_nan(row["loss"]):
            return row["round"]
    return None


def count_divergences(runs):
    return len([run for run in runs if divergence_round(run) is not None])


def round_that_reached(run, target_accuracy):
    for row in run["rounds"]:
        if row["accuracy"] >= target_accuracy:
            return row["round"]
    return None


def print_model_metrics(runs_by_strategy):
    print_title("Desempenho do modelo global (rodada final, conjunto de teste)")
    for strategy in STRATEGIES:
        runs = runs_by_strategy[strategy]
        print(f"{strategy} (n={len(runs)})")
        for metric in ["accuracy", "precision", "recall", "f1"]:
            values = final_values(runs, metric)
            rounded = [round(value, 3) for value in values]
            print(f"  {metric:10s} {mean_and_std(values)}   por semente: {rounded}")
        divergences = [divergence_round(run) for run in runs]
        reached = [round_that_reached(run, 0.95) for run in runs]
        print(f"  divergências: {count_divergences(runs)}/{len(runs)}   rodada do NaN: {divergences}")
        print(f"  rodada em que atingiu 95%: {reached}")


def print_statistical_tests(runs_by_strategy):
    print_title("Testes estatísticos pareados (acurácia final)")
    pairs = [("stackelberg", "shapley"), ("stackelberg", "fedavg"), ("shapley", "fedavg")]
    for first, second in pairs:
        a = final_values(runs_by_strategy[first], "accuracy")
        b = final_values(runs_by_strategy[second], "accuracy")
        differences = np.array(a) - np.array(b)
        print(f"{first} - {second}: diferenças {np.round(differences, 3)}")
        print(f"  t pareado p = {stats.ttest_rel(a, b).pvalue:.4f}   Wilcoxon p = {stats.wilcoxon(a, b).pvalue:.4f}"
              f"   d_z = {differences.mean() / differences.std(ddof=1):.2f}")


def detection_counts(runs):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for run in runs:
        for row in training_rounds(run):
            flagged = set(row["detected_free_riders"])
            true_positives += len(flagged & FREE_RIDER_IDS)
            false_positives += len(flagged - FREE_RIDER_IDS)
            false_negatives += len(FREE_RIDER_IDS - flagged)
    return true_positives, false_positives, false_negatives


def flag_rate(runs, client_ids):
    rates = []
    for run in runs:
        rounds = training_rounds(run)
        for client_id in client_ids:
            flagged = len([row for row in rounds if client_id in row["detected_free_riders"]])
            rates.append(flagged / len(rounds))
    return rates


def print_detection(runs_by_strategy):
    print_title("Detecção de free-riders")
    for strategy in MECHANISMS:
        runs = runs_by_strategy[strategy]
        true_positives, false_positives, false_negatives = detection_counts(runs)
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = 2 * true_positives / max(2 * true_positives + false_positives + false_negatives, 1)
        print(f"{strategy}: precisão {precision:.3f}  recall {recall:.3f}  F1 {f1:.3f}  falsos positivos {false_positives}")
        print(f"  free-riders sinalizados em {np.mean(flag_rate(runs, FREE_RIDER_IDS)):.2f} das rodadas")
        print(f"  honestos sinalizados em {np.mean(flag_rate(runs, HONEST_IDS)):.2f} das rodadas")


def print_heterogeneity(runs_by_strategy):
    print_title("Shapley: sinalização de honestos x tamanho da partição")
    sizes = []
    rates = []
    for run in runs_by_strategy["shapley"]:
        partition = partition_sizes(run["seed"])
        for client_id in HONEST_IDS:
            sizes.append(partition[client_id])
            rates.append(flag_rate([run], [client_id])[0])
    result = stats.spearmanr(sizes, rates)
    print(f"Spearman (n={len(sizes)}): rho = {result.statistic:.2f}  p = {result.pvalue:.4f}")


def cumulative_rewards(run):
    totals = {}
    for client_id in range(NUM_CLIENTS):
        totals[client_id] = 0.0
    for row in training_rounds(run):
        for client_id, value in row["metadata"]["rewards"].items():
            totals[int(client_id)] += value
    return totals


def print_reward_distance(runs_by_strategy):
    print_title("Recompensa: distância ao vetor de referência e parcela dos free-riders")
    for strategy in MECHANISMS:
        distances = []
        free_rider_shares = []
        for run in runs_by_strategy[strategy]:
            reference = reference_shares(partition_sizes(run["seed"]), NUM_FREE_RIDERS)
            shares = normalized_shares(cumulative_rewards(run))
            distances.append(squared_distance(shares, reference))
            free_rider_shares.append(sum(shares[client_id] for client_id in FREE_RIDER_IDS))
        print(f"{strategy}: distância acumulada {mean_and_std(distances)}"
              f"   parcela dos free-riders {np.mean(free_rider_shares) * 100:.1f}%")


def mean_aggregate_seconds(runs):
    seconds = []
    for run in runs:
        for row in training_rounds(run):
            seconds.append(row["metadata"]["aggregate_seconds"])
    return np.mean(seconds)


def print_cost_and_market(runs_by_strategy):
    print_title("Custo de agregação e jogo de Stackelberg")
    for strategy in MECHANISMS:
        print(f"{strategy}: tempo de agregação por rodada {mean_aggregate_seconds(runs_by_strategy[strategy]):.3f} s")

    rows = []
    for run in runs_by_strategy["stackelberg"]:
        rows.extend(training_rounds(run))
    prices = [row["metadata"]["price"] for row in rows]
    included = [row["metadata"]["num_participated"] for row in rows]
    honest_datasizes = []
    for row in rows:
        for client_id, value in row["metadata"]["datasizes"].items():
            if int(client_id) in HONEST_IDS and value > 0:
                honest_datasizes.append(value)
    total_times = sorted({row["metadata"]["total_time"] for row in rows})
    print(f"stackelberg: preço {np.mean(prices):.3f} ± {np.std(prices):.3f}   clientes incluídos por rodada {np.mean(included):.2f}")
    print(f"  amostras por honesto por rodada {np.mean(honest_datasizes):.0f}   t0 calibrado {total_times}")
    print(f"  pagamento médio por rodada {np.mean([row['metadata']['payment_cost'] for row in rows]):.1f}")


def print_zero_free_riders():
    print_title("Cenário sem free-riders (results/fr0)")
    for strategy in STRATEGIES:
        runs = load_runs(ZERO_FREE_RIDERS_DIR, strategy)
        if len(runs) == 0:
            print(f"{strategy}: sem resultados")
            continue
        flagged = 0
        rounds = 0
        for run in runs:
            for row in training_rounds(run):
                flagged += len(row.get("detected_free_riders", []))
                rounds += 1
        accuracy = mean_and_std(final_values(runs, "accuracy"))
        print(f"{strategy} (n={len(runs)}): acurácia {accuracy}   honestos sinalizados {flagged} ({flagged / rounds:.1f} por rodada)")


def main():
    runs_by_strategy = {}
    for strategy in STRATEGIES:
        runs_by_strategy[strategy] = load_runs(RESULTS_DIR, strategy)

    print_model_metrics(runs_by_strategy)
    print_statistical_tests(runs_by_strategy)
    print_detection(runs_by_strategy)
    print_heterogeneity(runs_by_strategy)
    print_reward_distance(runs_by_strategy)
    print_cost_and_market(runs_by_strategy)
    print_zero_free_riders()


if __name__ == "__main__":
    main()
