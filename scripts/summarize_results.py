import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flapp.fairness import normalized_shares, reference_shares, squared_distance
from flapp.task import load_partition_dataset

SETTINGS = {"results_dir": Path("results"), "dataset": "mnist"}

FEDAVG = "fedavg"
SHAPLEY = "shapley (V=0.3)"
STACKELBERG = "stackelberg"
SHAPLEY_V02 = "shapley (V=0.2)"
SHAPLEY_V0 = "shapley (V=0, sem exclusão)"
FEDAVG_CONTROL = "fedavg controle"



def group_sources():
    results_dir = SETTINGS["results_dir"]
    return {
        FEDAVG: (results_dir, "fedavg"),
        SHAPLEY: (results_dir, "shapley"),
        STACKELBERG: (results_dir, "stackelberg"),
        SHAPLEY_V02: (results_dir / "shapley_v02", "shapley"),
        SHAPLEY_V0: (results_dir / "shapley_v0", "shapley"),
        FEDAVG_CONTROL: (results_dir / "control", "fedavg"),
    }


MECHANISM_GROUPS = [SHAPLEY, SHAPLEY_V02, STACKELBERG]
REWARD_GROUPS = [SHAPLEY, SHAPLEY_V02, SHAPLEY_V0, STACKELBERG]
COMPARISONS = [
    (STACKELBERG, SHAPLEY),
    (STACKELBERG, FEDAVG),
    (SHAPLEY, FEDAVG),
    (STACKELBERG, FEDAVG_CONTROL),
    (FEDAVG, FEDAVG_CONTROL),
    (SHAPLEY, SHAPLEY_V0),
    (SHAPLEY, SHAPLEY_V02),
]
METRICS = ["accuracy", "precision", "recall", "f1"]
SEEDS = [42, 43, 44, 45, 46]
NUM_CLIENTS = 10
NUM_FREE_RIDERS = 3
NUM_ROUNDS = 50
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


def load_groups():
    groups = {}
    for label, (folder, strategy) in group_sources().items():
        runs = load_runs(folder, strategy)
        if len(runs) > 0:
            groups[label] = runs
    return groups


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
        sizes[client_id] = len(load_partition_dataset(client_id, NUM_CLIENTS, ALPHA, seed, SETTINGS["dataset"]))
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


def print_four_metrics(runs):
    for metric in METRICS:
        values = final_values(runs, metric)
        rounded = [round(value, 3) for value in values]
        print(f"  {metric:10s} {mean_and_std(values)}   por semente: {rounded}")


def print_compact_table(groups):
    print_title("Resumo: rodada final, média ± desvio-padrão entre as sementes")
    print(f"{'grupo':32s} {'n':>2s}  {'acurácia':>15s}  {'precisão':>15s}  {'recall':>15s}  {'F1':>15s}")
    for label, runs in groups.items():
        cells = []
        for metric in METRICS:
            cells.append(mean_and_std(final_values(runs, metric)))
        print(f"{label:32s} {len(runs):>2d}  {cells[0]:>15s}  {cells[1]:>15s}  {cells[2]:>15s}  {cells[3]:>15s}")


def print_model_metrics(groups):
    print_title("Desempenho do modelo global (rodada final, conjunto de teste)")
    for label, runs in groups.items():
        print(f"{label} (n={len(runs)})")
        print_four_metrics(runs)
        divergences = [divergence_round(run) for run in runs]
        reached = [round_that_reached(run, 0.95) for run in runs]
        print(f"  divergências: {count_divergences(runs)}/{len(runs)}   rodada do NaN: {divergences}")
        print(f"  rodada em que atingiu 95%: {reached}")


def paired_test(first_runs, second_runs, metric):
    first = {run["seed"]: run["rounds"][-1][metric] for run in first_runs}
    second = {run["seed"]: run["rounds"][-1][metric] for run in second_runs}
    seeds = sorted(set(first) & set(second))
    a = [first[seed] for seed in seeds]
    b = [second[seed] for seed in seeds]
    differences = np.array(a) - np.array(b)
    if len(seeds) < 3:
        return None
    t_p = stats.ttest_rel(a, b).pvalue
    try:
        wilcoxon_p = stats.wilcoxon(a, b).pvalue
    except ValueError:
        wilcoxon_p = float("nan")
    if differences.std(ddof=1) > 1e-9:
        effect = differences.mean() / differences.std(ddof=1)
    else:
        effect = float("nan")
    return differences.mean(), t_p, wilcoxon_p, effect


def print_statistical_tests(groups):
    print_title("Testes pareados por semente (rodada final): diferença média, t pareado, Wilcoxon, d_z")
    for first, second in COMPARISONS:
        if first not in groups or second not in groups:
            continue
        print(f"{first} - {second}")
        for metric in METRICS:
            result = paired_test(groups[first], groups[second], metric)
            if result is None:
                print(f"  {metric:10s} poucas sementes em comum")
                continue
            difference, t_p, wilcoxon_p, effect = result
            print(f"  {metric:10s} diferença {difference:+.4f}   t p = {t_p:.4f}   Wilcoxon p = {wilcoxon_p:.4f}   d_z = {effect:.2f}")


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


ROUNDS_NOT_REACHED = NUM_ROUNDS + 1


def rounds_to_95_by_seed(runs):
    values = {}
    for run in runs:
        reached = round_that_reached(run, 0.95)
        if reached is None:
            reached = ROUNDS_NOT_REACHED
        values[run["seed"]] = reached
    return values


def metric_by_seed(runs, metric):
    return {run["seed"]: run["rounds"][-1][metric] for run in runs}


def gains_by_seed(mechanism_runs, baseline_runs, metric):
    if metric == "rounds_to_95":
        mechanism = rounds_to_95_by_seed(mechanism_runs)
        baseline = rounds_to_95_by_seed(baseline_runs)
        sign = -1
    else:
        mechanism = metric_by_seed(mechanism_runs, metric)
        baseline = metric_by_seed(baseline_runs, metric)
        sign = 1
    gains = {}
    for seed in sorted(set(mechanism) & set(baseline)):
        gains[seed] = sign * (mechanism[seed] - baseline[seed])
    return gains


GAIN_PAIRS = [
    ("shapley (V=0.3) sobre shapley (V=0)", SHAPLEY, SHAPLEY_V0),
    ("stackelberg sobre fedavg controle", STACKELBERG, FEDAVG_CONTROL),
]


def print_gains(groups):
    print_title("Ganho da exclusão sobre a linha de base de cada mecanismo (mesmos dados e mesmos pesos)")
    first_label, first_mechanism, first_baseline = GAIN_PAIRS[0]
    second_label, second_mechanism, second_baseline = GAIN_PAIRS[1]
    needed = [first_mechanism, first_baseline, second_mechanism, second_baseline]
    if any(label not in groups for label in needed):
        print("faltam grupos para calcular os ganhos")
        return
    print("ganho = mecanismo - linha de base (para rounds_to_95: rodadas economizadas; não atingiu = 51)")
    for metric in METRICS + ["rounds_to_95"]:
        first = gains_by_seed(groups[first_mechanism], groups[first_baseline], metric)
        second = gains_by_seed(groups[second_mechanism], groups[second_baseline], metric)
        seeds = sorted(set(first) & set(second))
        print(f"{metric}")
        print(f"  {first_label}: {mean_and_std(list(first.values()))}")
        print(f"  {second_label}: {mean_and_std(list(second.values()))}")
        if len(seeds) < 3:
            print("  poucas sementes em comum para o teste")
            continue
        a = [first[seed] for seed in seeds]
        b = [second[seed] for seed in seeds]
        difference = np.mean(np.array(a) - np.array(b))
        p_value = stats.ttest_rel(a, b).pvalue
        print(f"  diferença entre os ganhos (shapley - stackelberg) {difference:+.4f}   t pareado p = {p_value:.4f}")


def print_detection(groups):
    print_title("Detecção de free-riders")
    for label in MECHANISM_GROUPS:
        if label not in groups:
            continue
        runs = groups[label]
        true_positives, false_positives, false_negatives = detection_counts(runs)
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = 2 * true_positives / max(2 * true_positives + false_positives + false_negatives, 1)
        print(f"{label}: precisão {precision:.3f}  recall {recall:.3f}  F1 {f1:.3f}  falsos positivos {false_positives}")
        print(f"  free-riders sinalizados em {np.mean(flag_rate(runs, FREE_RIDER_IDS)):.2f} das rodadas")
        print(f"  honestos sinalizados em {np.mean(flag_rate(runs, HONEST_IDS)):.2f} das rodadas")


def print_heterogeneity(groups):
    print_title("Shapley: sinalização de honestos x tamanho da partição")
    if SHAPLEY not in groups:
        return
    sizes = []
    rates = []
    for run in groups[SHAPLEY]:
        partition = partition_sizes(run["seed"])
        for client_id in HONEST_IDS:
            sizes.append(partition[client_id])
            rates.append(flag_rate([run], [client_id])[0])
    result = stats.spearmanr(sizes, rates)
    print(f"Spearman (n={len(sizes)}): rho = {result.statistic:.2f}  p = {result.pvalue:.4f}")


def uniform_honest_reference():
    reference = {}
    for client_id in range(NUM_CLIENTS):
        if client_id in FREE_RIDER_IDS:
            reference[client_id] = 0.0
        else:
            reference[client_id] = 1 / len(HONEST_IDS)
    return reference


def cumulative_rewards(run):
    totals = {}
    for client_id in range(NUM_CLIENTS):
        totals[client_id] = 0.0
    for row in training_rounds(run):
        for client_id, value in row["metadata"]["rewards"].items():
            totals[int(client_id)] += value
    return totals


def print_reward_distance(groups):
    print_title("Recompensa: parcela dos free-riders e distância a duas referências")
    for label in REWARD_GROUPS:
        if label not in groups:
            continue
        proportional = []
        uniform = []
        free_rider_shares = []
        for run in groups[label]:
            shares = normalized_shares(cumulative_rewards(run))
            reference = reference_shares(partition_sizes(run["seed"]), NUM_FREE_RIDERS)
            proportional.append(squared_distance(shares, reference))
            uniform.append(squared_distance(shares, uniform_honest_reference()))
            free_rider_shares.append(sum(shares[client_id] for client_id in FREE_RIDER_IDS))
        print(f"{label}: parcela dos free-riders {np.mean(free_rider_shares) * 100:.1f}%")
        print(f"  distância à referência proporcional aos dados {mean_and_std(proportional)}")
        print(f"  distância à referência uniforme entre honestos {mean_and_std(uniform)}")


def print_datasize_fraction(groups):
    print_title("Stackelberg: fração dos dados locais usada por rodada (honestos)")
    if STACKELBERG not in groups:
        return
    fractions = []
    for run in groups[STACKELBERG]:
        partition = partition_sizes(run["seed"])
        for row in training_rounds(run):
            for client_id, datasize in row["metadata"]["datasizes"].items():
                if int(client_id) in HONEST_IDS and datasize > 0:
                    fractions.append(datasize / partition[int(client_id)])
    print(f"média {np.mean(fractions) * 100:.1f}%   mínimo {np.min(fractions) * 100:.1f}%   máximo {np.max(fractions) * 100:.1f}%")
    print("FedAvg e Shapley treinam com 100% dos dados locais; o FedAvg de controle usa a mesma quantidade do Stackelberg.")


def server_seconds_of(row):
    metadata = row["metadata"]
    if "server_seconds" in metadata:
        return metadata["server_seconds"]
    return metadata["aggregate_seconds"]


def print_server_time(groups):
    print_title("Tempo do servidor por rodada (aggregate + preço; calibração de t0 dividida pelas rodadas)")
    for label, runs in groups.items():
        seconds = []
        calibration = []
        for run in runs:
            for row in training_rounds(run):
                seconds.append(server_seconds_of(row))
                calibration.append(row["metadata"].get("calibration_seconds", 0.0))
        per_round = np.mean(seconds) + np.mean(calibration) / NUM_ROUNDS
        print(f"{label}: {per_round:.4f} s por rodada   (server_seconds {np.mean(seconds):.4f} s, calibração única {np.mean(calibration):.3f} s)")


def print_stackelberg_market(groups):
    print_title("Jogo de Stackelberg: preço, clientes incluídos e pagamento")
    if STACKELBERG not in groups:
        return
    rows = []
    for run in groups[STACKELBERG]:
        rows.extend(training_rounds(run))
    prices = [row["metadata"]["price"] for row in rows]
    included = [row["metadata"]["num_participated"] for row in rows]
    payments = [row["metadata"]["payment_cost"] for row in rows]
    total_times = sorted({round(row["metadata"]["total_time"], 1) for row in rows})
    print(f"preço {np.mean(prices):.3f} ± {np.std(prices):.3f}   clientes incluídos por rodada {np.mean(included):.2f}")
    print(f"pagamento médio por rodada {np.mean(payments):.1f}   t0 calibrado {total_times}")


def print_zero_free_riders():
    print_title("Cenário sem free-riders (pasta fr0)")
    groups = {}
    sources = group_sources()
    for label in [FEDAVG, SHAPLEY, STACKELBERG]:
        strategy = sources[label][1]
        runs = load_runs(SETTINGS["results_dir"] / "fr0", strategy)
        if len(runs) > 0:
            groups[label] = runs
    if len(groups) == 0:
        print("sem resultados")
        return
    for label, runs in groups.items():
        flagged = 0
        rounds = 0
        for run in runs:
            for row in training_rounds(run):
                flagged += len(row.get("detected_free_riders", []))
                rounds += 1
        print(f"{label} (n={len(runs)}): honestos sinalizados {flagged} ({flagged / rounds:.1f} por rodada)")
        print_four_metrics(runs)


def main():
    parser = argparse.ArgumentParser(description="Print the numbers of the thesis tables.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--dataset", choices=["mnist", "fashion-mnist"], default="mnist")
    args = parser.parse_args()
    SETTINGS["results_dir"] = args.results_dir
    SETTINGS["dataset"] = args.dataset

    groups = load_groups()

    print_compact_table(groups)
    print_model_metrics(groups)
    print_statistical_tests(groups)
    print_gains(groups)
    print_detection(groups)
    print_heterogeneity(groups)
    print_reward_distance(groups)
    print_datasize_fraction(groups)
    print_server_time(groups)
    print_stackelberg_market(groups)
    print_zero_free_riders()


if __name__ == "__main__":
    main()
