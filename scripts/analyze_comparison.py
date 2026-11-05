"""Compara FedAvg, Shapley e Stackelberg nos 8 cenarios de results/ e escreve tabelas em markdown.

Os tres usam 100% dos dados locais: o Stackelberg da comparacao principal e a variante com dados completos
(pasta full_data/). O Stackelberg original, com os dados escolhidos pelo preco, entra numa tabela a parte.

Uso (no WSL, precisa de torch/datasets): python scripts/analyze_comparison.py > results/tabelas_comparacao.md
Cada metrica e calculada por semente; os testes sao pareados por semente (n=5).
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_results as sr

ALPHAS = ["0.5", "0.1"]
ATTACKS = {"silent": 3, "low_effort": 3, "noisy": 3, "fr50": 5}
STRATEGIES = ["fedavg", "shapley", "stackelberg"]
PAIRS = [("shapley", "fedavg"), ("stackelberg", "fedavg"), ("stackelberg", "shapley")]
LAST_ROUNDS = 10


def by_seed(runs, function):
    return {run["seed"]: function(run) for run in runs}


def final_accuracy(run):
    return run["rounds"][-1]["accuracy"]


def final_metric(name):
    # Metricas do modelo global (conjunto de teste, media macro entre as 10 classes) na ultima rodada.
    def read(run, strategy=None):
        return run["rounds"][-1][name]

    return read


def stability(run):
    # Oscilacao da acuracia nas ultimas rodadas dentro da mesma execucao (menor = mais estavel).
    values = [row["accuracy"] for row in run["rounds"][-LAST_ROUNDS:]]
    return float(np.std(values))


def first_round_at_95(run):
    for row in sr.training_rounds(run):
        if row["accuracy"] >= 0.95:
            return int(row["round"])
    return sr.NUM_ROUNDS + 1


def free_rider_reward_share(run, strategy):
    if strategy == "fedavg":
        # FedAvg nao paga: o peso de agregacao e proporcional aos dados de cada cliente.
        sizes = sr.partition_sizes(run["seed"])
        total = sum(sizes[client_id] for client_id in range(sr.NUM_CLIENTS))
        return sum(sizes[client_id] for client_id in sr.FREE_RIDER_IDS) / total
    shares = sr.normalized_shares(sr.cumulative_rewards(run))
    return sum(shares[client_id] for client_id in sr.FREE_RIDER_IDS)


def server_seconds(run):
    return float(np.mean([sr.server_seconds_of(row) for row in sr.training_rounds(run)]))


def detection(run):
    true_positives, false_positives, false_negatives = sr.detection_counts([run])
    flagged = true_positives + false_positives
    precision = true_positives / flagged if flagged > 0 else float("nan")
    recall = true_positives / (true_positives + false_negatives)
    return precision, recall


def mean_sd(values, scale=1.0, digits=1):
    values = np.array(list(values), dtype=float) * scale
    return f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}"


def paired(first, second, scale=1.0, digits=2):
    seeds = sorted(set(first) & set(second))
    a = np.array([first[s] for s in seeds]) * scale
    b = np.array([second[s] for s in seeds]) * scale
    difference = a - b
    if np.allclose(difference, 0):
        return "0,0 (p=1)"
    p_value = stats.ttest_rel(a, b).pvalue
    wins = int((difference > 0).sum())
    return f"{difference.mean():+.{digits}f} (p={p_value:.3f}; {wins}/{len(seeds)})"


def load(alpha, attack):
    folder = Path("results") / f"a{alpha.replace('.', '')}_{attack}"
    sr.SETTINGS["results_dir"] = folder
    sr.set_num_free_riders(ATTACKS[attack])
    runs = {name: sr.load_runs(folder, name) for name in ["fedavg", "shapley"]}
    runs["stackelberg"] = sr.load_runs(folder / "full_data", "stackelberg")
    runs["stackelberg_price"] = sr.load_runs(folder, "stackelberg")
    return runs


def data_fraction_used(run):
    # Share of all the clients' data that took part in training, averaged over the rounds.
    sizes = sr.partition_sizes(run["seed"])
    total = sum(sizes[client_id] for client_id in range(sr.NUM_CLIENTS))
    fractions = []
    for row in sr.training_rounds(run):
        used = sum(row["metadata"]["datasizes"].values())
        fractions.append(used / total)
    return float(np.mean(fractions))


def total_payment(run):
    return float(sum(row["metadata"]["payment_cost"] for row in sr.training_rounds(run)))


def print_table(title, header, rows):
    print(f"\n#### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main():
    for alpha in ALPHAS:
        for attack in ATTACKS:
            runs = load(alpha, attack)
            print(f"\n### α={alpha} — {attack}")
            metrics = {
                "acurácia final (%)": (final_metric("accuracy"), 100, 1),
                "precisão do modelo (%)": (final_metric("precision"), 100, 1),
                "recall do modelo (%)": (final_metric("recall"), 100, 1),
                "F1 do modelo (%)": (final_metric("f1"), 100, 1),
                "oscilação nas últimas 10 rodadas (pp)": (lambda r, s: stability(r), 100, 2),
                "rodada em que atinge 95%": (lambda r, s: first_round_at_95(r), 1, 1),
                "parcela da recompensa dos free-riders (%)": (free_rider_reward_share, 100, 1),
                "tempo do servidor (s/rodada)": (lambda r, s: server_seconds(r), 1, 3),
            }
            rows = []
            tests = []
            for name, (function, scale, digits) in metrics.items():
                values = {}
                for strategy in STRATEGIES:
                    values[strategy] = {run["seed"]: function(run, strategy) for run in runs[strategy]}
                rows.append([name] + [mean_sd(values[s].values(), scale, digits) for s in STRATEGIES])
                tests.append([name] + [paired(values[a], values[b], scale, digits) for a, b in PAIRS])
            print_table("Médias ± desvio entre sementes", ["métrica", "FedAvg", "Shapley", "Stackelberg"], rows)
            print_table(
                "Diferença pareada (p do t pareado; sementes em que o primeiro é maior)",
                ["métrica", "Shapley − FedAvg", "Stackelberg − FedAvg", "Stackelberg − Shapley"],
                tests,
            )
            detection_rows = []
            for strategy in ["shapley", "stackelberg"]:
                values = [detection(run) for run in runs[strategy]]
                precisions = [v[0] for v in values if v[0] == v[0]]
                recall = np.mean([v[1] for v in values])
                if len(precisions) == 0:
                    precision_text = "sem sinalizações"
                else:
                    precision_text = f"{np.mean(precisions):.2f}"
                detection_rows.append([strategy, precision_text, f"{recall:.2f}"])
            print_table(
                "Detecção de free-riders (média das sementes; não é a precisão/recall do modelo)",
                ["mecanismo", "precisão da detecção", "recall da detecção"],
                detection_rows,
            )

            price_rows = []
            for label, key in [("Stackelberg com dados completos", "stackelberg"), ("Stackelberg com dados escolhidos pelo preço", "stackelberg_price")]:
                group = runs[key]
                price_rows.append([
                    label,
                    mean_sd([final_accuracy(r) for r in group], 100, 1),
                    mean_sd([r["rounds"][-1]["f1"] for r in group], 100, 1),
                    mean_sd([data_fraction_used(r) for r in group], 100, 1),
                    mean_sd([total_payment(r) for r in group], 1e-3, 0),
                ])
            print_table(
                "Stackelberg: dados completos × dados escolhidos pelo preço",
                ["variante", "acurácia final (%)", "F1 final (%)", "dados usados (% dos dados de todos os clientes)", "pagamento total (milhares)"],
                price_rows,
            )


if __name__ == "__main__":
    main()
