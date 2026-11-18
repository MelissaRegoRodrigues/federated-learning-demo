"""Teste da H4: o escore de contribuicao de cada mecanismo cresce com a contribuicao de referencia?

Contribuicao de referencia (a mesma de flapp.fairness.reference_shares): 0 para os free-riders e, para os
honestos, a fracao dos dados honestos que o cliente tem. Unidade = cliente x semente.

Parte 1 (escore): Spearman unilateral (rho > 0) entre o escore medio do cliente nas rodadas de treino
(phi do Shapley; Cm do Stackelberg com dados completos) e a referencia.
  - teste principal da H4: todos os clientes (free-riders devem ficar embaixo e, entre os honestos,
    quem tem mais dados deve ficar acima);
  - analise complementar: so honestos, para ver se a ordem entre honestos acompanha os dados.
Holm sobre os 16 testes de cada versao (8 cenarios x 2 mecanismos), em familias separadas.

Parte 2 (recompensa): distancia quadratica entre a parcela de recompensa acumulada e a referencia
(menor = mais proporcional). FedAvg nao paga: a parcela dele e o peso de agregacao (proporcional aos
dados). Teste t pareado por semente entre os mecanismos, Holm sobre os 24 testes (8 cenarios x 3 pares).

Uso (no WSL, mesmo ambiente dos outros scripts):
    python scripts/h4_proportionality.py > results/h4_proporcionalidade.md
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_results as sr
from flapp.fairness import normalized_shares, reference_shares, squared_distance

ALPHAS = ["0.5", "0.1"]
ATTACKS = {"silent": 3, "low_effort": 3, "noisy": 3, "fr50": 5}
# mecanismo -> subpasta onde estao os resultados da comparacao principal
MECHANISMS = {"shapley": "", "stackelberg": "full_data"}
PAIRS = [("shapley", "fedavg"), ("stackelberg", "fedavg"), ("stackelberg", "shapley")]


def mean_scores(run):
    # Escore medio de cada cliente nas rodadas de treino em que ele recebeu escore.
    # No Stackelberg, quem e barrado deixa de receber escore; conta so o que ele recebeu antes.
    scores = {}
    for row in sr.training_rounds(run):
        for client_id, value in row.get("client_scores", {}).items():
            scores.setdefault(int(client_id), []).append(float(value))
    means = {}
    for client_id, values in scores.items():
        means[client_id] = float(np.mean(values))
    return means


def spearman_greater(scores, references):
    # Sem variacao em um dos lados nao ha correlacao para medir.
    if len(set(scores)) < 2 or len(set(references)) < 2:
        return float("nan"), 1.0
    result = stats.spearmanr(scores, references, alternative="greater")
    return float(result.statistic), float(result.pvalue)


def score_test(runs, num_free_riders, honest_only):
    scores = []
    references = []
    for run in runs:
        reference = reference_shares(sr.partition_sizes(run["seed"]), num_free_riders)
        means = mean_scores(run)
        for client_id, value in means.items():
            if honest_only and client_id < num_free_riders:
                continue
            scores.append(value)
            references.append(reference[client_id])
    rho, p_value = spearman_greater(scores, references)
    return rho, p_value, len(scores)


def reward_shares(run, mechanism):
    if mechanism == "fedavg":
        # FedAvg nao paga: a parcela e o peso de agregacao, proporcional aos dados.
        sizes = sr.partition_sizes(run["seed"])
        return normalized_shares({client_id: float(size) for client_id, size in sizes.items()})
    return normalized_shares(sr.cumulative_rewards(run))


def reward_distances(runs, mechanism, num_free_riders):
    distances = {}
    for run in runs:
        reference = reference_shares(sr.partition_sizes(run["seed"]), num_free_riders)
        distances[run["seed"]] = squared_distance(reward_shares(run, mechanism), reference)
    return distances


def paired_t(first, second):
    seeds = sorted(set(first) & set(second))
    difference = np.array([first[s] - second[s] for s in seeds])
    if np.allclose(difference, 0):
        return 0.0, 1.0
    return float(difference.mean()), float(stats.ttest_rel([first[s] for s in seeds], [second[s] for s in seeds]).pvalue)


def set_scenario(alpha, num_free_riders):
    # O summarize_results guarda alpha e numero de free-riders em variaveis globais.
    sr.SETTINGS["alpha"] = float(alpha)
    sr.set_num_free_riders(num_free_riders)


def format_p(p_value):
    if p_value < 1e-4:
        return f"{p_value:.1e}"
    return f"{p_value:.3f}"


def main():
    score_rows = []
    distance_rows = []
    for alpha in ALPHAS:
        for attack, num_free_riders in ATTACKS.items():
            set_scenario(alpha, num_free_riders)
            folder = Path("results") / f"a{alpha.replace('.', '')}_{attack}"
            runs = {"fedavg": sr.load_runs(folder, "fedavg")}
            for mechanism, subfolder in MECHANISMS.items():
                runs[mechanism] = sr.load_runs(folder / subfolder if subfolder else folder, mechanism)
                for honest_only in [False, True]:
                    rho, p_value, n = score_test(runs[mechanism], num_free_riders, honest_only)
                    score_rows.append((alpha, attack, mechanism, honest_only, rho, p_value, n))
            distances = {name: reward_distances(runs[name], name, num_free_riders) for name in runs}
            means = {name: float(np.mean(list(values.values()))) for name, values in distances.items()}
            tests = [paired_t(distances[a], distances[b]) for a, b in PAIRS]
            distance_rows.append((alpha, attack, means, tests))

    print("<!-- Gerado por scripts/h4_proportionality.py. Unidade = cliente x semente. -->")
    print("\n### Parte 1: escore do mecanismo × contribuição de referência (Spearman unilateral, ρ > 0)")
    titles = {False: "Teste principal da H4 (todos os clientes)", True: "Análise complementar (só honestos)"}
    for honest_only in [False, True]:
        rows = [row for row in score_rows if row[3] == honest_only]
        adjusted = sr.holm_adjust([row[5] for row in rows])
        print(f"\n#### {titles[honest_only]}\n")
        print("| α | cenário | mecanismo | n | ρ | p | p Holm (16 testes) |")
        print("|---|---|---|---|---|---|---|")
        for (alpha, attack, mechanism, _, rho, p_value, n), p_holm in zip(rows, adjusted):
            rho_text = "n/a" if rho != rho else f"{rho:+.2f}"
            print(f"| {alpha} | {attack} | {mechanism} | {n} | {rho_text} | {format_p(p_value)} | {format_p(p_holm)} |")

    all_p = [p for _, _, _, tests in distance_rows for _, p in tests]
    distance_adjusted = iter(sr.holm_adjust(all_p))
    print("\n### Parte 2: distância da recompensa à referência (menor = mais proporcional)\n")
    print("| α | cenário | FedAvg | Shapley | Stackelberg | Shapley − FedAvg | Stackelberg − FedAvg | Stackelberg − Shapley |")
    print("|---|---|---|---|---|---|---|---|")
    for alpha, attack, means, tests in distance_rows:
        cells = []
        for difference, p_value in tests:
            cells.append(f"{difference:+.3f} (p={format_p(p_value)}; Holm {format_p(next(distance_adjusted))})")
        print(
            f"| {alpha} | {attack} | {means['fedavg']:.3f} | {means['shapley']:.3f} | {means['stackelberg']:.3f} | "
            + " | ".join(cells) + " |"
        )


if __name__ == "__main__":
    main()
