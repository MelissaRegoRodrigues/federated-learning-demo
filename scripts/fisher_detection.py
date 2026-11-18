"""Teste exato de Fisher da deteccao de free-riders (Shapley e Stackelberg da comparacao principal).

Pergunta: o mecanismo sinaliza free-riders com mais frequencia do que sinaliza honestos?
Para cada cenario e mecanismo monta-se a tabela 2x2 abaixo:

                  sinalizado   nao sinalizado
    free-rider        VP             FN
    honesto           FP             VN

A unidade de contagem depende de como o mecanismo decide:
- Shapley decide de novo a cada rodada, sem memoria: unidade = cliente x rodada de treino, somando as
  5 sementes (as mesmas contagens da precisao/recall da deteccao). Ressalva: as rodadas de uma mesma
  execucao nao sao independentes, entao o p e otimista (anticonservador).
- Stackelberg barra o cliente para sempre: a decisao da rodada 1 se repete em todas as seguintes, e contar
  rodadas multiplicaria uma unica decisao por 49. Unidade = cliente x semente (barrado ou nao no fim).

O teste e unilateral (razao de chances > 1). Sob deteccao ao acaso a razao de chances e 1.
A correcao de Holm e aplicada sobre os 16 testes (8 cenarios x 2 mecanismos).

Uso (no WSL, mesmo ambiente dos outros scripts): python scripts/fisher_detection.py
"""
import sys
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_results as sr

ALPHAS = ["0.5", "0.1"]
ATTACKS = {"silent": 3, "low_effort": 3, "noisy": 3, "fr50": 5}
# mecanismo -> (subpasta dos resultados, unidade de contagem)
MECHANISMS = {"shapley": ("", "rodada"), "stackelberg": ("full_data", "cliente")}


def count_flags(flagged_sets, free_rider_ids, num_clients):
    # Cada conjunto e uma observacao: quem foi sinalizado entre free-riders e honestos.
    free_riders = set(free_rider_ids)
    honest_ids = set(range(num_clients)) - free_riders
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0
    for flagged in flagged_sets:
        true_positives += len(flagged & free_riders)
        false_negatives += len(free_riders - flagged)
        false_positives += len(flagged & honest_ids)
        true_negatives += len(honest_ids - flagged)
    return [[true_positives, false_negatives], [false_positives, true_negatives]]


def contingency(runs, free_rider_ids, num_clients):
    # Unidade = cliente x rodada de treino (para mecanismos que decidem de novo a cada rodada).
    flagged_sets = []
    for run in runs:
        for row in sr.training_rounds(run):
            flagged_sets.append(set(row["detected_free_riders"]))
    return count_flags(flagged_sets, free_rider_ids, num_clients)


def contingency_by_client(runs, free_rider_ids, num_clients):
    # Unidade = cliente x semente: quem estava barrado na ultima rodada (exclusao permanente).
    flagged_sets = [set(run["rounds"][-1]["detected_free_riders"]) for run in runs]
    return count_flags(flagged_sets, free_rider_ids, num_clients)


def fisher_one_sided(table):
    # Sem nenhuma sinalizacao (ou sinalizando todo mundo) nao ha o que testar: p = 1.
    flagged_total = table[0][0] + table[1][0]
    not_flagged_total = table[0][1] + table[1][1]
    if flagged_total == 0 or not_flagged_total == 0:
        return float("nan"), 1.0
    result = stats.fisher_exact(table, alternative="greater")
    return result.statistic, result.pvalue


def main():
    rows = []
    for alpha in ALPHAS:
        for attack, num_free_riders in ATTACKS.items():
            folder = Path("results") / f"a{alpha.replace('.', '')}_{attack}"
            for mechanism, (subfolder, unit) in MECHANISMS.items():
                runs = sr.load_runs(folder / subfolder if subfolder else folder, mechanism)
                if unit == "cliente":
                    table = contingency_by_client(runs, range(num_free_riders), sr.NUM_CLIENTS)
                else:
                    table = contingency(runs, range(num_free_riders), sr.NUM_CLIENTS)
                odds_ratio, p_value = fisher_one_sided(table)
                rows.append((alpha, attack, mechanism, unit, table, odds_ratio, p_value))

    adjusted = sr.holm_adjust([row[6] for row in rows])
    print("| α | cenário | mecanismo | unidade | VP | FN | FP | VN | razão de chances | p (Fisher, unilateral) | p Holm |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (alpha, attack, mechanism, unit, table, odds_ratio, p_value), p_holm in zip(rows, adjusted):
        (tp, fn), (fp, tn) = table
        print(
            f"| {alpha} | {attack} | {mechanism} | {unit} | {tp} | {fn} | {fp} | {tn} | "
            f"{format_odds(odds_ratio)} | {format_p(p_value)} | {format_p(p_holm)} |"
        )


def format_odds(odds_ratio):
    if odds_ratio != odds_ratio:
        return "n/a"
    if odds_ratio == float("inf"):
        return "∞"
    return f"{odds_ratio:.2f}"


def format_p(p_value):
    # O p de uma separacao perfeita com muitas rodadas fica abaixo do menor float e aparece como 0.
    if p_value < 1e-300:
        return "< 1e-300"
    return f"{p_value:.2g}"


if __name__ == "__main__":
    main()
