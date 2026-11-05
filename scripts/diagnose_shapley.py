import glob
import json
import random
import statistics
import sys

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "results"
SCENARIOS = ["a01_silent", "a05_silent", "a01_noisy", "a05_noisy", "a05_low_effort", "a01_low_effort", "a05_fr50", "a01_fr50"]
ROUND_WINDOWS = [(0, 5), (5, 15), (15, 30), (30, 51)]
RANDOM_TIEBREAK_REPEATS = 20


def load_rounds(scenario):
    """Devolve uma lista de (num_free_riders, num_clients, rodada, {cliente: phi})."""
    rows = []
    for path in sorted(glob.glob(f"{RESULTS_DIR}/{scenario}/shapley_seed*.json")):
        with open(path) as file:
            run = json.load(file)
        for row in run["rounds"]:
            scores = row.get("client_scores")
            if not scores:
                continue
            phi = {}
            for client_id, value in scores.items():
                phi[int(client_id)] = float(value)
            rows.append((run["num_free_riders"], run["num_clients"], row["round"], phi))
    return rows


def voting_rate_for(scenario):
    # Mesma regra do experimento: V = 0.5 no fr50 e 0.3 nos demais.
    if "fr50" in scenario:
        return 0.5
    return 0.3


def recall_of(flagged, num_free_riders):
    real_free_riders = set(range(num_free_riders))
    return len(set(flagged) & real_free_riders) / num_free_riders


def flagged_with_id_tiebreak(phi, how_many):
    # Igual ao codigo: ordena por phi e desempata por client_id crescente.
    ranked = sorted(phi, key=lambda client_id: (phi[client_id], client_id))
    return ranked[:how_many]


def flagged_with_random_tiebreak(phi, how_many):
    # Embaralha antes de ordenar. O sort e estavel, entao empates ficam em ordem aleatoria.
    client_ids = list(phi)
    random.shuffle(client_ids)
    ranked = sorted(client_ids, key=lambda client_id: phi[client_id])
    return ranked[:how_many]


def report_detection(scenario, rows):
    how_many_rate = voting_rate_for(scenario)
    with_id = []
    with_random = []
    tied_at_cutoff = 0
    for num_free_riders, num_clients, _, phi in rows:
        how_many = int(how_many_rate * num_clients)
        with_id.append(recall_of(flagged_with_id_tiebreak(phi, how_many), num_free_riders))
        repeats = []
        for _ in range(RANDOM_TIEBREAK_REPEATS):
            repeats.append(recall_of(flagged_with_random_tiebreak(phi, how_many), num_free_riders))
        with_random.append(statistics.mean(repeats))
        ordered_values = sorted(phi.values())
        if ordered_values[how_many - 1] == ordered_values[how_many]:
            tied_at_cutoff += 1
    print(
        f"{scenario:16} recall(desempate por id)={statistics.mean(with_id):.2f}  "
        f"recall(desempate aleatorio)={statistics.mean(with_random):.2f}  "
        f"rodadas empatadas no corte={tied_at_cutoff / len(rows):.2f}"
    )


def report_by_window(scenario, rows):
    print(scenario)
    for start, end in ROUND_WINDOWS:
        free_rider_phi = []
        honest_phi = []
        rounds_with_ties = 0
        total_rounds = 0
        for num_free_riders, _, round_number, phi in rows:
            if not (start <= round_number < end):
                continue
            total_rounds += 1
            if len(set(phi.values())) < len(phi):
                rounds_with_ties += 1
            for client_id, value in phi.items():
                if client_id < num_free_riders:
                    free_rider_phi.append(value)
                else:
                    honest_phi.append(value)
        if total_rounds == 0:
            continue
        negative_share = sum(1 for value in free_rider_phi if value < 0) / len(free_rider_phi)
        print(
            f"  rodadas {start:2}-{end:2}: "
            f"media phi FR={statistics.mean(free_rider_phi):+.4f} honestos={statistics.mean(honest_phi):+.4f} | "
            f"media |phi| FR={statistics.mean(abs(v) for v in free_rider_phi):.4f} "
            f"honestos={statistics.mean(abs(v) for v in honest_phi):.4f} | "
            f"P(phi_FR<0)={negative_share:.2f} | rodadas com empate={rounds_with_ties / total_rounds:.2f}"
        )


def main():
    random.seed(0)
    print("Resolucao da acuracia: 1 amostra de 6000 = 0.000167\n")
    print("== Deteccao: desempate por id contra aleatorio ==")
    loaded = {}
    for scenario in SCENARIOS:
        rows = load_rounds(scenario)
        if rows:
            loaded[scenario] = rows
            report_detection(scenario, rows)
    print("\n== phi dos free-riders (FR) e honestos por janela de rodadas ==")
    for scenario in ["a01_silent", "a05_silent", "a01_noisy", "a05_noisy"]:
        if scenario in loaded:
            report_by_window(scenario, loaded[scenario])


if __name__ == "__main__":
    main()
