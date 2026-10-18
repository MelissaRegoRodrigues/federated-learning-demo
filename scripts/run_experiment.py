import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flwr.simulation import run_simulation

from flapp.aggregate import results_filename
from flapp.client_app import build_client_app
from flapp.server_app import build_server_app


def parse_args():
    parser = argparse.ArgumentParser(description="Run one FL aggregation-strategy experiment.")
    parser.add_argument("--strategy", required=True, choices=["fedavg", "shapley", "stackelberg"])
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-rounds", type=int, default=50)
    parser.add_argument("--num-free-riders", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--shapley-voting-rate", type=float, default=0.2, help="Shapley: fraction of lowest contributions excluded.")
    parser.add_argument("--stackelberg-cm0", type=float, default=0.01, help="Stackelberg: compatibility threshold Cm0.")
    parser.add_argument("--stackelberg-model-size", type=int, default=18406, help="Stackelberg: model size in the cost model.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--client-cpus", type=int, default=1, help="CPUs per simulated client.")
    return parser.parse_args()


def check_run_is_complete(results_path, num_rounds):
    if not results_path.exists():
        sys.exit(f"ERRO: {results_path} não foi criado.")
    data = json.loads(results_path.read_text())
    logged_rounds = []
    for row in data["rounds"]:
        logged_rounds.append(row["round"])
    if num_rounds not in logged_rounds:
        last_round = max(logged_rounds, default=0)
        sys.exit(f"ERRO: {results_path} só tem rodadas até {last_round} de {num_rounds}.")


def prepare_results_path(args):
    results_path = Path(args.results_dir) / results_filename(args.strategy, args.seed)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if results_path.exists():
        results_path.unlink()
    return results_path


def make_client_app(args):
    return build_client_app(
        args.alpha,
        args.seed,
        args.num_free_riders,
        args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


def make_server_app(args, results_path):
    return build_server_app(
        args.strategy,
        args.num_rounds,
        results_path,
        args.num_clients,
        args.num_free_riders,
        args.seed,
        alpha=args.alpha,
        voting_rate=args.shapley_voting_rate,
        compatibility_threshold=args.stackelberg_cm0,
        model_size=args.stackelberg_model_size,
    )


def main():
    args = parse_args()
    results_path = prepare_results_path(args)
    run_simulation(
        server_app=make_server_app(args, results_path),
        client_app=make_client_app(args),
        num_supernodes=args.num_clients,
        backend_config={
            "client_resources": {"num_cpus": args.client_cpus, "num_gpus": 0.0},
            "init_args": {"num_gpus": 0},
        },
    )
    check_run_is_complete(results_path, args.num_rounds)


if __name__ == "__main__":
    main()
