"""Entrypoint: runs one strategy's federated simulation and writes
results/<strategy>.json. Replaces the `flwr run` CLI (see spec's
environment findings for why) — this script calls Flower's
run_simulation() directly.

MUST set CUDA_VISIBLE_DEVICES before importing flwr/ray: this machine's
NVIDIA GPU (passed through into WSL2) crashes Ray's pynvml-based
accelerator detection otherwise (confirmed during design).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flwr.simulation import run_simulation

from flapp.client_app import build_client_app
from flapp.server_app import build_server_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one FL aggregation-strategy experiment.")
    parser.add_argument("--strategy", required=True, choices=["fedavg", "shapley", "stackelberg"])
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-rounds", type=int, default=50)
    parser.add_argument("--num-free-riders", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_dir) / f"{args.strategy}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if results_path.exists():
        results_path.unlink()  # start each run from a clean file, not an appended one

    client_app = build_client_app(args.alpha, args.seed, args.num_free_riders, args.local_epochs)
    server_app = build_server_app(
        args.strategy, args.num_rounds, results_path, args.num_clients, args.num_free_riders
    )

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=args.num_clients,
        backend_config={
            "client_resources": {"num_cpus": 1, "num_gpus": 0.0},
            "init_args": {"num_gpus": 0},
        },
    )


if __name__ == "__main__":
    main()
