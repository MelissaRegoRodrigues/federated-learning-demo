"""Flower ServerApp: picks a Strategy by name from the registry and wires
in the centralized MNIST-test-set evaluator every strategy shares."""
from pathlib import Path

from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from flapp.strategies import STRATEGIES
from flapp.task import SimpleCNN, load_test_set, set_params, test


def make_centralized_eval_fn():
    test_loader = load_test_set()

    def evaluate_fn(server_round: int, ndarrays, config: dict):
        model = SimpleCNN()
        set_params(model, ndarrays)
        accuracy = test(model, test_loader)
        return 0.0, {"accuracy": accuracy}

    return evaluate_fn


def build_server_app(
    strategy_name: str,
    num_rounds: int,
    results_path: Path,
    num_clients: int,
    num_free_riders: int,
) -> ServerApp:
    def server_fn(context: Context):
        strategy_cls = STRATEGIES[strategy_name]
        strategy = strategy_cls(
            evaluate_fn=make_centralized_eval_fn(),
            results_path=results_path,
            num_clients=num_clients,
            num_free_riders=num_free_riders,
            fraction_fit=1.0,
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
        ) if strategy_name == "fedavg" else strategy_cls(
            evaluate_fn=make_centralized_eval_fn(),
            results_path=results_path,
            num_clients=num_clients,
            num_free_riders=num_free_riders,
        )
        return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=num_rounds))

    return ServerApp(server_fn=server_fn)
