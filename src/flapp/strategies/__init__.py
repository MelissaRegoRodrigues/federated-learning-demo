from flapp.strategies.fedavg import LoggingFedAvg
from flapp.strategies.shapley import ShapleyWeightedStrategy
from flapp.strategies.stackelberg import StackelbergStrategy

STRATEGIES = {
    "fedavg": LoggingFedAvg,
    "shapley": ShapleyWeightedStrategy,
    "stackelberg": StackelbergStrategy,
}
