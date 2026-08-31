"""Strategy registry: to add a 4th aggregation method later, write one
Strategy subclass and add one line here — nothing else in this codebase
needs to change."""
from flapp.strategies.fedavg import LoggingFedAvg
from flapp.strategies.shapley import ShapleyWeightedStrategy
from flapp.strategies.stackelberg import StackelbergStrategy

STRATEGIES = {
    "fedavg": LoggingFedAvg,
    "shapley": ShapleyWeightedStrategy,
    "stackelberg": StackelbergStrategy,
}
