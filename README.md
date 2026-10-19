# Federated Learning + Game Theory — Free-Rider Mitigation

This code compares three aggregation strategies on
MNIST split non-IID across 10 simulated clients, 3 of which are free-riders:

- **FedAvg**: plain weighted average (baseline).
- **Shapley**: leave-one-out contributions; the lowest 30% (the free-rider rate) are excluded
  from aggregation and reward (ConDA).
- **Stackelberg**: EntroCFL game where the server sets a payment per data, each client picks how
  much data to train on, and clients incompatible with the others are excluded without payment.

Free-riders return the global model without training.

Built on [Flower](https://flower.ai/) and PyTorch.



## Setup

Run inside **WSL2 (Ubuntu)**: Flower's Ray simulation backend does not work on native
Windows.

```bash
python3 -m venv ~/.venvs/fl-demo
source ~/.venvs/fl-demo/bin/activate
pip install -r requirements.txt
```

If `python3 -m venv` fails because `ensurepip` is missing
## Run

One experiment (defaults: 10 clients, 3 free-riders, 50 rounds, Dirichlet α = 0.5):
```bash
python scripts/run_experiment.py --strategy stackelberg --seed 42
```
This writes `results/stackelberg_seed42.json`. Run one experiment at a time: two at once do
not fit in the 15 GB WSL memory.

All experiments of the thesis. The function runs the whole comparison of one scenario; the
Stackelberg of a seed runs before its control, because the control reads the price of each round from it.
```bash
run_comparison() {
  base=$1; seeds=$2; shift 2
  for seed in $seeds; do
    python scripts/run_experiment.py --strategy stackelberg --seed $seed --results-dir $base "$@"
    python scripts/run_experiment.py --strategy fedavg --seed $seed --results-dir $base "$@"
    python scripts/run_experiment.py --strategy shapley --seed $seed --results-dir $base "$@"
    python scripts/run_experiment.py --strategy shapley --seed $seed --shapley-voting-rate 0 --results-dir $base/shapley_v0 "$@"
    python scripts/run_experiment.py --strategy fedavg --seed $seed --results-dir $base/control --price-from $base/stackelberg_seed$seed.json "$@"
  done
}

run_comparison results "42 43 44 45 46"
run_comparison results/fashion_low_effort "42 43 44" --dataset fashion-mnist --free-rider-type low-effort
```

Two scenarios: MNIST with silent free-riders (5 seeds, about 8 hours) and Fashion-MNIST with low-effort
free-riders (3 seeds, about 5 hours). Run one experiment at a time.

Optional extras, not needed for the main comparison: Shapley with voting rate 0.2
(`--shapley-voting-rate 0.2 --results-dir results/shapley_v02`) and a scenario without free-riders
(`--num-free-riders 0 --results-dir results/fr0`). Other combinations of `--dataset` and
`--free-rider-type` also work.

- `results/`: the three mechanisms (Shapley with voting rate 0.3).
- `results/shapley_v0/`: Shapley without exclusion (uniform weights), the baseline of the Shapley
  mechanism.
- `results/control/`: FedAvg where every client trains on the amount of data it would choose in the
  Stackelberg game (the game without the compatibility check), the baseline of the Stackelberg mechanism.
- `results/fashion_low_effort/`: the same comparison on Fashion-MNIST, with low-effort free-riders
  (they train 1 epoch instead of 2).

Run `python scripts/run_experiment.py --help` to see every option.

## Analyze

```bash
python scripts/compare_runs.py results/*_seed*.json
python scripts/summarize_results.py
# other scenarios: add --results-dir results/fashion --dataset fashion-mnist
python scripts/analyze_cm0.py results/stackelberg_seed*.json
```

- `compare_runs.py`: plots of accuracy, detection, precision/recall/F1 and reward distance in
  `results/`.
- `summarize_results.py`: every number of the thesis tables: accuracy, precision, recall and F1 of
  every group, paired tests on the four metrics, the gain of the exclusion of each mechanism over
  its own baseline, detection, reward distance and server time.
- `analyze_cm0.py`: range of the Stackelberg threshold Cm0 that gives exactly the same runs.

## Tests

```bash
pytest tests/
```

## Project structure

```
src/flapp/
├── task.py             model, MNIST loading, training and metrics
├── partition.py        Dirichlet non-IID split
├── free_rider.py       which clients are free-riders (they return the global model)
├── economics.py        cost and payment model of the Stackelberg game (EntroCFL)
├── fairness.py         distance between rewards and the reference vector
├── aggregate.py        mean and std across seeds
├── client_app.py       Flower client
├── server_app.py       Flower server
└── strategies/         fedavg.py, shapley.py, stackelberg.py and shared code
scripts/                run_experiment.py and the analysis scripts above
tests/                  pytest suite
docs/                   formulas, limitations of the comparison, background
```
