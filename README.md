# Federated Learning + Game Theory — Free-Rider Mitigation

This code compares three aggregation strategies on
MNIST split non-IID across 10 simulated clients, 3 of which are free-riders:

- **FedAvg**: plain weighted average (baseline).
- **Shapley**: leave-one-out contributions; the lowest 20% are excluded from aggregation and
  reward (ConDA).
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

All experiments of the thesis:
```bash
for seed in 42 43 44 45 46; do
  for strategy in fedavg shapley stackelberg; do
    python scripts/run_experiment.py --strategy $strategy --seed $seed
  done
done

for seed in 42 43 44; do
  for strategy in fedavg shapley stackelberg; do
    python scripts/run_experiment.py --strategy $strategy --seed $seed --num-free-riders 0 --results-dir results/fr0
  done
done
```

Run `python scripts/run_experiment.py --help` to see every option.

## Analyze

```bash
python scripts/compare_runs.py results/*_seed*.json
python scripts/summarize_results.py
python scripts/analyze_cm0.py results/stackelberg_seed*.json
```

- `compare_runs.py`: plots of accuracy, detection, precision/recall/F1 and reward distance in
  `results/`.
- `summarize_results.py`: every number of the thesis tables.
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
├── free_rider.py       silent, noisy and lazy behaviors
├── economics.py        cost and payment model of the Stackelberg game (EntroCFL)
├── fairness.py         distance between rewards and the reference vector
├── aggregate.py        mean and std across seeds
├── client_app.py       Flower client
├── server_app.py       Flower server
└── strategies/         fedavg.py, shapley.py, stackelberg.py and shared code
scripts/                run_experiment.py and the analysis scripts above
tests/                  pytest suite
docs/                   formulas and background
```
