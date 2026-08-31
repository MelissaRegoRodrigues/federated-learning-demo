# Federated Learning + Game Theory — Free-Rider Mitigation

Code companion to a TCC (undergraduate thesis) on Federated Learning (FL)
combined with Game Theory. It compares three aggregation strategies —
**FedAvg** (naive baseline), **Shapley-Weighted** (post-hoc contribution
scoring), and **Stackelberg** (reputation-gated pricing) — on the same
experiment: MNIST split non-IID across simulated clients, some of whom
are deliberately dishonest ("free-riders"), to see which aggregation
method best detects and neutralizes them.

## What is Federated Learning?

In FL, many clients (phones, hospitals, sensors, ...) each train a model
on their own local data and send only the *trained model weights* to a
server, which aggregates them into one global model — the raw data never
leaves the client. This is great for privacy, but it opens a door: a
client can misbehave (send garbage, train too little, or send noise)
and the server has to infer this purely from the weights it receives.

## What is Flower, and why this project uses it

[Flower](https://flower.ai/) is the field's standard open-source FL
framework: it provides real `ClientApp`/`ServerApp`/`Strategy`
abstractions for simulating many clients and a pluggable aggregation
algorithm, instead of hand-rolling a training loop from scratch. This
project builds all three strategies as real Flower `Strategy` subclasses.

**A note on how it's launched:** Flower's current CLI (`flwr run`) runs
even a single-machine simulation through an auto-managed background
"SuperLink" process, with connection settings in a separate global config
file outside the project — useful for Flower's multi-machine/cloud
features, but unnecessary overhead for a solo research simulation like
this one. This project instead calls Flower's `run_simulation()`
function directly from `scripts/run_experiment.py` — the same real
`Strategy`/`ClientApp` objects, just invoked without the CLI's extra
machinery. (`run_simulation` is marked deprecated in favor of `flwr run`,
but remains fully functional, and is the pattern most existing
Flower-based papers and tutorials still use.)

## The three strategies

- **FedAvg** (`src/flapp/strategies/fedavg.py`): Flower's real,
  unmodified averaging — every client's update counts equally, regardless
  of quality. Simple, but a free-rider's bad update pollutes the global
  model exactly as much as an honest client's good one.

- **Shapley-Weighted** (`src/flapp/strategies/shapley.py`): computes each
  client's marginal contribution via leave-one-out — aggregate with
  everyone, aggregate again without client *i*, and the accuracy
  difference is client *i*'s Shapley value. A free-rider's presence
  actively *hurts* accuracy, so its Shapley value goes negative. These
  values are then shifted to be non-negative and normalized to sum to 1,
  so a free-rider's aggregation weight approaches (but never goes below)
  zero — a signed-weight version was tried first and found, during
  validation, to occasionally destabilize training badly enough that it
  never recovered; shifting to non-negative weights makes this exactly
  as numerically stable as FedAvg's plain averaging. Cost: this needs
  `n+1` full model evaluations every round (1 with everyone + 1 per
  client left out) — not Monte Carlo, but not cheap either.

- **Stackelberg** (`src/flapp/strategies/stackelberg.py`): a cheaper,
  approximate alternative. Each round, every client's update is compared
  (via cosine similarity) against the coordinate-wise *median* update
  across all clients — a free-rider's update tends to diverge from this
  robust "consensus direction." That similarity feeds a running
  reputation score (an exponential moving average) per client. Clients
  whose reputation falls into the bottom 25% are excluded from a
  price-equilibrium game that decides final participation among the rest.
  No extra model evaluations are needed (unlike Shapley), but it takes a
  few rounds of history to build a reliable reputation, and it's
  approximate, not exact.

Both Shapley and Stackelberg flag likely free-riders using the same
25th-percentile threshold rule, so their "detection rate" numbers are
directly comparable in `results/detection_rate.png`.

**Reward equity** (`src/flapp/equity.py`): a separate, post-hoc question
from detection — not "did we flag the free-rider?" but "does the score
itself (Shapley value / reputation) actually track true honesty?".
Computed as the point-biserial correlation between each round's
per-client scores and the (ground-truth, ignored by the strategies
themselves) honest/free-rider label: +1 means the score perfectly ranks
every honest client above every free-rider, 0 means no discriminative
power, negative means the mechanism is rewarding free-riders more than
honest clients. Plotted in `results/equity_index.png`.

## Non-IID data and free-rider simulation

- **Non-IID partitioning** (`src/flapp/partition.py`): MNIST is split
  across clients using a Dirichlet distribution — a lower `alpha` means
  each client's slice of digits is more skewed/unrealistic-looking
  (heterogeneous), which is what makes free-rider detection non-trivial
  in the first place.
- **Free-rider behaviors** (`src/flapp/free_rider.py`): the first
  `--num-free-riders` clients (by partition id) cycle through three
  dishonest behaviors — `silent` (sends an all-zero update), `noisy`
  (sends its real update plus large Gaussian noise), and `lazy` (sends
  its update scaled down to ~30%, as if it trained far less than it
  claims). Every other client is honest.

## Project structure

```
src/flapp/
├── task.py             # SimpleCNN + MNIST load/train/test/get_params/set_params
├── partition.py         # Dirichlet non-IID partitioning
├── free_rider.py        # silent/noisy/lazy behavior corruption
├── equity.py            # reward equity index (post-hoc analysis, not used by strategies)
├── client_app.py        # Flower ClientApp (entrypoint)
├── server_app.py        # Flower ServerApp (entrypoint)
└── strategies/
    ├── base.py           # shared Strategy scaffolding for Shapley/Stackelberg
    ├── logging_utils.py  # unified JSON result schema
    ├── fedavg.py         # LoggingFedAvg — real, unmodified FedAvg + logging
    ├── shapley.py        # ShapleyWeightedStrategy
    └── stackelberg.py    # StackelbergStrategy
scripts/
├── run_experiment.py     # entrypoint: runs one strategy, writes results/<strategy>.json
└── compare_runs.py       # reads the 3 result JSONs, plots them together
```

Imports flow one way only: `client_app.py`/`server_app.py`/`scripts/*.py`
are the entrypoints; everything else (`task.py`, `partition.py`,
`free_rider.py`, `strategies/*`) is a leaf module that never imports an
entrypoint or anything that imports it back.

## How to run

This must run **inside WSL2 (Ubuntu)**, not native Windows PowerShell —
Flower's Ray-based simulation backend crashes on native Windows.

**One-time setup** (this Ubuntu install has no `sudo`-installed
`python3-pip`, so the venv is bootstrapped without relying on
`ensurepip`):
```bash
mkdir -p ~/.venvs
python3 -m venv --without-pip ~/.venvs/fl-demo
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
~/.venvs/fl-demo/bin/python /tmp/get-pip.py --quiet
cd /path/to/federated-learning-demo
~/.venvs/fl-demo/bin/pip install -r requirements.txt
```

**Run all three strategies** (fixed result files:
`results/fedavg.json`, `results/shapley.json`, `results/stackelberg.json`).
Defaults already match this project's documented methodology (10 clients,
50 rounds, 3 free-riders = 30%, alpha=0.5, 2 local epochs) — only
`--strategy` is required:
```bash
~/.venvs/fl-demo/bin/python scripts/run_experiment.py --strategy fedavg
~/.venvs/fl-demo/bin/python scripts/run_experiment.py --strategy shapley
~/.venvs/fl-demo/bin/python scripts/run_experiment.py --strategy stackelberg
```

**Compare them:**
```bash
~/.venvs/fl-demo/bin/python scripts/compare_runs.py results/fedavg.json results/shapley.json results/stackelberg.json
```
This writes `results/convergence.png` (accuracy per round, all three
strategies), `results/detection_rate.png` (free-rider detection rate,
Shapley vs. Stackelberg), and `results/equity_index.png` (does each
strategy's score actually track true honesty, Shapley vs. Stackelberg).

**Run the test suite:**
```bash
~/.venvs/fl-demo/bin/pytest tests/ -v
```
