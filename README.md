# Federated Learning + Game Theory — Free-Rider Mitigation

This code compares three aggregation strategies on
MNIST split non-IID across 10 simulated clients, 3 of which are free-riders:

- **FedAvg**: plain weighted average (baseline).
- **Shapley**: leave-one-out contributions; the lowest 30% (the free-rider rate) are excluded
  from aggregation and reward (ConDA).
- **Stackelberg**: EntroCFL game where the server sets a payment per data, each client picks how
  much data to train on, and clients incompatible with the others are excluded without payment.

Free-riders are clients 0-2 (0-4 in the 50% scenario). Three behaviours are implemented: **silent**
(returns the global model without training), **low-effort** (trains 1 local epoch instead of 2) and
**noisy** (returns the global model plus Gaussian noise).

Built on [Flower](https://flower.ai/) and PyTorch.

## Setup

Run inside **WSL2 (Ubuntu)**: Flower's Ray simulation backend does not work on native
Windows. The results were produced with Python 3.14.4 and the exact versions pinned in
`requirements.txt` (`flwr==1.35.0`).

```bash
python3 -m venv ~/.venvs/fl-demo
source ~/.venvs/fl-demo/bin/activate
pip install -r requirements.txt
```

If `python3 -m venv` fails because `ensurepip` is missing, see `docs/background.md`.

## Run

One experiment (defaults: 10 clients, 3 silent free-riders, 50 rounds, Dirichlet α = 0.5, the small
CNN with 18,406 parameters used in every result of the thesis):
```bash
python scripts/run_experiment.py --strategy stackelberg --seed 42
```
This writes `results/stackelberg_seed42.json`. Run one experiment at a time: two at once do
not fit in the 15 GB WSL memory.

### Reproducing the thesis (8 scenarios)

The thesis uses α ∈ {0.5, 0.1} × {silent, low_effort, noisy, fr50} with seeds 42-46. The whole battery
is in one script (one experiment at a time; `--resume` skips experiments that are already complete, so
an interrupted run can simply be started again):

```bash
bash scripts/run_all.sh                 # every scenario, every mechanism and control
bash scripts/run_all.sh full_data       # only the Stackelberg of the main comparison
bash scripts/run_all.sh shapley_loss    # optional ablation: Shapley with loss utility
```

For one scenario and one seed, these are the runs the script makes (example: α = 0.1, noisy):
```bash
dir=results/a01_noisy; args="--alpha 0.1 --seed 42 --num-free-riders 3 --free-rider-type noisy"
python scripts/run_experiment.py --resume --model small --results-dir $dir --strategy stackelberg $args
python scripts/run_experiment.py --resume --model small --results-dir $dir/full_data --strategy stackelberg --full-data $args
python scripts/run_experiment.py --resume --model small --results-dir $dir --strategy fedavg $args
python scripts/run_experiment.py --resume --model small --results-dir $dir --strategy shapley --shapley-voting-rate 0.3 $args
python scripts/run_experiment.py --resume --model small --results-dir $dir/shapley_v0 --strategy shapley --shapley-voting-rate 0 $args
python scripts/run_experiment.py --resume --model small --results-dir $dir/control --strategy fedavg --price-from $dir/stackelberg_seed42.json $args
```

| Scenario | Folder | Extra arguments |
|---|---|---|
| silent | `results/a05_silent`, `results/a01_silent` | `--num-free-riders 3` |
| low-effort | `results/a05_low_effort`, `results/a01_low_effort` | `--num-free-riders 3 --free-rider-type low-effort` |
| noisy | `results/a05_noisy`, `results/a01_noisy` | `--num-free-riders 3 --free-rider-type noisy` |
| fr50 | `results/a05_fr50`, `results/a01_fr50` | `--num-free-riders 5`, and Shapley with `--shapley-voting-rate 0.5` |

In silent and fr50 the script also runs the **perfect detector** (FedAvg that ignores the real
free-riders): `--strategy fedavg --exclude-clients 0,1,2 --results-dir $dir/oracle` (`0,1,2,3,4` in fr50).
In silent it also runs `fr0` (no free-riders) and `shapley_v02` (voting rate 0.2).

- `full_data/`: Stackelberg where the clients accepted by the game train with all their data. This is the
  Stackelberg of the main comparison, so the three mechanisms use the same data.
- `shapley_v0/`: Shapley without exclusion (uniform weights), the baseline of the Shapley mechanism.
- `control/`: FedAvg where every client trains on the amount of data it would choose in the
  Stackelberg game, the baseline of the price-chosen Stackelberg.

The raw JSON results (320 files) are not versioned in this repository. They are attached to the
[`resultados-tcc` release](https://github.com/MelissaRegoRodrigues/federated-learning-demo/releases/tag/resultados-tcc)
as `resultados_brutos.zip`: extract it at the repository root (it already contains `results/`) and run
the commands of the next section to regenerate every table without running the experiments again.

Run `python scripts/run_experiment.py --help` to see every option.

## Analyze

```bash
bash scripts/summarize_all.sh > results/resumo_geral.md
python scripts/analyze_comparison.py > results/tabelas_comparacao.md
python scripts/diagnose_shapley.py
python scripts/analyze_cm0.py results/a05_silent/stackelberg_seed*.json
```

- `summarize_all.sh` / `summarize_results.py`: every number of one scenario: accuracy, precision, recall
  and F1, paired tests with Holm correction and d_z, detection, reward distance, Spearman correlation and
  server time.
- `analyze_comparison.py`: the comparison tables of the thesis (FedAvg × Shapley × Stackelberg with full
  data) over the 8 scenarios, with 95% CI and Holm over all 192 tests.
- `diagnose_shapley.py`: saturation and ties of the Shapley contributions, from the saved results.
- `analyze_cm0.py`: range of the Stackelberg threshold Cm0 that gives exactly the same runs.

## Tests

```bash
pytest tests/
```

## Project structure

```
src/flapp/
├── task.py             model, MNIST loading, training and metrics
├── partition.py        Dirichlet non-IID split (Flower's DirichletPartitioner)
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
