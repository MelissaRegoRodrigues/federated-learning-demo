# Background

## What is Federated Learning?

In FL, many clients (phones, hospitals, sensors, ...) each train a model
on their own local data and send only the *trained model weights* to a
server, which aggregates them into one global model — the raw data never
leaves the client. This is great for privacy, but it opens a door: a
client can misbehave, for example by returning the global model without training it
(a free-rider), and the server has to infer this purely from the weights it receives.

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

## Setup without ensurepip

On an Ubuntu install without `python3-pip` or `python3-venv` (no `sudo` to install them),
`python3 -m venv` fails. Create the environment without pip and bootstrap it:

```bash
mkdir -p ~/.venvs
python3 -m venv --without-pip ~/.venvs/fl-demo
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
~/.venvs/fl-demo/bin/python /tmp/get-pip.py --quiet
source ~/.venvs/fl-demo/bin/activate
pip install -r requirements.txt
```
