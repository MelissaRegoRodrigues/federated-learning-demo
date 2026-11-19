"""Figura da particao de Dirichlet usada nos experimentos.

Para cada alfa (0,5 e 0,1), um mapa de calor com quantas amostras de cada digito (colunas) cada cliente
(linhas) recebeu. A particao e a mesma que os experimentos usam (flapp.task.partition_all_clients):
os 10% de validacao saem antes, e o resto e dividido pelo DirichletPartitioner com a mesma semente.

Uso (no WSL, mesmo ambiente dos experimentos):
    python scripts/plot_partition.py                # semente 42
    python scripts/plot_partition.py --seed 43
A figura vai para results/particao_dirichlet_seed<semente>.png.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem janela: so salva o arquivo
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from flapp.task import load_train_set, partition_all_clients  # noqa: E402

NUM_CLIENTS = 10
NUM_CLASSES = 10
ALPHAS = [0.5, 0.1]
# Clientes 0-2 sao free-riders em todos os cenarios; 3 e 4 so no fr50.
ALWAYS_FREE_RIDERS = {0, 1, 2}
FR50_ONLY = {3, 4}


def class_counts(alpha, seed):
    # Matriz clientes x digitos com o numero de amostras de treino de cada cliente.
    labels = load_train_set().targets.numpy()
    counts = np.zeros((NUM_CLIENTS, NUM_CLASSES), dtype=int)
    for client_id, indices in enumerate(partition_all_clients(NUM_CLIENTS, alpha, seed)):
        counts[client_id] = np.bincount(labels[indices], minlength=NUM_CLASSES)
    return counts


def client_label(client_id, total):
    marker = ""
    if client_id in ALWAYS_FREE_RIDERS:
        marker = " (FR)"
    elif client_id in FR50_ONLY:
        marker = " (FR*)"
    return f"cliente {client_id}{marker}  n={total:,}".replace(",", ".")


def draw_panel(axis, counts, alpha, vmax):
    image = axis.imshow(counts, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    for row in range(NUM_CLIENTS):
        for column in range(NUM_CLASSES):
            value = counts[row, column]
            if value == 0:
                continue
            # Texto branco nas celulas escuras, escuro nas claras, para dar para ler.
            color = "white" if value > vmax * 0.55 else "#1f2933"
            axis.text(column, row, str(value), ha="center", va="center", fontsize=6.5, color=color)
    axis.set_xticks(range(NUM_CLASSES))
    axis.set_xlabel("dígito (classe)")
    axis.set_yticks(range(NUM_CLIENTS))
    axis.set_yticklabels([client_label(c, counts[c].sum()) for c in range(NUM_CLIENTS)], fontsize=8)
    axis.set_title(f"α = {str(alpha).replace('.', ',')}", fontsize=11)
    # Linha separando os free-riders fixos (0-2) dos demais.
    axis.axhline(2.5, color="#b42318", linewidth=1.2)
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_counts = {alpha: class_counts(alpha, args.seed) for alpha in ALPHAS}
    vmax = max(counts.max() for counts in all_counts.values())

    figure, axes = plt.subplots(1, len(ALPHAS), figsize=(13, 5.2), constrained_layout=True)
    for axis, alpha in zip(axes, ALPHAS):
        image = draw_panel(axis, all_counts[alpha], alpha, vmax)
    colorbar = figure.colorbar(image, ax=axes, shrink=0.85, pad=0.01)
    colorbar.set_label("amostras de treino")
    colorbar.outline.set_visible(False)
    figure.suptitle(
        f"Partição de Dirichlet do MNIST entre 10 clientes (semente {args.seed}; 54.000 amostras de treino)",
        fontsize=12,
    )
    figure.text(
        0.01, -0.02,
        "FR = free-rider em todos os cenários; FR* = free-rider só no cenário fr50. "
        "Células vazias = nenhuma amostra daquele dígito.",
        fontsize=8, color="#52606d",
    )

    output = Path("results") / f"particao_dirichlet_seed{args.seed}.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Figura salva em {output}")
    for alpha, counts in all_counts.items():
        print(f"alpha={alpha}: amostras por cliente = {counts.sum(axis=1).tolist()}")


if __name__ == "__main__":
    main()
