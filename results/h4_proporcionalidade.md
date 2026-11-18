<!-- Gerado por scripts/h4_proportionality.py. Unidade = cliente x semente. -->

### Parte 1: escore do mecanismo × contribuição de referência (Spearman unilateral, ρ > 0)

| α | cenário | mecanismo | clientes | n | ρ | p | p Holm (32 testes) |
|---|---|---|---|---|---|---|---|
| 0.5 | silent | shapley | todos | 50 | +0.78 | 9.0e-12 | 2.6e-10 |
| 0.5 | silent | shapley | só honestos | 35 | +0.75 | 8.1e-08 | 1.9e-06 |
| 0.5 | silent | stackelberg | todos | 50 | +0.55 | 1.8e-05 | 0.000 |
| 0.5 | silent | stackelberg | só honestos | 35 | -0.28 | 0.947 | 1.000 |
| 0.5 | low_effort | shapley | todos | 50 | +0.58 | 5.5e-06 | 0.000 |
| 0.5 | low_effort | shapley | só honestos | 35 | +0.81 | 2.5e-09 | 6.3e-08 |
| 0.5 | low_effort | stackelberg | todos | 50 | -0.11 | 0.767 | 1.000 |
| 0.5 | low_effort | stackelberg | só honestos | 35 | -0.34 | 0.975 | 1.000 |
| 0.5 | noisy | shapley | todos | 50 | +0.92 | 1.9e-21 | 6.1e-20 |
| 0.5 | noisy | shapley | só honestos | 35 | +0.82 | 8.6e-10 | 2.3e-08 |
| 0.5 | noisy | stackelberg | todos | 50 | +0.72 | 1.4e-09 | 3.7e-08 |
| 0.5 | noisy | stackelberg | só honestos | 35 | +0.24 | 0.078 | 0.941 |
| 0.5 | fr50 | shapley | todos | 50 | +0.88 | 1.1e-17 | 3.3e-16 |
| 0.5 | fr50 | shapley | só honestos | 25 | +0.69 | 7.1e-05 | 0.001 |
| 0.5 | fr50 | stackelberg | todos | 50 | +0.79 | 5.1e-12 | 1.5e-10 |
| 0.5 | fr50 | stackelberg | só honestos | 25 | -0.48 | 0.992 | 1.000 |
| 0.1 | silent | shapley | todos | 50 | +0.15 | 0.147 | 1.000 |
| 0.1 | silent | shapley | só honestos | 35 | +0.24 | 0.083 | 0.941 |
| 0.1 | silent | stackelberg | todos | 50 | +0.48 | 0.000 | 0.004 |
| 0.1 | silent | stackelberg | só honestos | 35 | -0.48 | 0.998 | 1.000 |
| 0.1 | low_effort | shapley | todos | 50 | +0.40 | 0.002 | 0.031 |
| 0.1 | low_effort | shapley | só honestos | 35 | +0.47 | 0.002 | 0.031 |
| 0.1 | low_effort | stackelberg | todos | 50 | -0.21 | 0.924 | 1.000 |
| 0.1 | low_effort | stackelberg | só honestos | 35 | -0.08 | 0.686 | 1.000 |
| 0.1 | noisy | shapley | todos | 50 | +0.63 | 5.5e-07 | 1.3e-05 |
| 0.1 | noisy | shapley | só honestos | 35 | +0.54 | 0.000 | 0.007 |
| 0.1 | noisy | stackelberg | todos | 50 | +0.62 | 7.7e-07 | 1.7e-05 |
| 0.1 | noisy | stackelberg | só honestos | 35 | -0.05 | 0.619 | 1.000 |
| 0.1 | fr50 | shapley | todos | 50 | +0.40 | 0.002 | 0.028 |
| 0.1 | fr50 | shapley | só honestos | 25 | +0.32 | 0.062 | 0.810 |
| 0.1 | fr50 | stackelberg | todos | 50 | +0.76 | 8.8e-11 | 2.5e-09 |
| 0.1 | fr50 | stackelberg | só honestos | 25 | -0.69 | 1.000 | 1.000 |

### Parte 2: distância da recompensa à referência (menor = mais proporcional)

| α | cenário | FedAvg | Shapley | Stackelberg | Shapley − FedAvg | Stackelberg − FedAvg | Stackelberg − Shapley |
|---|---|---|---|---|---|---|---|
| 0.5 | silent | 0.050 | 0.066 | 0.000 | +0.016 (p=0.142; Holm 1.000) | -0.050 (p=0.003; Holm 0.072) | -0.066 (p=0.003; Holm 0.059) |
| 0.5 | low_effort | 0.050 | 0.065 | 0.050 | +0.015 (p=0.061; Holm 0.919) | +0.000 (p=1.000; Holm 1.000) | -0.015 (p=0.061; Holm 0.919) |
| 0.5 | noisy | 0.050 | 0.035 | 0.050 | -0.015 (p=0.062; Holm 0.919) | +0.000 (p=1.000; Holm 1.000) | +0.015 (p=0.062; Holm 0.919) |
| 0.5 | fr50 | 0.138 | 0.116 | 0.000 | -0.022 (p=0.388; Holm 1.000) | -0.138 (p=0.001; Holm 0.014) | -0.116 (p=0.017; Holm 0.318) |
| 0.1 | silent | 0.070 | 0.105 | 0.000 | +0.035 (p=0.026; Holm 0.443) | -0.070 (p=0.042; Holm 0.670) | -0.105 (p=0.002; Holm 0.056) |
| 0.1 | low_effort | 0.070 | 0.084 | 0.070 | +0.014 (p=0.555; Holm 1.000) | +0.000 (p=0.374; Holm 1.000) | -0.014 (p=0.555; Holm 1.000) |
| 0.1 | noisy | 0.070 | 0.068 | 0.070 | -0.002 (p=0.950; Holm 1.000) | +0.000 (p=1.000; Holm 1.000) | +0.002 (p=0.950; Holm 1.000) |
| 0.1 | fr50 | 0.130 | 0.137 | 0.000 | +0.007 (p=0.838; Holm 1.000) | -0.130 (p=0.018; Holm 0.319) | -0.137 (p=0.003; Holm 0.072) |
