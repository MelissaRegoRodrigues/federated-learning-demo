
### α=0.5 — silent

#### Médias ± desvio entre sementes

| métrica | FedAvg | Shapley | Stackelberg |
|---|---|---|---|
| acurácia final (%) | 97.9 ± 0.1 | 97.8 ± 0.1 | 98.0 ± 0.1 |
| precisão do modelo (%) | 97.9 ± 0.1 | 97.8 ± 0.1 | 98.0 ± 0.1 |
| recall do modelo (%) | 97.9 ± 0.1 | 97.8 ± 0.1 | 98.0 ± 0.1 |
| F1 do modelo (%) | 97.9 ± 0.1 | 97.8 ± 0.1 | 98.0 ± 0.1 |
| oscilação nas últimas 10 rodadas (pp) | 0.11 ± 0.04 | 0.09 ± 0.03 | 0.09 ± 0.04 |
| rodada em que atinge 95% | 11.6 ± 3.2 | 12.2 ± 3.8 | 9.4 ± 2.9 |
| parcela da recompensa dos free-riders (%) | 29.7 ± 4.8 | 28.7 ± 2.0 | 0.0 ± 0.0 |
| tempo do servidor (s/rodada) | 0.004 ± 0.000 | 7.290 ± 0.308 | 0.180 ± 0.017 |

#### Diferença pareada (p do t pareado; sementes em que o primeiro é maior)

| métrica | Shapley − FedAvg | Stackelberg − FedAvg | Stackelberg − Shapley |
|---|---|---|---|
| acurácia final (%) | -0.1 (p=0.231; 1/5) | +0.1 (p=0.003; 5/5) | +0.2 (p=0.053; 5/5) |
| precisão do modelo (%) | -0.1 (p=0.199; 1/5) | +0.1 (p=0.005; 5/5) | +0.2 (p=0.050; 5/5) |
| recall do modelo (%) | -0.1 (p=0.247; 1/5) | +0.1 (p=0.004; 5/5) | +0.2 (p=0.052; 5/5) |
| F1 do modelo (%) | -0.1 (p=0.238; 1/5) | +0.1 (p=0.004; 5/5) | +0.2 (p=0.055; 5/5) |
| oscilação nas últimas 10 rodadas (pp) | -0.02 (p=0.170; 1/5) | -0.01 (p=0.122; 1/5) | +0.01 (p=0.509; 3/5) |
| rodada em que atinge 95% | +0.6 (p=0.374; 2/5) | -2.2 (p=0.004; 0/5) | -2.8 (p=0.025; 0/5) |
| parcela da recompensa dos free-riders (%) | -1.0 (p=0.726; 1/5) | -29.7 (p=0.000; 0/5) | -28.7 (p=0.000; 0/5) |
| tempo do servidor (s/rodada) | +7.285 (p=0.000; 5/5) | +0.176 (p=0.000; 5/5) | -7.110 (p=0.000; 0/5) |

#### Detecção de free-riders (média das sementes; não é a precisão/recall do modelo)

| mecanismo | precisão da detecção | recall da detecção |
|---|---|---|
| shapley | 0.33 | 0.33 |
| stackelberg | 1.00 | 1.00 |

#### Stackelberg: dados completos × dados escolhidos pelo preço

| variante | acurácia final (%) | F1 final (%) | dados usados (% dos dados de todos os clientes) | pagamento total (milhares) |
|---|---|---|---|---|
| Stackelberg com dados completos | 98.0 ± 0.1 | 98.0 ± 0.1 | 70.9 ± 4.7 | 4443 ± 549 |
| Stackelberg com dados escolhidos pelo preço | 92.8 ± 1.2 | 92.7 ± 1.2 | 7.9 ± 1.5 | 501 ± 147 |

### α=0.5 — low_effort

#### Médias ± desvio entre sementes

| métrica | FedAvg | Shapley | Stackelberg |
|---|---|---|---|
| acurácia final (%) | 98.2 ± 0.1 | 98.2 ± 0.1 | 98.2 ± 0.1 |
| precisão do modelo (%) | 98.2 ± 0.1 | 98.2 ± 0.1 | 98.2 ± 0.1 |
| recall do modelo (%) | 98.2 ± 0.1 | 98.2 ± 0.1 | 98.2 ± 0.1 |
| F1 do modelo (%) | 98.2 ± 0.1 | 98.2 ± 0.1 | 98.2 ± 0.1 |
| oscilação nas últimas 10 rodadas (pp) | 0.12 ± 0.09 | 0.09 ± 0.02 | 0.12 ± 0.09 |
| rodada em que atinge 95% | 9.0 ± 2.0 | 10.2 ± 2.3 | 9.0 ± 2.0 |
| parcela da recompensa dos free-riders (%) | 29.7 ± 4.8 | 29.5 ± 3.1 | 29.7 ± 4.8 |
| tempo do servidor (s/rodada) | 0.004 ± 0.000 | 5.972 ± 0.211 | 0.180 ± 0.010 |

#### Diferença pareada (p do t pareado; sementes em que o primeiro é maior)

| métrica | Shapley − FedAvg | Stackelberg − FedAvg | Stackelberg − Shapley |
|---|---|---|---|
| acurácia final (%) | +0.0 (p=0.929; 2/5) | +0.0 (p=0.717; 2/5) | +0.0 (p=1.000; 3/5) |
| precisão do modelo (%) | +0.0 (p=0.908; 2/5) | +0.0 (p=0.652; 3/5) | +0.0 (p=0.991; 3/5) |
| recall do modelo (%) | +0.0 (p=0.914; 2/5) | +0.0 (p=0.706; 2/5) | -0.0 (p=0.994; 3/5) |
| F1 do modelo (%) | +0.0 (p=0.889; 2/5) | +0.0 (p=0.679; 2/5) | -0.0 (p=0.981; 3/5) |
| oscilação nas últimas 10 rodadas (pp) | -0.03 (p=0.559; 2/5) | -0.00 (p=0.846; 3/5) | +0.03 (p=0.594; 2/5) |
| rodada em que atinge 95% | +1.2 (p=0.109; 3/5) | 0,0 (p=1) | -1.2 (p=0.109; 0/5) |
| parcela da recompensa dos free-riders (%) | -0.2 (p=0.931; 2/5) | 0,0 (p=1) | +0.2 (p=0.931; 3/5) |
| tempo do servidor (s/rodada) | +5.968 (p=0.000; 5/5) | +0.175 (p=0.000; 5/5) | -5.793 (p=0.000; 0/5) |

#### Detecção de free-riders (média das sementes; não é a precisão/recall do modelo)

| mecanismo | precisão da detecção | recall da detecção |
|---|---|---|
| shapley | 0.31 | 0.31 |
| stackelberg | sem sinalizações | 0.00 |

#### Stackelberg: dados completos × dados escolhidos pelo preço

| variante | acurácia final (%) | F1 final (%) | dados usados (% dos dados de todos os clientes) | pagamento total (milhares) |
|---|---|---|---|---|
| Stackelberg com dados completos | 98.2 ± 0.1 | 98.2 ± 0.1 | 100.0 ± 0.0 | 5751 ± 335 |
| Stackelberg com dados escolhidos pelo preço | 91.6 ± 1.8 | 91.5 ± 1.8 | 9.1 ± 1.3 | 528 ± 104 |

### α=0.5 — noisy

#### Médias ± desvio entre sementes

| métrica | FedAvg | Shapley | Stackelberg |
|---|---|---|---|
| acurácia final (%) | 97.0 ± 0.4 | 97.5 ± 0.2 | 97.0 ± 0.3 |
| precisão do modelo (%) | 97.0 ± 0.3 | 97.5 ± 0.2 | 97.0 ± 0.3 |
| recall do modelo (%) | 97.0 ± 0.3 | 97.5 ± 0.2 | 97.0 ± 0.3 |
| F1 do modelo (%) | 97.0 ± 0.4 | 97.5 ± 0.2 | 97.0 ± 0.3 |
| oscilação nas últimas 10 rodadas (pp) | 0.19 ± 0.08 | 0.16 ± 0.04 | 0.18 ± 0.08 |
| rodada em que atinge 95% | 14.4 ± 5.0 | 12.4 ± 2.9 | 14.4 ± 5.0 |
| parcela da recompensa dos free-riders (%) | 29.7 ± 4.8 | 15.2 ± 1.8 | 29.7 ± 4.8 |
| tempo do servidor (s/rodada) | 0.004 ± 0.000 | 6.021 ± 0.177 | 0.204 ± 0.028 |

#### Diferença pareada (p do t pareado; sementes em que o primeiro é maior)

| métrica | Shapley − FedAvg | Stackelberg − FedAvg | Stackelberg − Shapley |
|---|---|---|---|
| acurácia final (%) | +0.5 (p=0.009; 5/5) | -0.0 (p=0.904; 2/5) | -0.5 (p=0.006; 0/5) |
| precisão do modelo (%) | +0.5 (p=0.007; 5/5) | -0.0 (p=0.884; 2/5) | -0.5 (p=0.004; 0/5) |
| recall do modelo (%) | +0.5 (p=0.010; 5/5) | -0.0 (p=0.920; 2/5) | -0.5 (p=0.007; 0/5) |
| F1 do modelo (%) | +0.5 (p=0.010; 5/5) | -0.0 (p=0.916; 2/5) | -0.5 (p=0.006; 0/5) |
| oscilação nas últimas 10 rodadas (pp) | -0.03 (p=0.568; 3/5) | -0.00 (p=0.620; 2/5) | +0.03 (p=0.645; 2/5) |
| rodada em que atinge 95% | -2.0 (p=0.129; 0/5) | 0,0 (p=1) | +2.0 (p=0.129; 4/5) |
| parcela da recompensa dos free-riders (%) | -14.5 (p=0.006; 0/5) | 0,0 (p=1) | +14.5 (p=0.006; 5/5) |
| tempo do servidor (s/rodada) | +6.017 (p=0.000; 5/5) | +0.199 (p=0.000; 5/5) | -5.817 (p=0.000; 0/5) |

#### Detecção de free-riders (média das sementes; não é a precisão/recall do modelo)

| mecanismo | precisão da detecção | recall da detecção |
|---|---|---|
| shapley | 0.65 | 0.65 |
| stackelberg | sem sinalizações | 0.00 |

#### Stackelberg: dados completos × dados escolhidos pelo preço

| variante | acurácia final (%) | F1 final (%) | dados usados (% dos dados de todos os clientes) | pagamento total (milhares) |
|---|---|---|---|---|
| Stackelberg com dados completos | 97.0 ± 0.3 | 97.0 ± 0.3 | 100.0 ± 0.0 | 5751 ± 335 |
| Stackelberg com dados escolhidos pelo preço | 88.2 ± 2.5 | 88.1 ± 2.6 | 9.1 ± 1.3 | 528 ± 104 |

### α=0.5 — fr50

#### Médias ± desvio entre sementes

| métrica | FedAvg | Shapley | Stackelberg |
|---|---|---|---|
| acurácia final (%) | 96.5 ± 0.6 | 96.9 ± 0.5 | 97.2 ± 0.4 |
| precisão do modelo (%) | 96.6 ± 0.5 | 96.9 ± 0.5 | 97.3 ± 0.4 |
| recall do modelo (%) | 96.5 ± 0.6 | 96.9 ± 0.5 | 97.2 ± 0.4 |
| F1 do modelo (%) | 96.5 ± 0.6 | 96.9 ± 0.5 | 97.2 ± 0.4 |
| oscilação nas últimas 10 rodadas (pp) | 0.17 ± 0.08 | 0.17 ± 0.05 | 0.18 ± 0.11 |
| rodada em que atinge 95% | 26.0 ± 8.2 | 18.8 ± 6.0 | 13.0 ± 3.2 |
| parcela da recompensa dos free-riders (%) | 54.1 ± 5.2 | 39.1 ± 12.4 | 0.0 ± 0.0 |
| tempo do servidor (s/rodada) | 0.004 ± 0.000 | 7.834 ± 2.990 | 0.163 ± 0.025 |

#### Diferença pareada (p do t pareado; sementes em que o primeiro é maior)

| métrica | Shapley − FedAvg | Stackelberg − FedAvg | Stackelberg − Shapley |
|---|---|---|---|
| acurácia final (%) | +0.4 (p=0.085; 5/5) | +0.7 (p=0.006; 5/5) | +0.3 (p=0.023; 5/5) |
| precisão do modelo (%) | +0.3 (p=0.101; 4/5) | +0.7 (p=0.007; 5/5) | +0.3 (p=0.018; 5/5) |
| recall do modelo (%) | +0.4 (p=0.086; 5/5) | +0.7 (p=0.007; 5/5) | +0.3 (p=0.034; 5/5) |
| F1 do modelo (%) | +0.4 (p=0.087; 5/5) | +0.7 (p=0.007; 5/5) | +0.3 (p=0.027; 5/5) |
| oscilação nas últimas 10 rodadas (pp) | -0.00 (p=0.900; 3/5) | +0.00 (p=0.925; 2/5) | +0.01 (p=0.893; 2/5) |
| rodada em que atinge 95% | -7.2 (p=0.008; 0/5) | -13.0 (p=0.005; 0/5) | -5.8 (p=0.021; 0/5) |
| parcela da recompensa dos free-riders (%) | -15.0 (p=0.034; 1/5) | -54.1 (p=0.000; 0/5) | -39.1 (p=0.002; 0/5) |
| tempo do servidor (s/rodada) | +7.830 (p=0.004; 5/5) | +0.159 (p=0.000; 5/5) | -7.671 (p=0.005; 0/5) |

#### Detecção de free-riders (média das sementes; não é a precisão/recall do modelo)

| mecanismo | precisão da detecção | recall da detecção |
|---|---|---|
| shapley | 0.61 | 0.61 |
| stackelberg | 1.00 | 1.00 |

#### Stackelberg: dados completos × dados escolhidos pelo preço

| variante | acurácia final (%) | F1 final (%) | dados usados (% dos dados de todos os clientes) | pagamento total (milhares) |
|---|---|---|---|---|
| Stackelberg com dados completos | 97.2 ± 0.4 | 97.2 ± 0.4 | 47.0 ± 5.1 | 3134 ± 408 |
| Stackelberg com dados escolhidos pelo preço | 92.1 ± 0.6 | 91.9 ± 0.6 | 6.7 ± 1.0 | 456 ± 113 |

### α=0.1 — silent

#### Médias ± desvio entre sementes

| métrica | FedAvg | Shapley | Stackelberg |
|---|---|---|---|
| acurácia final (%) | 91.0 ± 6.0 | 93.3 ± 2.0 | 93.3 ± 2.7 |
| precisão do modelo (%) | 92.2 ± 4.2 | 93.4 ± 2.0 | 93.8 ± 2.1 |
| recall do modelo (%) | 91.0 ± 5.9 | 93.2 ± 2.1 | 93.3 ± 2.6 |
| F1 do modelo (%) | 90.5 ± 6.5 | 93.1 ± 2.1 | 93.1 ± 2.7 |
| oscilação nas últimas 10 rodadas (pp) | 0.46 ± 0.45 | 0.57 ± 0.21 | 0.37 ± 0.17 |
| rodada em que atinge 95% | 50.8 ± 0.4 | 47.2 ± 6.1 | 46.6 ± 6.2 |
| parcela da recompensa dos free-riders (%) | 29.7 ± 4.8 | 31.9 ± 8.4 | 0.0 ± 0.0 |
| tempo do servidor (s/rodada) | 0.004 ± 0.000 | 6.953 ± 1.279 | 0.183 ± 0.016 |

#### Diferença pareada (p do t pareado; sementes em que o primeiro é maior)

| métrica | Shapley − FedAvg | Stackelberg − FedAvg | Stackelberg − Shapley |
|---|---|---|---|
| acurácia final (%) | +2.3 (p=0.302; 5/5) | +2.4 (p=0.200; 5/5) | +0.0 (p=0.933; 4/5) |
| precisão do modelo (%) | +1.3 (p=0.313; 4/5) | +1.6 (p=0.165; 5/5) | +0.3 (p=0.217; 4/5) |
| recall do modelo (%) | +2.2 (p=0.305; 5/5) | +2.3 (p=0.196; 5/5) | +0.1 (p=0.842; 4/5) |
| F1 do modelo (%) | +2.6 (p=0.291; 5/5) | +2.6 (p=0.210; 5/5) | -0.0 (p=0.977; 4/5) |
| oscilação nas últimas 10 rodadas (pp) | +0.11 (p=0.590; 4/5) | -0.09 (p=0.608; 2/5) | -0.20 (p=0.097; 1/5) |
| rodada em que atinge 95% | -3.6 (p=0.230; 0/5) | -4.2 (p=0.194; 0/5) | -0.6 (p=0.788; 1/5) |
| parcela da recompensa dos free-riders (%) | +2.2 (p=0.565; 3/5) | -29.7 (p=0.000; 0/5) | -31.9 (p=0.001; 0/5) |
| tempo do servidor (s/rodada) | +6.949 (p=0.000; 5/5) | +0.179 (p=0.000; 5/5) | -6.770 (p=0.000; 0/5) |

#### Detecção de free-riders (média das sementes; não é a precisão/recall do modelo)

| mecanismo | precisão da detecção | recall da detecção |
|---|---|---|
| shapley | 0.26 | 0.26 |
| stackelberg | 1.00 | 1.00 |

#### Stackelberg: dados completos × dados escolhidos pelo preço

| variante | acurácia final (%) | F1 final (%) | dados usados (% dos dados de todos os clientes) | pagamento total (milhares) |
|---|---|---|---|---|
| Stackelberg com dados completos | 93.3 ± 2.7 | 93.1 ± 2.7 | 69.5 ± 11.6 | 4364 ± 986 |
| Stackelberg com dados escolhidos pelo preço | 83.6 ± 3.7 | 82.1 ± 5.0 | 7.9 ± 1.5 | 501 ± 147 |

### α=0.1 — low_effort

#### Médias ± desvio entre sementes

| métrica | FedAvg | Shapley | Stackelberg |
|---|---|---|---|
| acurácia final (%) | 96.2 ± 0.7 | 95.5 ± 0.7 | 96.2 ± 0.7 |
| precisão do modelo (%) | 96.3 ± 0.7 | 95.7 ± 0.7 | 96.3 ± 0.7 |
| recall do modelo (%) | 96.1 ± 0.7 | 95.5 ± 0.7 | 96.2 ± 0.7 |
| F1 do modelo (%) | 96.1 ± 0.8 | 95.5 ± 0.7 | 96.1 ± 0.8 |
| oscilação nas últimas 10 rodadas (pp) | 0.33 ± 0.20 | 1.09 ± 1.18 | 0.35 ± 0.23 |
| rodada em que atinge 95% | 33.4 ± 11.0 | 34.4 ± 8.7 | 33.8 ± 11.1 |
| parcela da recompensa dos free-riders (%) | 29.7 ± 4.8 | 26.3 ± 7.4 | 31.1 ± 11.8 |
| tempo do servidor (s/rodada) | 0.004 ± 0.000 | 6.141 ± 0.093 | 0.207 ± 0.024 |

#### Diferença pareada (p do t pareado; sementes em que o primeiro é maior)

| métrica | Shapley − FedAvg | Stackelberg − FedAvg | Stackelberg − Shapley |
|---|---|---|---|
| acurácia final (%) | -0.7 (p=0.122; 1/5) | +0.0 (p=0.307; 3/5) | +0.7 (p=0.111; 4/5) |
| precisão do modelo (%) | -0.6 (p=0.108; 1/5) | +0.0 (p=0.282; 3/5) | +0.6 (p=0.096; 4/5) |
| recall do modelo (%) | -0.7 (p=0.132; 1/5) | +0.0 (p=0.295; 3/5) | +0.7 (p=0.121; 4/5) |
| F1 do modelo (%) | -0.6 (p=0.144; 1/5) | +0.0 (p=0.276; 3/5) | +0.7 (p=0.130; 4/5) |
| oscilação nas últimas 10 rodadas (pp) | +0.76 (p=0.247; 4/5) | +0.01 (p=0.431; 3/5) | -0.75 (p=0.255; 1/5) |
| rodada em que atinge 95% | +1.0 (p=0.721; 3/5) | +0.4 (p=0.178; 2/5) | -0.6 (p=0.832; 1/5) |
| parcela da recompensa dos free-riders (%) | -3.4 (p=0.450; 2/5) | +1.4 (p=0.759; 2/5) | +4.8 (p=0.405; 3/5) |
| tempo do servidor (s/rodada) | +6.136 (p=0.000; 5/5) | +0.202 (p=0.000; 5/5) | -5.934 (p=0.000; 0/5) |

#### Detecção de free-riders (média das sementes; não é a precisão/recall do modelo)

| mecanismo | precisão da detecção | recall da detecção |
|---|---|---|
| shapley | 0.39 | 0.39 |
| stackelberg | 1.00 | 0.07 |

#### Stackelberg: dados completos × dados escolhidos pelo preço

| variante | acurácia final (%) | F1 final (%) | dados usados (% dos dados de todos os clientes) | pagamento total (milhares) |
|---|---|---|---|---|
| Stackelberg com dados completos | 96.2 ± 0.7 | 96.1 ± 0.8 | 100.0 ± 0.0 | 5767 ± 308 |
| Stackelberg com dados escolhidos pelo preço | 87.1 ± 1.2 | 86.7 ± 1.3 | 9.0 ± 1.5 | 523 ± 114 |

### α=0.1 — noisy

#### Médias ± desvio entre sementes

| métrica | FedAvg | Shapley | Stackelberg |
|---|---|---|---|
| acurácia final (%) | 87.0 ± 8.7 | 92.2 ± 2.4 | 87.1 ± 8.8 |
| precisão do modelo (%) | 88.8 ± 6.7 | 92.6 ± 2.2 | 88.9 ± 6.7 |
| recall do modelo (%) | 87.2 ± 8.3 | 92.1 ± 2.4 | 87.2 ± 8.4 |
| F1 do modelo (%) | 86.0 ± 9.8 | 92.0 ± 2.5 | 86.1 ± 9.9 |
| oscilação nas últimas 10 rodadas (pp) | 1.01 ± 0.60 | 0.98 ± 0.58 | 1.01 ± 0.58 |
| rodada em que atinge 95% | 51.0 ± 0.0 | 49.2 ± 4.0 | 51.0 ± 0.0 |
| parcela da recompensa dos free-riders (%) | 29.7 ± 4.8 | 22.5 ± 5.3 | 31.1 ± 11.8 |
| tempo do servidor (s/rodada) | 0.004 ± 0.000 | 6.821 ± 0.706 | 0.183 ± 0.019 |

#### Diferença pareada (p do t pareado; sementes em que o primeiro é maior)

| métrica | Shapley − FedAvg | Stackelberg − FedAvg | Stackelberg − Shapley |
|---|---|---|---|
| acurácia final (%) | +5.2 (p=0.178; 5/5) | +0.0 (p=0.567; 2/5) | -5.2 (p=0.186; 0/5) |
| precisão do modelo (%) | +3.8 (p=0.189; 5/5) | +0.0 (p=0.349; 3/5) | -3.7 (p=0.194; 0/5) |
| recall do modelo (%) | +4.9 (p=0.174; 5/5) | +0.0 (p=0.603; 2/5) | -4.9 (p=0.182; 0/5) |
| F1 do modelo (%) | +5.9 (p=0.179; 5/5) | +0.0 (p=0.601; 2/5) | -5.9 (p=0.187; 0/5) |
| oscilação nas últimas 10 rodadas (pp) | -0.03 (p=0.920; 2/5) | +0.00 (p=0.929; 3/5) | +0.03 (p=0.918; 3/5) |
| rodada em que atinge 95% | -1.8 (p=0.374; 0/5) | 0,0 (p=1) | +1.8 (p=0.374; 1/5) |
| parcela da recompensa dos free-riders (%) | -7.2 (p=0.094; 1/5) | +1.4 (p=0.758; 2/5) | +8.6 (p=0.197; 4/5) |
| tempo do servidor (s/rodada) | +6.817 (p=0.000; 5/5) | +0.179 (p=0.000; 5/5) | -6.638 (p=0.000; 0/5) |

#### Detecção de free-riders (média das sementes; não é a precisão/recall do modelo)

| mecanismo | precisão da detecção | recall da detecção |
|---|---|---|
| shapley | 0.47 | 0.47 |
| stackelberg | sem sinalizações | 0.00 |

#### Stackelberg: dados completos × dados escolhidos pelo preço

| variante | acurácia final (%) | F1 final (%) | dados usados (% dos dados de todos os clientes) | pagamento total (milhares) |
|---|---|---|---|---|
| Stackelberg com dados completos | 87.1 ± 8.8 | 86.1 ± 9.9 | 100.0 ± 0.0 | 5751 ± 335 |
| Stackelberg com dados escolhidos pelo preço | 76.9 ± 4.2 | 74.5 ± 5.9 | 9.0 ± 1.5 | 523 ± 114 |

### α=0.1 — fr50

#### Médias ± desvio entre sementes

| métrica | FedAvg | Shapley | Stackelberg |
|---|---|---|---|
| acurácia final (%) | 85.4 ± 9.6 | 88.9 ± 6.7 | 89.1 ± 6.4 |
| precisão do modelo (%) | 86.4 ± 8.9 | 88.5 ± 8.4 | 88.6 ± 8.4 |
| recall do modelo (%) | 85.5 ± 9.3 | 88.8 ± 6.7 | 89.2 ± 6.3 |
| F1 do modelo (%) | 83.4 ± 12.0 | 87.4 ± 8.7 | 87.8 ± 8.1 |
| oscilação nas últimas 10 rodadas (pp) | 0.50 ± 0.34 | 0.55 ± 0.19 | 0.47 ± 0.17 |
| rodada em que atinge 95% | 51.0 ± 0.0 | 51.0 ± 0.0 | 51.0 ± 0.0 |
| parcela da recompensa dos free-riders (%) | 54.1 ± 5.2 | 37.2 ± 14.3 | 0.0 ± 0.0 |
| tempo do servidor (s/rodada) | 0.005 ± 0.001 | 8.446 ± 4.193 | 0.187 ± 0.062 |

#### Diferença pareada (p do t pareado; sementes em que o primeiro é maior)

| métrica | Shapley − FedAvg | Stackelberg − FedAvg | Stackelberg − Shapley |
|---|---|---|---|
| acurácia final (%) | +3.6 (p=0.121; 4/5) | +3.8 (p=0.132; 5/5) | +0.2 (p=0.688; 2/5) |
| precisão do modelo (%) | +2.0 (p=0.122; 4/5) | +2.1 (p=0.130; 5/5) | +0.1 (p=0.720; 2/5) |
| recall do modelo (%) | +3.3 (p=0.119; 4/5) | +3.7 (p=0.126; 5/5) | +0.4 (p=0.496; 3/5) |
| F1 do modelo (%) | +4.0 (p=0.141; 4/5) | +4.4 (p=0.173; 5/5) | +0.5 (p=0.540; 2/5) |
| oscilação nas últimas 10 rodadas (pp) | +0.05 (p=0.727; 3/5) | -0.03 (p=0.779; 3/5) | -0.08 (p=0.253; 2/5) |
| rodada em que atinge 95% | 0,0 (p=1) | 0,0 (p=1) | 0,0 (p=1) |
| parcela da recompensa dos free-riders (%) | -16.9 (p=0.083; 1/5) | -54.1 (p=0.000; 0/5) | -37.2 (p=0.004; 0/5) |
| tempo do servidor (s/rodada) | +8.442 (p=0.011; 5/5) | +0.182 (p=0.003; 5/5) | -8.259 (p=0.011; 0/5) |

#### Detecção de free-riders (média das sementes; não é a precisão/recall do modelo)

| mecanismo | precisão da detecção | recall da detecção |
|---|---|---|
| shapley | 0.63 | 0.63 |
| stackelberg | 1.00 | 1.00 |

#### Stackelberg: dados completos × dados escolhidos pelo preço

| variante | acurácia final (%) | F1 final (%) | dados usados (% dos dados de todos os clientes) | pagamento total (milhares) |
|---|---|---|---|---|
| Stackelberg com dados completos | 89.1 ± 6.4 | 87.8 ± 8.1 | 53.2 ± 12.9 | 3621 ± 1210 |
| Stackelberg com dados escolhidos pelo preço | 81.2 ± 8.1 | 79.2 ± 10.1 | 6.7 ± 1.0 | 455 ± 113 |
