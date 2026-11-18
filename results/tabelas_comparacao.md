<!-- Holm-Bonferroni sobre 192 testes (cenários x métricas x pares). IC 95% da diferença entre colchetes. -->

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

#### Diferença pareada [IC 95%] (p do t pareado; p ajustado por Holm; sementes em que o primeiro é maior)

| métrica | Shapley − FedAvg | Stackelberg − FedAvg | Stackelberg − Shapley |
|---|---|---|---|
| acurácia final (%) | -0.1 [-0.3, +0.1] (p=0.231; Holm 1.000; 1/5) | +0.1 [+0.1, +0.2] (p=0.003; Holm 0.511; 5/5) | +0.2 [-0.0, +0.5] (p=0.053; Holm 1.000; 5/5) |
| precisão do modelo (%) | -0.1 [-0.3, +0.1] (p=0.199; Holm 1.000; 1/5) | +0.1 [+0.1, +0.2] (p=0.005; Holm 0.714; 5/5) | +0.2 [-0.0, +0.5] (p=0.050; Holm 1.000; 5/5) |
| recall do modelo (%) | -0.1 [-0.3, +0.1] (p=0.247; Holm 1.000; 1/5) | +0.1 [+0.1, +0.2] (p=0.004; Holm 0.613; 5/5) | +0.2 [-0.0, +0.5] (p=0.052; Holm 1.000; 5/5) |
| F1 do modelo (%) | -0.1 [-0.3, +0.1] (p=0.238; Holm 1.000; 1/5) | +0.1 [+0.1, +0.2] (p=0.004; Holm 0.688; 5/5) | +0.2 [-0.0, +0.5] (p=0.055; Holm 1.000; 5/5) |
| oscilação nas últimas 10 rodadas (pp) | -0.02 [-0.05, +0.01] (p=0.170; Holm 1.000; 1/5) | -0.01 [-0.03, +0.01] (p=0.122; Holm 1.000; 1/5) | +0.01 [-0.02, +0.03] (p=0.509; Holm 1.000; 3/5) |
| rodada em que atinge 95% | +0.6 [-1.1, +2.3] (p=0.374; Holm 1.000; 2/5) | -2.2 [-3.2, -1.2] (p=0.004; Holm 0.686; 0/5) | -2.8 [-5.0, -0.6] (p=0.025; Holm 1.000; 0/5) |
| parcela da recompensa dos free-riders (%) | -1.0 [-8.0, +6.1] (p=0.726; Holm 1.000; 1/5) | -29.7 [-35.6, -23.8] (p=0.000; Holm 0.027; 0/5) | -28.7 [-31.2, -26.3] (p=0.000; Holm 0.001; 0/5) |
| tempo do servidor (s/rodada) | +7.285 [+6.903, +7.667] (p=0.000; Holm 0.000; 5/5) | +0.176 [+0.155, +0.197] (p=0.000; Holm 0.004; 5/5) | -7.110 [-7.512, -6.707] (p=0.000; Holm 0.000; 0/5) |

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

#### Diferença pareada [IC 95%] (p do t pareado; p ajustado por Holm; sementes em que o primeiro é maior)

| métrica | Shapley − FedAvg | Stackelberg − FedAvg | Stackelberg − Shapley |
|---|---|---|---|
| acurácia final (%) | +0.0 [-0.1, +0.1] (p=0.929; Holm 1.000; 2/5) | +0.0 [-0.0, +0.0] (p=0.717; Holm 1.000; 2/5) | +0.0 [-0.1, +0.1] (p=1.000; Holm 1.000; 3/5) |
| precisão do modelo (%) | +0.0 [-0.1, +0.1] (p=0.908; Holm 1.000; 2/5) | +0.0 [-0.0, +0.0] (p=0.652; Holm 1.000; 3/5) | +0.0 [-0.1, +0.1] (p=0.991; Holm 1.000; 3/5) |
| recall do modelo (%) | +0.0 [-0.1, +0.1] (p=0.914; Holm 1.000; 2/5) | +0.0 [-0.0, +0.0] (p=0.706; Holm 1.000; 2/5) | -0.0 [-0.1, +0.1] (p=0.994; Holm 1.000; 3/5) |
| F1 do modelo (%) | +0.0 [-0.1, +0.1] (p=0.889; Holm 1.000; 2/5) | +0.0 [-0.0, +0.0] (p=0.679; Holm 1.000; 2/5) | -0.0 [-0.1, +0.1] (p=0.981; Holm 1.000; 3/5) |
| oscilação nas últimas 10 rodadas (pp) | -0.03 [-0.15, +0.09] (p=0.559; Holm 1.000; 2/5) | -0.00 [-0.02, +0.02] (p=0.846; Holm 1.000; 3/5) | +0.03 [-0.10, +0.15] (p=0.594; Holm 1.000; 2/5) |
| rodada em que atinge 95% | +1.2 [-0.4, +2.8] (p=0.109; Holm 1.000; 3/5) | 0,0 (p=1) | -1.2 [-2.8, +0.4] (p=0.109; Holm 1.000; 0/5) |
| parcela da recompensa dos free-riders (%) | -0.2 [-4.9, +4.6] (p=0.931; Holm 1.000; 2/5) | 0,0 (p=1) | +0.2 [-4.6, +4.9] (p=0.931; Holm 1.000; 3/5) |
| tempo do servidor (s/rodada) | +5.968 [+5.706, +6.230] (p=0.000; Holm 0.000; 5/5) | +0.175 [+0.163, +0.187] (p=0.000; Holm 0.000; 5/5) | -5.793 [-6.056, -5.529] (p=0.000; Holm 0.000; 0/5) |

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

#### Diferença pareada [IC 95%] (p do t pareado; p ajustado por Holm; sementes em que o primeiro é maior)

| métrica | Shapley − FedAvg | Stackelberg − FedAvg | Stackelberg − Shapley |
|---|---|---|---|
| acurácia final (%) | +0.5 [+0.2, +0.8] (p=0.009; Holm 1.000; 5/5) | -0.0 [-0.0, +0.0] (p=0.904; Holm 1.000; 2/5) | -0.5 [-0.8, -0.2] (p=0.006; Holm 0.967; 0/5) |
| precisão do modelo (%) | +0.5 [+0.2, +0.7] (p=0.007; Holm 1.000; 5/5) | -0.0 [-0.0, +0.0] (p=0.884; Holm 1.000; 2/5) | -0.5 [-0.7, -0.2] (p=0.004; Holm 0.707; 0/5) |
| recall do modelo (%) | +0.5 [+0.2, +0.8] (p=0.010; Holm 1.000; 5/5) | -0.0 [-0.0, +0.0] (p=0.920; Holm 1.000; 2/5) | -0.5 [-0.7, -0.2] (p=0.007; Holm 0.986; 0/5) |
| F1 do modelo (%) | +0.5 [+0.2, +0.8] (p=0.010; Holm 1.000; 5/5) | -0.0 [-0.0, +0.0] (p=0.916; Holm 1.000; 2/5) | -0.5 [-0.8, -0.2] (p=0.006; Holm 0.982; 0/5) |
| oscilação nas últimas 10 rodadas (pp) | -0.03 [-0.17, +0.10] (p=0.568; Holm 1.000; 3/5) | -0.00 [-0.03, +0.02] (p=0.620; Holm 1.000; 2/5) | +0.03 [-0.12, +0.17] (p=0.645; Holm 1.000; 2/5) |
| rodada em que atinge 95% | -2.0 [-4.9, +0.9] (p=0.129; Holm 1.000; 0/5) | 0,0 (p=1) | +2.0 [-0.9, +4.9] (p=0.129; Holm 1.000; 4/5) |
| parcela da recompensa dos free-riders (%) | -14.5 [-22.1, -6.9] (p=0.006; Holm 0.967; 0/5) | 0,0 (p=1) | +14.5 [+6.9, +22.1] (p=0.006; Holm 0.967; 5/5) |
| tempo do servidor (s/rodada) | +6.017 [+5.797, +6.236] (p=0.000; Holm 0.000; 5/5) | +0.199 [+0.164, +0.235] (p=0.000; Holm 0.017; 5/5) | -5.817 [-6.018, -5.617] (p=0.000; Holm 0.000; 0/5) |

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

#### Diferença pareada [IC 95%] (p do t pareado; p ajustado por Holm; sementes em que o primeiro é maior)

| métrica | Shapley − FedAvg | Stackelberg − FedAvg | Stackelberg − Shapley |
|---|---|---|---|
| acurácia final (%) | +0.4 [-0.1, +0.9] (p=0.085; Holm 1.000; 5/5) | +0.7 [+0.3, +1.0] (p=0.006; Holm 0.982; 5/5) | +0.3 [+0.1, +0.5] (p=0.023; Holm 1.000; 5/5) |
| precisão do modelo (%) | +0.3 [-0.1, +0.8] (p=0.101; Holm 1.000; 4/5) | +0.7 [+0.3, +1.0] (p=0.007; Holm 0.982; 5/5) | +0.3 [+0.1, +0.5] (p=0.018; Holm 1.000; 5/5) |
| recall do modelo (%) | +0.4 [-0.1, +0.9] (p=0.086; Holm 1.000; 5/5) | +0.7 [+0.3, +1.1] (p=0.007; Holm 0.982; 5/5) | +0.3 [+0.0, +0.5] (p=0.034; Holm 1.000; 5/5) |
| F1 do modelo (%) | +0.4 [-0.1, +0.9] (p=0.087; Holm 1.000; 5/5) | +0.7 [+0.3, +1.1] (p=0.007; Holm 0.982; 5/5) | +0.3 [+0.1, +0.5] (p=0.027; Holm 1.000; 5/5) |
| oscilação nas últimas 10 rodadas (pp) | -0.00 [-0.09, +0.08] (p=0.900; Holm 1.000; 3/5) | +0.00 [-0.07, +0.07] (p=0.925; Holm 1.000; 2/5) | +0.01 [-0.12, +0.13] (p=0.893; Holm 1.000; 2/5) |
| rodada em que atinge 95% | -7.2 [-11.3, -3.1] (p=0.008; Holm 1.000; 0/5) | -13.0 [-19.5, -6.5] (p=0.005; Holm 0.782; 0/5) | -5.8 [-10.1, -1.5] (p=0.021; Holm 1.000; 0/5) |
| parcela da recompensa dos free-riders (%) | -15.0 [-28.1, -1.9] (p=0.034; Holm 1.000; 1/5) | -54.1 [-60.5, -47.7] (p=0.000; Holm 0.004; 0/5) | -39.1 [-54.5, -23.7] (p=0.002; Holm 0.356; 0/5) |
| tempo do servidor (s/rodada) | +7.830 [+4.118, +11.542] (p=0.004; Holm 0.687; 5/5) | +0.159 [+0.128, +0.190] (p=0.000; Holm 0.024; 5/5) | -7.671 [-11.388, -3.953] (p=0.005; Holm 0.722; 0/5) |

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
| parcela da recompensa dos free-riders (%) | 31.1 ± 11.8 | 31.9 ± 8.4 | 0.0 ± 0.0 |
| tempo do servidor (s/rodada) | 0.004 ± 0.000 | 6.953 ± 1.279 | 0.183 ± 0.016 |

#### Diferença pareada [IC 95%] (p do t pareado; p ajustado por Holm; sementes em que o primeiro é maior)

| métrica | Shapley − FedAvg | Stackelberg − FedAvg | Stackelberg − Shapley |
|---|---|---|---|
| acurácia final (%) | +2.3 [-3.1, +7.8] (p=0.302; Holm 1.000; 5/5) | +2.4 [-1.9, +6.6] (p=0.200; Holm 1.000; 5/5) | +0.0 [-1.3, +1.3] (p=0.933; Holm 1.000; 4/5) |
| precisão do modelo (%) | +1.3 [-1.8, +4.3] (p=0.313; Holm 1.000; 4/5) | +1.6 [-1.0, +4.3] (p=0.165; Holm 1.000; 5/5) | +0.3 [-0.3, +1.0] (p=0.217; Holm 1.000; 4/5) |
| recall do modelo (%) | +2.2 [-3.0, +7.4] (p=0.305; Holm 1.000; 5/5) | +2.3 [-1.8, +6.4] (p=0.196; Holm 1.000; 5/5) | +0.1 [-1.2, +1.3] (p=0.842; Holm 1.000; 4/5) |
| F1 do modelo (%) | +2.6 [-3.4, +8.7] (p=0.291; Holm 1.000; 5/5) | +2.6 [-2.3, +7.5] (p=0.210; Holm 1.000; 5/5) | -0.0 [-1.3, +1.3] (p=0.977; Holm 1.000; 4/5) |
| oscilação nas últimas 10 rodadas (pp) | +0.11 [-0.42, +0.65] (p=0.590; Holm 1.000; 4/5) | -0.09 [-0.52, +0.35] (p=0.608; Holm 1.000; 2/5) | -0.20 [-0.46, +0.06] (p=0.097; Holm 1.000; 1/5) |
| rodada em que atinge 95% | -3.6 [-10.7, +3.5] (p=0.230; Holm 1.000; 0/5) | -4.2 [-11.7, +3.3] (p=0.194; Holm 1.000; 0/5) | -0.6 [-6.4, +5.2] (p=0.788; Holm 1.000; 1/5) |
| parcela da recompensa dos free-riders (%) | +0.8 [-6.6, +8.1] (p=0.785; Holm 1.000; 3/5) | -31.1 [-45.8, -16.4] (p=0.004; Holm 0.686; 0/5) | -31.9 [-42.3, -21.4] (p=0.001; Holm 0.180; 0/5) |
| tempo do servidor (s/rodada) | +6.949 [+5.361, +8.537] (p=0.000; Holm 0.045; 5/5) | +0.179 [+0.160, +0.198] (p=0.000; Holm 0.002; 5/5) | -6.770 [-8.377, -5.164] (p=0.000; Holm 0.052; 0/5) |

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
| parcela da recompensa dos free-riders (%) | 31.1 ± 11.8 | 26.3 ± 7.4 | 31.1 ± 11.8 |
| tempo do servidor (s/rodada) | 0.004 ± 0.000 | 6.141 ± 0.093 | 0.207 ± 0.024 |

#### Diferença pareada [IC 95%] (p do t pareado; p ajustado por Holm; sementes em que o primeiro é maior)

| métrica | Shapley − FedAvg | Stackelberg − FedAvg | Stackelberg − Shapley |
|---|---|---|---|
| acurácia final (%) | -0.7 [-1.6, +0.3] (p=0.122; Holm 1.000; 1/5) | +0.0 [-0.1, +0.1] (p=0.307; Holm 1.000; 3/5) | +0.7 [-0.3, +1.7] (p=0.111; Holm 1.000; 4/5) |
| precisão do modelo (%) | -0.6 [-1.4, +0.2] (p=0.108; Holm 1.000; 1/5) | +0.0 [-0.0, +0.1] (p=0.282; Holm 1.000; 3/5) | +0.6 [-0.2, +1.4] (p=0.096; Holm 1.000; 4/5) |
| recall do modelo (%) | -0.7 [-1.6, +0.3] (p=0.132; Holm 1.000; 1/5) | +0.0 [-0.1, +0.1] (p=0.295; Holm 1.000; 3/5) | +0.7 [-0.3, +1.7] (p=0.121; Holm 1.000; 4/5) |
| F1 do modelo (%) | -0.6 [-1.6, +0.3] (p=0.144; Holm 1.000; 1/5) | +0.0 [-0.1, +0.1] (p=0.276; Holm 1.000; 3/5) | +0.7 [-0.3, +1.6] (p=0.130; Holm 1.000; 4/5) |
| oscilação nas últimas 10 rodadas (pp) | +0.76 [-0.80, +2.32] (p=0.247; Holm 1.000; 4/5) | +0.01 [-0.03, +0.05] (p=0.431; Holm 1.000; 3/5) | -0.75 [-2.31, +0.81] (p=0.255; Holm 1.000; 1/5) |
| rodada em que atinge 95% | +1.0 [-6.2, +8.2] (p=0.721; Holm 1.000; 3/5) | +0.4 [-0.3, +1.1] (p=0.178; Holm 1.000; 2/5) | -0.6 [-8.0, +6.8] (p=0.832; Holm 1.000; 1/5) |
| parcela da recompensa dos free-riders (%) | -4.8 [-19.2, +9.6] (p=0.405; Holm 1.000; 2/5) | -0.0 [-0.0, +0.0] (p=0.374; Holm 1.000; 1/5) | +4.8 [-9.6, +19.2] (p=0.405; Holm 1.000; 3/5) |
| tempo do servidor (s/rodada) | +6.136 [+6.020, +6.252] (p=0.000; Holm 0.000; 5/5) | +0.202 [+0.172, +0.232] (p=0.000; Holm 0.008; 5/5) | -5.934 [-6.077, -5.791] (p=0.000; Holm 0.000; 0/5) |

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
| parcela da recompensa dos free-riders (%) | 31.1 ± 11.8 | 22.5 ± 5.3 | 31.1 ± 11.8 |
| tempo do servidor (s/rodada) | 0.004 ± 0.000 | 6.821 ± 0.706 | 0.183 ± 0.019 |

#### Diferença pareada [IC 95%] (p do t pareado; p ajustado por Holm; sementes em que o primeiro é maior)

| métrica | Shapley − FedAvg | Stackelberg − FedAvg | Stackelberg − Shapley |
|---|---|---|---|
| acurácia final (%) | +5.2 [-3.7, +14.1] (p=0.178; Holm 1.000; 5/5) | +0.0 [-0.2, +0.2] (p=0.567; Holm 1.000; 2/5) | -5.2 [-14.2, +3.8] (p=0.186; Holm 1.000; 0/5) |
| precisão do modelo (%) | +3.8 [-2.8, +10.3] (p=0.189; Holm 1.000; 5/5) | +0.0 [-0.1, +0.1] (p=0.349; Holm 1.000; 3/5) | -3.7 [-10.3, +2.9] (p=0.194; Holm 1.000; 0/5) |
| recall do modelo (%) | +4.9 [-3.4, +13.2] (p=0.174; Holm 1.000; 5/5) | +0.0 [-0.2, +0.2] (p=0.603; Holm 1.000; 2/5) | -4.9 [-13.3, +3.5] (p=0.182; Holm 1.000; 0/5) |
| F1 do modelo (%) | +5.9 [-4.2, +16.1] (p=0.179; Holm 1.000; 5/5) | +0.0 [-0.2, +0.3] (p=0.601; Holm 1.000; 2/5) | -5.9 [-16.2, +4.4] (p=0.187; Holm 1.000; 0/5) |
| oscilação nas últimas 10 rodadas (pp) | -0.03 [-0.84, +0.78] (p=0.920; Holm 1.000; 2/5) | +0.00 [-0.05, +0.06] (p=0.929; Holm 1.000; 3/5) | +0.03 [-0.80, +0.86] (p=0.918; Holm 1.000; 3/5) |
| rodada em que atinge 95% | -1.8 [-6.8, +3.2] (p=0.374; Holm 1.000; 0/5) | 0,0 (p=1) | +1.8 [-3.2, +6.8] (p=0.374; Holm 1.000; 1/5) |
| parcela da recompensa dos free-riders (%) | -8.6 [-24.0, +6.8] (p=0.197; Holm 1.000; 1/5) | 0,0 (p=1) | +8.6 [-6.8, +24.0] (p=0.197; Holm 1.000; 4/5) |
| tempo do servidor (s/rodada) | +6.817 [+5.941, +7.693] (p=0.000; Holm 0.005; 5/5) | +0.179 [+0.155, +0.202] (p=0.000; Holm 0.005; 5/5) | -6.638 [-7.497, -5.780] (p=0.000; Holm 0.005; 0/5) |

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
| parcela da recompensa dos free-riders (%) | 47.7 ± 13.2 | 37.2 ± 14.3 | 0.0 ± 0.0 |
| tempo do servidor (s/rodada) | 0.005 ± 0.001 | 8.446 ± 4.193 | 0.187 ± 0.062 |

#### Diferença pareada [IC 95%] (p do t pareado; p ajustado por Holm; sementes em que o primeiro é maior)

| métrica | Shapley − FedAvg | Stackelberg − FedAvg | Stackelberg − Shapley |
|---|---|---|---|
| acurácia final (%) | +3.6 [-1.5, +8.6] (p=0.121; Holm 1.000; 4/5) | +3.8 [-1.8, +9.3] (p=0.132; Holm 1.000; 5/5) | +0.2 [-1.1, +1.5] (p=0.688; Holm 1.000; 2/5) |
| precisão do modelo (%) | +2.0 [-0.8, +4.9] (p=0.122; Holm 1.000; 4/5) | +2.1 [-1.0, +5.3] (p=0.130; Holm 1.000; 5/5) | +0.1 [-0.9, +1.2] (p=0.720; Holm 1.000; 2/5) |
| recall do modelo (%) | +3.3 [-1.3, +7.9] (p=0.119; Holm 1.000; 4/5) | +3.7 [-1.6, +8.9] (p=0.126; Holm 1.000; 5/5) | +0.4 [-1.0, +1.7] (p=0.496; Holm 1.000; 3/5) |
| F1 do modelo (%) | +4.0 [-2.0, +9.9] (p=0.141; Holm 1.000; 4/5) | +4.4 [-3.0, +11.9] (p=0.173; Holm 1.000; 5/5) | +0.5 [-1.5, +2.5] (p=0.540; Holm 1.000; 2/5) |
| oscilação nas últimas 10 rodadas (pp) | +0.05 [-0.33, +0.43] (p=0.727; Holm 1.000; 3/5) | -0.03 [-0.29, +0.23] (p=0.779; Holm 1.000; 3/5) | -0.08 [-0.25, +0.09] (p=0.253; Holm 1.000; 2/5) |
| rodada em que atinge 95% | 0,0 (p=1) | 0,0 (p=1) | 0,0 (p=1) |
| parcela da recompensa dos free-riders (%) | -10.5 [-32.8, +11.8] (p=0.260; Holm 1.000; 2/5) | -47.7 [-64.1, -31.4] (p=0.001; Holm 0.213; 0/5) | -37.2 [-55.0, -19.4] (p=0.004; Holm 0.704; 0/5) |
| tempo do servidor (s/rodada) | +8.442 [+3.237, +13.647] (p=0.011; Holm 1.000; 5/5) | +0.182 [+0.106, +0.259] (p=0.003; Holm 0.444; 5/5) | -8.259 [-13.392, -3.127] (p=0.011; Holm 1.000; 0/5) |

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
