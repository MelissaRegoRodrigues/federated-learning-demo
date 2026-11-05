# Análise dos resultados: FedAvg × Shapley × Stackelberg

Comparação dos três mecanismos contra free-riders em aprendizado federado com dados não-IID (MNIST, Flower).
Todas as tabelas vêm de `results/` e são reproduzíveis com `scripts/analyze_comparison.py` (tabelas completas, com todas as métricas e os testes pareados, em `results/tabelas_comparacao.md`) e `scripts/summarize_all.sh` (resumo detalhado em `results/resumo_geral.txt`).

## 1. Configuração do experimento

| Item | Valor |
|---|---|
| Base | MNIST, 10 clientes, 50 rodadas, rede pequena (18.406 parâmetros) |
| Heterogeneidade | Dirichlet α = 0,5 (moderada) e α = 0,1 (forte) |
| Sementes | 42 a 46 (5 por configuração, pareadas entre mecanismos) |
| Cenários de ataque | **silent** (devolve o modelo global), **low-effort** (1 época em vez de 2), **noisy** (modelo global + ruído gaussiano σ = 0,05, ataque ativo, checagem de sanidade), **fr50** (5 de 10 clientes free-riders, silenciosos) |
| Free-riders | 3 clientes (0, 1, 2), 30%; no fr50 são 5 clientes |
| FedAvg | média ponderada pelo número de exemplos; todos os clientes treinam com 100% dos dados locais |
| Shapley | leave-one-out sobre acurácia de validação; sinaliza os int(V·N) piores por rodada (V = 0,3; V = 0,5 no fr50); pesos uniformes entre os não sinalizados; 100% dos dados locais |
| Stackelberg | jogo preço × dados (EntroCFL [29]); teste de compatibilidade por entropia + cosseno, limiar Cm0 = 0,01 **fixo em todos os cenários**; barramento permanente |

**Mesmos dados para os três.** Na comparação principal os três mecanismos treinam com **100% dos dados locais** dos clientes que participam. Para isso, o Stackelberg da coluna principal é a variante com dados completos (`--full-data`): o jogo continua decidindo **quem participa** (quem é barrado), mas quem participa usa todos os seus dados. O Stackelberg original, em que cada cliente escolhe a quantidade de dados pelo preço (resposta ótima), entra numa tabela complementar (seção 6), porque usa só ~7 a 9% dos dados de todos os clientes e isso, sozinho, derruba a acurácia.

**Métricas do modelo.** Acurácia, precisão, recall e F1 (média macro entre as 10 classes) do modelo global no conjunto de teste, na última rodada. Elas são distintas da precisão e do recall **da detecção** (seção 4).

**Como ler os testes.** Com n = 5 o menor p possível do Wilcoxon é 0,0625, então ele nunca é "significativo" a 5%. Por isso usamos o t pareado por semente, o tamanho do efeito e a contagem de sementes em que um mecanismo supera o outro (por exemplo "5/5"). Não há correção para comparações múltiplas; trate p entre 0,01 e 0,05 como indício, não como prova.

## 2. Desempenho do modelo (acurácia, precisão, recall e F1)

Média ± desvio entre as 5 sementes, em %. Precisão, recall e F1 são do modelo global.

### 2.1 Free-riders passivos: silent e fr50 (onde o Stackelberg detecta)

| Cenário | Métrica | FedAvg | Shapley | Stackelberg |
|---|---|---|---|---|
| α=0,5 silent | acurácia | 97,9 ± 0,1 | 97,8 ± 0,1 | **98,0 ± 0,1** |
| | precisão | 97,9 ± 0,1 | 97,8 ± 0,1 | 98,0 ± 0,1 |
| | recall | 97,9 ± 0,1 | 97,8 ± 0,1 | 98,0 ± 0,1 |
| | F1 | 97,9 ± 0,1 | 97,8 ± 0,1 | 98,0 ± 0,1 |
| α=0,5 fr50 | acurácia | 96,5 ± 0,6 | 96,9 ± 0,5 | **97,2 ± 0,4** |
| | precisão | 96,6 ± 0,5 | 96,9 ± 0,5 | 97,3 ± 0,4 |
| | recall | 96,5 ± 0,6 | 96,9 ± 0,5 | 97,2 ± 0,4 |
| | F1 | 96,5 ± 0,6 | 96,9 ± 0,5 | 97,2 ± 0,4 |
| α=0,1 silent | acurácia | 91,0 ± 6,0 | 93,3 ± 2,0 | 93,3 ± 2,7 |
| | precisão | 92,2 ± 4,2 | 93,4 ± 2,0 | 93,8 ± 2,1 |
| | recall | 91,0 ± 5,9 | 93,2 ± 2,1 | 93,3 ± 2,6 |
| | F1 | 90,5 ± 6,5 | 93,1 ± 2,1 | 93,1 ± 2,7 |
| α=0,1 fr50 | acurácia | 85,4 ± 9,6 | 88,9 ± 6,7 | 89,1 ± 6,4 |
| | precisão | 86,4 ± 8,9 | 88,5 ± 8,4 | 88,6 ± 8,4 |
| | recall | 85,5 ± 9,3 | 88,8 ± 6,7 | 89,2 ± 6,3 |
| | F1 | 83,4 ± 12,0 | 87,4 ± 8,7 | 87,8 ± 8,1 |

Diferenças pareadas em pontos percentuais (p do t pareado; sementes em que o primeiro vence):

| Cenário | Métrica | Shapley − FedAvg | Stackelberg − FedAvg | Stackelberg − Shapley |
|---|---|---|---|---|
| α=0,5 silent | acurácia | −0,1 (0,231; 1/5) | **+0,1 (0,003; 5/5)** | +0,2 (0,053; 5/5) |
| | F1 | −0,1 (0,238; 1/5) | **+0,1 (0,004; 5/5)** | +0,2 (0,055; 5/5) |
| α=0,5 fr50 | acurácia | +0,4 (0,085; 5/5) | **+0,7 (0,006; 5/5)** | **+0,3 (0,023; 5/5)** |
| | F1 | +0,4 (0,087; 5/5) | **+0,7 (0,007; 5/5)** | **+0,3 (0,027; 5/5)** |
| α=0,1 silent | acurácia | +2,3 (0,302; 5/5) | +2,4 (0,200; 5/5) | +0,0 (0,933; 4/5) |
| | F1 | +2,6 (0,291; 5/5) | +2,6 (0,210; 5/5) | −0,0 (0,977; 4/5) |
| α=0,1 fr50 | acurácia | +3,6 (0,121; 4/5) | +3,8 (0,132; 5/5) | +0,2 (0,688; 2/5) |
| | F1 | +4,0 (0,141; 4/5) | +4,4 (0,173; 5/5) | +0,5 (0,540; 2/5) |

(Precisão e recall seguem o mesmo padrão; estão em `results/tabelas_comparacao.md`.)

**Leitura.**
- Com α=0,5 o Stackelberg fica **ligeiramente acima** do FedAvg nas quatro métricas (+0,1 pp no silent e +0,7 pp no fr50, 5/5 sementes, p ≤ 0,007). É diferença pequena, mas consistente. Sobre o Shapley, a vantagem é significativa só no fr50. O ganho vem de excluir quem só dilui o modelo; o efeito cresce com a proporção de free-riders.
- Com α=0,1 o Stackelberg e o Shapley ficam 2 a 4 pp acima do FedAvg em média (5/5 e 4 a 5/5 sementes), **mas sem significância** (p entre 0,12 e 0,30), porque o FedAvg tem desvio de 6 a 10 pp. Stackelberg e Shapley não se distinguem entre si.
- **Onde precisão, recall e F1 dizem algo além da acurácia:** só com α=0,1. O FedAvg tem F1 menor que a acurácia (83,4% contra 85,4% no fr50; 90,5% contra 91,0% no silent) e desvio maior no F1 (12,0 pp no fr50), o que indica erro concentrado em algumas classes quando os dados são muito concentrados. Stackelberg e Shapley reduzem essa lacuna (F1 87,8% e 87,4%), isto é, melhoram também as classes mais prejudicadas, não só a média. Com α=0,5 as quatro métricas são praticamente idênticas e a acurácia basta.
- **Todos os testes aqui são pareados e com n = 5.** As diferenças com α=0,1 são tendência consistente em direção, não prova.

### 2.2 Low-effort (passivo parcial): ninguém ganha do FedAvg

| Cenário | Métrica | FedAvg | Shapley | Stackelberg |
|---|---|---|---|---|
| α=0,5 low-effort | acurácia / precisão / recall / F1 | 98,2 (todas) | 98,2 (todas) | 98,2 (todas) |
| α=0,1 low-effort | acurácia | 96,2 ± 0,7 | 95,5 ± 0,7 | 96,2 ± 0,7 |
| | precisão | 96,3 ± 0,7 | 95,7 ± 0,7 | 96,3 ± 0,7 |
| | recall | 96,1 ± 0,7 | 95,5 ± 0,7 | 96,2 ± 0,7 |
| | F1 | 96,1 ± 0,8 | 95,5 ± 0,7 | 96,1 ± 0,8 |

- Pareado: Shapley − FedAvg = −0,7 pp com α=0,1 (p = 0,122; 1/5), e 0,0 com α=0,5. Stackelberg − FedAvg ≈ 0,0 (p ≥ 0,3).
- O Stackelberg **não barra ninguém** (Cm0 fixo não pega esse ataque; no α=0,1 chega a barrar um cliente em 7% dos casos), então ele se comporta como FedAvg. O Shapley fica um pouco abaixo e oscila mais (1,09 contra 0,33 pp no α=0,1). A conclusão é que contra um free-rider que treina 1 época nenhum mecanismo ajuda, e o dano do próprio ataque ao FedAvg já é pequeno (seção 3 do documento de taxonomia).

### 2.3 Noisy (ataque ativo): só o Shapley ajuda

| Cenário | Métrica | FedAvg | Shapley | Stackelberg |
|---|---|---|---|---|
| α=0,5 noisy | acurácia | 97,0 ± 0,4 | **97,5 ± 0,2** | 97,0 ± 0,3 |
| | precisão | 97,0 ± 0,3 | 97,5 ± 0,2 | 97,0 ± 0,3 |
| | recall | 97,0 ± 0,3 | 97,5 ± 0,2 | 97,0 ± 0,3 |
| | F1 | 97,0 ± 0,4 | 97,5 ± 0,2 | 97,0 ± 0,3 |
| α=0,1 noisy | acurácia | 87,0 ± 8,7 | **92,2 ± 2,4** | 87,1 ± 8,8 |
| | precisão | 88,8 ± 6,7 | 92,6 ± 2,2 | 88,9 ± 6,7 |
| | recall | 87,2 ± 8,3 | 92,1 ± 2,4 | 87,2 ± 8,4 |
| | F1 | 86,0 ± 9,8 | 92,0 ± 2,5 | 86,1 ± 9,9 |

- Shapley − FedAvg: **+0,5 pp** com α=0,5 (p = 0,009; 5/5, nas quatro métricas) e **+5,2 pp** com α=0,1 (5/5, p = 0,178; F1 +5,9 pp). Stackelberg ≈ FedAvg (não sinaliza ninguém).
- É o único cenário em que o Shapley se destaca de forma clara. O ruído derruba a acurácia de validação de quem o envia, e o leave-one-out percebe isso (detecção 0,65 com α=0,5).
- O noisy é um **ataque ativo**, não free-riding passivo (documento de taxonomia). Por isso ele é reportado à parte, como checagem de sanidade, e não entra na resposta a H1.

## 3. Estabilidade

**(a) Oscilação dentro de uma execução** (desvio-padrão da acurácia nas últimas 10 rodadas, em pp):

| Cenário | FedAvg | Shapley | Stackelberg |
|---|---|---|---|
| α=0,5 silent | 0,11 | 0,09 | 0,09 |
| α=0,5 low-effort | 0,12 | 0,09 | 0,12 |
| α=0,5 noisy | 0,19 | 0,16 | 0,18 |
| α=0,5 fr50 | 0,17 | 0,17 | 0,18 |
| α=0,1 silent | 0,46 | 0,57 | 0,37 |
| α=0,1 low-effort | 0,33 | 1,09 | 0,35 |
| α=0,1 noisy | 1,01 | 0,98 | 1,01 |
| α=0,1 fr50 | 0,50 | 0,55 | 0,47 |

Com os mesmos dados, **nenhuma diferença significativa** de oscilação entre os três (a menor é Stackelberg − Shapley no α=0,1 silent: −0,20 pp, p = 0,097). A oscilação maior que eu havia observado no Stackelberg era da variante com preço, em que cada cliente treina com uma fração variável dos dados (seção 6), e não do mecanismo de exclusão.

**(b) Variabilidade entre sementes** (desvio-padrão da acurácia final, α=0,1):

| Cenário | FedAvg | Shapley | Stackelberg |
|---|---|---|---|
| silent | ±6,0 | ±2,0 | ±2,7 |
| noisy | ±8,7 | ±2,4 | ±8,8 |
| fr50 | ±9,6 | ±6,7 | ±6,4 |

Com α=0,1 o FedAvg às vezes tem uma semente ruim, e os mecanismos que excluem ou pesam de forma uniforme reduzem isso onde há efeito (Shapley em todos; Stackelberg onde detecta). É descritivo: com n = 5 não dá para testar variâncias. Parte do efeito do Shapley vem do peso uniforme e não da detecção (seção 6). Com α=0,5 todos os desvios ficam abaixo de 1 pp.

**Rodadas até 95% de acurácia.** Onde o Stackelberg detecta, ele chega mais cedo: silent α=0,5, 9,4 rodadas contra 11,6 (FedAvg, p = 0,004) e 12,2 (Shapley, p = 0,025); fr50 α=0,5, 13,0 contra 26,0 (FedAvg, p = 0,005) e 18,8 (Shapley, p = 0,021). Nos demais cenários não há diferença significativa.

## 4. Detecção de free-riders

Precisão e recall **da detecção**, não do modelo. No Shapley os dois são iguais por construção (o número de sinalizados é fixo e igual ao de free-riders). A referência do acaso é 0,30 (3 de 10) e 0,50 no fr50.

| Cenário | Shapley | Stackelberg (precisão / recall) |
|---|---|---|
| α=0,5 silent | 0,33 | **1,00 / 1,00** |
| α=0,5 low-effort | 0,31 | não sinaliza ninguém / 0,00 |
| α=0,5 noisy | **0,65** | não sinaliza ninguém / 0,00 |
| α=0,5 fr50 | 0,61 | **1,00 / 1,00** |
| α=0,1 silent | 0,26 | **1,00 / 1,00** |
| α=0,1 low-effort | 0,39 | 1,00 / 0,07 |
| α=0,1 noisy | 0,47 | não sinaliza ninguém / 0,00 |
| α=0,1 fr50 | 0,63 | **1,00 / 1,00** |

- **Stackelberg:** perfeito contra free-riders silenciosos e no fr50 (zero falsos positivos). Não detecta low-effort nem noisy: o limiar fixo (Cm0 = 0,01) está entre o Cm dos silenciosos (0) e o dos honestos (≥ 0,067), e esses ataques não ficam abaixo dele.
- **Shapley:** no silent e no low-effort fica no nível do acaso (abaixo dele no silent com α=0,1). Só se afasta do acaso no noisy com α=0,5 (0,65) e no fr50 (0,61 e 0,63, contra 0,50 do acaso). Sinaliza honestos em 15 a 39% das rodadas e sinaliza mais os clientes com pouco dado.
- **Sem free-riders (fr0, α=0,5):** o Shapley sinaliza 3 honestos por rodada, sem perda de acurácia (98,3%, igual ao FedAvg).

## 5. Recompensa e custo do servidor

**Parcela da recompensa que vai para os free-riders (%).** No FedAvg não há pagamento explícito; usamos o peso de agregação, proporcional ao tamanho dos dados (≈ 29,7%, e 54,1% no fr50).

| Cenário | FedAvg | Shapley | Stackelberg |
|---|---|---|---|
| α=0,5 silent | 29,7 | 28,7 | **0,0** |
| α=0,5 low-effort | 29,7 | 29,5 | 29,7 |
| α=0,5 noisy | 29,7 | **15,2** | 29,7 |
| α=0,5 fr50 | 54,1 | 39,1 | **0,0** |
| α=0,1 silent | 29,7 | 31,9 | **0,0** |
| α=0,1 low-effort | 29,7 | 26,3 | 31,1 |
| α=0,1 noisy | 29,7 | 22,5 | 31,1 |
| α=0,1 fr50 | 54,1 | 37,2 | **0,0** |

O Stackelberg elimina totalmente a recompensa dos free-riders onde os detecta (silent e fr50), a diferença mais forte e mais clara do experimento (p ≤ 0,004 contra os dois outros). O Shapley reduz a recompensa no noisy com α=0,5 (−14,5 pp, p = 0,006) e no fr50 (−15 pp, p = 0,034), mas não no silent nem no low-effort. Vale ressalvar que a recompensa do FedAvg e a do Shapley são pesos de agregação, enquanto a do Stackelberg é um pagamento: os três não pagam na mesma unidade, então a comparação é de **parcela**, não de valor absoluto.

**Custo do servidor por rodada (segundos, medido e relativo).**

| | FedAvg | Shapley | Stackelberg |
|---|---|---|---|
| típico | 0,004 | 6 a 8,4 | 0,16 a 0,21 (+ ~2,4 s de calibração única) |

O Shapley custa cerca de **1.500 vezes** o FedAvg (o leave-one-out exige uma avaliação do modelo por cliente a cada rodada) e cerca de 30 a 50 vezes o Stackelberg. A diferença é significativa em todos os cenários.

## 6. Stackelberg: dados completos × dados escolhidos pelo preço

A coluna principal usa dados completos para que os três mecanismos treinem com a mesma quantidade de dados. O Stackelberg como definido na literatura [29] deixa cada cliente escolher a quantidade de dados que treina (resposta ótima ao preço). A tabela abaixo mostra as duas variantes lado a lado. **Os pagamentos só são comparáveis dentro da mesma variante:** na de dados completos o pagamento é calculado sobre dados que o cliente não escolheu, então não representa o equilíbrio do jogo.

Dados usados = % dos dados de **todos** os clientes (inclusive os free-riders) que participaram do treino, média das rodadas. Pagamento em milhares de unidades sintéticas.

| Cenário | Variante | Acurácia (%) | F1 (%) | Dados usados (%) | Pagamento (mil) |
|---|---|---|---|---|---|
| α=0,5 silent | dados completos | 98,0 ± 0,1 | 98,0 ± 0,1 | 70,9 ± 4,7 | 4443 ± 549 |
| | preço | 92,8 ± 1,2 | 92,7 ± 1,2 | 7,9 ± 1,5 | 501 ± 147 |
| α=0,5 low-effort | dados completos | 98,2 ± 0,1 | 98,2 ± 0,1 | 100,0 | 5751 ± 335 |
| | preço | 91,6 ± 1,8 | 91,5 ± 1,8 | 9,1 ± 1,3 | 528 ± 104 |
| α=0,5 noisy | dados completos | 97,0 ± 0,3 | 97,0 ± 0,3 | 100,0 | 5751 ± 335 |
| | preço | 88,2 ± 2,5 | 88,1 ± 2,6 | 9,1 ± 1,3 | 528 ± 104 |
| α=0,5 fr50 | dados completos | 97,2 ± 0,4 | 97,2 ± 0,4 | 47,0 ± 5,1 | 3134 ± 408 |
| | preço | 92,1 ± 0,6 | 91,9 ± 0,6 | 6,7 ± 1,0 | 456 ± 113 |
| α=0,1 silent | dados completos | 93,3 ± 2,7 | 93,1 ± 2,7 | 69,5 ± 11,6 | 4364 ± 986 |
| | preço | 83,6 ± 3,7 | 82,1 ± 5,0 | 7,9 ± 1,5 | 501 ± 147 |
| α=0,1 low-effort | dados completos | 96,2 ± 0,7 | 96,1 ± 0,8 | 100,0 | 5767 ± 308 |
| | preço | 87,1 ± 1,2 | 86,7 ± 1,3 | 9,0 ± 1,5 | 523 ± 114 |
| α=0,1 noisy | dados completos | 87,1 ± 8,8 | 86,1 ± 9,9 | 100,0 | 5751 ± 335 |
| | preço | 76,9 ± 4,2 | 74,5 ± 5,9 | 9,0 ± 1,5 | 523 ± 114 |
| α=0,1 fr50 | dados completos | 89,1 ± 6,4 | 87,8 ± 8,1 | 53,2 ± 12,9 | 3621 ± 1210 |
| | preço | 81,2 ± 8,1 | 79,2 ± 10,1 | 6,7 ± 1,0 | 455 ± 113 |

**Leitura.**
- A versão com preço usa de 7 a 9% dos dados de todos os clientes e paga de 456 a 528 mil, contra 47 a 100% dos dados e de 3.134 a 5.767 mil com dados completos. **É um compromisso real:** o jogo original economiza cerca de 90% dos dados e do pagamento, ao custo de 5 a 11 pp de acurácia (e até 12 pp de F1 com α=0,1).
- Onde o Stackelberg não barra ninguém (low-effort, noisy), os dados completos dão o mesmo resultado do FedAvg e a versão com preço perde ~9 a 11 pp. A perda é, portanto, do **volume de dados**, não da detecção.
- A leitura "o Stackelberg perde em todos os cenários" vale **só para a versão com preço**, e só quando se compara a acurácia isolada. Ao comparar também o custo (dados e pagamento), o resultado é um equilíbrio entre acurácia e economia, e não uma derrota.

## 7. O que explica cada resultado (controles e ablações)

Controles rodados em todos os cenários (5 sementes cada): **`shapley_v0`** (Shapley sem exclusão, V=0), **`control`** (FedAvg com o mesmo volume de dados do jogo com preço) e, nos cenários silent e fr50, o **detector perfeito** (FedAvg ignorando os free-riders reais). Acurácia final (%):

| Cenário | FedAvg | Shapley | Shapley V=0 | Stackelberg (dados completos) | Stackelberg (preço) | FedAvg controle (dados do preço) | Detector perfeito |
|---|---|---|---|---|---|---|---|
| α=0,5 silent | 97,9 | 97,8 | 97,8 | 98,0 | 92,8 | 91,6 | 98,0 |
| α=0,5 noisy | 97,0 | 97,5 | 96,9 | 97,0 | 88,2 | 88,2 | n/a |
| α=0,5 fr50 | 96,5 | 96,9 | 96,7 | 97,2 | 92,1 | 89,5 | 97,2 |
| α=0,1 silent | 91,0 | 93,3 | 92,4 | 93,3 | 83,6 | 81,4 | 93,3 |
| α=0,1 noisy | 87,0 | 92,2 | 89,3 | 87,1 | 76,9 | 77,0 | n/a |
| α=0,1 fr50 | 85,4 | 88,9 | 87,9 | 89,1 | 81,2 | 77,2 | 89,2 |

- **Stackelberg com dados completos = detector perfeito** (98,0; 97,2; 93,3; 89,1 contra 98,0; 97,2; 93,3; 89,2). Isso confirma que a detecção é correta, e que a diferença para o FedAvg é o tamanho do que o detector pode recuperar.
- **Stackelberg com preço = FedAvg controle** (quando não sinaliza ninguém, idêntico; quando sinaliza, a exclusão devolve de 1,2 a 4 pp). A perda vem do preço.
- **Shapley com α=0,1:** o ganho sobre o FedAvg vem em boa parte do **peso uniforme entre clientes** (V=0 já dá 92,4 no silent contra 91,0 do FedAvg). A exclusão acrescenta de 0,9 pp (silent) a 2,9 pp (noisy); no noisy é o único caso em que ela contribui de forma clara.
- **Detector perfeito:** com α=0,5 ele só ganha 0,1 pp sobre o FedAvg no silent, então a folga que um detector poderia recuperar é minúscula e não deve ser reportada como "recuperação" nesse cenário. Com α=0,1 a folga é de 2 a 4 pp.
- **Sobre a perda do silent com α=0,1 (5,4 pp contra 10 honestos):** cerca de 3,1 pp são só os dados dos 3 clientes que faltam, e uns 2,3 pp vêm da atualização vazia puxando o modelo para trás.

## 8. Síntese comparativa

| Critério | FedAvg | Shapley | Stackelberg (dados completos) |
|---|---|---|---|
| Acurácia/precisão/recall/F1, α=0,5, passivos | referência | empatado +0,1 a +0,7 pp (silent, fr50), pequeno |
| Acurácia/F1, α=0,1, passivos | pior em média | melhor em média, n.s. | melhor em média, n.s. |
| Low-effort | referência | −0,7 pp (α=0,1), n.s. | igual ao FedAvg |
| Noisy (ataque ativo) | referência | **+0,5 / +5,2 pp** | igual ao FedAvg |
| Oscilação entre rodadas | baixa | baixa | baixa |
| Detecção silent / fr50 | nenhuma | acaso a 0,63 | **perfeita** |
| Detecção low-effort / noisy | nenhuma | 0,31 a 0,65 | nenhuma |
| Recompensa que vai a free-riders | proporcional aos dados | −0 a −15 pp | **0** quando detecta |
| Custo do servidor | mínimo | ~1.500× | ~40× |
| Variante com preço | n/a | n/a | economiza ~90% dos dados e do pagamento, −5 a −11 pp |

Nenhum mecanismo domina os outros. O Stackelberg ganha em recompensa justa e em acurácia nos cenários em que detecta, com custo de servidor baixo; o Shapley é o único que ajuda contra ruído e perde por custo e por detecção fraca; o FedAvg ganha em simplicidade e não perde nada onde o ataque é leve.

## 9. Conclusões que o experimento sustenta

1. **O Stackelberg detecta com precisão e recall de 100% os free-riders silenciosos (3 de 10 e 5 de 10)** e, com os mesmos dados dos demais mecanismos, iguala ou supera o FedAvg e o Shapley nas quatro métricas. Com α=0,5 o efeito é estatisticamente detectável, mas pequeno em MNIST (efeito de teto, acurácia perto de 98%); com α=0,1 a diferença é maior em pontos percentuais, mas inconclusiva (alta variância do FedAvg). Os p-valores devem ser lidos com a correção de Holm e o IC 95% da diferença, impressos por `scripts/summarize_results.py`.
2. **O custo do jogo com preço é de dados e acurácia:** ele usa ~7 a 9% dos dados e paga ~90% menos, e perde de 5 a 11 pp de acurácia. A perda vem do volume de dados, não da detecção (seção 7).
3. **O Shapley com leave-one-out é um detector fraco** (nível do acaso para silent e low-effort), custa ~1.500× o FedAvg e seu ganho com α=0,1 é consistente em direção mas não significativo, em parte do peso uniforme. Só se destaca contra ruído, que é ataque ativo. Isso é compatível com [18], que descreve o leave-one-out como "moderadamente eficaz".
4. **Nenhum mecanismo ajuda contra low-effort.** Esse tipo é o único sem respaldo direto na literatura (o número de épocas é escolha do trabalho), por isso é análise de sensibilidade.
5. **H1 (o FedAvg degrada com free-riders)** deve ser testada nos tipos passivos (silent e low-effort), com o ruidoso reportado à parte. Com α=0,5 a degradação por passivos é pequena (0,1 a 0,4 pp); com α=0,1 chega a ~5 pp no silent, parte dela só pela falta de dados.
6. **Com α=0,5 e MNIST os efeitos são pequenos** (a acurácia satura em ~98%); os mecanismos só fazem diferença maior com heterogeneidade forte, proporção alta de free-riders ou ataque ativo.

## 10. Limitações (para declarar no texto)

- **n = 5 sementes por configuração.** O Wilcoxon não alcança significância e várias diferenças em α=0,1 têm p alto por causa do desvio do FedAvg. Mais sementes dariam mais poder.
- **Stackelberg com dados completos não é o mecanismo da literatura:** o cliente deixa de jogar a resposta ótima e o pagamento é calculado sobre dados que ele não escolheu. Serve para igualar os dados e isolar a detecção; o Stackelberg original aparece na seção 6.
- **O limiar Cm0 é fixo e não foi calibrado por cenário.** Ajustá-lo ao ataque seria otimizar em cima do conjunto de teste, e na prática o servidor não conhece o ataque. Com ele, o Stackelberg só detecta silenciosos. Uma varredura de Cm0 fica como análise de sensibilidade complementar, não executada.
- **"Shapley" é leave-one-out**, não o valor de Shapley exato nem o aproximado por Monte Carlo ou GTG-Shapley [40]. Não foi testado.
- **A utilidade do Shapley (acurácia) satura, e isso explica a detecção no nível do acaso.** Diagnóstico a partir dos `client_scores` já salvos (`scripts/diagnose_shapley.py`, sem novas execuções). Com α=0,5 e free-riders silenciosos, nas rodadas 0 a 5 o φ dos free-riders é negativo (média −0,012; 92% deles com φ<0), como esperado. A partir da rodada 5, o |φ| cai para cerca de 0,0005 a 0,001 (3 a 6 amostras em 6.000) e 96% das rodadas têm empate em φ, de modo que o ranking vira ruído. O desempate do código é por `client_id` crescente, e os free-riders são os ids 0 a 2; com desempate aleatório o recall cai de 0,33 para 0,29 (acaso: 0,30), ou seja, o 0,33 é em parte artefato do desempate. Com α=0,1 não há saturação (|φ| de 0,005 a 0,01), mas o φ dos silenciosos é positivo depois da rodada 5 (só 3 a 9% com φ<0): tudo indica que, com label skew forte, o modelo sem mudança que o silencioso devolve age como âncora e removê-lo piora a média (inferência, não testada isoladamente). Com ruído, o φ médio dos free-riders é negativo em todas as janelas. A variante com perda (`--shapley-utility loss`) é a correção natural para a saturação; seus resultados estão em `results/*/shapley_loss/` quando executada (`bash scripts/run_all.sh shapley_loss`).
- **O V do Shapley depende do cenário** (0,3; 0,5 no fr50), seguindo a regra de escolher V ligeiramente acima da fração de free-riders. Isso pressupõe conhecer essa fração, o que é uma vantagem frente ao Stackelberg.
- **Informação e recompensa diferentes entre os mecanismos:** o Shapley usa um conjunto de validação no servidor; o Stackelberg usa só os gradientes; o FedAvg não paga. A parcela de recompensa do FedAvg é inferida do peso de agregação, e os pagamentos do Stackelberg estão em unidades sintéticas.
- **Free-riders não estratégicos:** eles têm comportamento fixo e não reagem ao mecanismo.
- **Sem baseline robusto "sem jogo"** (por exemplo agregação por mediana ou Krum) na comparação. O FedAvg é o único baseline sem mecanismo.
- **Base e rede:** apenas MNIST com uma CNN pequena. A rede maior, usada nas primeiras rodadas, está arquivada em `results_rede_grande/` e não entra nesta análise. A saturação de acurácia do MNIST reduz a diferença visível entre mecanismos.
- **Sem correção para comparações múltiplas.**

## 11. Reprodução

```bash
bash scripts/run_all.sh                # bateria completa (uma execução por vez)
bash scripts/run_full_data.sh          # Stackelberg com dados completos (silent)
bash scripts/run_full_data_rest.sh     # idem para fr50, noisy e low-effort
bash scripts/summarize_all.sh > results/resumo_geral.txt
python scripts/analyze_comparison.py > results/tabelas_comparacao.md   # no WSL (precisa de torch)
```
