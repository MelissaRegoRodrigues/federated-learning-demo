# Taxonomia de free-riders (rascunho para a seção 2.3)

Rascunho em português para adaptar ao texto do TCC. Cada afirmação indica de onde vem: **[RSL]** = artigo da revisão sistemática (conferido no PDF), **[externo]** = fonte que não está entre os 67 artigos da RSL, **[implementação]** = o que o código faz (`src/flapp/client_app.py`).

## 1. Dois eixos para classificar o comportamento

Um cliente que não ajuda o modelo pode ser descrito por dois eixos independentes:

| Eixo | Pergunta | Valores |
|---|---|---|
| Intenção observável | O cliente apenas deixa de contribuir, ou tenta piorar/disfarçar o modelo? | passivo × ativo |
| Esforço de treino | Quanto ele treina de fato? | nenhum, parcial |

O termo **free-rider** é usado aqui só para o comportamento **passivo**: obter o modelo global sem contribuir (ou contribuindo menos do que declara). O comportamento **ativo** (enviar atualização aleatória ou prejudicial) é um ataque, e não faz parte da definição de free-rider.

## 2. Os três tipos implementados e como cada um se justifica

| Tipo | O que o cliente envia | Classe | Base |
|---|---|---|---|
| **Silencioso** | o modelo global sem alteração (atualização zero) | free-rider passivo | **[RSL]** [17]: *"free riding means that edge nodes directly upload the global model to be trained to the central coordinator without training."* |
| **Low-effort** | modelo treinado por 1 época em vez de 2, declarando o tamanho de dados completo | free-rider passivo parcial | **[RSL]** [04] e [16] simulam usuários que *"do not respect the required number of local iterations and compute fewer epochs"*. O número de épocas (1 em vez de 2) é **escolha do trabalho**, e nenhum artigo da RSL usa esse termo nem esse valor. |
| **Ruidoso** | modelo global somado a ruído gaussiano (σ = 0,05) | **ataque ativo / free-rider disfarçado** | **[RSL]** um único artigo (*Incentive-Aware Decentralized Data Collaboration*) separa os dois conceitos: free-riders que obtêm parâmetros *"without doing their part"* e partes maliciosas que fornecem *"useless or harmful model parameters to degrade"* o modelo. O disfarce com ruído vem de Lin et al. (2019) e Fraboni et al. (2021): **[externo]**, ver a seção 4. |

Cenários de proporção: 30% de free-riders nos cenários principais e 50% no `fr50`, como análise de sensibilidade.

## 3. O que isso muda na pergunta H1 ("o FedAvg degrada com free-riders")

Os dados já existem: o FedAvg foi rodado com cada tipo isolado, 5 sementes, nas duas heterogeneidades, e há uma referência sem free-riders (10 clientes honestos). Diferença da acurácia final do FedAvg contra a referência, em pontos percentuais:

| Tipo | α = 0,5 | α = 0,1 |
|---|---|---|
| Referência, 10 honestos | 98,3 | 96,4 |
| Silencioso (30%) | −0,42 (p = 0,009; pior em 5/5) | −5,38 (p = 0,095; pior em 5/5) |
| Low-effort (30%) | −0,09 (p = 0,040; 5/5) | −0,19 (p = 0,086; 5/5) |
| Ruidoso (30%) | **−1,29** (p = 0,003; 5/5) | **−9,33** (p = 0,064; 5/5) |
| Silencioso (50%) | −1,75 (p = 0,004; 5/5) | −10,97 (p = 0,052; 5/5) |

Leituras possíveis, e quais são seguras:
- **Seguro:** em todos os tipos e cenários o FedAvg fica pior que a referência nas 5 sementes. Com α = 0,5 a perda dos tipos **passivos é pequena** (0,1 a 0,4 pp).
- **A suspeita da banca tem fundamento:** o tipo que mais degrada o FedAvg é o **ruidoso**, que é o ativo. Com α = 0,5 ele degrada cerca de 3 vezes mais que o silencioso, e com α = 0,1 cerca de 1,7 vez. Se H1 for sustentada só pelo ruidoso, ela vira "o FedAvg degrada com ruído adversarial". Por isso H1 deve ser testada nos tipos passivos (silencioso e low-effort), e o ruidoso reportado à parte.
- **Parte da perda não é do "ataque", e sim da ausência de dados.** Com α = 0,1, o FedAvg só com os 7 clientes honestos (o "detector perfeito") chega a 93,3%, contra 96,4% com os 10 honestos e 91,0% com o silencioso. Então, dos 5,4 pp perdidos no silencioso, cerca de 3,1 pp são só os dados dos 3 clientes que faltam, e uns 2,3 pp vêm de a atualização vazia puxar o modelo para trás. Com α = 0,5 a diferença é de 0,3 pp contra 0,1 pp.
- **Cautela estatística:** com α = 0,1 os valores de p ficam em torno de 0,05 a 0,10 (5 sementes e desvio grande do FedAvg), então nem a perda de 5,4 pp do silencioso passa de 5%.

## 4. Fontes fora da RSL: o que foi conferido

- **Lin et al. (2019), "Free-riders in Federated Learning: Attacks and Defenses", arXiv:1911.12560** e **Fraboni et al. (2021), "Free-rider Attacks on Model Aggregation in Federated Learning", AISTATS 2021**: **não estão entre os 67 PDFs da RSL**. Aparecem apenas nas referências de dois artigos da RSL: *Quantifying Bytes* (cita os dois) e *Incentive-Aware Decentralized Data Collaboration* (cita Fraboni, e é nele que a frase sobre "free-riders [11]" se apoia). São, portanto, **fontes secundárias, citadas por artigos da RSL**, e precisam ser marcadas como fora da RSL se forem usadas.
- Não é o mesmo Fraboni: o artigo *Clustered Sampling*, citado em outro trabalho da RSL, é outro paper e não trata de free-riders.
- **Não li esses dois artigos.** O que sei sobre o conteúdo (que descrevem free-riders que enviam pesos aleatórios ou com ruído para se disfarçar) é de memória e **não está verificado**. Antes de citar, é preciso ler os PDFs. Nenhum trecho das duas obras foi usado nas tabelas acima.
- A planilha da RSL não está no repositório, então **não consegui conferir** se ela sustenta essa taxonomia.

## 5. Recomendação de redação

1. Definir **free-rider** como passivo (silencioso e low-effort) e citar [17], [04] e [16] para isso.
2. Apresentar o **ruidoso** como **free-rider disfarçado / ataque ativo**, incluído como checagem de sanidade (os mecanismos reagem a um cliente que realmente prejudica o modelo), e dizer que a justificativa do disfarce vem de fonte externa à RSL.
3. Declarar que o número de épocas do low-effort é uma escolha do trabalho, tratada como análise de sensibilidade.
4. Responder H1 com os tipos passivos, e reportar o ruidoso separadamente.
