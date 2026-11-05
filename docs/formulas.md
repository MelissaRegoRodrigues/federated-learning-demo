# Formulas

Every formula below is what the code computes, with the paper and equation it comes from.
Each table links the math symbol to the name in the code and, when there is one, to the
command-line option of `scripts/run_experiment.py`. Choices that the source papers do not
specify are listed in [Declared choices](#declared-choices).

## Notation

| Symbol | Meaning | Name in the code |
|---|---|---|
| $N$ | number of clients | `num_clients` (`--num-clients`, 10) |
| $t$ | communication round | `server_round` (`--num-rounds`, 50) |
| $\omega(t)$ | global model at the start of round $t$ | `current_global` |
| $\omega_i(t+1)$ | model sent back by client $i$ | `models[i]` |
| $n_i$ | number of training samples of client $i$ | `len(dataset)` |
| $N_s$ | clients that take part in the current round (not barred, $D_i^* > 0$) | `participants`, `active_client_params()` (Stackelberg) |
| $\lvert D\rvert$ | total number of samples, $\sum_i n_i$ | `sum(sizes)` |
| $\bar{D}_i$ | samples client $i$ has available in the game (equal to $n_i$) | `dataset_sizes`, `max_datasize` |
| $\bar{\vartheta}, \bar{\epsilon}, \bar{\Gamma}$ | averages over the clients of $\vartheta_i$, $\epsilon_i$, $\Gamma_i$ | `average_parameters` (`economics.py`) |
| $m$ | client that takes the most time in a round | `slowest_client` (`economics.py`) |

Symbols that the papers use for different things are renamed here, so that each symbol has one
meaning in this file: $\alpha_D$ (Dirichlet concentration) and $\lambda$ (entropy weight, $\alpha$ in
[29]); $\kappa_i$ (data transfer limit, $C_i$ in [29]) and $C_i$ (merged contribution of Shapley);
$N$ and $N_s$ (clients) and $\lvert D\rvert$ (samples), where FedAvg and [29] both write $n$;
$\boldsymbol{\pi}_k$ (Dirichlet shares) and $q_i$ (entropy share, $p_i$ in [29]), so that $\bar{p}$
is only the price.

## Non-IID split: Dirichlet label skew (`src/flapp/partition.py`)

For every class $k$, the share of that class given to each of the $N$ clients is drawn from a
symmetric Dirichlet distribution, and the samples of class $k$ are cut in those shares
(the split of [63] Secure Shapley Value, "Dirichlet distribution of order $n$ with scaling
parameters of 0.5"; also [18]):

$$
\boldsymbol{\pi}_k = (\pi_{k,1}, \dots, \pi_{k,N}) \sim \mathrm{Dir}_N(\alpha_D)
$$

| Symbol | Meaning | Name in the code |
|---|---|---|
| $\alpha_D$ | concentration: smaller means more skewed clients | `alpha` (`--alpha`; 0.5 moderate and 0.1 strong skew are both run) |
| $\boldsymbol{\pi}_k$ | shares of class $k$ across the clients | no variable in this repo: drawn inside Flower's `DirichletPartitioner` |

The split is made by Flower's `DirichletPartitioner` (`flwr-datasets`), applied to the labels, with
`partition_by="label"`, `min_partition_size=10`, `self_balancing=False` and `shuffle=True`. If any client
ends up with fewer than 10 samples, the partitioner draws the whole split again, at most 10 times, and then
raises an error.

Before splitting, 10% of the training set (`VALIDATION_FRACTION`) is held out on the server as
the set used to measure Shapley contributions, so no client trains on it.

## Free-riders (`src/flapp/free_rider.py`, `src/flapp/client_app.py`)

The first `--num-free-riders` clients (default 3) are free-riders. Following [17]: *"free riding
means that edge nodes directly upload the global model to be trained to the central coordinator
without training."* A free-rider returns $\omega(t)$ unchanged, so its update is
$\omega_i(t+1) - \omega(t) = 0$ (`--free-rider-type silent`, the default).

**Low-effort variant** (`--free-rider-type low-effort`). The free-rider trains for `--low-effort-epochs`
local epochs (default 1) instead of `--local-epochs` (default 2) and sends the trained model, declaring
its data size as an honest client. This follows [04] and [16], which simulate *"unreliable users [that]
do not respect the required number of local iterations τ and compute fewer epochs"* (25%, 50% and 75% of
the users). Its update is not zero and points in a direction similar to an honest update, so a test on
the similarity of the updates does not detect it for free.

**Noisy variant** (`--free-rider-type noisy`, `--noise-std`, default 0.05). The free-rider sends the global
model plus Gaussian noise $\mathcal{N}(0, \sigma^2)$ on every weight, without training. Its update is random,
so it harms the average instead of only adding nothing. This type is **not** in the original objectives: it
is a sanity check that a mechanism does react to a free-rider that damages the model. Each run has a single
free-rider type; the types are never mixed in the same run.

**Proportion.** 3 of 10 clients (30%, clients 0, 1 and 2) in the main scenarios, and 5 of 10 (50%, clients
0 to 4) in the scenario `fr50` (`--num-free-riders 5`).

## Datasets and models

MNIST (default) and Fashion-MNIST (`--dataset`), the datasets used in [29], [38], [04] and [16]. Both
have 60,000 training and 10,000 test images of 28x28 pixels and 10 classes, so the same networks are used. The
pixels are normalized with the mean and standard deviation of the training set: 0.1307 and 0.3081 for
MNIST, 0.2860 and 0.3530 for Fashion-MNIST (computed from the training pixels).

**Only MNIST is used in the experiments reported in the results.** Fashion-MNIST is implemented, and only
pilot runs were made with it (`results_piloto/`); they are not part of the comparison.

Two networks are available (`--model`): `cnn` (`SimpleCNN`, 421,642 parameters, used in the first rounds of
experiments, archived in `results_rede_grande/`) and `small` (`SmallCNN`, conv 1 to 8, conv 8 to 8, pooling,
fully connected 392 to 44 to 10, exactly **18,406 parameters**). All the results reported use `small`.

## FedAvg (`src/flapp/strategies/fedavg.py`)

[McMahan et al., 2017]:

$$
\omega(t+1) = \sum_{i=1}^{N} \frac{n_i}{\lvert D\rvert}\, \omega_i(t+1), \qquad \lvert D\rvert = \sum_i n_i
$$

## Shapley: leave-one-out contribution with contribution-aware aggregation (`src/flapp/strategies/shapley.py`)

**Score** ([18] Contributions Estimation in FL, §4.2, and §5.2.2 multiple-round reuse). Each
round, the server builds the model of a coalition $S$ from the received models and measures its
accuracy on the held-out set:

$$
\phi_i = v(\mathcal{N}) - v(\mathcal{N} \setminus \{i\}),
\qquad
v(S) = \text{accuracy of } \sum_{j \in S} \frac{n_j}{\lvert D_S\rvert}\, \omega_j(t+1)
$$

**Penalty and aggregation** ([21] ConDA, Eqs. 10–12 and the ConAvg variant). The updates ranked
in the bottom $V\%$ of the contributions are voted as low-contribution; with a single validator
(the server) a voted update has its contribution reassigned to 0 and every other update gets 1.
The new model is the contribution-weighted average:

$$
C_i = \begin{cases} 0 & \text{if } i \text{ is in the bottom } V\% \\ 1 & \text{otherwise} \end{cases},
\qquad
\omega(t+1) = \sum_i \frac{C_i}{\sum_k C_k}\, \omega_i(t+1)
$$

The reward of client $i$ is its weight $C_i / \sum_k C_k$. Clients with $C_i = 0$ are reported as
detected free-riders.

In the code this means that, every round, exactly $\lfloor V \cdot N \rfloor$ clients are flagged (3 of 10 with
$V = 0.3$; there is no percentile cut), the weights of the others are **uniform** (not proportional to the
marginal contribution and not to the data size), and the decision has no memory between rounds. The mechanism
without exclusion exists only as an ablation ($V = 0$: full data, uniform weights, nobody flagged). An
optional utility based on the loss instead of the accuracy (`--shapley-utility loss`) was tested in pilots and
is not used in the main runs.

| Symbol | Meaning | Name in the code |
|---|---|---|
| $v(S)$ | utility of coalition $S$ | `evaluate_utility`, `coalition_model` |
| $\lvert D_S\rvert$ | samples of the clients in $S$, $\sum_{j\in S} n_j$ | `sizes` |
| $\phi_i$ | leave-one-out contribution | `leave_one_out_values` |
| $V$ | voting rate | `voting_rate` (`--shapley-voting-rate`; 0.3 main, 0.5 in `fr50`, 0.2 sensitivity, 0 ablation) |
| $C_i$ | merged contribution | `merged_contributions` |

## Stackelberg game: EntroCFL (`src/flapp/strategies/stackelberg.py`, `src/flapp/economics.py`)

Everything in this section is from [29] EntroCFL (Tu et al., 2025). Parameter values are from its
Table II.

**Stage 2, clients (followers).** The server announces a payment per data $\bar{p}$. Client $i$
picks how many samples $D_i$ to train on in round $t$ by maximizing (Eq. 14):

$$
U_i(D_i) = \rho^{t-1}\bar{p}\,D_i
- \Bigl(\vartheta_i D_i^{3/2} + \frac{\gamma \epsilon_i D_i \tilde{M}}{\Gamma_i}
+ \frac{W\bigl(2^{\kappa_i/W} - 1\bigr)\tilde{M}\gamma}{\kappa_i h_i}\Bigr)
$$

The three costs are local data cost (Eq. 7), computing cost (Eq. 10) and transmission cost
(Eq. 11). The best response is (Theorem 1, Eqs. 22–23):

$$
\tilde{D}_i = \frac{4}{9\vartheta_i^2}\Bigl(\rho^{t-1}\bar{p} - \frac{\gamma\epsilon_i\tilde{M}}{\Gamma_i}\Bigr)^2,
\qquad
D_i^* = \begin{cases} \min\{\tilde{D}_i, \bar{D}_i\} & \text{if } U_i(\tilde{D}_i) > 0 \\ 0 & \text{otherwise} \end{cases}
$$

**Stage 1, server (leader).** The optimal payment per data $\bar{p}^*$ is the root of
(Theorem 2, Eq. 24):

$$
-\frac{3\chi\bar{\vartheta}}{2\sqrt{N_s t_0}}\sqrt{\tilde{M}\kappa_m}\,(\bar{p}^*)^{-3/2}
+ \frac{8\chi\tilde{M}\bigl(\bar{p}^* - \frac{\gamma\epsilon_m\tilde{M}}{\Gamma_m}\bigr)}{9 t_0 \vartheta_m^2 \Gamma_m}
+ \Bigl(\bar{p}^* - \frac{\gamma\bar{\epsilon}\tilde{M}}{\bar{\Gamma}}\Bigr)\frac{2 N_s t_0 \vartheta_m^2 \Gamma_m}{\tilde{M}\bar{\vartheta}} = 0
$$

where barred quantities are the clients' averages and index $m$ is the client that takes the most
time per round. The left side is increasing in $\bar{p}^*$, so the root is unique and found by
bisection. The server recomputes $\bar{p}^*$ every round with the current number of participants $N_s$.

**Payment** (Algorithm 2): an included client receives $\rho^{t-1}\bar{p}^* D_i^*$.

**Aggregation** (Eq. 18): $\omega(t+1) = \sum_i \frac{D_i^*}{\sum_k D_k^*}\,\omega_i(t+1)$ over the
included clients.

**Exclusion of free-riders** (EntroCFL-CV, §IV-D). With the gradient
$\nabla F_i = \omega(t) - \omega_i(t+1)$ (Eq. 17, up to the learning rate, which cancels out),
the similarity of two clients is (Eqs. 20–21, 39):

$$
q_i = \frac{\nabla F_i^\top \nabla F_i}{\nabla F_i^\top \nabla F_i + \nabla F_j^\top \nabla F_j},
\qquad
\mathrm{Entro}_{i,j} = -\lambda\bigl(q_i \log q_i + q_j \log q_j\bigr)
+ (1-\lambda)\frac{\langle \nabla F_i, \nabla F_j\rangle}{\lVert \nabla F_i\rVert\,\lVert \nabla F_j\rVert}
$$

and the compatibility of client $i$ is the mean over the other members (Eq. 45):
$Cm_i = \frac{1}{N_s - 1}\sum_{j \neq i} \mathrm{Entro}_{i,j}$. A client with $Cm_i < Cm_0$ has its
model ignored, receives no payment and is barred from all later rounds. A free-rider that returns the
global model sends a zero gradient, so its $Cm_i$ is exactly 0. The check is a similarity test between
gradients: there is no reputation score against the median and no product of cosine and norm ratio. The
entropy term depends only on the share of each gradient's squared norm, and enters the score as a weighted sum
with the cosine.

| Symbol | Meaning | Value / name in the code |
|---|---|---|
| $\gamma$ | economic cost per unit of energy | 0.06, `ENERGY_PRICE` |
| $\tilde{M}$ | model size in the cost model | 18406, `--stackelberg-model-size` |
| $\Gamma_i$ | computing power | $[3000, 5000]$, `computing_power` |
| $\kappa_i$ | data transfer limit | $[30, 60]$, `transfer_limit` |
| $\chi$ | economic impact of the loss | $10^{11}$, `LOSS_WEIGHT` |
| $\epsilon_i$ | energy efficiency coefficient | $[0.3, 0.5]$, `energy_efficiency` |
| $W$ | bandwidth | 150000, `BANDWIDTH` |
| $h_i$ | channel power gain | 50, `CHANNEL_GAIN` |
| $\vartheta_i$ | local data preprocessing coefficient | $[0.05, 0.07]$, `preprocessing` |
| $\rho$ | discount factor | 0.998, `DISCOUNT` |
| $\lambda$ | weight of the entropy term | 0.6, `ENTROPY_WEIGHT` |
| $t_0$ | total time budget | calibrated, `total_time` |
| $Cm_0$ | compatibility threshold | `--stackelberg-cm0`, 0.01 |

## Evaluation metrics

- **Accuracy, precision, recall, F1** of the global model on the MNIST test set, macro-averaged over the 10
  classes, at the last round.
- **Detection precision and recall** compare the flagged clients with the true free-riders. They are not the
  precision and recall of the model. For Shapley they are equal by construction, because the number of
  flagged clients is fixed.
- **Stability**: standard deviation of the accuracy over the last 10 rounds of a run, and the standard
  deviation of the final accuracy between seeds.
- **Rounds to 95%**, **share of the reward that goes to free-riders** (for FedAvg, the share of the
  aggregation weight, which is proportional to the data size), **server time per round**, and, for
  Stackelberg, the **fraction of the data used** and the **total payment**.
- **Controls and ablations**: `shapley_v0` (no exclusion), `control` (FedAvg with the data size of the
  game, `--price-from`), `oracle` (FedAvg ignoring the true free-riders, `--exclude-clients`), `fr0` (no
  free-riders, silent scenario) and `full_data` (Stackelberg with all the data, `--full-data`).
- **Reward distance** ([55] Volatility of Shapley-based contribution metrics; [47]): squared
  Euclidean distance between the normalized cumulative rewards $R_i / \sum_k R_k$ and a reference
  vector $g$, where $g_i = n_i / \sum_{j \text{ honest}} n_j$ for honest clients ([55] uses the data
  size as ground truth) and $g_i = 0$ for free-riders ([17]: free-riders are not accepted in the
  incentive allocation):

$$
d = \sum_i \Bigl(\frac{R_i}{\sum_k R_k} - g_i\Bigr)^2
$$

[55] warns that a small distance does not guarantee that every client is paid fairly.

## Declared choices

What the source papers leave open, and what the code does:

| Item | Choice | Why |
|---|---|---|
| $Cm_0$ ([29] gives no value) | 0.01 | `scripts/analyze_cm0.py` shows that every value between the largest free-rider $Cm$ and the smallest honest $Cm$ excludes exactly the same clients. $Cm_0 = 0$ would keep free-riders, because [29] keeps clients with $Cm_i \geq Cm_0$ and a free-rider has $Cm_i = 0$. |
| $t_0$ ([29] gives no value for 10 clients) | calibrated | The smallest $t_0$ for which [29]'s own model (Eq. 2, Algorithm 2) runs exactly the number of rounds of the experiment. |
| $\tilde{M}$ | 18406 (Table II of [29]) | With the real size of the first CNN (421,642 parameters) the transmission cost makes every client's utility negative and nobody participates. The `small` network used in the reported results has exactly 18,406 parameters, so there the value is the real size. |
| $t_0$ is calibrated with all 10 clients active in every round | calibration before any exclusion | The server does not know the free-riders in advance. After the first round the free-riders are barred and the price is recomputed with the remaining clients, so the real run no longer follows the calibrated model exactly. |
| Price of round 1 | computed with all 10 clients | The server does not know who the free-riders are before the first compatibility test. |
| Number of local epochs of a low-effort free-rider | 1 (instead of 2) | [04] and [16] do not say how many epochs fewer; 1 is the smallest integer reduction that still trains. |
| Normalization of Fashion-MNIST | mean 0.2860, std 0.3530 | Computed from the training pixels, like the constants of MNIST. |
| Noise of the noisy free-rider | $\sigma = 0.05$ on every weight | No source gives a value; it is a sanity check, small enough not to be trivial to see from the norm alone. |
| $Cm_0$ in every scenario | 0.01, never tuned per scenario | The server does not know the type of attack. Adjusting the threshold after seeing the results would be tuning on the test scenario. With this value the check catches silent free-riders only (see limitation 13). |
| Voting rate in `fr50` | $V = 0.5$ | Same rule as the main runs: the value slightly above the free-rider rate (50% here). |
| Number of seeds | 5 (42 to 46), paired between mechanisms | Each run takes about 5 to 7 minutes on the reference machine and uses about 9 to 10 GB of memory, so runs are made one at a time. |
| Number of clusters | 1 | There is a single data distribution (the clustering layer of [29] separates rotated images). |
| EntroCFL-SV (z-score credit) | not used | [29] gives no window $R_v$ and does not say how to turn a parameter vector into a scalar z-score. |
| Client parameters in Table II ranges | uniform draw per client, fixed by the seed | [29] gives ranges only. |
| $\log$ in Entro | natural logarithm | [29] does not give the base. |
| $0 \log 0$, cosine with a zero vector, Entro of two zero vectors | 0 | Standard conventions; the formulas are undefined there. |
| Client $m$ in Eq. 24 | found by iterating "price → slowest client → price" | [29] defines $m$ as the client that takes the most time, which depends on the price. |
| Time of a round | computing time (Eq. 8) + transmission time (Eq. 12) of the slowest client | [29] says the round time is the longest training time. |
| $D_i^*$ | rounded down to an integer | samples are counted in units. |
| Clients with $D_i^* = 0$ | do not train and are left out of $Cm$ | [29] Eq. 25: such a client has no cost and no reward. |
| Coalition model in [18] | weighted by data size | [18] reuses the clients' updates without fixing the weights; FedAvg weights are the usual aggregation. |
| Validators in [21] | one (the server) | [21] uses several validation clients. |
| Voting rate $V$ of Shapley | 0.3 in the main runs, 0.2 as sensitivity | [21] finds the best $V$ slightly above the malicious rate (10% in its experiments, $V = 20\%$); here the free-rider rate is 30%. |
| Baselines of the gain | Stackelberg: FedAvg control (same data size, weights proportional to the declared size, no exclusion). Shapley: the same mechanism with $V = 0$ (full data, uniform weights, no exclusion) | Each mechanism is compared with its own no-exclusion baseline with the same weight rule, so that the gain measures the exclusion only. The ablation of a component follows the design of [21] (ConAvg and KDAvg against FedAvg); the baselines themselves are not part of any source paper. |
| Minimum 10 samples per client, no self-balancing | the defaults of Flower's `DirichletPartitioner`; self-balancing is not part of the formulation of [63] | no effect on seeds 42–46 (smallest client: 917 samples). |

## Limitations of the comparison

These points come from the design of the mechanisms and must be stated together with the
results.

**1. The voting rate of Shapley depends on the proportion of free-riders.** [21] sets $V = 20\%$
with 10% of attackers and observes that the best result is obtained when $V$ *slightly exceeds* the
malicious rate. Here 30% of the clients are free-riders, so the main runs use $V = 0.3$, which flags
$\lfloor 0.3 \cdot 10 \rfloor = 3$ clients per round. This works because the proportion of free-riders
is known; the same rule flags 3 clients when there are no free-riders, so in the scenario without
free-riders it excludes 3 honest clients every round. The runs with $V = 0.2$ (the value of the
experiments of [21]) can be added as a sensitivity analysis: they flag only $\lfloor 0.2 \cdot 10 \rfloor = 2$
clients, so the detection recall is at most $2/3 \approx 0.67$ by construction and one free-rider
always keeps a positive weight.

**2. Different assumptions about the server.** The Shapley mechanism needs an IID validation set
held out on the server (10% of the training data, 6,000 samples). The Stackelberg mechanism needs no
data on the server, but needs the cost parameters of the clients (Table II of [29]), which are
assumed to be known to the server through their averages.

**3. Free-riders do not react to the price.** A free-rider always returns the global model,
whatever the payment. Only honest clients play the Stackelberg game. In FedAvg and Shapley a
free-rider declares the size of its whole local dataset ($n_i$); in the Stackelberg game it declares
the data size $D_i^*$ that the game would give it, but it is excluded by the compatibility test, so it
is never paid.

**4. Clients train on a fraction of their data in the Stackelberg game.** The optimal data size
$D_i^*$ (Eqs. 22–23) is chosen by each client from its costs, so it is a fraction of the local data and
not the whole dataset. This fraction and the price are not constants: they depend on the seed (the cost
parameters and the partition sizes), on $t_0$ (which is calibrated for the number of rounds of the
experiment) and on how many clients are still active. They are measured in every run and reported by
`scripts/summarize_results.py`; no fixed value should be quoted from a single run or from a run with a
different number of rounds. FedAvg and Shapley train on 100% of the local data. To separate the effect of
the data size from the effect of the mechanism, the control runs (`--price-from`) train FedAvg with the data
size that every client would choose in the Stackelberg game at the price of that round, without the
compatibility check. The accuracy of the Stackelberg runs must be compared with the control runs, not only
with the full-data runs.

**5. The cost model uses the model size of [29].** $\tilde{M} = 18406$ (Table II of [29], the model for
EMNIST). With the first CNN (421,642 parameters) this value was only nominal; with the `small` network used
in the reported results it equals the real size. The model size only enters the cost model of
the game (computing, transmission and round time); the CNN is trained at its real size.

**6. The reward distance is not neutral between the mechanisms.** The reference vector $g$ is
proportional to the data size $n_i$ of the honest clients ([55] uses the data size as ground truth).
Shapley pays equal weights $C_i / \sum_k C_k$ to the clients it does not exclude, and Stackelberg pays
in proportion to $D_i^*$, which depends on the client's costs and not on $n_i$. Neither mechanism
targets $g$, and Shapley can never reach $d \approx 0$ because its weights are uniform. The distance is
therefore reported together with the share of the reward that goes to free-riders, which does not depend
on the reference, and with the distance to a uniform reference among the honest clients. [55] also warns
that a small distance does not guarantee that every client is paid fairly.

**7. Server cost.** `aggregate_seconds` covers only `aggregate_fit`. The result files also record
`server_seconds`, which for Stackelberg adds the computation of the price (`configure_fit`, every
round), and `calibration_seconds`, the one-time calibration of $t_0$ in the constructor.
`scripts/summarize_results.py` reports the server time per round as `server_seconds` plus the
calibration divided by the number of rounds. For Shapley the time includes the evaluations of the
coalition models on the validation set.

**8. The aggregation weights differ between mechanisms.** FedAvg weights the clients by $n_i$, Shapley
gives equal weight to the clients it does not exclude, and Stackelberg weights the included clients by
$D_i^*$. These are the rules of McMahan et al., [21] and [29]. The exclusion also differs in time: in
Stackelberg a client that fails the compatibility check is barred from all later rounds ([29]), while
Shapley decides again in every round, without memory ([21]).

**9. Adding a mechanism is not only a subclass.** A new mechanism needs a `Strategy` subclass, an
entry in `STRATEGIES`, a branch in `make_strategy` (`src/flapp/server_app.py`) and an entry in the
`choices` of `--strategy` (`scripts/run_experiment.py`).

**10. The Shapley value is approximated by leave-one-out.** The conclusions about "Shapley" in this
work are conclusions about the leave-one-out approximation (with the voting of [21]), not about the
exact value or about GTG-Shapley. [18] reports that leave-one-out is *"the least effective in
quantity-skewed FL, only slightly outperforming Random, but has moderate effectiveness in label-skewed
FL"*; the partition used here has both kinds of skew. [21] notes that deletion-based approaches *"overlook
coordination with other clients and are prone to inaccuracies"*, and [47] that they compress the
contribution of small datasets. The reason for the choice is cost: evaluating one coalition on the 6,000
validation samples takes about 1.08 s, so leave-one-out ($N + 1 = 11$ evaluations) takes about 10 minutes
of evaluations per run, while the exact value ($2^{10} = 1024$ coalitions per round) would take about
15 hours per run. Estimators such as GTG-Shapley [40] and Monte Carlo sampling of permutations were not run and are left as
future work. The choice
pushes the comparison in two opposite directions: the server cost of Shapley is the lowest one among its
estimators, so the cost gap to Stackelberg is a lower bound, while the detection quality of Shapley may be
underestimated.

**11. The detection in the Stackelberg mechanism does not come from the game.** In [29] the game
decides the payment per data and the data size of each client; the exclusion of clients comes from the
compatibility check (EntroCFL-CV, a similarity test between gradients). The control run without
exclusion (`results/control`, the game without the check, in the spirit of the "no clustering" baseline of
[29]) separates the two layers.

**12. Silent free-riders are easy for a similarity test.** A free-rider that returns the global model
has a zero update, which any test based on the similarity or the norm of the updates detects without
evaluating a model, while the marginal contribution of a client to the accuracy of the average of ten
similar models is a weak signal. The difference in detection between the two mechanisms therefore reflects
the type of attack and the type of detector, and not only the mechanisms. The low-effort free-riders
(fewer local epochs, as simulated in [04] and [16]) are evaluated in a separate scenario for this reason.
Both types are still simple: they do not adapt to the mechanism, and only these two behaviors were tested.

**13. The compatibility threshold of Stackelberg only catches silent free-riders.** $Cm_0 = 0.01$ lies between
the $Cm$ of silent free-riders (0) and the smallest $Cm$ of honest clients, and the low-effort and noisy
free-riders are not below it, so in those scenarios Stackelberg flags nobody (recall 0). The threshold was
not tuned per scenario (see declared choices). A sweep of $Cm_0$ is left as a sensitivity analysis: the
detection can be recomputed offline from the logged $Cm$, but the accuracy needs new runs.

**14. The accuracy lost by Stackelberg comes from the price, not from the detection.** Training the clients
that the game accepts with all their data (`--full-data`, ablation) brings Stackelberg to the accuracy of
FedAvg with a perfect detector in the silent scenario (about 93.3% with $\alpha_D = 0.1$ and 98.0% with
$\alpha_D = 0.5$), while the game as designed reaches 83.6% and 92.8%. The ablation is not the mechanism of
[29]: the client no longer chooses its best response, and the payment is computed on data it did not choose,
so the payments are not comparable with the original runs. It only isolates the effect of the data size.

**15. Few seeds.** With 5 seeds the Wilcoxon test cannot go below $p = 0.0625$, so the comparisons use the
paired $t$ test and the number of seeds in which one mechanism beats the other. No correction for multiple
comparisons is made.

## References (numbers from the systematic review)

- [17] Contract-Theory-Based Incentive Mechanism for Federated Learning in Health CrowdSensing.
- [18] Contributions Estimation in Federated Learning: A Comprehensive Experimental Evaluation.
- [21] Defending Federated Recommender Systems against Untargeted Attacks: A Contribution-Aware Robust Aggregation Scheme (ConDA).
- [29] Tu et al. EntroCFL: Entropy-Based Clustered Federated Learning With Incentive Mechanism. IEEE IoT Journal, 2025.
- [47] Intrinsic Performance Influence-based Participant Contribution Estimation for Horizontal Federated Learning.
- [55] On the Volatility of Shapley-Based Contribution Metrics in Federated Learning.
- [63] Secure Shapley Value for Cross-Silo Federated Learning.
- McMahan, B. et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS.
