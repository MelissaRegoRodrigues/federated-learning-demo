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
| $\omega(t)$ | global model at the start of round $t$ | `global_weights`, `current_global` |
| $\omega_i(t+1)$ | model sent back by client $i$ | `models[i]` |
| $n_i$ | number of training samples of client $i$ | `len(dataset)` |

## Non-IID split: Dirichlet label skew (`src/flapp/partition.py`)

For every class $k$, the share of that class given to each of the $N$ clients is drawn from a
symmetric Dirichlet distribution, and the samples of class $k$ are cut in those shares
(the split of [63] Secure Shapley Value, "Dirichlet distribution of order $n$ with scaling
parameters of 0.5"; also [18]):

$$
\mathbf{p}_k = (p_{k,1}, \dots, p_{k,N}) \sim \mathrm{Dir}_N(\alpha)
$$

| Symbol | Meaning | Name in the code |
|---|---|---|
| $\alpha$ | concentration: smaller means more skewed clients | `alpha` (`--alpha`, 0.5) |
| $\mathbf{p}_k$ | shares of class $k$ across the clients | `proportions_per_class[k]` |

The split is implemented in this repository (`partition_dirichlet`); it is not Flower's
`DirichletPartitioner`. If any client ends up with fewer than 10 samples (`MIN_PARTITION_SIZE`),
the whole split is drawn again, at most 1000 times, and then an error is raised.

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

## Datasets

MNIST (default) and Fashion-MNIST (`--dataset`), the datasets used in [29], [38], [04] and [16]. Both
have 60,000 training and 10,000 test images of 28x28 pixels and 10 classes, so the same CNN is used. The
pixels are normalized with the mean and standard deviation of the training set: 0.1307 and 0.3081 for
MNIST, 0.2860 and 0.3530 for Fashion-MNIST (computed from the training pixels).

## FedAvg (`src/flapp/strategies/fedavg.py`)

[McMahan et al., 2017]:

$$
\omega(t+1) = \sum_{i=1}^{N} \frac{n_i}{n}\, \omega_i(t+1), \qquad n = \sum_i n_i
$$

## Shapley: leave-one-out contribution with contribution-aware aggregation (`src/flapp/strategies/shapley.py`)

**Score** ([18] Contributions Estimation in FL, §4.2, and §5.2.2 multiple-round reuse). Each
round, the server builds the model of a coalition $S$ from the received models and measures its
accuracy on the held-out set:

$$
\phi_i = v(\mathcal{N}) - v(\mathcal{N} \setminus \{i\}),
\qquad
v(S) = \text{accuracy of } \sum_{j \in S} \frac{n_j}{n_S}\, \omega_j(t+1)
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

| Symbol | Meaning | Name in the code |
|---|---|---|
| $v(S)$ | utility of coalition $S$ | `evaluate_utility`, `coalition_model` |
| $\phi_i$ | leave-one-out contribution | `leave_one_out_values` |
| $V$ | voting rate | `voting_rate` (`--shapley-voting-rate`, 0.3) |
| $C_i$ | merged contribution | `merged_contributions` |

## Stackelberg game: EntroCFL (`src/flapp/strategies/stackelberg.py`, `src/flapp/economics.py`)

Everything in this section is from [29] EntroCFL (Tu et al., 2025). Parameter values are from its
Table II.

**Stage 2, clients (followers).** The server announces a payment per data $\bar{p}$. Client $i$
picks how many samples $D_i$ to train on in round $t$ by maximizing (Eq. 14):

$$
U_i(D_i) = \rho^{t-1}\bar{p}\,D_i
- \Bigl(\vartheta_i D_i^{3/2} + \frac{\gamma \epsilon_i D_i \tilde{M}}{\Gamma_i}
+ \frac{W\bigl(2^{C_i/W} - 1\bigr)\tilde{M}\gamma}{C_i h_i}\Bigr)
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
-\frac{3\chi\bar{\vartheta}}{2\sqrt{n t_0}}\sqrt{\tilde{M}C_m}\,(\bar{p}^*)^{-3/2}
+ \frac{8\chi\tilde{M}\bigl(\bar{p}^* - \frac{\gamma\epsilon_m\tilde{M}}{\Gamma_m}\bigr)}{9 t_0 \vartheta_m^2 \Gamma_m}
+ \Bigl(\bar{p}^* - \frac{\gamma\bar{\epsilon}\tilde{M}}{\bar{\Gamma}}\Bigr)\frac{2 n t_0 \vartheta_m^2 \Gamma_m}{\tilde{M}\bar{\vartheta}} = 0
$$

where barred quantities are the clients' averages and index $m$ is the client that takes the most
time per round. The left side is increasing in $\bar{p}^*$, so the root is unique and found by
bisection. The server recomputes $\bar{p}^*$ every round with the current number of clients $n$.

**Payment** (Algorithm 2): an included client receives $\rho^{t-1}\bar{p}^* D_i^*$.

**Aggregation** (Eq. 18): $\omega(t+1) = \sum_i \frac{D_i^*}{\sum_k D_k^*}\,\omega_i(t+1)$ over the
included clients.

**Exclusion of free-riders** (EntroCFL-CV, §IV-D). With the gradient
$\nabla F_i = \omega(t) - \omega_i(t+1)$ (Eq. 17, up to the learning rate, which cancels out),
the similarity of two clients is (Eqs. 20–21, 39):

$$
p_i = \frac{\nabla F_i^\top \nabla F_i}{\nabla F_i^\top \nabla F_i + \nabla F_j^\top \nabla F_j},
\qquad
\mathrm{Entro}_{i,j} = -\alpha\bigl(p_i \log p_i + p_j \log p_j\bigr)
+ (1-\alpha)\frac{\langle \nabla F_i, \nabla F_j\rangle}{\lVert \nabla F_i\rVert\,\lVert \nabla F_j\rVert}
$$

and the compatibility of client $i$ is the mean over the other members (Eq. 45):
$Cm_i = \frac{1}{N_s - 1}\sum_{j \neq i} \mathrm{Entro}_{i,j}$. A client with $Cm_i < Cm_0$ has its
model ignored, receives no payment and is barred from all later rounds. A free-rider sends a zero
gradient, so its $Cm_i$ is exactly 0.

| Symbol | Meaning | Value / name in the code |
|---|---|---|
| $\gamma$ | economic cost per unit of energy | 0.06, `ENERGY_PRICE` |
| $\tilde{M}$ | model size in the cost model | 18406, `--stackelberg-model-size` |
| $\Gamma_i$ | computing power | $[3000, 5000]$, `computing_power` |
| $C_i$ | data transfer limit | $[30, 60]$, `transfer_limit` |
| $\chi$ | economic impact of the loss | $10^{11}$, `LOSS_WEIGHT` |
| $\epsilon_i$ | energy efficiency coefficient | $[0.3, 0.5]$, `energy_efficiency` |
| $W$ | bandwidth | 150000, `BANDWIDTH` |
| $h_i$ | channel power gain | 50, `CHANNEL_GAIN` |
| $\vartheta_i$ | local data preprocessing coefficient | $[0.05, 0.07]$, `preprocessing` |
| $\rho$ | discount factor | 0.998, `DISCOUNT` |
| $\alpha$ | weight of the entropy term | 0.6, `ENTROPY_WEIGHT` |
| $t_0$ | total time budget | calibrated, `total_time` |
| $Cm_0$ | compatibility threshold | `--stackelberg-cm0`, 0.01 |

## Evaluation metrics

- **Accuracy, precision, recall, F1** on the MNIST test set, macro-averaged over the 10 classes.
- **Detection precision and recall** compare the flagged clients with the true free-riders.
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
| $\tilde{M}$ | 18406 (Table II of [29]) | With the real size of the CNN (421,642 parameters) the transmission cost makes every client's utility negative and nobody participates. |
| $t_0$ is calibrated with all 10 clients active in every round | calibration before any exclusion | The server does not know the free-riders in advance. After the first round the free-riders are barred and the price is recomputed with the remaining clients, so the real run no longer follows the calibrated model exactly. |
| Price of round 1 | computed with all 10 clients | The server does not know who the free-riders are before the first compatibility test. |
| Number of local epochs of a low-effort free-rider | 1 (instead of 2) | [04] and [16] do not say how many epochs fewer; 1 is the smallest integer reduction that still trains. |
| Normalization of Fashion-MNIST | mean 0.2860, std 0.3530 | Computed from the training pixels, like the constants of MNIST. |
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
| Minimum 10 samples per client, redrawing the whole split up to 1000 times | own implementation; 10 is also the default `min_partition_size` of Flower's `DirichletPartitioner` | no effect on seeds 42–46 (smallest client: 917 samples). |

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

**5. The cost model uses a nominal model size.** $\tilde{M} = 18406$ (Table II of [29], the model for
EMNIST), while the CNN trained here has 421,642 parameters. The model size only enters the cost model of
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
$D_i^*$. These are the rules of McMahan et al., [21] and [29].

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
15 hours per run. Estimators such as GTG-Shapley [40] were not run and are left as future work. The choice
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

## References (numbers from the systematic review)

- [17] Contract-Theory-Based Incentive Mechanism for Federated Learning in Health CrowdSensing.
- [18] Contributions Estimation in Federated Learning: A Comprehensive Experimental Evaluation.
- [21] Defending Federated Recommender Systems against Untargeted Attacks: A Contribution-Aware Robust Aggregation Scheme (ConDA).
- [29] Tu et al. EntroCFL: Entropy-Based Clustered Federated Learning With Incentive Mechanism. IEEE IoT Journal, 2025.
- [47] Intrinsic Performance Influence-based Participant Contribution Estimation for Horizontal Federated Learning.
- [55] On the Volatility of Shapley-Based Contribution Metrics in Federated Learning.
- [63] Secure Shapley Value for Cross-Silo Federated Learning.
- McMahan, B. et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS.
