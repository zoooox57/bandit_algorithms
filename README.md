# Bandit Algorithms Mini Benchmark

This is a small benchmark for multi-armed bandit algorithms. The goal is to compare how different action-selection strategies balance exploration and exploitation.

## Project Structure

```text
.
├── experiments/
│   └── 01_simple_baseline/      # Early random and epsilon-greedy scripts
├── results/
│   ├── 01_simple_baseline/      # Plots from the first simple experiment
│   ├── 02_stationary_benchmark/ # Fixed-probability benchmark results
│   ├── 03_dynamic_exp3/         # Dynamic probability benchmark results
│   ├── 04_exp3_gamma_sweep/     # EXP3 gamma sensitivity results
│   └── 05_contextual_bandit/    # Contextual bandit benchmark results
├── src/
│   ├── agents.py                # Bandit algorithms
│   ├── contextual_agents.py     # Contextual bandit algorithms
│   ├── contextual_envs.py       # Contextual bandit environment
│   ├── contextual_experiment.py
│   ├── envs.py                  # Stationary and dynamic environments
│   ├── experiment.py            # Stationary benchmark runner
│   ├── exp3_dynamic_experiment.py
│   └── exp3_gamma_sweep.py
└── README.md
```

The project has five experiment groups:

| Experiment | Purpose | Code | Results |
| --- | --- | --- | --- |
| Simple baseline | Introduce random and epsilon-greedy behavior | `experiments/01_simple_baseline/` | `results/01_simple_baseline/` |
| Stationary benchmark | Compare algorithms when arm probabilities are fixed | `src/experiment.py` | `results/02_stationary_benchmark/` |
| Dynamic EXP3 benchmark | Test algorithms when the best arm changes over time | `src/exp3_dynamic_experiment.py` | `results/03_dynamic_exp3/` |
| EXP3 gamma sweep | Compare different EXP3 exploration rates and update rules | `src/exp3_gamma_sweep.py` | `results/04_exp3_gamma_sweep/` |
| Contextual bandit | Compare contextual epsilon-greedy and LinUCB | `src/contextual_experiment.py` | `results/05_contextual_bandit/` |

## Environment

The experiment uses a Bernoulli bandit environment. Each arm has a fixed success probability that is hidden from the agent. The default setting is:

```python
probs = [0.1, 0.3, 0.5, 0.6, 0.8]
```

At each round, the agent selects one arm and receives a binary reward:

```text
reward = 1, with probability probs[arm]
reward = 0, otherwise
```

The optimal arm is the last arm, with true expected reward `0.8`. During learning, the algorithms only observe the rewards from their own selected arms. The true probabilities are used only for evaluation, such as computing pseudo-regret.

## Algorithms

The current benchmark in `src/experiment.py` compares five algorithms:

| Algorithm | Setting | Description |
| --- | --- | --- |
| Epsilon-Greedy | `epsilon=0.1` | Chooses a random arm with probability 10%; otherwise chooses the arm with the highest empirical mean reward |
| Decay Epsilon-Greedy | `epsilon_t = min(1, K / t)` | Explores more in early rounds and gradually reduces exploration over time |
| UCB | `c=2.0` | Uses an upper confidence bound to favor arms with high uncertainty |
| Thompson Sampling | Beta-Bernoulli posterior | Samples from each arm's posterior distribution and chooses the arm with the highest sample |
| EXP3 | `gamma=0.1` | Maintains exponential weights over arms and mixes them with uniform exploration |

## Metrics

The experiment tracks three main metrics:

| Metric | Meaning |
| --- | --- |
| Cumulative pseudo-regret | The cumulative gap between the expected reward of the optimal arm and the expected reward of the chosen arm; lower is better |
| Average reward | The average observed reward up to the current round; higher is better |
| Optimal arm selection rate | The fraction of rounds in which the optimal arm was selected; higher is better |

Pseudo-regret is computed as:

```text
instantaneous_regret = best_mean - chosen_arm_mean
cumulative_regret = sum(instantaneous_regret)
```

## Smoke Benchmark Results

The `results/02_stationary_benchmark/` directory contains a short smoke benchmark with horizon 200 and 3 random seeds. The final summary comes from `results/02_stationary_benchmark/summary.csv`:

| Rank | Algorithm | Cumulative pseudo-regret ↓ | Average reward ↑ | Optimal arm rate ↑ |
| --- | --- | ---: | ---: | ---: |
| 1 | Thompson Sampling | 10.77 ± 4.13 | 0.723 | 0.837 |
| 2 | Epsilon-Greedy (`epsilon=0.1`) | 20.03 ± 8.61 | 0.675 | 0.702 |
| 3 | Decay Epsilon-Greedy (`c=1`) | 20.40 ± 19.33 | 0.667 | 0.608 |
| 4 | UCB (`c=2`) | 41.83 ± 2.51 | 0.558 | 0.412 |
| 5 | EXP3 (`gamma=0.1`) | 77.37 ± 2.46 | 0.150 | 0.173 |

The averaged learning curves are saved as:

- `results/02_stationary_benchmark/cumulative_regret.png`
- `results/02_stationary_benchmark/average_reward.png`
- `results/02_stationary_benchmark/optimal_arm_rate.png`

## Dynamic EXP3 Experiment

The stationary benchmark above is not the best setting for explaining EXP3, because the reward probabilities are fixed. To test a changing environment, `src/exp3_dynamic_experiment.py` uses a piecewise-stationary Bernoulli bandit. The active probability vector changes every 200 rounds:

```python
prob_schedule = [
    [0.8, 0.2, 0.2, 0.2, 0.2],
    [0.2, 0.8, 0.2, 0.2, 0.2],
    [0.2, 0.2, 0.8, 0.2, 0.2],
    [0.2, 0.2, 0.2, 0.8, 0.2],
    [0.2, 0.2, 0.2, 0.2, 0.8],
]
```

This means the best arm changes over time. The dynamic experiment evaluates regret against the current best arm at each round:

```text
instantaneous_dynamic_regret = current_best_mean - chosen_arm_current_mean
```

The default dynamic run uses horizon 1000, 20 random seeds, and segment length 200. The results are saved in `results/03_dynamic_exp3/`:

| Rank | Algorithm | Cumulative dynamic pseudo-regret ↓ | Average reward ↑ | Current best arm rate ↑ |
| --- | --- | ---: | ---: | ---: |
| 1 | UCB (`c=2`) | 132.57 ± 16.05 | 0.667 | 0.779 |
| 2 | Thompson Sampling | 352.29 ± 41.42 | 0.447 | 0.413 |
| 3 | Restarted EXP3 (`gamma=0.1`, restart interval 200) | 365.34 ± 12.52 | 0.486 | 0.391 |
| 4 | Epsilon-Greedy (`epsilon=0.1`) | 423.00 ± 39.33 | 0.378 | 0.295 |
| 5 | EXP3.S-style (`gamma=0.1`, `alpha=0.01`) | 440.64 ± 12.36 | 0.383 | 0.266 |
| 6 | Discounted EXP3 (`gamma=0.1`, `decay=0.99`) | 441.39 ± 11.75 | 0.379 | 0.264 |
| 7 | Sliding-window EXP3 (`gamma=0.1`, window 200) | 463.44 ± 9.90 | 0.335 | 0.228 |
| 8 | Decay Epsilon-Greedy (`c=1`) | 482.37 ± 37.76 | 0.316 | 0.196 |
| 9 | Loss-based EXP3 (`gamma=0.1`) | 516.18 ± 4.34 | 0.272 | 0.140 |
| 10 | EXP3 (`gamma=0.1`) | 521.67 ± 5.46 | 0.271 | 0.131 |

In this switching-probability environment, vanilla EXP3 still does not perform best. This is an important distinction: standard EXP3 is designed for adversarial bandits, but it usually competes against the best fixed arm in hindsight. Among the EXP3 variants, Restarted EXP3 works best because the restart interval matches the environment's 200-round switching period.

## EXP3 Gamma Sweep

EXP3 uses `gamma` to control how much probability is reserved for uniform exploration. A larger `gamma` explores more, while a smaller `gamma` relies more heavily on the learned exponential weights.

The gamma sweep uses the same dynamic environment as above and compares two EXP3 update rules:

- reward-based EXP3: increases an arm's weight after high reward
- loss-based EXP3: decreases an arm's weight after low reward

| Rank | EXP3 setting | Cumulative dynamic pseudo-regret ↓ | Average reward ↑ | Current best arm rate ↑ |
| --- | --- | ---: | ---: | ---: |
| 1 | `gamma=0.5` | 484.08 ± 8.72 | 0.314 | 0.193 |
| 2 | loss-based `gamma=0.5` | 485.10 ± 7.54 | 0.312 | 0.192 |
| 3 | `gamma=0.01` | 496.20 ± 7.47 | 0.295 | 0.173 |
| 4 | loss-based `gamma=0.01` | 496.20 ± 6.77 | 0.296 | 0.173 |
| 5 | `gamma=0.2` | 500.94 ± 5.48 | 0.292 | 0.165 |
| 6 | loss-based `gamma=0.2` | 501.03 ± 4.37 | 0.291 | 0.165 |
| 7 | loss-based `gamma=0.05` | 515.40 ± 7.82 | 0.274 | 0.141 |
| 8 | loss-based `gamma=0.1` | 516.18 ± 4.34 | 0.272 | 0.140 |
| 9 | `gamma=0.05` | 518.70 ± 9.29 | 0.275 | 0.136 |
| 10 | `gamma=0.1` | 521.67 ± 5.46 | 0.271 | 0.131 |

In this setting, increasing `gamma` to `0.5` improves EXP3 because the best arm changes every 200 rounds. More uniform exploration gives EXP3 more chances to discover the new best arm after each switch. The loss-based update also helps for some smaller gamma values, because low rewards can push a selected arm's weight down instead of merely stopping it from increasing. However, even the best gamma in this sweep still performs worse than UCB in the dynamic benchmark, so changing the update rule helps but does not fully solve the lack of forgetting in vanilla EXP3.

## Contextual Bandit Experiment

The contextual experiment adds side information before each decision. Each round samples one of two user contexts:

```python
sports_user = [1.0, 0.0]
music_user = [0.0, 1.0]
```

There are three arms: a sports-like arm, a music-like arm, and a general arm. The reward probability is:

```text
P(reward = 1 | context, arm) = sigmoid(context @ theta_arm)
```

This makes the best arm depend on the current context. The benchmark compares:

- Contextual Epsilon-Greedy: mostly chooses the arm with the highest predicted reward, with random exploration
- LinUCB: chooses the arm with the highest predicted reward plus an uncertainty bonus

The default contextual run uses horizon 1000 and 20 random seeds. The results are saved in `results/05_contextual_bandit/`:

| Rank | Algorithm | Cumulative contextual pseudo-regret ↓ | Average reward ↑ | Contextual optimal arm rate ↑ |
| --- | --- | ---: | ---: | ---: |
| 1 | LinUCB (`alpha=1`) | 12.66 ± 29.11 | 0.868 | 0.957 |
| 2 | Contextual Epsilon-Greedy (`epsilon=0.1`) | 50.54 ± 24.07 | 0.831 | 0.867 |

LinUCB performs better in this simple contextual environment because it explores arms with high uncertainty in the current context. Contextual epsilon-greedy also learns the context-dependent best arms, but its exploration is random rather than uncertainty-directed.

## Observations

In this 200-round smoke benchmark, Thompson Sampling performs best. It has the lowest cumulative pseudo-regret, the highest average reward, and an optimal arm selection rate of about 83.7%. This suggests that the Beta-Bernoulli posterior helps the agent quickly concentrate its pulls on the true best arm while still maintaining useful uncertainty-driven exploration.

Epsilon-Greedy with `epsilon=0.1` ranks second. It can exploit the empirically best arm, but it always keeps a 10% random exploration rate. As a result, even after it has identified a good arm, it still pulls suboptimal arms occasionally, causing regret to continue increasing slowly.

Decay Epsilon-Greedy has a similar mean regret to fixed Epsilon-Greedy, but its standard deviation is much larger. It explores heavily at the beginning and then reduces exploration quickly. This can work well when early reward observations are helpful, but it can also lock onto a suboptimal arm if early noise is misleading.

UCB is weaker than the stochastic baselines in this short smoke benchmark. With `c=2.0`, the confidence bonus encourages substantial exploration. Over only 200 rounds, the cost of that exploration is not fully offset by later exploitation, so its cumulative pseudo-regret is higher than Epsilon-Greedy and Thompson Sampling.

EXP3 has the highest regret in this experiment. This is expected because EXP3 is designed for adversarial bandit settings, where rewards may be chosen by an adversary rather than sampled from fixed stochastic probabilities. In this Bernoulli benchmark, algorithms that exploit the stationary reward structure, especially Thompson Sampling, have a clear advantage.

In the dynamic experiment, EXP3 gives an unexpected result. Before running this experiment, I expected EXP3 to perform best because it is designed for adversarial bandit problems. However, vanilla EXP3 performs worst here, while UCB performs best. UCB is also the algorithm that recovers most clearly after round 200, when the best arm changes. From my perspective, this happens because UCB keeps enough exploration pressure through its confidence bonus, so it continues to try arms other than the one that looked best before the switch.

The other algorithms are more strongly affected by previous data. For EXP3 in particular, the arm weights can increase exponentially, but vanilla EXP3 does not forget old rewards. Therefore, after the first 200 rounds, the arm that was best in the first segment can still have a very large weight. Restarted EXP3 performs much better because it resets the weights at the same frequency as the environment changes. Discounted EXP3, sliding-window EXP3, and EXP3.S-style weight sharing also improve over vanilla EXP3, but they still do not match UCB in this simple piecewise-stationary setting.

## Reproducing the Experiment

If the dependencies are not installed yet, install them with:

```bash
python3 -m pip install numpy pandas matplotlib
```

If you use the local virtual environment, activate it first:

```bash
source .venv/bin/activate
```

Run the default benchmark:

```bash
python3 src/experiment.py
```

Run the short smoke benchmark:

```bash
python3 src/experiment.py --horizon 200 --seeds 3 --output-dir results/02_stationary_benchmark
```

Run the dynamic EXP3 benchmark:

```bash
python3 src/exp3_dynamic_experiment.py --horizon 1000 --seeds 20 --segment-length 200 --output-dir results/03_dynamic_exp3
```

Run the EXP3 gamma sweep:

```bash
python3 src/exp3_gamma_sweep.py --horizon 1000 --seeds 20 --segment-length 200 --output-dir results/04_exp3_gamma_sweep
```

Run the contextual bandit benchmark:

```bash
python3 src/contextual_experiment.py --horizon 1000 --seeds 20 --output-dir results/05_contextual_bandit
```

Run the simple baseline scripts:

```bash
python3 experiments/01_simple_baseline/random__.py
python3 experiments/01_simple_baseline/epsilon_greedy.py
```

The experiment writes:

- `summary.csv`: final performance summary for each algorithm
- `mean_curves.csv`: averaged metric curves over time
- `cumulative_regret.png`: cumulative pseudo-regret curve
- `average_reward.png`: average reward curve
- `optimal_arm_rate.png`: optimal arm selection rate curve
