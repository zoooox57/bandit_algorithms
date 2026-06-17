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
│   └── 03_dynamic_exp3/         # Dynamic probability benchmark results
├── src/
│   ├── agents.py                # Bandit algorithms
│   ├── envs.py                  # Stationary and dynamic environments
│   ├── experiment.py            # Stationary benchmark runner
│   └── exp3_dynamic_experiment.py
└── README.md
```

The project has three experiment groups:

| Experiment | Purpose | Code | Results |
| --- | --- | --- | --- |
| Simple baseline | Introduce random and epsilon-greedy behavior | `experiments/01_simple_baseline/` | `results/01_simple_baseline/` |
| Stationary benchmark | Compare algorithms when arm probabilities are fixed | `src/experiment.py` | `results/02_stationary_benchmark/` |
| Dynamic EXP3 benchmark | Test algorithms when the best arm changes over time | `src/exp3_dynamic_experiment.py` | `results/03_dynamic_exp3/` |

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
| 3 | Epsilon-Greedy (`epsilon=0.1`) | 423.00 ± 39.33 | 0.378 | 0.295 |
| 4 | Decay Epsilon-Greedy (`c=1`) | 482.37 ± 37.76 | 0.316 | 0.196 |
| 5 | EXP3 (`gamma=0.1`) | 521.67 ± 5.46 | 0.271 | 0.131 |

In this switching-probability environment, vanilla EXP3 still does not perform best. This is an important distinction: standard EXP3 is designed for adversarial bandits, but it usually competes against the best fixed arm in hindsight.

## Observations

In this 200-round smoke benchmark, Thompson Sampling performs best. It has the lowest cumulative pseudo-regret, the highest average reward, and an optimal arm selection rate of about 83.7%. This suggests that the Beta-Bernoulli posterior helps the agent quickly concentrate its pulls on the true best arm while still maintaining useful uncertainty-driven exploration.

Epsilon-Greedy with `epsilon=0.1` ranks second. It can exploit the empirically best arm, but it always keeps a 10% random exploration rate. As a result, even after it has identified a good arm, it still pulls suboptimal arms occasionally, causing regret to continue increasing slowly.

Decay Epsilon-Greedy has a similar mean regret to fixed Epsilon-Greedy, but its standard deviation is much larger. It explores heavily at the beginning and then reduces exploration quickly. This can work well when early reward observations are helpful, but it can also lock onto a suboptimal arm if early noise is misleading.

UCB is weaker than the stochastic baselines in this short smoke benchmark. With `c=2.0`, the confidence bonus encourages substantial exploration. Over only 200 rounds, the cost of that exploration is not fully offset by later exploitation, so its cumulative pseudo-regret is higher than Epsilon-Greedy and Thompson Sampling.

EXP3 has the highest regret in this experiment. This is expected because EXP3 is designed for adversarial bandit settings, where rewards may be chosen by an adversary rather than sampled from fixed stochastic probabilities. In this Bernoulli benchmark, algorithms that exploit the stationary reward structure, especially Thompson Sampling, have a clear advantage.

In the dynamic experiment, EXP3 gives an unexpected result. Before running this experiment, I expected EXP3 to perform best because it is designed for adversarial bandit problems. However, vanilla EXP3 performs worst here, while UCB performs best. UCB is also the only algorithm that clearly recovers after round 200, when the best arm changes. From my perspective, this happens because UCB keeps enough exploration pressure through its confidence bonus, so it continues to try arms other than the one that looked best before the switch.

The other algorithms are more strongly affected by previous data. For EXP3 in particular, the arm weights can increase exponentially, but they do not decrease or forget old rewards in the current implementation. Therefore, after the first 200 rounds, the arm that was best in the first segment can still have a very large weight. When the environment changes, EXP3 needs time to build up the weight of the new best arm. A better version for this experiment would add some form of forgetting, such as discounted weights, restarted EXP3, sliding-window EXP3, or EXP3.S.

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
