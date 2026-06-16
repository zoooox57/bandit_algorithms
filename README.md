# Bandit Algorithms Mini Benchmark

This is a small benchmark for multi-armed bandit algorithms. The goal is to compare how different action-selection strategies balance exploration and exploitation.

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

The current benchmark in `src/experiment.py` compares four algorithms:

| Algorithm | Setting | Description |
| --- | --- | --- |
| Epsilon-Greedy | `epsilon=0.1` | Chooses a random arm with probability 10%; otherwise chooses the arm with the highest empirical mean reward |
| Decay Epsilon-Greedy | `epsilon_t = min(1, K / t)` | Explores more in early rounds and gradually reduces exploration over time |
| UCB | `c=2.0` | Uses an upper confidence bound to favor arms with high uncertainty |
| Thompson Sampling | Beta-Bernoulli posterior | Samples from each arm's posterior distribution and chooses the arm with the highest sample |

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

The `results_smoke/` directory contains a short smoke benchmark with horizon 200 and 3 random seeds. The final summary comes from `results_smoke/summary.csv`:

| Rank | Algorithm | Cumulative pseudo-regret ↓ | Average reward ↑ | Optimal arm rate ↑ |
| --- | --- | ---: | ---: | ---: |
| 1 | Thompson Sampling | 10.77 ± 4.13 | 0.723 | 0.837 |
| 2 | Epsilon-Greedy (`epsilon=0.1`) | 20.03 ± 8.61 | 0.675 | 0.702 |
| 3 | Decay Epsilon-Greedy (`c=1`) | 20.40 ± 19.33 | 0.667 | 0.608 |
| 4 | UCB (`c=2`) | 41.83 ± 2.51 | 0.558 | 0.412 |

The averaged learning curves are saved as:

- `results_smoke/cumulative_regret.png`
- `results_smoke/average_reward.png`
- `results_smoke/optimal_arm_rate.png`

## Observations

In this 200-round smoke benchmark, Thompson Sampling performs best. It has the lowest cumulative pseudo-regret, the highest average reward, and an optimal arm selection rate of about 83.7%. This suggests that the Beta-Bernoulli posterior helps the agent quickly concentrate its pulls on the true best arm while still maintaining useful uncertainty-driven exploration.

Epsilon-Greedy with `epsilon=0.1` ranks second. It can exploit the empirically best arm, but it always keeps a 10% random exploration rate. As a result, even after it has identified a good arm, it still pulls suboptimal arms occasionally, causing regret to continue increasing slowly.

Decay Epsilon-Greedy has a similar mean regret to fixed Epsilon-Greedy, but its standard deviation is much larger. It explores heavily at the beginning and then reduces exploration quickly. This can work well when early reward observations are helpful, but it can also lock onto a suboptimal arm if early noise is misleading.

UCB performs worst in this short smoke benchmark. With `c=2.0`, the confidence bonus encourages substantial exploration. Over only 200 rounds, the cost of that exploration is not fully offset by later exploitation, leading to the highest cumulative pseudo-regret and the lowest optimal arm selection rate.

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
python3 src/experiment.py --output-dir results
```

Run the short smoke benchmark:

```bash
python3 src/experiment.py --horizon 200 --seeds 3 --output-dir results_smoke
```

The experiment writes:

- `summary.csv`: final performance summary for each algorithm
- `mean_curves.csv`: averaged metric curves over time
- `cumulative_regret.png`: cumulative pseudo-regret curve
- `average_reward.png`: average reward curve
- `optimal_arm_rate.png`: optimal arm selection rate curve
