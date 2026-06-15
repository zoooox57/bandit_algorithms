# bandit_algorithms
This is a small experiment on multi-armed bandit algorithms. The goal is to understand how different action-selection strategies balance exploration and exploitation

## Environment

I use a synthetic Bernoulli bandit environment. Each arm has a fixed but hidden success probability:

```python
probs = [0.5, 0.6, 0.8]

At each round, the agent selects one arm and receives a binary reward:
reward = 1 with probability probs[arm]
reward = 0 otherwise
The agent does not know the true probabilities. It only learns from observed rewards.
Algorithms

Random Agent
The random agent selects an arm uniformly at random at each round.
This agent does not learn from past rewards when choosing actions. It is useful as a simple baseline because it shows what happens when there is no learning strategy.

Expected behavior:
It keeps selecting suboptimal arms frequently.
Its cumulative regret usually grows almost linearly.
It does not become better over time.

Greedy Agent
The greedy agent selects the arm with the highest empirical mean reward:
argmax_i Q_i
where Q_i is the average reward observed from arm i so far.

Expected behavior:
At the beginning, regret may grow quickly because the agent has little information.
If the agent identifies the best arm, regret growth becomes much slower later.
However, pure greedy can be sensitive to early random rewards and may get stuck choosing a suboptimal arm.
Regret
I evaluate algorithms using cumulative pseudo-regret:
instant_regret = best_mean - chosen_arm_mean
where best_mean is the success probability of the best arm, and chosen_arm_mean is the true success probability of the selected arm.
Cumulative regret is the sum of instant regret over time.
Lower cumulative regret means the algorithm learns to choose better arms more efficiently.
Observation

In my experiment, the random agent keeps accumulating regret because it continues to choose arms randomly. It does not use reward feedback to improve its future decisions.
The greedy agent behaves differently. Its regret grows faster at the beginning, when its estimates are still inaccurate. After it has collected enough rewards, it often starts choosing the best arm more frequently. As a result, the cumulative regret curve becomes flatter later.
This shows the key difference between the two strategies:
Random agent explores all the time but does not exploit learned information.
Greedy agent exploits learned information but may not explore enough.

Next Step
A natural next step is to implement epsilon-greedy. It improves pure greedy by adding explicit exploration:
with probability epsilon: choose a random arm
otherwise: choose the arm with the highest empirical mean
This should reduce the risk of getting stuck with a suboptimal arm due to unlucky early rewards.
