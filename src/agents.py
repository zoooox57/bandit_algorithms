"""Bandit agents used in the mini-benchmark.

Each agent implements two methods:

select_arm(t): choose an arm at round t.
update(arm, reward): update internal statistics after observing reward.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BanditAgent(ABC):
    """Base class for all bandit algorithms."""

    name = "base"

    def __init__(self, n_arms: int, seed: int = 0):
        if n_arms <= 0:
            raise ValueError("n_arms must be positive.")
        self.n_arms = n_arms
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(n_arms, dtype=int)
        self.q_values = np.zeros(n_arms, dtype=float)

    @abstractmethod
    def select_arm(self, t: int) -> int:
        """Select an arm at round t."""

    def update(self, arm: int, reward: float) -> None:
        """Incrementally update the empirical mean reward of an arm."""

        self.counts[arm] += 1
        n = self.counts[arm]
        self.q_values[arm] += (reward - self.q_values[arm]) / n


class EpsilonGreedyAgent(BanditAgent):
    """Epsilon-greedy with a constant exploration probability."""

    def __init__(self, n_arms: int, epsilon: float = 0.1, seed: int = 0):
        super().__init__(n_arms=n_arms, seed=seed)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        self.epsilon = epsilon
        self.name = f"epsilon_greedy_{epsilon:g}"

    def select_arm(self, t: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))
        return int(np.argmax(self.q_values))


class DecayEpsilonGreedyAgent(BanditAgent):
    """Epsilon-greedy with epsilon_t = min(1, c * K / t)."""

    def __init__(self, n_arms: int, c: float = 1.0, seed: int = 0):
        super().__init__(n_arms=n_arms, seed=seed)
        if c <= 0:
            raise ValueError("c must be positive.")
        self.c = c
        self.name = f"decay_epsilon_c{c:g}"

    def epsilon(self, t: int) -> float:
        return min(1.0, self.c * self.n_arms / max(t, 1))

    def select_arm(self, t: int) -> int:
        if self.rng.random() < self.epsilon(t):
            return int(self.rng.integers(self.n_arms))
        return int(np.argmax(self.q_values))


class UCBAgent(BanditAgent):
    """Upper Confidence Bound agent.

    The score for arm i is:

        Q_i(t) + c * sqrt(log(t) / N_i(t))

    Arms that have never been selected are pulled once before using the formula.
    """

    def __init__(self, n_arms: int, c: float = 2.0, seed: int = 0):
        super().__init__(n_arms=n_arms, seed=seed)
        if c <= 0:
            raise ValueError("c must be positive.")
        self.c = c
        self.name = f"ucb_c{c:g}"

    def select_arm(self, t: int) -> int:
        for arm, count in enumerate(self.counts):
            if count == 0:
                return arm

        bonus = self.c * np.sqrt(np.log(max(t, 2)) / self.counts)
        return int(np.argmax(self.q_values + bonus))


class EXP3Agent(BanditAgent):
    """EXP3 agent for adversarial bandits with rewards in [0, 1]."""

    def __init__(self, n_arms: int, gamma: float = 0.1, seed: int = 0):
        super().__init__(n_arms=n_arms, seed=seed)
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        self.gamma = gamma
        self.log_weights = np.zeros(n_arms, dtype=float)
        self.probs = np.ones(n_arms, dtype=float) / n_arms
        self.name = f"exp3_gamma_{gamma:g}"

    def action_probabilities(self) -> np.ndarray:
        weights = np.exp(self.log_weights - np.max(self.log_weights))
        weight_probs = weights / np.sum(weights)
        return (1.0 - self.gamma) * weight_probs + self.gamma / self.n_arms

    def select_arm(self, t: int) -> int:
        self.probs = self.action_probabilities()
        return int(self.rng.choice(self.n_arms, p=self.probs))

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        estimated_reward = reward / self.probs[arm]
        self.log_weights[arm] += self.gamma * estimated_reward / self.n_arms


class LossEXP3Agent(BanditAgent):
    """Loss-based EXP3 variant.

    Standard reward-based EXP3 only increases the selected arm's weight when
    reward is high. This variant treats 1 - reward as a loss and decreases the
    selected arm's weight when the observed reward is low.
    """

    def __init__(self, n_arms: int, gamma: float = 0.1, seed: int = 0):
        super().__init__(n_arms=n_arms, seed=seed)
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        self.gamma = gamma
        self.log_weights = np.zeros(n_arms, dtype=float)
        self.probs = np.ones(n_arms, dtype=float) / n_arms
        self.name = f"loss_exp3_gamma_{gamma:g}"

    def action_probabilities(self) -> np.ndarray:
        weights = np.exp(self.log_weights - np.max(self.log_weights))
        weight_probs = weights / np.sum(weights)
        return (1.0 - self.gamma) * weight_probs + self.gamma / self.n_arms

    def select_arm(self, t: int) -> int:
        self.probs = self.action_probabilities()
        return int(self.rng.choice(self.n_arms, p=self.probs))

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        estimated_loss = (1.0 - reward) / self.probs[arm]
        self.log_weights[arm] -= self.gamma * estimated_loss / self.n_arms


class RestartedEXP3Agent(EXP3Agent):
    """EXP3 with periodic weight resets."""

    def __init__(
        self,
        n_arms: int,
        gamma: float = 0.1,
        restart_interval: int = 200,
        seed: int = 0,
    ):
        super().__init__(n_arms=n_arms, gamma=gamma, seed=seed)
        if restart_interval <= 0:
            raise ValueError("restart_interval must be positive.")
        self.restart_interval = restart_interval
        self.name = f"restarted_exp3_gamma_{gamma:g}_r{restart_interval:g}"

    def select_arm(self, t: int) -> int:
        if t > 1 and (t - 1) % self.restart_interval == 0:
            self.log_weights = np.zeros(self.n_arms, dtype=float)
        return super().select_arm(t)


class DiscountedEXP3Agent(EXP3Agent):
    """EXP3 with exponentially discounted historical weights."""

    def __init__(self, n_arms: int, gamma: float = 0.1, decay: float = 0.99, seed: int = 0):
        super().__init__(n_arms=n_arms, gamma=gamma, seed=seed)
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1].")
        self.decay = decay
        self.name = f"discounted_exp3_gamma_{gamma:g}_d{decay:g}"

    def update(self, arm: int, reward: float) -> None:
        self.log_weights *= self.decay
        super().update(arm, reward)


class SlidingWindowEXP3Agent(EXP3Agent):
    """EXP3 that only keeps estimated rewards from a recent window."""

    def __init__(self, n_arms: int, gamma: float = 0.1, window_size: int = 200, seed: int = 0):
        super().__init__(n_arms=n_arms, gamma=gamma, seed=seed)
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        self.window_size = window_size
        self.history = []
        self.name = f"sliding_exp3_gamma_{gamma:g}_w{window_size:g}"

    def update(self, arm: int, reward: float) -> None:
        BanditAgent.update(self, arm, reward)
        estimated_reward = reward / self.probs[arm]
        self.history.append((arm, estimated_reward))
        if len(self.history) > self.window_size:
            self.history.pop(0)

        self.log_weights = np.zeros(self.n_arms, dtype=float)
        for past_arm, past_estimated_reward in self.history:
            self.log_weights[past_arm] += self.gamma * past_estimated_reward / self.n_arms


class EXP3SAgent(EXP3Agent):
    """EXP3.S-style fixed-share variant for switching environments."""

    def __init__(self, n_arms: int, gamma: float = 0.1, alpha: float = 0.01, seed: int = 0):
        super().__init__(n_arms=n_arms, gamma=gamma, seed=seed)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        self.alpha = alpha
        self.name = f"exp3s_gamma_{gamma:g}_alpha_{alpha:g}"

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)

        weights = np.exp(self.log_weights - np.max(self.log_weights))
        shared_weight = np.sum(weights) / self.n_arms
        weights = (1.0 - self.alpha) * weights + self.alpha * shared_weight
        self.log_weights = np.log(weights)


class ThompsonSamplingAgent(BanditAgent):
    """Thompson Sampling for Bernoulli bandits using Beta posteriors."""

    name = "thompson_sampling"

    def __init__(self, n_arms: int, seed: int = 0):
        super().__init__(n_arms=n_arms, seed=seed)
        self.alpha = np.ones(n_arms, dtype=float)
        self.beta = np.ones(n_arms, dtype=float)

    def select_arm(self, t: int) -> int:
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward
