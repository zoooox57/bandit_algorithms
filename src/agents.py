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
