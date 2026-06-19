"""Contextual bandit environments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(x: float) -> float:
    """Map a real-valued score to a probability."""

    return float(1.0 / (1.0 + np.exp(-x)))


@dataclass(frozen=True)
class LogisticContextualBandit:
    """A simple contextual Bernoulli bandit with logistic rewards.

    Each round samples one of two user contexts:

        [1, 0]: sports-like user
        [0, 1]: music-like user

    Each arm has a hidden parameter vector. The reward probability is
    sigmoid(context @ theta_arm).
    """

    theta: np.ndarray
    seed: int = 0

    def __init__(self, theta=None, seed: int = 0):
        if theta is None:
            theta = [
                [2.0, -1.0],
                [-1.0, 2.0],
                [0.5, 0.5],
            ]

        theta_array = np.asarray(theta, dtype=float)
        if theta_array.ndim != 2:
            raise ValueError("theta must be a two-dimensional array.")
        if theta_array.shape[0] == 0 or theta_array.shape[1] == 0:
            raise ValueError("theta must contain at least one arm and one context feature.")

        object.__setattr__(self, "theta", theta_array)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "_rng", np.random.default_rng(seed))

    @property
    def n_arms(self) -> int:
        return self.theta.shape[0]

    @property
    def context_dim(self) -> int:
        return self.theta.shape[1]

    def sample_context(self) -> np.ndarray:
        """Sample a context vector for the current round."""

        if self._rng.random() < 0.5:
            return np.array([1.0, 0.0])
        return np.array([0.0, 1.0])

    def reward_prob(self, context: np.ndarray, arm: int) -> float:
        """Return P(reward = 1 | context, arm)."""

        self._validate_context(context)
        self._validate_arm(arm)
        return sigmoid(float(context @ self.theta[arm]))

    def reward_probs(self, context: np.ndarray) -> np.ndarray:
        """Return reward probabilities for all arms under this context."""

        self._validate_context(context)
        return np.array([self.reward_prob(context, arm) for arm in range(self.n_arms)])

    def best_arm(self, context: np.ndarray) -> int:
        return int(np.argmax(self.reward_probs(context)))

    def best_mean(self, context: np.ndarray) -> float:
        return float(np.max(self.reward_probs(context)))

    def pull(self, context: np.ndarray, arm: int) -> float:
        """Pull one arm and return a stochastic binary reward."""

        return float(self._rng.random() < self.reward_prob(context, arm))

    def _validate_context(self, context: np.ndarray) -> None:
        context = np.asarray(context, dtype=float)
        if context.shape != (self.context_dim,):
            raise ValueError(f"context must have shape ({self.context_dim},).")

    def _validate_arm(self, arm: int) -> None:
        if not 0 <= arm < self.n_arms:
            raise IndexError(f"arm must be in [0, {self.n_arms - 1}], got {arm}.")
