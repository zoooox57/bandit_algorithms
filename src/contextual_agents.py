"""Contextual bandit agents."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ContextualBanditAgent(ABC):
    """Base class for contextual bandit algorithms."""

    name = "contextual_base"

    def __init__(self, n_arms: int, context_dim: int, lambda_: float = 1.0, seed: int = 0):
        if n_arms <= 0:
            raise ValueError("n_arms must be positive.")
        if context_dim <= 0:
            raise ValueError("context_dim must be positive.")
        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive.")

        self.n_arms = n_arms
        self.context_dim = context_dim
        self.lambda_ = lambda_
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.a_matrices = np.array([lambda_ * np.eye(context_dim) for _ in range(n_arms)])
        self.b_vectors = np.zeros((n_arms, context_dim), dtype=float)
        self.counts = np.zeros(n_arms, dtype=int)

    @abstractmethod
    def select_arm(self, context: np.ndarray) -> int:
        """Select an arm after observing the current context."""

    def theta_hat(self, arm: int) -> np.ndarray:
        """Return the ridge-regression parameter estimate for one arm."""

        return np.linalg.solve(self.a_matrices[arm], self.b_vectors[arm])

    def predicted_reward(self, context: np.ndarray, arm: int) -> float:
        return float(context @ self.theta_hat(arm))

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """Update the selected arm's linear model."""

        self.counts[arm] += 1
        self.a_matrices[arm] += np.outer(context, context)
        self.b_vectors[arm] += reward * context


class ContextualEpsilonGreedyAgent(ContextualBanditAgent):
    """Contextual epsilon-greedy with one linear model per arm."""

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        epsilon: float = 0.1,
        lambda_: float = 1.0,
        seed: int = 0,
    ):
        super().__init__(n_arms=n_arms, context_dim=context_dim, lambda_=lambda_, seed=seed)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        self.epsilon = epsilon
        self.name = f"contextual_epsilon_greedy_{epsilon:g}"

    def select_arm(self, context: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))

        scores = [self.predicted_reward(context, arm) for arm in range(self.n_arms)]
        return int(np.argmax(scores))


class LinUCBAgent(ContextualBanditAgent):
    """LinUCB with disjoint linear models for each arm."""

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        alpha: float = 1.0,
        lambda_: float = 1.0,
        seed: int = 0,
    ):
        super().__init__(n_arms=n_arms, context_dim=context_dim, lambda_=lambda_, seed=seed)
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        self.alpha = alpha
        self.name = f"linucb_alpha_{alpha:g}"

    def select_arm(self, context: np.ndarray) -> int:
        scores = []
        for arm in range(self.n_arms):
            a_inv = np.linalg.inv(self.a_matrices[arm])
            theta = a_inv @ self.b_vectors[arm]
            predicted_reward = float(context @ theta)
            uncertainty = float(np.sqrt(context @ a_inv @ context))
            scores.append(predicted_reward + self.alpha * uncertainty)
        return int(np.argmax(scores))
