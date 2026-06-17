"""Bandit environments for the mini-benchmark.

The first environment is a Bernoulli multi-armed bandit. Each arm i has an
unknown success probability p_i. Pulling arm i returns reward 1 with
probability p_i and reward 0 otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BernoulliBandit:
    """A stochastic K-armed Bernoulli bandit.

    Parameters
    ----------
    probs:
        Success probabilities for all arms. Each value must be in [0, 1].
    seed:
        Random seed used by this environment.
    """

    probs: np.ndarray
    seed: int = 0

    def __init__(self, probs, seed: int = 0):
        probs_array = np.asarray(probs, dtype=float)
        if probs_array.ndim != 1:
            raise ValueError("probs must be a one-dimensional sequence.")
        if len(probs_array) == 0:
            raise ValueError("probs must contain at least one arm.")
        if np.any((probs_array < 0.0) | (probs_array > 1.0)):
            raise ValueError("All Bernoulli probabilities must be in [0, 1].")

        object.__setattr__(self, "probs", probs_array)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "_rng", np.random.default_rng(seed))

    @property
    def n_arms(self) -> int:
        return len(self.probs)

    @property
    def best_arm(self) -> int:
        return int(np.argmax(self.probs))

    @property
    def best_mean(self) -> float:
        return float(np.max(self.probs))

    def mean_reward(self, arm: int) -> float:
        """Return the true expected reward of an arm.

        The learner should not use this value during learning. It is only used
        by the experiment runner to compute pseudo-regret.
        """

        self._validate_arm(arm)
        return float(self.probs[arm])

    def pull(self, arm: int) -> float:
        """Pull one arm and return a stochastic reward in {0.0, 1.0}."""

        self._validate_arm(arm)
        return float(self._rng.random() < self.probs[arm])

    def _validate_arm(self, arm: int) -> None:
        if not 0 <= arm < self.n_arms:
            raise IndexError(f"arm must be in [0, {self.n_arms - 1}], got {arm}.")


@dataclass(frozen=True)
class DynamicBernoulliBandit:
    """A piecewise-stationary Bernoulli bandit.

    The environment cycles through a sequence of probability vectors. Each
    vector stays active for ``segment_length`` rounds, so the best arm can
    change over time.
    """

    prob_schedule: np.ndarray
    segment_length: int
    seed: int = 0

    def __init__(self, prob_schedule, segment_length: int, seed: int = 0):
        schedule = np.asarray(prob_schedule, dtype=float)
        if schedule.ndim != 2:
            raise ValueError("prob_schedule must be a two-dimensional sequence.")
        if schedule.shape[0] == 0 or schedule.shape[1] == 0:
            raise ValueError("prob_schedule must contain at least one segment and one arm.")
        if segment_length <= 0:
            raise ValueError("segment_length must be positive.")
        if np.any((schedule < 0.0) | (schedule > 1.0)):
            raise ValueError("All Bernoulli probabilities must be in [0, 1].")

        object.__setattr__(self, "prob_schedule", schedule)
        object.__setattr__(self, "segment_length", segment_length)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "_rng", np.random.default_rng(seed))

    @property
    def n_arms(self) -> int:
        return self.prob_schedule.shape[1]

    def current_probs(self, t: int) -> np.ndarray:
        """Return the active probability vector at round t, using 1-based t."""

        if t <= 0:
            raise ValueError("t must be positive.")
        segment = ((t - 1) // self.segment_length) % len(self.prob_schedule)
        return self.prob_schedule[segment]

    def best_arm(self, t: int) -> int:
        return int(np.argmax(self.current_probs(t)))

    def best_mean(self, t: int) -> float:
        return float(np.max(self.current_probs(t)))

    def mean_reward(self, arm: int, t: int) -> float:
        self._validate_arm(arm)
        return float(self.current_probs(t)[arm])

    def pull(self, arm: int, t: int) -> float:
        self._validate_arm(arm)
        return float(self._rng.random() < self.mean_reward(arm, t))

    def _validate_arm(self, arm: int) -> None:
        if not 0 <= arm < self.n_arms:
            raise IndexError(f"arm must be in [0, {self.n_arms - 1}], got {arm}.")
