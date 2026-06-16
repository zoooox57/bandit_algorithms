"""Run the bandit mini-benchmark.

Example:
    python src/experiment.py --horizon 5000 --seeds 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents import (
    DecayEpsilonGreedyAgent,
    EpsilonGreedyAgent,
    ThompsonSamplingAgent,
    UCBAgent,
)
from envs import BernoulliBandit


def make_agents(n_arms: int, seed: int):
    """Create the algorithms compared in the benchmark."""

    return [
        EpsilonGreedyAgent(n_arms=n_arms, epsilon=0.1, seed=seed),
        DecayEpsilonGreedyAgent(n_arms=n_arms, c=1.0, seed=seed),
        UCBAgent(n_arms=n_arms, c=2.0, seed=seed),
        ThompsonSamplingAgent(n_arms=n_arms, seed=seed),
    ]


def run_one_seed(agent, probs, horizon: int, seed: int) -> pd.DataFrame:
    """Run one agent in one Bernoulli bandit environment."""

    env = BernoulliBandit(probs=probs, seed=seed)

    cumulative_reward = 0.0
    cumulative_regret = 0.0
    optimal_pulls = 0
    rows = []

    for t in range(1, horizon + 1):
        arm = agent.select_arm(t)
        reward = env.pull(arm)
        agent.update(arm, reward)

        # Pseudo-regret uses the true means for evaluation only.
        instantaneous_regret = env.best_mean - env.mean_reward(arm)
        cumulative_regret += instantaneous_regret
        cumulative_reward += reward
        optimal_pulls += int(arm == env.best_arm)

        rows.append(
            {
                "algorithm": agent.name,
                "seed": seed,
                "t": t,
                "arm": arm,
                "reward": reward,
                "cumulative_reward": cumulative_reward,
                "average_reward": cumulative_reward / t,
                "instantaneous_regret": instantaneous_regret,
                "cumulative_regret": cumulative_regret,
                "optimal_arm_rate": optimal_pulls / t,
            }
        )

    return pd.DataFrame(rows)


def run_benchmark(probs, horizon: int, n_seeds: int) -> pd.DataFrame:
    """Run all algorithms over multiple random seeds."""

    all_runs = []
    for seed in range(n_seeds):
        for agent in make_agents(n_arms=len(probs), seed=seed):
            all_runs.append(run_one_seed(agent, probs, horizon, seed))
    return pd.concat(all_runs, ignore_index=True)


def summarize(final_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize final performance over seeds for each algorithm."""

    final_rows = final_df.loc[final_df.groupby(["algorithm", "seed"])["t"].idxmax()]
    summary = (
        final_rows.groupby("algorithm")
        .agg(
            mean_cumulative_regret=("cumulative_regret", "mean"),
            std_cumulative_regret=("cumulative_regret", "std"),
            mean_average_reward=("average_reward", "mean"),
            mean_optimal_arm_rate=("optimal_arm_rate", "mean"),
        )
        .reset_index()
        .sort_values("mean_cumulative_regret")
    )
    return summary


def plot_metric(mean_df: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> None:
    """Plot one metric as a function of time."""

    plt.figure(figsize=(8, 5))
    for algorithm, group in mean_df.groupby("algorithm"):
        plt.plot(group["t"], group[metric], label=algorithm)

    plt.xlabel("Round t")
    plt.ylabel(ylabel)
    plt.title(ylabel + " over time")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_results(results: pd.DataFrame, output_dir: Path) -> None:
    """Save raw curves, summary table, and plots."""

    output_dir.mkdir(parents=True, exist_ok=True)

    mean_curves = (
        results.groupby(["algorithm", "t"])
        .agg(
            cumulative_regret=("cumulative_regret", "mean"),
            average_reward=("average_reward", "mean"),
            optimal_arm_rate=("optimal_arm_rate", "mean"),
        )
        .reset_index()
    )

    summary = summarize(results)
    mean_curves.to_csv(output_dir / "mean_curves.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    plot_metric(
        mean_curves,
        metric="cumulative_regret",
        ylabel="Cumulative pseudo-regret",
        output_path=output_dir / "cumulative_regret.png",
    )
    plot_metric(
        mean_curves,
        metric="average_reward",
        ylabel="Average reward",
        output_path=output_dir / "average_reward.png",
    )
    plot_metric(
        mean_curves,
        metric="optimal_arm_rate",
        ylabel="Optimal arm selection rate",
        output_path=output_dir / "optimal_arm_rate.png",
    )

    print("\nFinal summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Bandit algorithms mini-benchmark")
    parser.add_argument(
        "--probs",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.6, 0.8],
        help="Bernoulli success probabilities for the arms.",
    )
    parser.add_argument("--horizon", type=int, default=5000, help="Number of rounds.")
    parser.add_argument("--seeds", type=int, default=50, help="Number of random seeds.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where CSV files and plots will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_benchmark(probs=args.probs, horizon=args.horizon, n_seeds=args.seeds)
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
