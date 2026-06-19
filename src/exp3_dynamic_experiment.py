"""Run EXP3 in a dynamic Bernoulli bandit environment.

Example:
    python src/exp3_dynamic_experiment.py --horizon 1000 --seeds 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents import (
    DecayEpsilonGreedyAgent,
    DiscountedEXP3Agent,
    EpsilonGreedyAgent,
    EXP3Agent,
    EXP3SAgent,
    LossEXP3Agent,
    RestartedEXP3Agent,
    SlidingWindowEXP3Agent,
    ThompsonSamplingAgent,
    UCBAgent,
)
from envs import DynamicBernoulliBandit


DEFAULT_PROB_SCHEDULE = [
    [0.8, 0.2, 0.2, 0.2, 0.2],
    [0.2, 0.8, 0.2, 0.2, 0.2],
    [0.2, 0.2, 0.8, 0.2, 0.2],
    [0.2, 0.2, 0.2, 0.8, 0.2],
    [0.2, 0.2, 0.2, 0.2, 0.8],
]


def make_agents(n_arms: int, seed: int, gamma: float, segment_length: int):
    """Create algorithms for the dynamic benchmark."""

    return [
        EXP3Agent(n_arms=n_arms, gamma=gamma, seed=seed),
        LossEXP3Agent(n_arms=n_arms, gamma=gamma, seed=seed),
        RestartedEXP3Agent(
            n_arms=n_arms,
            gamma=gamma,
            restart_interval=segment_length,
            seed=seed,
        ),
        DiscountedEXP3Agent(n_arms=n_arms, gamma=gamma, decay=0.99, seed=seed),
        SlidingWindowEXP3Agent(
            n_arms=n_arms,
            gamma=gamma,
            window_size=segment_length,
            seed=seed,
        ),
        EXP3SAgent(n_arms=n_arms, gamma=gamma, alpha=0.01, seed=seed),
        EpsilonGreedyAgent(n_arms=n_arms, epsilon=0.1, seed=seed),
        DecayEpsilonGreedyAgent(n_arms=n_arms, c=1.0, seed=seed),
        UCBAgent(n_arms=n_arms, c=2.0, seed=seed),
        ThompsonSamplingAgent(n_arms=n_arms, seed=seed),
    ]


def run_one_seed(agent, prob_schedule, segment_length: int, horizon: int, seed: int) -> pd.DataFrame:
    """Run one agent in one dynamic Bernoulli bandit environment."""

    env = DynamicBernoulliBandit(
        prob_schedule=prob_schedule,
        segment_length=segment_length,
        seed=seed,
    )

    cumulative_reward = 0.0
    cumulative_regret = 0.0
    optimal_pulls = 0
    rows = []

    for t in range(1, horizon + 1):
        arm = agent.select_arm(t)
        reward = env.pull(arm, t)
        agent.update(arm, reward)

        instantaneous_regret = env.best_mean(t) - env.mean_reward(arm, t)
        cumulative_regret += instantaneous_regret
        cumulative_reward += reward
        optimal_pulls += int(arm == env.best_arm(t))

        rows.append(
            {
                "algorithm": agent.name,
                "seed": seed,
                "t": t,
                "segment": (t - 1) // segment_length,
                "best_arm": env.best_arm(t),
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


def run_benchmark(prob_schedule, segment_length: int, horizon: int, n_seeds: int, gamma: float):
    """Run all algorithms over multiple random seeds."""

    all_runs = []
    n_arms = len(prob_schedule[0])
    for seed in range(n_seeds):
        for agent in make_agents(
            n_arms=n_arms,
            seed=seed,
            gamma=gamma,
            segment_length=segment_length,
        ):
            all_runs.append(run_one_seed(agent, prob_schedule, segment_length, horizon, seed))
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


def plot_metric(
    mean_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
    segment_length: int,
    horizon: int,
) -> None:
    """Plot one metric as a function of time."""

    plt.figure(figsize=(8, 5))
    for algorithm, group in mean_df.groupby("algorithm"):
        plt.plot(group["t"], group[metric], label=algorithm)

    for switch_t in range(segment_length + 1, horizon + 1, segment_length):
        plt.axvline(switch_t, color="black", linestyle="--", linewidth=0.8, alpha=0.25)

    plt.xlabel("Round t")
    plt.ylabel(ylabel)
    plt.title(ylabel + " over time")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_results(
    results: pd.DataFrame,
    output_dir: Path,
    prob_schedule,
    segment_length: int,
    horizon: int,
) -> None:
    """Save raw curves, summary table, schedule, and plots."""

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

    schedule = pd.DataFrame(prob_schedule)
    schedule.index.name = "segment"

    summary = summarize(results)
    mean_curves.to_csv(output_dir / "mean_curves.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    schedule.to_csv(output_dir / "prob_schedule.csv")

    plot_metric(
        mean_curves,
        metric="cumulative_regret",
        ylabel="Cumulative dynamic pseudo-regret",
        output_path=output_dir / "cumulative_regret.png",
        segment_length=segment_length,
        horizon=horizon,
    )
    plot_metric(
        mean_curves,
        metric="average_reward",
        ylabel="Average reward",
        output_path=output_dir / "average_reward.png",
        segment_length=segment_length,
        horizon=horizon,
    )
    plot_metric(
        mean_curves,
        metric="optimal_arm_rate",
        ylabel="Current best arm selection rate",
        output_path=output_dir / "optimal_arm_rate.png",
        segment_length=segment_length,
        horizon=horizon,
    )

    print("\nFinal summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="EXP3 dynamic bandit benchmark")
    parser.add_argument("--horizon", type=int, default=1000, help="Number of rounds.")
    parser.add_argument("--seeds", type=int, default=20, help="Number of random seeds.")
    parser.add_argument(
        "--segment-length",
        type=int,
        default=200,
        help="Number of rounds before switching to the next probability vector.",
    )
    parser.add_argument("--gamma", type=float, default=0.1, help="EXP3 exploration parameter.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/03_dynamic_exp3"),
        help="Directory where CSV files and plots will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_benchmark(
        prob_schedule=DEFAULT_PROB_SCHEDULE,
        segment_length=args.segment_length,
        horizon=args.horizon,
        n_seeds=args.seeds,
        gamma=args.gamma,
    )
    save_results(
        results,
        output_dir=args.output_dir,
        prob_schedule=DEFAULT_PROB_SCHEDULE,
        segment_length=args.segment_length,
        horizon=args.horizon,
    )


if __name__ == "__main__":
    main()
