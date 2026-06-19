"""Run a simple contextual bandit benchmark.

Example:
    python src/contextual_experiment.py --horizon 1000 --seeds 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from contextual_agents import ContextualEpsilonGreedyAgent, LinUCBAgent
from contextual_envs import LogisticContextualBandit


def make_agents(n_arms: int, context_dim: int, seed: int):
    """Create contextual bandit algorithms."""

    return [
        ContextualEpsilonGreedyAgent(
            n_arms=n_arms,
            context_dim=context_dim,
            epsilon=0.1,
            lambda_=1.0,
            seed=seed,
        ),
        LinUCBAgent(
            n_arms=n_arms,
            context_dim=context_dim,
            alpha=1.0,
            lambda_=1.0,
            seed=seed,
        ),
    ]


def run_one_seed(agent, horizon: int, seed: int) -> pd.DataFrame:
    """Run one contextual agent in one environment."""

    env = LogisticContextualBandit(seed=seed)
    cumulative_reward = 0.0
    cumulative_regret = 0.0
    optimal_pulls = 0
    rows = []

    for t in range(1, horizon + 1):
        context = env.sample_context()
        arm = agent.select_arm(context)
        reward = env.pull(context, arm)
        agent.update(context, arm, reward)

        best_arm = env.best_arm(context)
        instantaneous_regret = env.best_mean(context) - env.reward_prob(context, arm)
        cumulative_regret += instantaneous_regret
        cumulative_reward += reward
        optimal_pulls += int(arm == best_arm)

        rows.append(
            {
                "algorithm": agent.name,
                "seed": seed,
                "t": t,
                "context_0": context[0],
                "context_1": context[1],
                "best_arm": best_arm,
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


def run_benchmark(horizon: int, n_seeds: int) -> pd.DataFrame:
    """Run all contextual algorithms over multiple random seeds."""

    env = LogisticContextualBandit(seed=0)
    all_runs = []
    for seed in range(n_seeds):
        for agent in make_agents(env.n_arms, env.context_dim, seed):
            all_runs.append(run_one_seed(agent, horizon, seed))
    return pd.concat(all_runs, ignore_index=True)


def summarize(final_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize final performance over seeds for each algorithm."""

    final_rows = final_df.loc[final_df.groupby(["algorithm", "seed"])["t"].idxmax()]
    return (
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


def plot_metric(mean_df: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> None:
    """Plot one averaged metric over time."""

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
    """Save summary table, mean curves, and plots."""

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
        ylabel="Cumulative contextual pseudo-regret",
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
        ylabel="Contextual optimal arm selection rate",
        output_path=output_dir / "optimal_arm_rate.png",
    )

    print("\nFinal summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Contextual bandit benchmark")
    parser.add_argument("--horizon", type=int, default=1000, help="Number of rounds.")
    parser.add_argument("--seeds", type=int, default=20, help="Number of random seeds.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/05_contextual_bandit"),
        help="Directory where CSV files and plots will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_benchmark(horizon=args.horizon, n_seeds=args.seeds)
    save_results(results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
