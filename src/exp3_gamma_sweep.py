"""Sweep EXP3 gamma values in the dynamic Bernoulli bandit environment.

Example:
    python src/exp3_gamma_sweep.py --horizon 1000 --seeds 20 --gammas 0.01 0.05 0.1 0.2 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from agents import EXP3Agent, LossEXP3Agent
from envs import DynamicBernoulliBandit
from exp3_dynamic_experiment import DEFAULT_PROB_SCHEDULE, summarize


def run_one_seed(agent, prob_schedule, segment_length: int, horizon: int, seed: int) -> pd.DataFrame:
    """Run one EXP3 gamma setting in one dynamic bandit environment."""

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
                "gamma": agent.gamma,
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


def run_sweep(prob_schedule, segment_length: int, horizon: int, n_seeds: int, gammas):
    """Run EXP3 over several gamma values."""

    all_runs = []
    n_arms = len(prob_schedule[0])
    agent_classes = [EXP3Agent, LossEXP3Agent]
    for gamma in gammas:
        for seed in range(n_seeds):
            for agent_class in agent_classes:
                agent = agent_class(n_arms=n_arms, gamma=gamma, seed=seed)
                all_runs.append(run_one_seed(agent, prob_schedule, segment_length, horizon, seed))
    return pd.concat(all_runs, ignore_index=True)


def plot_metric(mean_df: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> None:
    """Plot one metric as a function of time for each gamma."""

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
    """Save summary, curves, and plots for the gamma sweep."""

    output_dir.mkdir(parents=True, exist_ok=True)

    mean_curves = (
        results.groupby(["gamma", "algorithm", "t"])
        .agg(
            cumulative_regret=("cumulative_regret", "mean"),
            average_reward=("average_reward", "mean"),
            optimal_arm_rate=("optimal_arm_rate", "mean"),
        )
        .reset_index()
    )

    summary = summarize(results)
    results.to_csv(output_dir / "raw_runs.csv", index=False)
    mean_curves.to_csv(output_dir / "mean_curves.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    plot_metric(
        mean_curves,
        metric="cumulative_regret",
        ylabel="Cumulative dynamic pseudo-regret",
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
        ylabel="Current best arm selection rate",
        output_path=output_dir / "optimal_arm_rate.png",
    )

    print("\nEXP3 gamma sweep summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="EXP3 gamma sweep in a dynamic bandit")
    parser.add_argument("--horizon", type=int, default=1000, help="Number of rounds.")
    parser.add_argument("--seeds", type=int, default=20, help="Number of random seeds.")
    parser.add_argument(
        "--segment-length",
        type=int,
        default=200,
        help="Number of rounds before switching to the next probability vector.",
    )
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.2, 0.5],
        help="EXP3 gamma values to compare.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/04_exp3_gamma_sweep"),
        help="Directory where CSV files and plots will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_sweep(
        prob_schedule=DEFAULT_PROB_SCHEDULE,
        segment_length=args.segment_length,
        horizon=args.horizon,
        n_seeds=args.seeds,
        gammas=args.gammas,
    )
    save_results(results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
