"""Render training summary figures from a train_mappo.py metrics.jsonl file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5c5b54"
GRID = "#e6e5e0"

OUTCOMES = ["joint_stag", "hare", "failed_stag", "timeout", "hare_stag_mismatch"]


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, smoothed])


def style_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left", pad=10)
    ax.set_xlabel("update", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in (args.run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    updates = np.array([record["update"] for record in records])
    n_episodes = sum(records[0]["outcomes"].values())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="white")
    fig.subplots_adjust(hspace=0.45, wspace=0.3)

    ax = axes[0][0]
    for index, outcome in enumerate(OUTCOMES):
        rate = np.array(
            [record["outcomes"].get(outcome, 0) / n_episodes for record in records]
        )
        smoothed = rolling_mean(rate, args.window)
        ax.plot(updates, smoothed, color=SERIES[index], linewidth=2, label=outcome)
    ax.set_ylim(0, 1)
    style_axis(ax, "Episode outcome rates (rolling mean)", "fraction of episodes")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY)

    ax = axes[0][1]
    for index, agent in enumerate(("agent_0", "agent_1")):
        values = np.array([record[f"mean_return_{agent}"] for record in records])
        ax.plot(
            updates,
            rolling_mean(values, args.window),
            color=SERIES[index],
            linewidth=2,
            label=agent,
        )
    for reference, label in ((4.0, "joint stag"), (2.0, "hare")):
        ax.axhline(reference, color=GRID, linewidth=1)
        ax.annotate(
            label,
            xy=(updates[-1], reference),
            fontsize=8,
            color=TEXT_SECONDARY,
            va="bottom",
            ha="right",
        )
    style_axis(ax, "Mean episode return (rolling mean)", "return")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY)

    ax = axes[1][0]
    lengths = np.array([record["mean_length"] for record in records])
    ax.plot(updates, rolling_mean(lengths, args.window), color=SERIES[0], linewidth=2)
    style_axis(ax, "Mean episode length", "steps")

    ax = axes[1][1]
    entropy = np.array([record["entropy"] for record in records])
    silence = np.array([record["silence_rate"] for record in records])
    ax.plot(
        updates,
        rolling_mean(entropy, args.window) / np.log(30),
        color=SERIES[0],
        linewidth=2,
        label="policy entropy (normalized)",
    )
    ax.plot(
        updates,
        rolling_mean(silence, args.window),
        color=SERIES[1],
        linewidth=2,
        label="message silence rate",
    )
    ax.set_ylim(0, 1.05)
    style_axis(ax, "Exploration and channel use", "value")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY)

    output = args.run_dir / "training_summary.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
