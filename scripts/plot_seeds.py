"""Aggregate training metrics and language analyses across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5c5b54"
GRID = "#e6e5e0"


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    return np.concatenate([np.full(window - 1, np.nan), smoothed])


def style_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)


def load_metrics(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()

    runs = {run_dir.name: load_metrics(run_dir) for run_dir in args.run_dirs}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="white")
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    ax = axes[0][0]
    curriculum_drawn = False
    for index, (name, records) in enumerate(runs.items()):
        updates = np.array([record["update"] for record in records])
        episodes = sum(records[0]["outcomes"].values())
        rate = np.array(
            [record["outcomes"].get("joint_stag", 0) / episodes for record in records]
        )
        ax.plot(
            updates,
            rolling_mean(rate, args.window),
            color=SERIES[index % len(SERIES)],
            linewidth=2,
            label=name.replace("curriculum_run_", ""),
        )
        if not curriculum_drawn and "failed_stag_reward" in records[0]:
            risk = np.array([record["failed_stag_reward"] for record in records])
            ax.plot(
                updates,
                risk / 2.0,
                color=TEXT_SECONDARY,
                linewidth=1,
                linestyle="--",
                label="failed_stag_reward / 2",
            )
            curriculum_drawn = True
    ax.set_ylim(0, 1)
    style_axis(ax, "Joint stag rate per seed", "update", "fraction of episodes")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY)

    ax = axes[0][1]
    for index, (name, records) in enumerate(runs.items()):
        updates = np.array([record["update"] for record in records])
        episodes = sum(records[0]["outcomes"].values())
        rate = np.array(
            [record["outcomes"].get("hare", 0) / episodes for record in records]
        )
        ax.plot(
            updates,
            rolling_mean(rate, args.window),
            color=SERIES[index % len(SERIES)],
            linewidth=2,
            label=name.replace("curriculum_run_", ""),
        )
    ax.set_ylim(0, 1)
    style_axis(ax, "Hare rate per seed", "update", "fraction of episodes")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY)

    ax = axes[1][0]
    for index, (name, records) in enumerate(runs.items()):
        updates = np.array([record["update"] for record in records])
        mean_return = np.array(
            [
                (record["mean_return_agent_0"] + record["mean_return_agent_1"]) / 2
                for record in records
            ]
        )
        ax.plot(
            updates,
            rolling_mean(mean_return, args.window),
            color=SERIES[index % len(SERIES)],
            linewidth=2,
            label=name.replace("curriculum_run_", ""),
        )
    for reference, label in ((4.0, "joint stag"), (2.0, "hare")):
        ax.axhline(reference, color=GRID, linewidth=1)
        ax.annotate(
            label, xy=(0.99, reference), xycoords=("axes fraction", "data"),
            fontsize=8, color=TEXT_SECONDARY, va="bottom", ha="right",
        )
    style_axis(ax, "Mean return (agent average) per seed", "update", "return")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY)

    ax = axes[1][1]
    plotted_mi = False
    for index, (name, _) in enumerate(runs.items()):
        analysis_path = args.run_dirs[index] / "language_analysis.json"
        if not analysis_path.exists():
            continue
        analysis = json.loads(analysis_path.read_text())
        steps = [
            int(entry["checkpoint"].split("_")[1])
            for entry in analysis["mi_by_checkpoint"]
        ]
        total_mi = [
            entry["mi_color_agent0"] + entry["mi_region_agent1"]
            for entry in analysis["mi_by_checkpoint"]
        ]
        ax.plot(
            steps,
            total_mi,
            color=SERIES[index % len(SERIES)],
            linewidth=2,
            marker="o",
            markersize=4,
            label=name.replace("curriculum_run_", ""),
        )
        plotted_mi = True
    if plotted_mi:
        ax.axhline(2.0, color=GRID, linewidth=1)
        ax.annotate(
            "both clues fully transmitted (2 bits)",
            xy=(0.99, 2.0), xycoords=("axes fraction", "data"),
            fontsize=8, color=TEXT_SECONDARY, va="bottom", ha="right",
        )
        ax.set_ylim(bottom=0)
    style_axis(
        ax,
        "Speaker informativeness per seed",
        "training update",
        "MI(clue; message), both agents summed (bits)",
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
