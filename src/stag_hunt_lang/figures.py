"""Simple, reusable plots of the environment and agent trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from stag_hunt_lang import EnvConfig, StagHuntLanguageEnv
from stag_hunt_lang.world_cli import scripted_actions

AGENT_COLORS = {"agent_0": "#168f87", "agent_1": "#7b2cbf"}
STAG_COLORS = {0: "#d1495b", 1: "#277da1"}
REGION_COLORS = ("#f5e6cc", "#dceef8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/figures"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    environment_path = args.output / "environment.png"
    trajectories_path = args.output / "agent_trajectories.png"
    plot_environment(seed=args.seed, output=environment_path, dpi=args.dpi)
    plot_agent_trajectories(seed=args.seed, output=trajectories_path, dpi=args.dpi)
    print(environment_path)
    print(trajectories_path)


def plot_environment(seed: int, output: Path, dpi: int = 180) -> None:
    """Plot one reset state with the information needed to interpret it."""

    env = StagHuntLanguageEnv(EnvConfig())
    env.reset(seed=seed)
    figure, axis = plt.subplots(figsize=(7.2, 7.0), constrained_layout=True)
    draw_world(axis, env, show_agents=True, show_correct=True)
    axis.set_title(
        "Sample environment\n"
        f"A0 clue: {env.color_name(env.correct_color)}  ·  "
        f"A1 clue: {env.region_name(env.correct_region)}",
        fontsize=14,
        weight="bold",
    )
    axis.legend(
        handles=world_legend(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=4,
        frameon=False,
    )
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_agent_trajectories(seed: int, output: Path, dpi: int = 180) -> None:
    """Plot the two paths from a scripted successful cooperative episode."""

    env, paths = simulate_cooperative_episode(seed)
    figure, axis = plt.subplots(figsize=(7.2, 7.0), constrained_layout=True)
    draw_world(axis, env, show_agents=False, show_correct=True)
    for agent, path in paths.items():
        draw_trajectory(axis, agent, path)
    axis.set_title(
        "Example agent trajectories\nboth agents converge on the correct stag",
        fontsize=14,
        weight="bold",
    )
    axis.legend(
        handles=trajectory_legend(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=4,
        frameon=False,
    )
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def draw_world(
    axis: Axes,
    env: StagHuntLanguageEnv,
    *,
    show_agents: bool = True,
    show_correct: bool = False,
) -> None:
    """Draw any current environment state on an existing Matplotlib axis."""

    config = env.config
    east_start_x = int(np.ceil(config.grid_size / config.n_regions))
    boundary = east_start_x - 0.5
    axis.axvspan(-0.5, boundary, color=REGION_COLORS[0], alpha=0.72, zorder=0)
    axis.axvspan(
        boundary,
        config.grid_size - 0.5,
        color=REGION_COLORS[1],
        alpha=0.72,
        zorder=0,
    )
    axis.text(0.03, 0.97, "WEST", transform=axis.transAxes, weight="bold", va="top")
    axis.text(0.84, 0.97, "EAST", transform=axis.transAxes, weight="bold", va="top")

    for index, (position, feature) in enumerate(
        zip(env.stag_positions, env.stag_features, strict=True)
    ):
        x, y = map(int, position)
        axis.scatter(
            x,
            y,
            marker="*",
            s=330,
            color=STAG_COLORS.get(int(feature[0]), "#6c757d"),
            edgecolor="black",
            linewidth=1.0,
            zorder=4,
        )
        axis.text(x, y + 0.42, f"S{index}", ha="center", fontsize=9, weight="bold")

    for index, position in enumerate(env.hare_positions):
        x, y = map(int, position)
        axis.scatter(
            x,
            y,
            marker="h",
            s=190,
            color="#9c6644",
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        axis.text(x, y + 0.38, f"H{index}", ha="center", fontsize=8, weight="bold")

    if show_correct:
        target_x, target_y = map(int, env.correct_target_position)
        axis.scatter(
            target_x,
            target_y,
            s=540,
            facecolors="none",
            edgecolors="#f4a261",
            linewidth=3.0,
            zorder=5,
        )

    if show_agents:
        for agent, position in env.agent_positions.items():
            draw_agent(axis, agent, position)

    axis.set_xlim(-0.5, config.grid_size - 0.5)
    axis.set_ylim(config.grid_size - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.set_xticks(range(config.grid_size))
    axis.set_yticks(range(config.grid_size))
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.grid(color="white", linewidth=1.5)


def simulate_cooperative_episode(
    seed: int,
) -> tuple[StagHuntLanguageEnv, dict[str, list[np.ndarray]]]:
    """Run the scripted joint-stag policy and retain both paths."""

    env = StagHuntLanguageEnv(EnvConfig())
    env.reset(seed=seed)
    paths = {
        agent: [position.copy()] for agent, position in env.agent_positions.items()
    }
    while env.agents:
        env.step(scripted_actions(env, "joint-stag"))
        for agent, position in env.agent_positions.items():
            paths[agent].append(position.copy())
    return env, paths


def draw_trajectory(axis: Axes, agent: str, path: list[np.ndarray]) -> None:
    """Overlay one agent path on an existing world plot."""

    points = compress_path(path)
    color = AGENT_COLORS[agent]
    axis.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        linewidth=3.0,
        marker="o",
        markersize=4,
        zorder=7,
    )
    draw_agent(axis, agent, points[0])
    if len(points) > 1:
        axis.annotate(
            "",
            xy=points[-1],
            xytext=points[-2],
            arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 2.5},
            zorder=8,
        )


def draw_agent(axis: Axes, agent: str, position: np.ndarray) -> None:
    x, y = map(float, position)
    label = "A0" if agent == "agent_0" else "A1"
    axis.scatter(
        x,
        y,
        s=180,
        color=AGENT_COLORS[agent],
        edgecolor="white",
        linewidth=2,
        zorder=9,
    )
    axis.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        color="white",
        fontsize=8,
        weight="bold",
        zorder=10,
    )


def compress_path(path: list[np.ndarray]) -> np.ndarray:
    unique = [path[0]]
    for point in path[1:]:
        if not np.array_equal(point, unique[-1]):
            unique.append(point)
    return np.stack(unique)


def world_legend() -> list[Patch | Line2D]:
    return [
        Patch(facecolor=REGION_COLORS[0], label="west"),
        Patch(facecolor=REGION_COLORS[1], label="east"),
        Line2D(
            [0], [0], marker="*", color="none", markerfacecolor=STAG_COLORS[0],
            markeredgecolor="black", markersize=11, label="red stag"
        ),
        Line2D(
            [0], [0], marker="*", color="none", markerfacecolor=STAG_COLORS[1],
            markeredgecolor="black", markersize=11, label="blue stag"
        ),
        Line2D(
            [0], [0], marker="h", color="none", markerfacecolor="#9c6644",
            markeredgecolor="black", markersize=9, label="hare"
        ),
    ]


def trajectory_legend() -> list[Line2D]:
    return [
        Line2D([0], [0], color=AGENT_COLORS["agent_0"], marker="o", label="A0"),
        Line2D([0], [0], color=AGENT_COLORS["agent_1"], marker="o", label="A1"),
        Line2D(
            [0], [0], marker="*", color="none", markerfacecolor="#d1495b",
            markeredgecolor="black", markersize=11, label="stag"
        ),
        Line2D(
            [0], [0], marker="h", color="none", markerfacecolor="#9c6644",
            markeredgecolor="black", markersize=9, label="hare"
        ),
    ]


if __name__ == "__main__":
    main()
