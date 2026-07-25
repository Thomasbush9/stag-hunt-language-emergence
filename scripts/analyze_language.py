"""Probe a trained checkpoint for emergent language.

Measures speaker informativeness (mutual information between each agent's
private clue and its sent messages), tracks it across checkpoints, and runs the
causal channel interventions from docs/experiment.md: intact, muted, and
randomized received messages.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from stag_hunt_lang import EnvConfig, StagHuntLanguageEnv
from stag_hunt_lang.device import resolve_torch_device
from stag_hunt_lang.models import RecurrentActor, actions_for_env
from stag_hunt_lang.observations import batch_observations

AGENTS = ("agent_0", "agent_1")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5c5b54"
GRID = "#e6e5e0"
OUTCOMES = ["joint_stag", "hare", "failed_stag", "timeout", "hare_stag_mismatch"]


def load_actor(path: Path, env_config: EnvConfig, device: torch.device) -> RecurrentActor:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    hidden_size = checkpoint["train_config"]["hidden_size"]
    actor = RecurrentActor(env_config, hidden_size=hidden_size).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor


def load_env_config(path: Path, device: torch.device) -> EnvConfig:
    """Evaluate in the run's own environment (hare reward, commit window, ...)."""

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    stored = dict(checkpoint.get("env_config", {}))
    known = {field for field in EnvConfig.__dataclass_fields__}
    return EnvConfig(**{key: value for key, value in stored.items() if key in known})


def run_eval(
    actor: RecurrentActor,
    env_config: EnvConfig,
    device: torch.device,
    episodes: int,
    seed_base: int,
    intervention: str,
    rng: np.random.Generator,
) -> dict:
    """Roll out episodes; optionally mute or randomize the received messages."""

    env = StagHuntLanguageEnv(env_config)
    outcomes: Counter[str] = Counter()
    returns = np.zeros((episodes, 2))
    # message counts conditioned on the sender's private clue
    color_message = np.zeros((env_config.n_colors, env_config.vocab_size + 1))
    region_message = np.zeros((env_config.n_regions, env_config.vocab_size + 1))

    for episode in range(episodes):
        observations, _ = env.reset(seed=seed_base + episode)
        color, region = env.correct_color, env.correct_region
        hidden = actor.initial_state(len(AGENTS), device=device)

        while env.agents:
            if intervention == "muted":
                for agent in AGENTS:
                    observations[agent]["received_message"] = 0
            elif intervention == "random":
                for agent in AGENTS:
                    observations[agent]["received_message"] = int(
                        rng.integers(env_config.vocab_size + 1)
                    )
            encoded = batch_observations(observations, list(AGENTS), env_config, device=device)
            with torch.no_grad():
                output = actor.act(encoded, hidden)
            hidden = output.hidden
            observations, rewards, _, _, _ = env.step(actions_for_env(output, list(AGENTS)))

            color_message[color, int(output.message[0].item())] += 1
            region_message[region, int(output.message[1].item())] += 1
            for index, agent in enumerate(AGENTS):
                returns[episode, index] += rewards[agent]

        outcomes[env.outcome] += 1

    return {
        "outcomes": dict(outcomes),
        "mean_return_agent_0": float(returns[:, 0].mean()),
        "mean_return_agent_1": float(returns[:, 1].mean()),
        "color_message": color_message,
        "region_message": region_message,
        "episodes": episodes,
    }


def mutual_information(counts: np.ndarray) -> float:
    """Plug-in MI in bits between the row variable and column variable."""

    total = counts.sum()
    if total == 0:
        return 0.0
    joint = counts / total
    row = joint.sum(axis=1, keepdims=True)
    col = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = joint * np.log2(joint / (row @ col))
    return float(np.nansum(terms))


def style_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left", pad=10)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def heatmap(ax: plt.Axes, matrix: np.ndarray, row_labels: list[str], title: str) -> None:
    probabilities = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    image = ax.imshow(probabilities, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(["silence"] + [f"m{index}" for index in range(1, matrix.shape[1])])
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(row_labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = probabilities[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 0.5 else TEXT_PRIMARY,
            )
    style_axis(ax, title)
    plt.colorbar(image, ax=ax, fraction=0.04, pad=0.02)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=10_000_000)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()

    device = torch.device(resolve_torch_device(args.device))
    rng = np.random.default_rng(args.seed)

    checkpoints = sorted(args.run_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        raise SystemExit(f"no checkpoints in {args.run_dir}")
    env_config = load_env_config(checkpoints[-1], device)
    env = StagHuntLanguageEnv(env_config)

    # MI trajectory across checkpoints (intact channel)
    mi_by_checkpoint = []
    for path in checkpoints:
        actor = load_actor(path, env_config, device)
        result = run_eval(actor, env_config, device, args.episodes, args.seed, "none", rng)
        mi_by_checkpoint.append(
            {
                "checkpoint": path.stem,
                "mi_color_agent0": mutual_information(result["color_message"]),
                "mi_region_agent1": mutual_information(result["region_message"]),
            }
        )
        print(f"{path.stem}: {mi_by_checkpoint[-1]}")

    # interventions on the final checkpoint
    actor = load_actor(checkpoints[-1], env_config, device)
    conditions = {}
    for intervention in ("none", "muted", "random"):
        conditions[intervention] = run_eval(
            actor, env_config, device, args.episodes, args.seed, intervention, rng
        )
        summary = {
            key: conditions[intervention][key]
            for key in ("outcomes", "mean_return_agent_0", "mean_return_agent_1")
        }
        print(f"intervention={intervention}: {summary}")

    final = conditions["none"]

    report = {
        "episodes_per_condition": args.episodes,
        "mi_by_checkpoint": mi_by_checkpoint,
        "chance_mi_bits": {
            "color": float(np.log2(env_config.n_colors)),
            "region": float(np.log2(env_config.n_regions)),
        },
        "interventions": {
            name: {
                "outcomes": result["outcomes"],
                "mean_return_agent_0": result["mean_return_agent_0"],
                "mean_return_agent_1": result["mean_return_agent_1"],
            }
            for name, result in conditions.items()
        },
    }
    (args.run_dir / "language_analysis.json").write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="white")
    fig.subplots_adjust(hspace=0.4, wspace=0.35)

    heatmap(
        axes[0][0],
        final["color_message"],
        [env.color_name(color) for color in range(env_config.n_colors)],
        "agent_0: P(message | private colour clue)",
    )
    heatmap(
        axes[0][1],
        final["region_message"],
        [env.region_name(region) for region in range(env_config.n_regions)],
        "agent_1: P(message | private region clue)",
    )

    ax = axes[1][0]
    steps = [int(entry["checkpoint"].split("_")[1]) for entry in mi_by_checkpoint]
    ax.plot(
        steps,
        [entry["mi_color_agent0"] for entry in mi_by_checkpoint],
        color=SERIES[0],
        linewidth=2,
        marker="o",
        markersize=5,
        label="agent_0: MI(colour; message)",
    )
    ax.plot(
        steps,
        [entry["mi_region_agent1"] for entry in mi_by_checkpoint],
        color=SERIES[1],
        linewidth=2,
        marker="o",
        markersize=5,
        label="agent_1: MI(region; message)",
    )
    ax.axhline(1.0, color=GRID, linewidth=1)
    ax.annotate(
        "perfect 1-bit clue transmission",
        xy=(steps[-1], 1.0),
        fontsize=8,
        color=TEXT_SECONDARY,
        va="bottom",
        ha="right",
    )
    ax.set_ylim(bottom=0)
    ax.set_xlabel("training update", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("mutual information (bits)", color=TEXT_SECONDARY, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    style_axis(ax, "Speaker informativeness across training")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY)

    ax = axes[1][1]
    condition_names = list(conditions)
    width = 0.25
    positions = np.arange(len(OUTCOMES))
    for index, name in enumerate(condition_names):
        result = conditions[name]
        rates = [
            result["outcomes"].get(outcome, 0) / result["episodes"] for outcome in OUTCOMES
        ]
        ax.bar(
            positions + (index - 1) * width,
            rates,
            width=width * 0.92,
            color=SERIES[index],
            label=f"channel: {name}",
            edgecolor="white",
            linewidth=1,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(OUTCOMES, fontsize=8)
    ax.set_ylabel("fraction of episodes", color=TEXT_SECONDARY, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    style_axis(ax, "Causal test: outcomes under channel intervention")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY)

    output = args.run_dir / "language_analysis.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
