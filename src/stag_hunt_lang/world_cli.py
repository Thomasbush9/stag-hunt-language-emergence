"""Inspect the world and replay scripted outcomes before training agents."""

from __future__ import annotations

import argparse

import numpy as np

from stag_hunt_lang import EnvConfig, Move, StagHuntLanguageEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=("reset", "joint-stag", "hare", "failed-stag"),
        default="reset",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    env = StagHuntLanguageEnv(EnvConfig())
    observations, _ = env.reset(seed=args.seed)
    print(env.render())

    print("\nWhat the policies actually receive:")
    for agent, observation in observations.items():
        print(
            f"  {agent}: private_clue={observation['private_clue'].tolist()}, "
            f"received_message={observation['received_message']}, "
            f"self={observation['self_position'].tolist()}, "
            f"other={observation['other_position'].tolist()}"
        )

    if args.scenario == "reset":
        return

    print(
        "\nScripted demonstration only: message meanings below are assigned by the "
        "script; learned agents receive no token semantics."
    )
    for step in range(args.max_steps):
        agents = env.agents.copy()
        actions = scripted_actions(env, args.scenario)
        _, rewards, terminations, truncations, _ = env.step(actions)
        print(f"\nSTEP {step + 1}: actions={actions} rewards={rewards}")
        print(env.render())
        if all(terminations.values()) or any(truncations.values()) or not env.agents:
            break
        if set(actions) != set(agents):
            raise RuntimeError("script did not act for every live agent")


def scripted_actions(
    env: StagHuntLanguageEnv, scenario: str
) -> dict[str, dict[str, int]]:
    positions = env.agent_positions
    messages = {
        "agent_0": 1 + env.correct_color,
        "agent_1": 1 + env.correct_region,
    }

    if scenario == "joint-stag":
        targets = {agent: env.correct_target_position for agent in env.possible_agents}
        interact_when_arrived = {agent: True for agent in env.possible_agents}
    elif scenario == "failed-stag":
        targets = {
            "agent_0": env.correct_target_position,
            "agent_1": positions["agent_1"],
        }
        interact_when_arrived = {"agent_0": True, "agent_1": False}
    elif scenario == "hare":
        targets = {
            "agent_0": env.hare_positions[0],
            "agent_1": positions["agent_1"],
        }
        interact_when_arrived = {"agent_0": True, "agent_1": False}
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    actions = {}
    all_arrived = all(np.array_equal(positions[agent], targets[agent]) for agent in env.agents)
    for agent in env.agents:
        if all_arrived and interact_when_arrived[agent]:
            move = Move.INTERACT
        else:
            move = move_toward(positions[agent], targets[agent])
        actions[agent] = {"move": int(move), "message": messages[agent]}
    return actions


def move_toward(position: np.ndarray, target: np.ndarray) -> Move:
    x, y = map(int, position)
    target_x, target_y = map(int, target)
    if x < target_x:
        return Move.RIGHT
    if x > target_x:
        return Move.LEFT
    if y < target_y:
        return Move.DOWN
    if y > target_y:
        return Move.UP
    return Move.STAY


if __name__ == "__main__":
    main()

