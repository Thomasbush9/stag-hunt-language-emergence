"""Small random-rollout CLI for environment smoke testing."""

from __future__ import annotations

import argparse

from stag_hunt_lang import EnvConfig, StagHuntLanguageEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    env = StagHuntLanguageEnv(
        EnvConfig(), render_mode="ansi" if args.render else None
    )

    for episode in range(args.episodes):
        _, _ = env.reset(seed=args.seed + episode)
        returns = {agent: 0.0 for agent in env.possible_agents}
        steps = 0
        while env.agents:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            _, rewards, _, _, _ = env.step(actions)
            for agent, reward in rewards.items():
                returns[agent] += reward
            steps += 1
        print(
            f"episode={episode} steps={steps} outcome={env.outcome} returns={returns}"
        )

    env.close()


if __name__ == "__main__":
    main()

