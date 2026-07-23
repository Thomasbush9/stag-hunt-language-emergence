"""Roll out the untrained recurrent PyTorch policy on MPS, CUDA, or CPU."""

from __future__ import annotations

import argparse

import torch

from stag_hunt_lang import EnvConfig, StagHuntLanguageEnv
from stag_hunt_lang.device import resolve_torch_device
from stag_hunt_lang.models import RecurrentActor, actions_for_env
from stag_hunt_lang.observations import batch_observations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--deterministic", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_torch_device(args.device)
    torch.manual_seed(args.seed)

    config = EnvConfig()
    env = StagHuntLanguageEnv(config)
    actor = RecurrentActor(config).to(device)
    actor.eval()

    print(f"device={device}")
    for episode in range(args.episodes):
        observations, _ = env.reset(seed=args.seed + episode)
        hidden = actor.initial_state(len(env.agents), device=device)
        returns = {agent: 0.0 for agent in env.possible_agents}
        steps = 0

        while env.agents:
            agents = env.agents.copy()
            encoded = batch_observations(observations, agents, config, device=device)
            with torch.no_grad():
                output = actor.act(encoded, hidden, deterministic=args.deterministic)
            hidden = output.hidden
            observations, rewards, _, _, _ = env.step(actions_for_env(output, agents))
            for agent, reward in rewards.items():
                returns[agent] += reward
            steps += 1

        print(
            f"episode={episode} steps={steps} outcome={env.outcome} returns={returns}"
        )


if __name__ == "__main__":
    main()
