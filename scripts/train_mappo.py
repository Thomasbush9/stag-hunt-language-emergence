"""Exploratory MAPPO baseline: shared recurrent actor, per-agent centralized critics.

Collects full episodes, recomputes the GRU forward pass over padded sequences
during the PPO update, and keeps the individual Stag Hunt payoffs (no team
reward collapse). Outputs config, per-update metrics (JSONL), checkpoints, and
summary figures to --output-dir.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from stag_hunt_lang import EnvConfig, StagHuntLanguageEnv
from stag_hunt_lang.device import resolve_torch_device
from stag_hunt_lang.models import CentralizedCritic, RecurrentActor, actions_for_env
from stag_hunt_lang.observations import batch_observations

AGENTS = ("agent_0", "agent_1")


@dataclass(slots=True)
class TrainConfig:
    updates: int = 300
    episodes_per_update: int = 32
    ppo_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    hidden_size: int = 128
    seed: int = 0
    device: str = "auto"
    # Risk curriculum: failed_stag_reward starts at curriculum_start and anneals
    # linearly to the EnvConfig default (0.0) over curriculum_updates updates.
    # curriculum_updates = 0 disables the curriculum.
    curriculum_start: float = 0.0
    curriculum_updates: int = 0
    # Joint proximity shaping: while BOTH agents are within Chebyshev distance 1
    # of the correct stag, both receive proximity_bonus per step (training
    # signal only, never part of the environment). Anneals linearly to zero by
    # proximity_updates. proximity_bonus = 0 disables it.
    proximity_bonus: float = 0.0
    proximity_updates: int = 0
    # Entropy coefficient for the message head; falls back to entropy_coef.
    message_entropy_coef: float | None = None
    # "ppo" (clipped surrogate, GAE, critics) or "reinforce" (single epoch,
    # discounted return-to-go with whitening baseline, critics unused).
    algo: str = "ppo"


@dataclass(slots=True)
class EpisodeBuffer:
    """One episode of trajectories for both agents."""

    observations: torch.Tensor  # [T, 2, obs_dim]
    states: torch.Tensor  # [T, state_dim]
    moves: torch.Tensor  # [T, 2]
    messages: torch.Tensor  # [T, 2]
    log_probs: torch.Tensor  # [T, 2]
    rewards: torch.Tensor  # [T, 2]
    bootstrap_value: torch.Tensor  # [2] value of final state (0 if terminated)
    outcome: str
    length: int


def collect_episode(
    env: StagHuntLanguageEnv,
    actor: RecurrentActor,
    critics: nn.ModuleList,
    config: EnvConfig,
    device: torch.device,
    seed: int | None,
    proximity_bonus: float = 0.0,
) -> EpisodeBuffer:
    observations, _ = env.reset(seed=seed)
    hidden = actor.initial_state(len(AGENTS), device=device)

    obs_steps, state_steps, move_steps, message_steps = [], [], [], []
    log_prob_steps, reward_steps = [], []

    truncated = False
    while env.agents:
        encoded = batch_observations(observations, list(AGENTS), config, device=device)
        state = torch.as_tensor(env.state(), dtype=torch.float32, device=device)
        with torch.no_grad():
            output = actor.act(encoded, hidden)
        hidden = output.hidden

        observations, rewards, _, truncations, _ = env.step(
            actions_for_env(output, list(AGENTS))
        )
        truncated = any(truncations.values())

        if proximity_bonus > 0.0:
            target = env.correct_target_position
            if all(
                np.abs(position - target).max() <= 1
                for position in env.agent_positions.values()
            ):
                for agent in AGENTS:
                    rewards[agent] += proximity_bonus

        obs_steps.append(encoded)
        state_steps.append(state)
        move_steps.append(output.move)
        message_steps.append(output.message)
        log_prob_steps.append(output.log_prob)
        reward_steps.append(
            torch.tensor([rewards[agent] for agent in AGENTS], dtype=torch.float32, device=device)
        )

    if truncated:
        final_state = torch.as_tensor(env.state(), dtype=torch.float32, device=device)
        with torch.no_grad():
            bootstrap = torch.stack([critic(final_state) for critic in critics])
    else:
        bootstrap = torch.zeros(len(AGENTS), device=device)

    return EpisodeBuffer(
        observations=torch.stack(obs_steps),
        states=torch.stack(state_steps),
        moves=torch.stack(move_steps),
        messages=torch.stack(message_steps),
        log_probs=torch.stack(log_prob_steps),
        rewards=torch.stack(reward_steps),
        bootstrap_value=bootstrap,
        outcome=env.outcome,
        length=len(obs_steps),
    )


def pad_episodes(
    episodes: list[EpisodeBuffer], device: torch.device
) -> dict[str, torch.Tensor]:
    batch = len(episodes)
    max_len = max(episode.length for episode in episodes)
    obs_dim = episodes[0].observations.shape[-1]
    state_dim = episodes[0].states.shape[-1]

    out = {
        "observations": torch.zeros(batch, max_len, 2, obs_dim, device=device),
        "states": torch.zeros(batch, max_len, state_dim, device=device),
        "moves": torch.zeros(batch, max_len, 2, dtype=torch.long, device=device),
        "messages": torch.zeros(batch, max_len, 2, dtype=torch.long, device=device),
        "log_probs": torch.zeros(batch, max_len, 2, device=device),
        "rewards": torch.zeros(batch, max_len, 2, device=device),
        "mask": torch.zeros(batch, max_len, device=device),
        "bootstrap": torch.zeros(batch, 2, device=device),
    }
    for index, episode in enumerate(episodes):
        length = episode.length
        out["observations"][index, :length] = episode.observations
        out["states"][index, :length] = episode.states
        out["moves"][index, :length] = episode.moves
        out["messages"][index, :length] = episode.messages
        out["log_probs"][index, :length] = episode.log_probs
        out["rewards"][index, :length] = episode.rewards
        out["mask"][index, :length] = 1.0
        out["bootstrap"][index] = episode.bootstrap_value
    return out


def compute_gae(
    rewards: torch.Tensor,  # [B, T, 2]
    values: torch.Tensor,  # [B, T, 2]
    bootstrap: torch.Tensor,  # [B, 2]
    mask: torch.Tensor,  # [B, T]
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, horizon, _ = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(batch, 2, device=rewards.device)
    next_value = bootstrap
    for t in reversed(range(horizon)):
        step_mask = mask[:, t].unsqueeze(-1)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_advantage = (delta + gamma * gae_lambda * last_advantage) * step_mask
        advantages[:, t] = last_advantage
        next_value = values[:, t] * step_mask + next_value * (1 - step_mask)
    returns = advantages + values
    return advantages, returns


def compute_returns_to_go(
    rewards: torch.Tensor,  # [B, T, 2]
    mask: torch.Tensor,  # [B, T]
    gamma: float,
) -> torch.Tensor:
    batch, horizon, _ = rewards.shape
    returns = torch.zeros_like(rewards)
    running = torch.zeros(batch, 2, device=rewards.device)
    for t in reversed(range(horizon)):
        step_mask = mask[:, t].unsqueeze(-1)
        running = (rewards[:, t] + gamma * running) * step_mask
        returns[:, t] = running
    return returns


def sequence_forward(
    actor: RecurrentActor,
    observations: torch.Tensor,  # [B, T, 2, obs_dim]
    moves: torch.Tensor,
    messages: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-run the GRU over full sequences; agents folded into the batch."""

    batch, horizon = observations.shape[:2]
    flat_obs = observations.permute(0, 2, 1, 3).reshape(batch * 2, horizon, -1)
    flat_moves = moves.permute(0, 2, 1).reshape(batch * 2, horizon)
    flat_messages = messages.permute(0, 2, 1).reshape(batch * 2, horizon)

    hidden = actor.initial_state(batch * 2, device=device)
    log_probs, move_entropies, message_entropies = [], [], []
    for t in range(horizon):
        move_dist, message_dist, hidden = actor.distributions(flat_obs[:, t], hidden)
        log_probs.append(
            move_dist.log_prob(flat_moves[:, t]) + message_dist.log_prob(flat_messages[:, t])
        )
        move_entropies.append(move_dist.entropy())
        message_entropies.append(message_dist.entropy())

    def to_sequence(steps: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(steps, dim=1).reshape(batch, 2, horizon).permute(0, 2, 1)

    return to_sequence(log_probs), to_sequence(move_entropies), to_sequence(message_entropies)


def ppo_update(
    actor: RecurrentActor,
    critics: nn.ModuleList,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    train_config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    mask = batch["mask"]
    mask_agents = mask.unsqueeze(-1).expand(-1, -1, 2)
    n_steps = mask_agents.sum()
    message_coef = (
        train_config.message_entropy_coef
        if train_config.message_entropy_coef is not None
        else train_config.entropy_coef
    )

    with torch.no_grad():
        if train_config.algo == "reinforce":
            # Discounted return-to-go, no bootstrap, whitening as the baseline.
            advantages = compute_returns_to_go(
                batch["rewards"], mask, train_config.gamma
            )
            returns = None
        else:
            values = torch.stack(
                [critic(batch["states"]) for critic in critics], dim=-1
            )  # [B, T, 2]
            advantages, returns = compute_gae(
                batch["rewards"],
                values,
                batch["bootstrap"],
                mask,
                train_config.gamma,
                train_config.gae_lambda,
            )
        flat_adv = advantages[mask_agents.bool()]
        advantages = (advantages - flat_adv.mean()) / (flat_adv.std() + 1e-8)

    epochs = 1 if train_config.algo == "reinforce" else train_config.ppo_epochs
    totals: Counter[str] = Counter()
    for _ in range(epochs):
        log_probs, move_entropy, message_entropy = sequence_forward(
            actor, batch["observations"], batch["moves"], batch["messages"], device
        )
        if train_config.algo == "reinforce":
            policy_loss = -(log_probs * advantages * mask_agents).sum() / n_steps
        else:
            ratio = torch.exp(log_probs - batch["log_probs"])
            surrogate = ratio * advantages
            clipped = torch.clamp(
                ratio, 1 - train_config.clip_range, 1 + train_config.clip_range
            ) * advantages
            policy_loss = -(torch.min(surrogate, clipped) * mask_agents).sum() / n_steps
        entropy_bonus = (
            train_config.entropy_coef * (move_entropy * mask_agents).sum()
            + message_coef * (message_entropy * mask_agents).sum()
        ) / n_steps

        loss = policy_loss - entropy_bonus
        if returns is not None:
            new_values = torch.stack(
                [critic(batch["states"]) for critic in critics], dim=-1
            )
            value_loss = (((new_values - returns) ** 2) * mask_agents).sum() / n_steps
            loss = loss + train_config.value_coef * value_loss
            totals["value_loss"] += float(value_loss.item())

        optimizer.zero_grad()
        loss.backward()
        parameters = list(actor.parameters()) + list(critics.parameters())
        torch.nn.utils.clip_grad_norm_(parameters, train_config.max_grad_norm)
        optimizer.step()

        totals["policy_loss"] += float(policy_loss.item())
        totals["entropy"] += float(
            ((move_entropy + message_entropy) * mask_agents).sum().item() / n_steps.item()
        )

    averaged = {key: value / epochs for key, value in totals.items()}
    averaged.setdefault("value_loss", 0.0)
    return averaged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--episodes-per-update", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--curriculum-start", type=float, default=0.0)
    parser.add_argument("--curriculum-updates", type=int, default=0)
    parser.add_argument("--proximity-bonus", type=float, default=0.0)
    parser.add_argument("--proximity-updates", type=int, default=0)
    parser.add_argument("--message-entropy-coef", type=float, default=None)
    parser.add_argument("--algo", choices=("ppo", "reinforce"), default="ppo")
    parser.add_argument("--hare-reward", type=float, default=2.0)
    parser.add_argument("--commit-window", type=int, default=0)
    args = parser.parse_args()

    train_config = TrainConfig(
        updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        entropy_coef=args.entropy_coef,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        curriculum_start=args.curriculum_start,
        curriculum_updates=args.curriculum_updates,
        proximity_bonus=args.proximity_bonus,
        proximity_updates=args.proximity_updates,
        message_entropy_coef=args.message_entropy_coef,
        algo=args.algo,
    )
    device = torch.device(resolve_torch_device(train_config.device))
    torch.manual_seed(train_config.seed)

    env_config = EnvConfig(hare_reward=args.hare_reward, commit_window=args.commit_window)
    env = StagHuntLanguageEnv(env_config)
    actor = RecurrentActor(env_config, hidden_size=train_config.hidden_size).to(device)
    critics = nn.ModuleList(
        [CentralizedCritic(env_config, hidden_size=train_config.hidden_size) for _ in AGENTS]
    ).to(device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critics.parameters()),
        lr=train_config.learning_rate,
    )

    run_dir = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {"train": asdict(train_config), "env": asdict(env_config), "device": str(device)},
            indent=2,
        )
    )
    metrics_path = run_dir / "metrics.jsonl"
    metrics_file = metrics_path.open("w")

    print(f"device={device} output={run_dir}")
    episode_seed = train_config.seed * 1_000_000
    start = time.time()

    for update in range(train_config.updates):
        if train_config.curriculum_updates > 0:
            progress = min(update / train_config.curriculum_updates, 1.0)
            failed_stag_reward = round(
                train_config.curriculum_start * (1.0 - progress)
                + env_config.failed_stag_reward * progress,
                6,
            )
            if failed_stag_reward != env.config.failed_stag_reward:
                env = StagHuntLanguageEnv(
                    replace(env_config, failed_stag_reward=failed_stag_reward)
                )

        if train_config.proximity_updates > 0:
            proximity_progress = min(update / train_config.proximity_updates, 1.0)
            current_bonus = train_config.proximity_bonus * (1.0 - proximity_progress)
        else:
            current_bonus = train_config.proximity_bonus

        episodes = []
        for _ in range(train_config.episodes_per_update):
            episode_seed += 1
            episodes.append(
                collect_episode(
                    env,
                    actor,
                    critics,
                    env_config,
                    device,
                    episode_seed,
                    proximity_bonus=current_bonus,
                )
            )

        batch = pad_episodes(episodes, device)
        losses = ppo_update(actor, critics, optimizer, batch, train_config, device)

        outcomes = Counter(episode.outcome for episode in episodes)
        returns = np.array(
            [episode.rewards.sum(dim=0).cpu().numpy() for episode in episodes]
        )  # [B, 2]
        messages = torch.cat([episode.messages.reshape(-1) for episode in episodes])
        silence_rate = float((messages == 0).float().mean().item())

        record = {
            "update": update,
            "elapsed_s": round(time.time() - start, 1),
            "failed_stag_reward": env.config.failed_stag_reward,
            "proximity_bonus": round(current_bonus, 6),
            "mean_return_agent_0": float(returns[:, 0].mean()),
            "mean_return_agent_1": float(returns[:, 1].mean()),
            "mean_length": float(np.mean([episode.length for episode in episodes])),
            "outcomes": dict(outcomes),
            "silence_rate": silence_rate,
            **{key: round(value, 4) for key, value in losses.items()},
        }
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()

        if update % 10 == 0 or update == train_config.updates - 1:
            print(
                f"update={update:4d} return=({record['mean_return_agent_0']:.2f}, "
                f"{record['mean_return_agent_1']:.2f}) len={record['mean_length']:.1f} "
                f"outcomes={dict(outcomes)} entropy={losses['entropy']:.3f}"
            )

        if (update + 1) % args.checkpoint_every == 0 or update == train_config.updates - 1:
            torch.save(
                {
                    "actor": actor.state_dict(),
                    "critics": critics.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_config": asdict(train_config),
                    "env_config": asdict(env_config),
                    "update": update,
                },
                run_dir / f"checkpoint_{update + 1:05d}.pt",
            )

    metrics_file.close()
    print(f"done in {time.time() - start:.0f}s; metrics at {metrics_path}")


if __name__ == "__main__":
    main()
