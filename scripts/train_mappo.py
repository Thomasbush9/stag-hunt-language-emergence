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
from stag_hunt_lang.models import CentralizedCritic, RecurrentActor
from stag_hunt_lang.observations import encode_observation_np, received_message_slice

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
    # Positive-signaling bias: adds -coef * MI(private clue; message policy)
    # per speaker role, estimated from the batch. 0 disables.
    signaling_coef: float = 0.0
    # Positive-listening bias: adds -coef * L1(move policy | real messages,
    # move policy | muted messages), pushing the listener to attend to the
    # channel. Costs one extra BPTT forward per epoch. 0 disables.
    listening_coef: float = 0.0
    # Decoupled agents: one RecurrentActor per agent instead of shared weights.
    separate_actors: bool = False
    tied_symbols: bool = False


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
    # Presence-mode targeting diagnostics ("correct"/"wrong"/None, step count).
    first_joint_presence: str | None
    wrong_presence_steps: int
    clue: torch.Tensor  # [2] episode clue: (correct color, correct region)
    color_holder: int  # agent index privately holding the color clue


def collect_batch(
    envs: list[StagHuntLanguageEnv],
    actors: list[RecurrentActor],
    critics: nn.ModuleList,
    config: EnvConfig,
    device: torch.device,
    seeds: list[int],
    proximity_bonus: float = 0.0,
) -> list[EpisodeBuffer]:
    """Collect one episode per env, stepping all envs in lockstep.

    The policy runs one batched forward per agent per timestep (instead of a
    batch-1 call per env), observations are encoded in NumPy with a single
    host-to-device transfer per step, and actions come back with a single
    device-to-host sync per step. Buffers are returned on the CPU;
    ``pad_episodes`` moves them to the device.
    """

    n = len(envs)
    observations: list[dict] = []
    for env, seed in zip(envs, seeds, strict=True):
        obs, _ = env.reset(seed=seed)
        observations.append(obs)

    hidden = [actor.initial_state(n, device=device) for actor in actors]
    steps: list[dict[str, list]] = [
        {"obs": [], "state": [], "move": [], "message": [], "logp": [], "reward": []}
        for _ in range(n)
    ]
    active = list(range(n))
    truncated: set[int] = set()

    while active:
        encoded_np = np.stack(
            [
                np.stack(
                    [
                        encode_observation_np(observations[index][agent], config)
                        for agent in AGENTS
                    ]
                )
                for index in active
            ]
        )  # [n_active, 2, obs_dim]
        states_np = np.stack([envs[index].state() for index in active])
        encoded = torch.from_numpy(encoded_np).to(device)
        rows = torch.as_tensor(active, device=device)

        agent_moves, agent_messages, agent_log_probs = [], [], []
        with torch.no_grad():
            for agent_index, actor in enumerate(actors):
                output = actor.act(encoded[:, agent_index], hidden[agent_index][rows])
                hidden[agent_index][rows] = output.hidden
                agent_moves.append(output.move)
                agent_messages.append(output.message)
                agent_log_probs.append(output.log_prob)
        moves = torch.stack(agent_moves, dim=1).cpu().numpy()  # [n_active, 2]
        messages = torch.stack(agent_messages, dim=1).cpu().numpy()
        log_probs = torch.stack(agent_log_probs, dim=1).cpu()

        still_active = []
        for row, index in enumerate(active):
            env = envs[index]
            observations[index], rewards, _, truncations, _ = env.step(
                {
                    agent: {
                        "move": int(moves[row, agent_index]),
                        "message": int(messages[row, agent_index]),
                    }
                    for agent_index, agent in enumerate(AGENTS)
                }
            )

            if proximity_bonus > 0.0:
                target = env.correct_target_position
                if all(
                    np.abs(position - target).max() <= 1
                    for position in env.agent_positions.values()
                ):
                    for agent in AGENTS:
                        rewards[agent] += proximity_bonus

            record = steps[index]
            record["obs"].append(encoded_np[row])
            record["state"].append(states_np[row])
            record["move"].append(moves[row])
            record["message"].append(messages[row])
            record["logp"].append(log_probs[row])
            record["reward"].append([rewards[agent] for agent in AGENTS])

            if env.agents:
                still_active.append(index)
            elif any(truncations.values()):
                truncated.add(index)
        active = still_active

    bootstraps = {index: torch.zeros(len(AGENTS)) for index in range(n)}
    if truncated:
        order = sorted(truncated)
        finals = torch.as_tensor(
            np.stack([envs[index].state() for index in order]),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            values = torch.stack([critic(finals) for critic in critics], dim=-1).cpu()
        for row, index in enumerate(order):
            bootstraps[index] = values[row]

    episodes = []
    for index, env in enumerate(envs):
        record = steps[index]
        episodes.append(
            EpisodeBuffer(
                observations=torch.from_numpy(np.stack(record["obs"])),
                states=torch.from_numpy(np.stack(record["state"])),
                moves=torch.as_tensor(np.stack(record["move"]), dtype=torch.long),
                messages=torch.as_tensor(np.stack(record["message"]), dtype=torch.long),
                log_probs=torch.stack(record["logp"]),
                rewards=torch.as_tensor(record["reward"], dtype=torch.float32),
                bootstrap_value=bootstraps[index],
                outcome=env.outcome,
                length=len(record["obs"]),
                first_joint_presence=env.first_joint_presence,
                wrong_presence_steps=env.wrong_presence_steps,
                clue=torch.tensor([env.correct_color, env.correct_region], dtype=torch.long),
                color_holder=AGENTS.index(env.color_holder),
            )
        )
    return episodes


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
        "clues": torch.zeros(batch, 2, dtype=torch.long, device=device),
        "color_holder": torch.zeros(batch, dtype=torch.long, device=device),
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
        out["clues"][index] = episode.clue
        out["color_holder"][index] = episode.color_holder
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
    actors: list[RecurrentActor],
    observations: torch.Tensor,  # [B, T, 2, obs_dim]
    moves: torch.Tensor,
    messages: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-run the GRUs over full sequences, one pass per agent.

    Returns per-step log-probs and head entropies [B, T, 2] plus the full
    move/message probability tensors [B, T, 2, n_actions] for the
    communication bias losses. With shared weights the two passes are
    mathematically identical to folding both agents into one batch.
    """

    batch, horizon = observations.shape[:2]
    per_agent: list[list[list[torch.Tensor]]] = []
    for index, actor in enumerate(actors):
        agent_obs = observations[:, :, index, :]
        agent_moves = moves[:, :, index]
        agent_messages = messages[:, :, index]
        hidden = actor.initial_state(batch, device=device)
        steps: list[list[torch.Tensor]] = [[], [], [], [], []]
        for t in range(horizon):
            move_dist, message_dist, hidden = actor.distributions(agent_obs[:, t], hidden)
            steps[0].append(
                move_dist.log_prob(agent_moves[:, t])
                + message_dist.log_prob(agent_messages[:, t])
            )
            steps[1].append(move_dist.entropy())
            steps[2].append(message_dist.entropy())
            steps[3].append(move_dist.probs)
            steps[4].append(message_dist.probs)
        per_agent.append(steps)

    def stack(slot: int) -> torch.Tensor:
        # [B, T] per agent -> [B, T, 2]; [B, T, n] per agent -> [B, T, 2, n]
        agent_tensors = [torch.stack(agent[slot], dim=1) for agent in per_agent]
        return torch.stack(agent_tensors, dim=2)

    return stack(0), stack(1), stack(2), stack(3), stack(4)


def clue_message_mi(
    probs: torch.Tensor,  # [B, T, vocab+1] — message policy of the attribute holder
    values: torch.Tensor,  # [B] attribute value per episode
    mask: torch.Tensor,  # [B, T]
    n_values: int,
) -> torch.Tensor:
    """Batch estimate of MI(held attribute; message policy) for one attribute.

    MI = H(marginal message dist) - sum_c w_c H(message dist | value = c),
    computed from the policy's step-wise probabilities (differentiable).
    """

    eps = 1e-8
    weights = mask.unsqueeze(-1)
    total = mask.sum().clamp_min(1.0)

    marginal = (probs * weights).sum(dim=(0, 1)) / total
    marginal_entropy = -(marginal * (marginal + eps).log()).sum()

    conditional_entropy = torch.zeros((), device=probs.device)
    for value in range(n_values):
        selected = (values == value).float().unsqueeze(1) * mask  # [B, T]
        count = selected.sum()
        if count.item() == 0:
            continue
        conditional = (probs * selected.unsqueeze(-1)).sum(dim=(0, 1)) / count
        entropy = -(conditional * (conditional + eps).log()).sum()
        conditional_entropy = conditional_entropy + (count / total) * entropy

    return marginal_entropy - conditional_entropy


def ppo_update(
    actors: list[RecurrentActor],
    critics: nn.ModuleList,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    train_config: TrainConfig,
    env_config: EnvConfig,
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
    message_slice = received_message_slice(env_config)
    for _ in range(epochs):
        log_probs, move_entropy, message_entropy, move_probs, message_probs = sequence_forward(
            actors, batch["observations"], batch["moves"], batch["messages"], device
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

        if train_config.signaling_coef > 0.0:
            # Group the MI estimate by which agent held each attribute this
            # episode (fixed assignment reduces to the old per-role grouping).
            batch_index = torch.arange(message_probs.shape[0], device=device)
            holder = batch["color_holder"]
            color_probs = message_probs[batch_index, :, holder, :]
            region_probs = message_probs[batch_index, :, 1 - holder, :]
            signaling_mi = clue_message_mi(
                color_probs, batch["clues"][:, 0], mask, env_config.n_colors
            ) + clue_message_mi(
                region_probs, batch["clues"][:, 1], mask, env_config.n_regions
            )
            loss = loss - train_config.signaling_coef * signaling_mi
            totals["signaling_mi_bits"] += float(signaling_mi.item()) / float(np.log(2))

        if train_config.listening_coef > 0.0:
            muted = batch["observations"].clone()
            muted[..., message_slice] = 0.0
            muted[..., message_slice.start] = 1.0  # one-hot silence
            _, _, _, muted_move_probs, _ = sequence_forward(
                actors, muted, batch["moves"], batch["messages"], device
            )
            divergence = (move_probs - muted_move_probs).abs().sum(-1)  # [B, T, 2]
            listening = (divergence * mask_agents).sum() / n_steps
            loss = loss - train_config.listening_coef * listening
            totals["listening_l1"] += float(listening.item())

        if returns is not None:
            new_values = torch.stack(
                [critic(batch["states"]) for critic in critics], dim=-1
            )
            value_loss = (((new_values - returns) ** 2) * mask_agents).sum() / n_steps
            loss = loss + train_config.value_coef * value_loss
            totals["value_loss"] += float(value_loss.item())

        optimizer.zero_grad()
        loss.backward()
        parameters = [p for actor in dict.fromkeys(actors) for p in actor.parameters()]
        parameters += list(critics.parameters())
        torch.nn.utils.clip_grad_norm_(parameters, train_config.max_grad_norm)
        optimizer.step()

        totals["policy_loss"] += float(policy_loss.item())
        totals["entropy"] += float(
            ((move_entropy + message_entropy) * mask_agents).sum().item() / n_steps.item()
        )

    averaged = {key: value / epochs for key, value in totals.items()}
    averaged.setdefault("value_loss", 0.0)
    averaged.setdefault("signaling_mi_bits", 0.0)
    averaged.setdefault("listening_l1", 0.0)
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
    parser.add_argument("--n-hares", type=int, default=2)
    parser.add_argument("--capture-mode", choices=("interact", "presence"), default="interact")
    parser.add_argument("--hide-other-position", action="store_true")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--sticky-messages", action="store_true")
    parser.add_argument("--talk-phase-steps", type=int, default=0)
    parser.add_argument("--signaling-coef", type=float, default=0.0)
    parser.add_argument("--listening-coef", type=float, default=0.0)
    parser.add_argument("--separate-actors", action="store_true")
    parser.add_argument("--randomize-clues", action="store_true")
    parser.add_argument("--n-colors", type=int, default=2)
    parser.add_argument("--n-regions", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=4)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--co-observation-prob", type=float, default=0.0)
    parser.add_argument("--tied-symbols", action="store_true")
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
        signaling_coef=args.signaling_coef,
        listening_coef=args.listening_coef,
        separate_actors=args.separate_actors,
        tied_symbols=args.tied_symbols,
    )
    device = torch.device(resolve_torch_device(train_config.device))
    torch.manual_seed(train_config.seed)

    env_config = EnvConfig(
        hare_reward=args.hare_reward,
        commit_window=args.commit_window,
        n_hares=args.n_hares,
        capture_mode=args.capture_mode,
        observe_other_position=not args.hide_other_position,
        horizon=args.horizon,
        sticky_messages=args.sticky_messages,
        talk_phase_steps=args.talk_phase_steps,
        randomize_clue_assignment=args.randomize_clues,
        n_colors=args.n_colors,
        n_regions=args.n_regions,
        vocab_size=args.vocab_size,
        co_observation_prob=args.co_observation_prob,
    )
    envs = [StagHuntLanguageEnv(env_config) for _ in range(train_config.episodes_per_update)]
    def build_actor() -> RecurrentActor:
        return RecurrentActor(
            env_config,
            hidden_size=train_config.hidden_size,
            tied_symbols=train_config.tied_symbols,
        ).to(device)

    if train_config.separate_actors:
        actors = [build_actor() for _ in AGENTS]
    else:
        shared = build_actor()
        actors = [shared, shared]
    critics = nn.ModuleList(
        [CentralizedCritic(env_config, hidden_size=train_config.hidden_size) for _ in AGENTS]
    ).to(device)
    actor_parameters = [p for actor in dict.fromkeys(actors) for p in actor.parameters()]
    optimizer = torch.optim.Adam(
        actor_parameters + list(critics.parameters()),
        lr=train_config.learning_rate,
    )

    if args.init_checkpoint is not None:
        state = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        if "actors" in state:
            for actor, actor_state in zip(dict.fromkeys(actors), state["actors"]):
                actor.load_state_dict(actor_state)
        else:
            actors[0].load_state_dict(state["actor"])
        critics.load_state_dict(state["critics"])
        optimizer.load_state_dict(state["optimizer"])
        print(f"warm-started from {args.init_checkpoint} (saved at update {state['update'] + 1})")

    run_dir = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train": asdict(train_config),
                "env": asdict(env_config),
                "device": str(device),
                "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
            },
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
            if failed_stag_reward != envs[0].config.failed_stag_reward:
                shifted = replace(env_config, failed_stag_reward=failed_stag_reward)
                envs = [
                    StagHuntLanguageEnv(shifted)
                    for _ in range(train_config.episodes_per_update)
                ]

        if train_config.proximity_updates > 0:
            proximity_progress = min(update / train_config.proximity_updates, 1.0)
            current_bonus = train_config.proximity_bonus * (1.0 - proximity_progress)
        else:
            current_bonus = train_config.proximity_bonus

        seeds = [episode_seed + 1 + offset for offset in range(train_config.episodes_per_update)]
        episode_seed += train_config.episodes_per_update
        episodes = collect_batch(
            envs,
            actors,
            critics,
            env_config,
            device,
            seeds,
            proximity_bonus=current_bonus,
        )

        batch = pad_episodes(episodes, device)
        losses = ppo_update(actors, critics, optimizer, batch, train_config, env_config, device)

        outcomes = Counter(episode.outcome for episode in episodes)
        returns = np.array(
            [episode.rewards.sum(dim=0).cpu().numpy() for episode in episodes]
        )  # [B, 2]
        messages = torch.cat([episode.messages.reshape(-1) for episode in episodes])
        silence_rate = float((messages == 0).float().mean().item())
        # One-shot targeting: of the episodes whose first joint stag presence
        # happened at all, how many aimed at the correct stag first? Chance is
        # 1/n_stags with independent clue-consistent guessing.
        first_presences = Counter(
            episode.first_joint_presence
            for episode in episodes
            if episode.first_joint_presence is not None
        )
        attempted = sum(first_presences.values())

        record = {
            "update": update,
            "elapsed_s": round(time.time() - start, 1),
            "failed_stag_reward": envs[0].config.failed_stag_reward,
            "proximity_bonus": round(current_bonus, 6),
            "mean_return_agent_0": float(returns[:, 0].mean()),
            "mean_return_agent_1": float(returns[:, 1].mean()),
            "mean_length": float(np.mean([episode.length for episode in episodes])),
            "outcomes": dict(outcomes),
            "silence_rate": silence_rate,
            "first_presence_counts": dict(first_presences),
            "first_presence_accuracy": (
                round(first_presences.get("correct", 0) / attempted, 4) if attempted else None
            ),
            "mean_wrong_presence_steps": float(
                np.mean([episode.wrong_presence_steps for episode in episodes])
            ),
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
            if train_config.separate_actors:
                actor_state = {"actors": [actor.state_dict() for actor in actors]}
            else:
                actor_state = {"actor": actors[0].state_dict()}
            torch.save(
                {
                    **actor_state,
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
