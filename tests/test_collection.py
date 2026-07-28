"""Vectorized collection: encoder parity and replay determinism."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from stag_hunt_lang import EnvConfig, StagHuntLanguageEnv
from stag_hunt_lang.models import RecurrentActor
from stag_hunt_lang.observations import encode_observation, encode_observation_np

_SPEC = importlib.util.spec_from_file_location(
    "train_mappo", Path(__file__).resolve().parent.parent / "scripts" / "train_mappo.py"
)
train_mappo = importlib.util.module_from_spec(_SPEC)
sys.modules["train_mappo"] = train_mappo
_SPEC.loader.exec_module(train_mappo)  # type: ignore[union-attr]


def test_numpy_encoder_matches_torch_encoder_exactly() -> None:
    for config in (
        EnvConfig(),
        EnvConfig(n_hares=0, capture_mode="presence", observe_other_position=False,
                  randomize_clue_assignment=True),
    ):
        env = StagHuntLanguageEnv(config)
        observations, _ = env.reset(seed=11)
        env.step({a: {"move": 1, "message": 2} for a in env.agents})
        observations = {a: env._observation(a) for a in env.possible_agents}
        for agent in env.possible_agents:
            torch_encoded = encode_observation(observations[agent], config).numpy()
            np_encoded = encode_observation_np(observations[agent], config)
            np.testing.assert_array_equal(torch_encoded, np_encoded)


def test_collect_batch_replays_deterministically() -> None:
    config = EnvConfig(
        n_hares=0, capture_mode="presence", observe_other_position=False,
        randomize_clue_assignment=True, horizon=12,
    )
    torch.manual_seed(0)
    actors = [RecurrentActor(config), RecurrentActor(config)]
    critics = torch.nn.ModuleList(
        [train_mappo.CentralizedCritic(config) for _ in range(2)]
    )
    envs = [StagHuntLanguageEnv(config) for _ in range(6)]
    seeds = list(range(100, 106))

    episodes = train_mappo.collect_batch(
        envs, actors, critics, config, torch.device("cpu"), seeds
    )

    assert len(episodes) == 6
    for episode, seed in zip(episodes, seeds, strict=True):
        # Replay the recorded actions in a fresh env: every stored step must match.
        replay = StagHuntLanguageEnv(config)
        observations, _ = replay.reset(seed=seed)
        for t in range(episode.length):
            for index, agent in enumerate(replay.possible_agents):
                stored = episode.observations[t, index].numpy()
                recomputed = encode_observation_np(observations[agent], config)
                np.testing.assert_array_equal(stored, recomputed)
            np.testing.assert_array_equal(
                episode.states[t].numpy(), replay.state()
            )
            actions = {
                agent: {
                    "move": int(episode.moves[t, index]),
                    "message": int(episode.messages[t, index]),
                }
                for index, agent in enumerate(replay.possible_agents)
            }
            observations, rewards, _, _, _ = replay.step(actions)
            np.testing.assert_allclose(
                episode.rewards[t].numpy(),
                [rewards[a] for a in replay.possible_agents],
            )
        assert replay.outcome == episode.outcome
        assert (replay.outcome == "joint_stag") == bool(
            (episode.rewards.sum() > 0).item()
        )
