"""Convert structured environment observations into PyTorch tensors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from stag_hunt_lang.config import EnvConfig


def observation_vector_size(config: EnvConfig) -> int:
    """Return the width produced by :func:`encode_observation`."""

    return (
        len(("agent_0", "agent_1"))
        + 2
        + 2
        + config.n_stags * 2
        + config.n_stags * 2
        + config.n_hares * 2
        + 2
        + config.vocab_size
        + 1
        + 1
    )


def encode_observation_np(observation: Mapping[str, Any], config: EnvConfig) -> "np.ndarray":
    """NumPy twin of :func:`encode_observation` for fast CPU-side collection.

    Must stay field-for-field identical to the torch encoder; the test suite
    asserts exact equality.
    """

    agent_id = np.zeros(2, dtype=np.float32)
    agent_id[int(observation["agent_id"])] = 1.0
    grid = config.grid_size - 1
    self_position = np.asarray(observation["self_position"], dtype=np.float32) / grid
    other_position = np.asarray(observation["other_position"], dtype=np.float32) / grid
    stag_positions = np.asarray(observation["stag_positions"], dtype=np.float32).ravel() / grid
    feature_scale = np.asarray(
        [config.n_colors - 1, config.n_regions - 1], dtype=np.float32
    )
    stag_features = (
        np.asarray(observation["stag_features"], dtype=np.float32) / feature_scale
    ).ravel()
    hare_positions = np.asarray(observation["hare_positions"], dtype=np.float32).ravel() / grid
    clue_scale = np.asarray([config.n_colors, config.n_regions], dtype=np.float32)
    private_clue = np.asarray(observation["private_clue"], dtype=np.float32) / clue_scale
    received = np.zeros(config.vocab_size + 1, dtype=np.float32)
    received[int(observation["received_message"])] = 1.0
    timestep = np.asarray(
        [float(observation["timestep"]) / config.horizon], dtype=np.float32
    )

    encoded = np.concatenate(
        [
            agent_id,
            self_position,
            other_position,
            stag_positions,
            stag_features,
            hare_positions,
            private_clue,
            received,
            timestep,
        ]
    )
    if encoded.shape != (observation_vector_size(config),):
        raise RuntimeError(f"encoded observation has shape {encoded.shape}")
    return encoded


def received_message_slice(config: EnvConfig) -> slice:
    """Locate the received-message one-hot inside the encoded observation."""

    start = (
        len(("agent_0", "agent_1"))
        + 2
        + 2
        + config.n_stags * 2
        + config.n_stags * 2
        + config.n_hares * 2
        + 2
    )
    stop = start + config.vocab_size + 1
    if stop + 1 != observation_vector_size(config):
        raise RuntimeError("received_message_slice is out of sync with encode_observation")
    return slice(start, stop)


def global_state_size(config: EnvConfig) -> int:
    """Return the width of ``StagHuntLanguageEnv.state()``."""

    return (
        11
        + 4 * config.n_stags
        + 2 * config.n_hares
        + (1 if config.randomize_clue_assignment else 0)
        + (2 if config.co_observation_prob > 0.0 else 0)
    )


def encode_observation(
    observation: Mapping[str, Any],
    config: EnvConfig,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Flatten and normalize one structured observation as float32."""

    target_device = torch.device(device)
    agent_id = F.one_hot(
        torch.as_tensor(observation["agent_id"], device=target_device), num_classes=2
    ).to(torch.float32)
    self_position = _float_tensor(observation["self_position"], target_device) / (
        config.grid_size - 1
    )
    other_position = _float_tensor(observation["other_position"], target_device) / (
        config.grid_size - 1
    )
    stag_positions = _float_tensor(observation["stag_positions"], target_device).ravel()
    stag_positions = stag_positions / (config.grid_size - 1)

    stag_features = _float_tensor(observation["stag_features"], target_device)
    feature_scale = torch.tensor(
        [config.n_colors - 1, config.n_regions - 1],
        dtype=torch.float32,
        device=target_device,
    )
    stag_features = (stag_features / feature_scale).ravel()

    hare_positions = _float_tensor(observation["hare_positions"], target_device).ravel()
    hare_positions = hare_positions / (config.grid_size - 1)

    private_clue = _float_tensor(observation["private_clue"], target_device)
    clue_scale = torch.tensor(
        [config.n_colors, config.n_regions], dtype=torch.float32, device=target_device
    )
    private_clue = private_clue / clue_scale

    received_message = F.one_hot(
        torch.as_tensor(observation["received_message"], device=target_device),
        num_classes=config.vocab_size + 1,
    ).to(torch.float32)
    timestep = torch.tensor(
        [float(observation["timestep"]) / config.horizon],
        dtype=torch.float32,
        device=target_device,
    )

    encoded = torch.cat(
        [
            agent_id,
            self_position,
            other_position,
            stag_positions,
            stag_features,
            hare_positions,
            private_clue,
            received_message,
            timestep,
        ]
    )
    expected = observation_vector_size(config)
    if encoded.shape != (expected,):
        raise RuntimeError(f"encoded observation has shape {encoded.shape}, expected {(expected,)}")
    return encoded


def batch_observations(
    observations: Mapping[str, Mapping[str, Any]],
    agents: Sequence[str],
    config: EnvConfig,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Encode observations in an explicit, stable agent order."""

    return torch.stack(
        [encode_observation(observations[agent], config, device=device) for agent in agents]
    )


def _float_tensor(value: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)

