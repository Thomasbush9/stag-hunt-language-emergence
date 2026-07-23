from __future__ import annotations

import torch

from stag_hunt_lang import EnvConfig, StagHuntLanguageEnv
from stag_hunt_lang.models import CentralizedCritic, RecurrentActor, actions_for_env
from stag_hunt_lang.observations import (
    batch_observations,
    global_state_size,
    observation_vector_size,
)


def test_observation_batch_has_declared_width() -> None:
    config = EnvConfig()
    env = StagHuntLanguageEnv(config)
    observations, _ = env.reset(seed=10)

    encoded = batch_observations(observations, env.agents, config)

    assert encoded.shape == (2, observation_vector_size(config))
    assert encoded.dtype == torch.float32
    assert torch.isfinite(encoded).all()


def test_recurrent_actor_samples_valid_joint_actions_and_backpropagates() -> None:
    config = EnvConfig()
    env = StagHuntLanguageEnv(config)
    observations, _ = env.reset(seed=11)
    encoded = batch_observations(observations, env.agents, config)

    actor = RecurrentActor(config, hidden_size=32)
    hidden = actor.initial_state(len(env.agents))
    output = actor.act(encoded, hidden)
    actions = actions_for_env(output, env.agents)

    assert all(env.action_space(agent).contains(actions[agent]) for agent in env.agents)
    assert output.log_prob.shape == (2,)
    assert output.entropy.shape == (2,)
    assert output.hidden.shape == (2, 32)

    loss = -output.log_prob.mean() - 0.01 * output.entropy.mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in actor.parameters())


def test_centralized_critic_accepts_environment_state() -> None:
    config = EnvConfig()
    env = StagHuntLanguageEnv(config)
    env.reset(seed=12)
    state = torch.from_numpy(env.state()).unsqueeze(0)
    critic = CentralizedCritic(config, hidden_size=32)

    value = critic(state)

    assert state.shape == (1, global_state_size(config))
    assert value.shape == (1,)

