from __future__ import annotations

import torch

from stag_hunt_lang import EnvConfig, StagHuntLanguageEnv
from stag_hunt_lang.models import CentralizedCritic, RecurrentActor, actions_for_env
from stag_hunt_lang.observations import (
    batch_observations,
    global_state_size,
    observation_vector_size,
    received_message_slice,
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



def test_tied_symbols_share_one_embedding_for_speaking_and_hearing() -> None:
    config = EnvConfig(vocab_size=4)
    actor = RecurrentActor(config, hidden_size=32, tied_symbols=True)
    assert not hasattr(actor, "message_head")

    observations = torch.zeros(2, observation_vector_size(config))
    hidden = actor.initial_state(2)
    output = actor.act(observations, hidden)
    assert output.message.shape == (2,)

    # The shared embedding carries gradient from BOTH pathways: producing a
    # message and hearing one must move the same parameter.
    _, message_distribution, _ = actor.distributions(observations, hidden)
    message_distribution.logits.sum().backward()
    speak_grad = actor.symbol_embedding.grad.clone()
    actor.zero_grad()

    heard = observations.clone()
    heard[:, received_message_slice(config)] = 0.0
    heard[:, received_message_slice(config).start + 2] = 1.0
    move_distribution, _, _ = actor.distributions(heard, hidden)
    move_distribution.logits.sum().backward()
    hear_grad = actor.symbol_embedding.grad

    assert speak_grad.abs().sum() > 0
    assert hear_grad.abs().sum() > 0


def test_untied_actor_keeps_independent_message_head() -> None:
    config = EnvConfig(vocab_size=4)
    actor = RecurrentActor(config, hidden_size=32)
    assert hasattr(actor, "message_head")
    assert not hasattr(actor, "symbol_embedding")
    output = actor.act(torch.zeros(2, observation_vector_size(config)), actor.initial_state(2))
    assert output.message.shape == (2,)
