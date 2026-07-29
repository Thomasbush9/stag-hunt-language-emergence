"""PyTorch modules for recurrent decentralized actors and centralized critics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from stag_hunt_lang.config import EnvConfig
from stag_hunt_lang.env import Move
from stag_hunt_lang.observations import (
    global_state_size,
    observation_vector_size,
    received_message_slice,
)


@dataclass(slots=True)
class PolicyOutput:
    """One batched policy step for physical and communication actions."""

    move: torch.Tensor
    message: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    hidden: torch.Tensor


class RecurrentActor(nn.Module):
    """A shared recurrent actor with separate move and message distributions.

    With ``tied_symbols`` the same symbol-embedding matrix is used to produce
    messages and to encode heard ones (as in tied input/output embeddings for
    language models). Learning to *say* a symbol informatively then also moves
    the representation used to *hear* it, so production and comprehension stop
    being independent problems — a motor-theory-of-perception scaffold that
    lives in the architecture rather than in the loss.
    """

    def __init__(
        self,
        config: EnvConfig,
        hidden_size: int = 128,
        *,
        tied_symbols: bool = False,
        symbol_dim: int = 32,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.tied_symbols = tied_symbols
        self.encoder = nn.Sequential(
            nn.Linear(observation_vector_size(config), hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
        )
        self.memory = nn.GRUCell(hidden_size, hidden_size)
        self.move_head = nn.Linear(hidden_size, len(Move))
        if tied_symbols:
            self.message_slice = received_message_slice(config)
            self.symbol_embedding = nn.Parameter(
                torch.randn(config.vocab_size + 1, symbol_dim) * 0.1
            )
            self.speak_projection = nn.Linear(hidden_size, symbol_dim)
            self.hear_projection = nn.Linear(symbol_dim, hidden_size)
            self.message_bias = nn.Parameter(torch.zeros(config.vocab_size + 1))
        else:
            self.message_head = nn.Linear(hidden_size, config.vocab_size + 1)

    def _encode(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode observations, routing heard symbols through the tied embedding."""

        if not self.tied_symbols:
            return self.encoder(observations)
        heard = observations[..., self.message_slice]
        masked = observations.clone()
        masked[..., self.message_slice] = 0.0
        return self.encoder(masked) + self.hear_projection(heard @ self.symbol_embedding)

    def _message_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.tied_symbols:
            return self.message_head(hidden)
        return self.speak_projection(hidden) @ self.symbol_embedding.t() + self.message_bias

    def initial_state(
        self, batch_size: int, *, device: str | torch.device | None = None
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def distributions(
        self, observations: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[Categorical, Categorical, torch.Tensor]:
        encoded = self._encode(observations)
        next_hidden = self.memory(encoded, hidden)
        return (
            Categorical(logits=self.move_head(next_hidden)),
            Categorical(logits=self._message_logits(next_hidden)),
            next_hidden,
        )

    def act(
        self,
        observations: torch.Tensor,
        hidden: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> PolicyOutput:
        move_distribution, message_distribution, next_hidden = self.distributions(
            observations, hidden
        )
        if deterministic:
            move = move_distribution.logits.argmax(dim=-1)
            message = message_distribution.logits.argmax(dim=-1)
        else:
            move = move_distribution.sample()
            message = message_distribution.sample()

        return PolicyOutput(
            move=move,
            message=message,
            log_prob=move_distribution.log_prob(move) + message_distribution.log_prob(message),
            entropy=move_distribution.entropy() + message_distribution.entropy(),
            hidden=next_hidden,
        )

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        hidden: torch.Tensor,
        move: torch.Tensor,
        message: torch.Tensor,
    ) -> PolicyOutput:
        """Evaluate fixed rollout actions for a future PPO update."""

        move_distribution, message_distribution, next_hidden = self.distributions(
            observations, hidden
        )
        return PolicyOutput(
            move=move,
            message=message,
            log_prob=move_distribution.log_prob(move) + message_distribution.log_prob(message),
            entropy=move_distribution.entropy() + message_distribution.entropy(),
            hidden=next_hidden,
        )


class CentralizedCritic(nn.Module):
    """Value function over the environment's global state vector."""

    def __init__(self, config: EnvConfig, hidden_size: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(global_state_size(config), hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)


def actions_for_env(
    output: PolicyOutput, agents: list[str]
) -> dict[str, dict[str, int]]:
    """Convert a batched policy sample to PettingZoo action dictionaries."""

    if output.move.shape[0] != len(agents) or output.message.shape[0] != len(agents):
        raise ValueError("policy output batch size must match the number of agents")
    return {
        agent: {
            "move": int(output.move[index].item()),
            "message": int(output.message[index].item()),
        }
        for index, agent in enumerate(agents)
    }

