# ADR 0001: PyTorch policies over a PettingZoo environment, with optional EGG integration

## Status

Accepted for the first experimental prototype.

## Context

The experiment is a sequential two-agent POMDP. Each agent repeatedly chooses a
physical action and a discrete message, receives delayed rewards, and maintains
memory across an episode. Development begins on Apple MPS and later moves to a
CUDA HPC system.

EGG provides valuable PyTorch components for emergent communication, especially
sender/receiver games, discrete channels, population sampling, and analysis. Its
standard training contract is organized around dataset batches and a game
forward pass rather than environment rollouts.

## Decision

- PettingZoo's parallel API defines the simulation boundary.
- PyTorch is a core dependency for policies and training.
- The first learner will use a recurrent decentralized actor and centralized
  critic suitable for PPO/MAPPO.
- Device selection is runtime configuration: CUDA, then MPS, then CPU.
- EGG is not a core environment or checkpoint dependency. A pinned optional
  adapter may later provide one-step comparison games and protocol analysis.

## Consequences

- The same environment can be driven by scripted, random, PyTorch, or future EGG
  adapters.
- Delayed reward and physical/message action log-probabilities remain under one
  sequential RL objective.
- EGG integration requires an explicit adapter instead of direct reuse of its
  standard trainer.
- The experiment is not coupled to a dirty local EGG checkout or its broad
  dependency set.

