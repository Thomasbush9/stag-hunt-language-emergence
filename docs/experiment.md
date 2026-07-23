# Experimental contract

## Primary question

Does emergent discrete communication help agents select the payoff-dominant
cooperative equilibrium as the risk of failed coordination increases?

## What version 0 establishes

The environment creates a genuine information gap: the correct target is the
intersection of two private clues, and each agent observes only one. A symbolic
protocol can therefore improve joint decision-making, but the environment does
not assume any token meanings.

The environment deliberately exposes the other agent's position by default.
This preserves embodied signaling as a competing communication modality. Setting
`observe_other_position=False` removes that channel for causal isolation.

## Reward semantics

- Both agents interact at the correct stag on the same step: each receives
  `stag_reward` and the episode terminates.
- An agent interacts at a hare: that agent receives `hare_reward` and the episode
  terminates.
- An agent commits to any stag without successful joint capture: it receives
  `failed_stag_reward` and the episode terminates.
- Empty-cell interaction does not commit and does not end the episode.
- Reaching the horizon truncates the episode.

The terminal commitment prevents an agent from safely trying a hare and then
joining the stag later. It makes the spatial task implement the strategic choice
represented by the matrix game.

## Planned causal evaluations

Evaluation wrappers should support:

1. muting all messages;
2. replacing messages with a constant symbol;
3. shuffling messages across episodes;
4. replacing individual messages counterfactually;
5. limiting the channel to one direction;
6. hiding partner positions;
7. supplying oracle access to both clues.

Task reward alone is not evidence of language. The central communication metric
will be the change in action distribution and return under message intervention.

## Planned training stages

1. Scripted oracle and no-communication baselines.
2. A small parameter-sharing PyTorch MAPPO baseline with a centralized critic
   and decentralized recurrent actors.
3. Risk sweep with paired seeds.
4. Message-intervention analysis.
5. One-way versus two-way communication.
6. Held-out target combinations, unseen layouts, and cross-play.
7. Fresh-partner acquisition and population turnover.

## Portability

The simulation boundary depends on NumPy, Gymnasium, and PettingZoo; policies and
training use PyTorch. Training configuration must keep device,
precision, batch size, number of environments, and output paths explicit so that
local MPS smoke runs and HPC CUDA runs use the same experiment definitions.

Checkpoints should contain model and optimizer state, configuration, RNG states,
source revision, and environment version. Large outputs and checkpoints are
ignored by Git.

## EGG integration boundary

EGG directly motivates the discrete communication design and is a good candidate
for later one-step sender/receiver baselines, population sampling, and protocol
analysis. Its standard game wrappers optimize communication inside a batched
sender/receiver forward pass. The Stag Hunt instead emits delayed rewards over a
recurrent PettingZoo rollout in which every agent jointly samples a physical
action and a message at every timestep.

The sequential baseline therefore owns its PyTorch categorical distributions and
PPO objective. EGG should remain an optional adapter rather than a dependency of
the environment or the checkpoint format. If integrated, it should be pinned to
an upstream revision and tested separately from the local EGG checkout.

