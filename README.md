# Stag Hunt Language Emergence

An embodied, partially observable multi-agent environment for testing whether
discrete messages causally help agents select the cooperative equilibrium in a
Stag Hunt.

The project uses PyTorch for policies and training while retaining a small,
deterministic PettingZoo `ParallelEnv` as the simulation boundary. The first
milestone provides:

- two agents with complementary private clues;
- factorized stag targets (`colour × region`);
- discrete messages delivered on the next environment step;
- visible or hidden partner positions;
- individual Stag Hunt rewards;
- terminal commitment through stag or hare interaction;
- a global state vector for centralized critics;
- a recurrent parameter-sharing PyTorch actor with distinct movement and
  discrete-message heads;
- a centralized PyTorch value network for a future MAPPO baseline;
- tests for reward semantics, private information, messaging, and API behavior.

## Setup

```bash
uv sync
uv run pytest
uv run ruff check .
uv run stag-hunt-random-rollout --episodes 3 --render
uv run stag-hunt-torch-rollout --episodes 3 --device auto
```

Generate the experiment figures with the optional visualization dependency:

```bash
uv sync --extra viz
uv run stag-hunt-figures --output docs/figures --seed 7
```

The generator produces two simple figures: `environment.png` and
`agent_trajectories.png`. The public `draw_world(ax, env, ...)` and
`draw_trajectory(ax, agent, path)` helpers can be reused for future learned
policies, interventions, and evaluation layouts.

The repository pins Python `3.12.12`. A broader `3.12` pin selected a local
Miniconda 3.12.4 interpreter whose native `readline` module crashes during
pytest startup on this machine. The exact patch pin also makes local setup more
reproducible before moving to the HPC.


`resolve_torch_device()` chooses CUDA first, then Apple MPS, then CPU. An explicit
device can be provided by the future training CLI. This keeps experiment code
portable between the local Apple Silicon machine and a Linux/CUDA HPC system.

## Environment sketch

At reset, every combination of colour and region is assigned to one stag target.
`agent_0` privately observes the correct colour and `agent_1` privately observes
the correct region. Both agents see target features and positions, but neither can
identify the rewarding stag alone.

Each step, an agent selects a physical action and a message:

```python
{
    "move": 0,     # stay, up, right, down, left, or interact
    "message": 0,  # silence or one of vocab_size symbols
}
```

Messages are received by the partner in the next observation. Movement and
communication happen simultaneously, allowing later experiments to compare
symbolic messages with visible trajectory-based signaling.

The default payoff structure is:

| agent_0 / agent_1 | Stag | Hare |
|---|---:|---:|
| **Stag** | 4, 4 | 0, 2 |
| **Hare** | 2, 0 | 2, 2 |

A stag interaction is a commitment: unless both agents interact with the correct
stag from its cell on the same step, the attempt fails and the episode ends.

## Quick use

```python
from stag_hunt_lang import EnvConfig, StagHuntLanguageEnv

env = StagHuntLanguageEnv(EnvConfig())
observations, infos = env.reset(seed=42)

while env.agents:
    actions = {
        agent: env.action_space(agent).sample()
        for agent in env.agents
    }
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

See [`docs/experiment.md`](docs/experiment.md) for the experimental contract and
the next implementation milestones.

## Training and analysis

The exploratory MAPPO baseline lives in `scripts/`:

```bash
uv run python scripts/train_mappo.py --output-dir <run-dir> \
    --updates 2000 --episodes-per-update 32 --seed 0 \
    --curriculum-start 1.8 --curriculum-updates 1200   # risk curriculum (optional)
uv run python scripts/plot_training.py --run-dir <run-dir>
uv run python scripts/analyze_language.py --run-dir <run-dir>
```

`train_mappo.py` implements recurrent PPO with a parameter-sharing actor and
per-agent centralized critics; the optional risk curriculum anneals
`failed_stag_reward` from `--curriculum-start` to the default over
`--curriculum-updates` updates. `analyze_language.py` measures MI between each
agent's private clue and its messages and runs the mute/randomize channel
interventions from the experimental contract. Results so far are documented in
[`docs/experiments/2026-07-23-risk-curriculum.md`](docs/experiments/2026-07-23-risk-curriculum.md).

## Why EGG is not the core trainer

[EGG](https://github.com/facebookresearch/EGG) remains highly relevant for its
discrete-channel wrappers, sender–receiver baselines, population games, and
protocol-analysis conventions. Its standard trainer is organized around
dataset batches and a sender/receiver game forward pass. This project instead
needs recurrent environment rollouts, temporal credit assignment, and joint
physical/message actions.

The first sequential baseline will therefore be native PyTorch PPO/MAPPO. An EGG
adapter is planned after that baseline is validated, for comparable one-step
games and language analysis. Keeping it optional also avoids coupling HPC runs
to a mutable local checkout or EGG's broad research dependency set.
