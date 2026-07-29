"""Do the bias-free runs use the channel for something other than the clue?

Several runs show captures collapsing under the random-message intervention
while MI(clue; message) is ~0. If the channel is causally load-bearing but not
referential, the messages should carry information about something else that is
decision-relevant — most plausibly the speaker's own position, since agents are
blind to each other (`--hide-other-position`).
"""

import sys
from collections import Counter
from importlib import util
from pathlib import Path

import numpy as np
import torch

REPO = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/stag-hunt-language-emergence")
FILES = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/stag-hunt-files")
sys.path.insert(0, str(REPO / "src"))

spec = util.spec_from_file_location("analyze_language", REPO / "scripts" / "analyze_language.py")
analyze = util.module_from_spec(spec)
sys.modules["analyze_language"] = analyze
spec.loader.exec_module(analyze)

from stag_hunt_lang import StagHuntLanguageEnv  # noqa: E402
from stag_hunt_lang.observations import batch_observations  # noqa: E402

AGENTS = ("agent_0", "agent_1")


def mi_bits(joint: np.ndarray) -> float:
    """Mutual information of a joint count table, in bits."""
    total = joint.sum()
    if total == 0:
        return 0.0
    p = joint / total
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = p * np.log2(p / (px * py))
    return float(np.nansum(term))


def probe(run: str, episodes: int = 400) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = sorted((FILES / run).glob("checkpoint_*.pt"))
    env_config = analyze.load_env_config(checkpoints[-1], device)
    actors = analyze.load_actors(checkpoints[-1], env_config, device)
    env = StagHuntLanguageEnv(env_config)
    grid = env_config.grid_size
    vocab = env_config.vocab_size + 1

    # candidate referents for the emitted symbol
    quadrant = np.zeros((4, vocab))          # speaker's own position (2x2 quadrant)
    x_half = np.zeros((2, vocab))            # speaker's own column half
    y_half = np.zeros((2, vocab))            # speaker's own row half
    timestep = np.zeros((6, vocab))          # coarse phase of the episode
    clue = np.zeros((max(env_config.n_colors, env_config.n_regions), vocab))
    partner_seen = np.zeros((2, vocab))      # whether partner is on the same half
    silence = Counter()

    for episode in range(episodes):
        observations, _ = env.reset(seed=100_000 + episode)
        hiddens = [actor.initial_state(1, device=device) for actor in actors]
        color_index = AGENTS.index(env.color_holder)
        step = 0
        while env.agents:
            encoded = batch_observations(observations, list(AGENTS), env_config, device=device)
            actions, messages = {}, []
            with torch.no_grad():
                for index, agent in enumerate(AGENTS):
                    out = actors[index].act(encoded[index : index + 1], hiddens[index])
                    hiddens[index] = out.hidden
                    actions[agent] = {"move": int(out.move.item()), "message": int(out.message.item())}
                    messages.append(int(out.message.item()))

            for index, agent in enumerate(AGENTS):
                m = messages[index]
                pos = observations[agent]["self_position"]
                qx, qy = int(pos[0] >= grid / 2), int(pos[1] >= grid / 2)
                quadrant[qy * 2 + qx, m] += 1
                x_half[qx, m] += 1
                y_half[qy, m] += 1
                timestep[min(step * 6 // env_config.horizon, 5), m] += 1
                held = observations[agent]["private_clue"]
                value = int(held[0] - 1) if index == color_index else int(held[1] - 1)
                if value >= 0:
                    clue[value, m] += 1
                other = env._agent_positions[AGENTS[1 - index]]
                partner_seen[int(other[0] >= grid / 2), m] += 1
                silence[m == 0] += 1

            observations, _, _, _, _ = env.step(actions)
            step += 1

    total = sum(silence.values())
    return {
        "MI(message; speaker quadrant)": mi_bits(quadrant),
        "MI(message; speaker x-half)": mi_bits(x_half),
        "MI(message; speaker y-half)": mi_bits(y_half),
        "MI(message; episode phase)": mi_bits(timestep),
        "MI(message; own clue)": mi_bits(clue),
        "MI(message; partner x-half)": mi_bits(partner_seen),
        "silence rate": silence[True] / total,
    }


if __name__ == "__main__":
    for run in sys.argv[1:]:
        print(f"\n=== {run}")
        for key, value in probe(run).items():
            print(f"  {key:<32} {value:.3f}")
