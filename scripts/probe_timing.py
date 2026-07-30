"""Is the code temporally concentrated?

`analyze_language.py` pools message counts over every timestep of the episode,
so a protocol that says one informative thing early and then babbles has its MI
diluted by the horizon. This probe recomputes MI(held clue; message) per
timestep, which distinguishes "no code" from "a code used briefly".
"""

import sys
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


def resolve_run(name: str) -> Path:
    """Find a run directory whether it is still at the top level or filed
    into a dated experiment folder."""
    direct = FILES / name
    if direct.is_dir():
        return direct
    matches = sorted(FILES.glob(f"*/{name}"))
    if not matches:
        raise SystemExit(f"run directory not found: {name}")
    return matches[-1]


def mi_bits(joint: np.ndarray) -> float:
    total = joint.sum()
    if total == 0:
        return 0.0
    p = joint / total
    px, py = p.sum(axis=1, keepdims=True), p.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum(p * np.log2(p / (px * py))))


def probe(name: str, episodes: int = 500) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = sorted(resolve_run(name).glob("checkpoint_*.pt"))[-1]
    config = analyze.load_env_config(checkpoint, device)
    actors = analyze.load_actors(checkpoint, config, device)
    env = StagHuntLanguageEnv(config)
    vocab = config.vocab_size + 1
    horizon = config.horizon

    colour = np.zeros((horizon, config.n_colors, vocab))
    region = np.zeros((horizon, config.n_regions, vocab))

    for episode in range(episodes):
        observations, _ = env.reset(seed=300_000 + episode)
        hiddens = [actor.initial_state(1, device=device) for actor in actors]
        colour_index = AGENTS.index(env.color_holder)
        step = 0
        while env.agents:
            encoded = batch_observations(observations, list(AGENTS), config, device=device)
            actions, messages = {}, []
            with torch.no_grad():
                for index, agent in enumerate(AGENTS):
                    out = actors[index].act(encoded[index : index + 1], hiddens[index])
                    hiddens[index] = out.hidden
                    actions[agent] = {"move": int(out.move.item()), "message": int(out.message.item())}
                    messages.append(int(out.message.item()))
            colour[step, env.correct_color, messages[colour_index]] += 1
            region[step, env.correct_region, messages[1 - colour_index]] += 1
            observations, _, _, _, _ = env.step(actions)
            step += 1

    per_step_c = np.array([mi_bits(colour[t]) for t in range(horizon)])
    per_step_r = np.array([mi_bits(region[t]) for t in range(horizon)])
    pooled_c = mi_bits(colour.sum(axis=0))
    pooled_r = mi_bits(region.sum(axis=0))

    print(f"\n=== {name}")
    print(f"  pooled over all steps : colour {pooled_c:.2f}  region {pooled_r:.2f}")
    print(f"  best single step      : colour {per_step_c.max():.2f} (t={per_step_c.argmax()})"
          f"  region {per_step_r.max():.2f} (t={per_step_r.argmax()})")
    head = " ".join(f"{v:.2f}" for v in per_step_c[:8])
    print(f"  colour MI by step 0-7 : {head}")
    head = " ".join(f"{v:.2f}" for v in per_step_r[:8])
    print(f"  region MI by step 0-7 : {head}")
    # Codebook at the most informative colour step.
    t = int(per_step_c.argmax())
    table = colour[t]
    rows = table.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.nan_to_num(table / rows)
    names = ["red", "blue", "green", "amber"][: config.n_colors]
    print(f"  P(message | colour) at t={t}  [silence, m1..m{config.vocab_size}]:")
    for i, row in enumerate(cond):
        print(f"    {names[i]:<6} " + " ".join(f"{v:.2f}" for v in row))


if __name__ == "__main__":
    for name in sys.argv[1:]:
        probe(name)
