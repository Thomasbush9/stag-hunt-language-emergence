"""Distribution-matched causal test for the channel.

`muted` and `random` interventions both change the *statistics* of what the
listener hears, so a performance drop can be pure input-distribution shift (seen
four times in this project). This probe instead replays a real message stream
from a DIFFERENT episode at the same timestep: the listener hears exactly the
kind of messages it was trained on, in the right temporal position, but the
content refers to another episode's target. A collapse here is referential
dependence, full stop.
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


def rollout(actors, config, device, episodes, seed_base, donor=None):
    """Roll out; if `donor` is given, substitute each heard message from it."""
    env = StagHuntLanguageEnv(config)
    outcomes: Counter[str] = Counter()
    hits = tried = 0
    streams = []  # per-episode [T, 2] messages actually emitted
    rng = np.random.default_rng(12345)

    for episode in range(episodes):
        observations, _ = env.reset(seed=seed_base + episode)
        hiddens = [actor.initial_state(1, device=device) for actor in actors]
        emitted = []
        # pick a donor episode whose target differs, so the content is wrong
        donor_stream = None
        if donor is not None:
            for _ in range(20):
                candidate = donor[int(rng.integers(len(donor)))]
                if candidate["target"] != (env.correct_color, env.correct_region):
                    donor_stream = candidate["messages"]
                    break
            if donor_stream is None:
                donor_stream = donor[0]["messages"]

        step = 0
        while env.agents:
            if donor_stream is not None:
                # feed the partner's message from the donor episode, same step
                src = donor_stream[min(step, len(donor_stream) - 1)]
                for index, agent in enumerate(AGENTS):
                    observations[agent]["received_message"] = int(src[1 - index])
            encoded = batch_observations(observations, list(AGENTS), config, device=device)
            actions, messages = {}, []
            with torch.no_grad():
                for index, agent in enumerate(AGENTS):
                    out = actors[index].act(encoded[index : index + 1], hiddens[index])
                    hiddens[index] = out.hidden
                    actions[agent] = {"move": int(out.move.item()), "message": int(out.message.item())}
                    messages.append(int(out.message.item()))
            emitted.append(messages)
            observations, _, _, _, _ = env.step(actions)
            step += 1

        outcomes[env.outcome] += 1
        if env.first_joint_presence is not None:
            tried += 1
            hits += env.first_joint_presence == "correct"
        streams.append({
            "messages": np.asarray(emitted),
            "target": (env.correct_color, env.correct_region),
        })

    capture = outcomes["joint_stag"] / episodes
    accuracy = hits / tried if tried else float("nan")
    return capture, accuracy, streams


def probe(name: str, episodes: int = 500) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = sorted((FILES / name).glob("checkpoint_*.pt"))[-1]
    config = analyze.load_env_config(checkpoint, device)
    actors = analyze.load_actors(checkpoint, config, device)

    intact_cap, intact_acc, streams = rollout(actors, config, device, episodes, 400_000)
    swap_cap, swap_acc, _ = rollout(actors, config, device, episodes, 400_000, donor=streams)

    print(f"\n=== {name}")
    print(f"  intact          captures {intact_cap:6.1%}   targeting {intact_acc:6.1%}")
    print(f"  swapped stream  captures {swap_cap:6.1%}   targeting {swap_acc:6.1%}")
    drop = (intact_cap - swap_cap) / intact_cap if intact_cap else 0.0
    print(f"  -> {drop:.0%} of captures depend on message CONTENT "
          f"(statistics held fixed)")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        probe(name)
