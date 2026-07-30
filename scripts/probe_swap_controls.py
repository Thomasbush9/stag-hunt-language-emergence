"""Isolate message CONTENT from message COHERENCE.

`probe_shuffle.py` replaces the heard stream with one from a different episode.
That breaks two things at once: the content refers to the wrong target, AND the
stream is no longer temporally coherent with the listener's own trajectory. A
collapse could therefore mean "meaning matters" or merely "synchrony matters".

This probe adds the decisive control: a donor episode with the SAME target
(same colour and region) but a different layout and different agent starts.
Coherence is broken exactly as much as in the different-target condition, but
the content is still true.

  intact                 -> upper bound
  same-target swap       -> content right, coherence broken
  different-target swap  -> content wrong, coherence broken

If same-target ≈ intact and different-target collapses, the channel carries
meaning. If both swaps collapse equally, the channel is carrying synchronisation
and our "communication" claim does not survive.
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


def resolve_run(name: str) -> Path:
    direct = FILES / name
    if direct.is_dir():
        return direct
    matches = sorted(FILES.glob(f"*/{name}"))
    if not matches:
        raise SystemExit(f"run directory not found: {name}")
    return matches[-1]


def rollout(actors, config, device, episodes, seed_base, donor=None, match_target=False):
    env = StagHuntLanguageEnv(config)
    outcomes: Counter[str] = Counter()
    hits = tried = 0
    returns = 0.0
    streams = []
    rng = np.random.default_rng(999)

    for episode in range(episodes):
        observations, _ = env.reset(seed=seed_base + episode)
        target = (env.correct_color, env.correct_region)
        hiddens = [actor.initial_state(1, device=device) for actor in actors]
        emitted = []

        donor_stream = None
        if donor is not None:
            # Sample a donor with the same target (control) or a different one.
            pool = [d for d in donor if (d["target"] == target) == match_target]
            if pool:
                donor_stream = pool[int(rng.integers(len(pool)))]["messages"]
            else:
                donor_stream = donor[0]["messages"]

        step = 0
        episode_return = 0.0
        while env.agents:
            if donor_stream is not None:
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
            observations, rewards, _, _, _ = env.step(actions)
            episode_return += rewards[AGENTS[0]]
            step += 1

        outcomes[env.outcome] += 1
        returns += episode_return
        if env.first_joint_presence is not None:
            tried += 1
            hits += env.first_joint_presence == "correct"
        streams.append({"messages": np.asarray(emitted), "target": target})

    return {
        "capture": outcomes["joint_stag"] / episodes,
        "accuracy": hits / tried if tried else float("nan"),
        "return": returns / episodes,
        "streams": streams,
    }


def probe(name: str, episodes: int = 500) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = sorted(resolve_run(name).glob("checkpoint_*.pt"))[-1]
    config = analyze.load_env_config(checkpoint, device)
    actors = analyze.load_actors(checkpoint, config, device)

    intact = rollout(actors, config, device, episodes, 500_000)
    same = rollout(actors, config, device, episodes, 500_000,
                   donor=intact["streams"], match_target=True)
    diff = rollout(actors, config, device, episodes, 500_000,
                   donor=intact["streams"], match_target=False)

    print(f"\n=== {name}")
    for label, r in (("intact", intact), ("same-target swap", same), ("diff-target swap", diff)):
        print(f"  {label:<18} captures {r['capture']:6.1%}  targeting {r['accuracy']:6.1%}"
              f"  return {r['return']:5.2f}")
    coherence_cost = 1 - same["capture"] / intact["capture"] if intact["capture"] else 0
    content_cost = 1 - diff["capture"] / same["capture"] if same["capture"] else 0
    print(f"  -> coherence alone costs {coherence_cost:5.0%};"
          f" CONTENT costs a further {content_cost:5.0%}")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        probe(name)
