"""Capture rate split by who could see the target.

If any information flows between agents, episodes where exactly ONE agent
co-observes should beat episodes where neither does. If captures are confined
to the both-informed episodes, nothing was transmitted.
"""
import sys
from collections import Counter
from importlib import util
from pathlib import Path

import torch

REPO = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/stag-hunt-language-emergence")
FILES = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/stag-hunt-files")
sys.path.insert(0, str(REPO / "src"))
spec = util.spec_from_file_location("analyze_language", REPO / "scripts" / "analyze_language.py")
analyze = util.module_from_spec(spec); sys.modules["analyze_language"] = analyze
spec.loader.exec_module(analyze)
from stag_hunt_lang import StagHuntLanguageEnv
from stag_hunt_lang.observations import batch_observations

AGENTS = ("agent_0", "agent_1")

def run(name, episodes=600):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = sorted((FILES / name).glob("checkpoint_*.pt"))[-1]
    cfg = analyze.load_env_config(ck, device)
    actors = analyze.load_actors(ck, cfg, device)
    env = StagHuntLanguageEnv(cfg)
    tot, cap, tried, hit = Counter(), Counter(), Counter(), Counter()
    for ep in range(episodes):
        obs, infos = env.reset(seed=200_000 + ep)
        n_informed = sum(infos["agent_0"]["co_observes"].values())
        hid = [a.initial_state(1, device=device) for a in actors]
        while env.agents:
            enc = batch_observations(obs, list(AGENTS), cfg, device=device)
            acts = {}
            with torch.no_grad():
                for i, ag in enumerate(AGENTS):
                    o = actors[i].act(enc[i:i+1], hid[i]); hid[i] = o.hidden
                    acts[ag] = {"move": int(o.move.item()), "message": int(o.message.item())}
            obs, _, _, _, _ = env.step(acts)
        tot[n_informed] += 1
        if env.outcome == "joint_stag":
            cap[n_informed] += 1
        fp = env.first_joint_presence
        if fp is not None:
            tried[n_informed] += 1
            if fp == "correct":
                hit[n_informed] += 1
    print(f"\n=== {name}")
    for k, label in [(0, "neither informed"), (1, "ONE informed"), (2, "both informed")]:
        if tot[k]:
            acc = f"{hit[k]/tried[k]:5.1%}" if tried[k] else "  n/a"
            print(f"  {label:<18} capture {cap[k]/tot[k]:5.1%}   targeting {acc}  (chance 12.5%)")

for name in sys.argv[1:]:
    run(name)
