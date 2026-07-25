# Experiment log — exploratory MAPPO baseline and risk-curriculum sweep

Date: 2026-07-23. Hardware: single A100-SXM4-40GB (MIG), 1 CPU core, CUDA 12.9
driver, torch 2.11.0+cu128.

## Setup notes

- The default PyPI torch wheel (2.13.0, cu130) fails CUDA initialization on the
  cluster's 12.9 driver. `pyproject.toml` now pins the `pytorch-cu128` index for
  Linux only; macOS continues to resolve torch from PyPI.
- All artifacts (checkpoints, metrics, figures, analysis JSON) are written
  outside the repo to
  `/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/stag-hunt-files/`.

## Trainer (`scripts/train_mappo.py`)

MAPPO baseline as planned in `docs/experiment.md` stage 2:

- one parameter-sharing `RecurrentActor` (GRU, hidden 128) acting for both
  agents, joint move × message categorical heads;
- two `CentralizedCritic` networks, one per agent, over `env.state()` — this
  preserves the individual Stag Hunt payoffs instead of collapsing them into a
  team reward;
- full-episode collection, GAE (γ=0.99, λ=0.95), PPO (clip 0.2, 4 epochs)
  with the GRU re-run over padded episode sequences (BPTT) at update time;
- entropy bonus 0.02 shared by both heads; Adam 3e-4; grad-norm clip 0.5.

Per-update metrics stream to `metrics.jsonl` (returns per agent, outcome
counts, episode length, silence rate, losses, entropy, current
`failed_stag_reward`).

## Exploratory run (`exploratory_run_01`)

500 updates × 32 episodes, default `EnvConfig` (grid 7, horizon 30, 2×2
targets, vocab 4, payoffs 4/2/0), seed 0, ~8 min wall time.

Result: convergence to the risk-dominant hare equilibrium.

- Outcomes drift from ~45% `failed_stag` early to ~95% `hare` by update 400;
  `joint_stag` is essentially never found. Episode length collapses 18 → 5.
- Language probe (`scripts/analyze_language.py`, 500 eval episodes per
  condition): MI(private clue; sent message) ≈ 0.002 bits for both agents (1.0
  bit ceiling), flat across checkpoints; muting or randomizing the received
  message changes neither outcomes nor returns. The channel is causally inert,
  consistent with the invariant that task reward alone is not evidence of
  communication.

Interpretation: hare hunting needs no partner information, and the joint-stag
event (both agents INTERACT on the correct cell on the same step) is too rare
under random exploration to seed a protocol. This is the expected negative
baseline for the causal story.

## Risk-curriculum sweep (`curriculum_run_seed{0..4}`)

Hypothesis: if failed stag commitments are cheap early in training, agents
explore stags long enough for the joint-stag reward to be discovered; the
information gap then gives the channel gradient pressure, and a protocol formed
early may survive as the risk anneals to the target payoff structure.

Design (trainer flags `--curriculum-start 1.8 --curriculum-updates 1200`):

- `failed_stag_reward` starts at 1.8 (stag attempts cost only 0.2 versus the
  2.0 hare) and anneals linearly to the default 0.0 over updates 0–1200, then
  holds. Payoff ordering `stag > hare > failed_stag` holds throughout.
- 5 seeds (0–4), 2000 updates × 32 episodes each (64k episodes/seed),
  checkpoints every 400 updates. Everything else identical to the exploratory
  run.
- Seeds run sequentially because the interactive allocation has a single CPU
  core (environment stepping is CPU-bound NumPy; the networks run on CUDA).
  Request `--cpus-per-task≥8` next time to parallelize seeds.

Launch command per seed:

```bash
uv run python scripts/train_mappo.py \
  --output-dir .../stag-hunt-files/curriculum_run_seed${seed} \
  --updates 2000 --episodes-per-update 32 --seed ${seed} \
  --curriculum-start 1.8 --curriculum-updates 1200 --checkpoint-every 400
```

## Analysis plan

Per seed, after training:

1. `scripts/plot_training.py` — outcome rates, returns, length,
   entropy/silence.
2. `scripts/analyze_language.py` — evaluated at the **default** payoffs
   (`failed_stag_reward = 0`): MI(clue; message) per checkpoint, message-given-
   clue heatmaps, and the mute/randomize channel interventions.

Across seeds: joint-stag rate and MI trajectories with per-seed lines
(5 seeds is too few for meaningful error bands); the key readouts are
(a) whether any seed sustains joint stag after the anneal completes, and
(b) whether MI > 0 precedes or accompanies joint-stag success, and
(c) whether channel interventions now causally degrade stag outcomes.

Failure modes to expect: protocol collapse back to hare once risk rises
(curriculum too fast), or stag success via movement signaling alone
(`observe_other_position=True` leaves that channel open — the planned
follow-up is the same sweep with hidden partner positions).

## Risk-curriculum result (5 seeds)

Negative, replicated across all seeds (cross-seed figure:
`stag-hunt-files/curriculum_sweep_summary.png`): agents explore stags
individually while risk is low (~55% `failed_stag`) but almost never sample
the joint-capture event: 12–18 `joint_stag` episodes per 64k (rate ≤ 0.03%),
invisible on the outcome plot, in every seed. Hare rate reaches ~1.0 in every seed; the switch point varies with seed
(update ~350 for the earliest, ~700 for the latest) but always occurs while
`failed_stag_reward` is still ≥ 0.7, i.e. well before the anneal completes.
Eval at true payoffs: 97–99% hare, mean returns ~1.0–1.2 per agent.
MI(clue; message) ≈ 0.0003–0.003 bits at every checkpoint of every seed;
mute/randomize interventions shift no outcome by more than ~1% (noise).

Diagnosis: the bottleneck is not the risk level but *sampling* the joint event.
Both agents must INTERACT on the correct cell on the same step, and any solo
stag commitment terminates the episode, so exploration never reaches the
cooperative payoff and no gradient ever flows toward the channel.

## Joint-proximity sweep (`prox_{ppo,reinforce}_seed{0..4}`)

Launched 2026-07-23 after the curriculum sweep. Hypothesis: densifying the
reward *around* the joint event (rather than lowering its risk) lets
exploration find joint capture, after which the information gap can recruit
the channel.

Design, on top of the unchanged risk curriculum:

- **Joint proximity bonus** (trainer-side shaping, never in the environment):
  while BOTH agents are within Chebyshev distance 1 of the correct stag, both
  receive +0.05 per step. Joint by construction — not farmable alone, so hare
  defection stays strictly better unless the partner cooperates. Max
  accumulable over the 30-step horizon is 1.5 < hare's 2.0. Anneals linearly
  to zero by update 1500, so the final 500 updates train on the pure game.
  Caveat on record: the bonus conditions on the correct stag, injecting target
  information into the learning signal (not observations); the cheap shortcut
  it enables is movement signaling, which the later
  `observe_other_position=False` variant will isolate.
- **Per-head entropy**: message head 0.05, move head 0.02, so the channel
  stays alive while movement converges.
- **Algorithm comparison**: the full 5-seed sweep runs twice — `--algo ppo`
  (as before) and `--algo reinforce` (single epoch, discounted return-to-go,
  whitening baseline instead of critics, no clipping — the classic emergent-
  communication estimator, e.g. EGG). Same seeds, curriculum, and shaping.

Note: `mean_return_*` in these runs' `metrics.jsonl` includes the shaping
bonus until it anneals away; `language_analysis.py` evaluations are always
unshaped and at the default payoffs.

Queue runner: `stag-hunt-files/launch_prox_sweeps.sh` (detached via setsid;
log `launch_prox_sweeps.log`).

### Proximity result (PPO arm, seeds 0–3; REINFORCE pending)

Correction after counting cumulative captures (`joint_stag_progression.png`):
the proximity bonus did **not** increase joint-stag sampling — PPO seeds
produced 11–15 joint-stag episodes per 64k, statistically identical to the
curriculum sweep's 12–18. It moved agents *near* the correct stag but the
same-cell-same-tick synchronization barrier absorbed all of that gain. Every
seed ended in hare lock-in with MI ≈ 0.001–0.003 bits and null
interventions. The first intervention that genuinely moved sampling was the
commitment window (88 captures, ~6×, see below).

## Commitment window (`commit_window`, env change)

Diagnosis after the proximity sweep: even with both agents adjacent to the
correct stag, joint capture requires both agents on the *same cell* choosing
INTERACT on the *same tick*, and any near-miss (solo INTERACT) terminates the
episode. The synchronization, not the risk or the proximity, is the deepest
bottleneck.

Motivation from human experiments: communication in Stag Hunt play rises with
risk, and specifically when one player is already at the stag *waiting* for
the partner. The current environment makes that state unreachable — pouncing
alone instantly ends the episode.

New `EnvConfig.commit_window` (default 0 = exact original semantics, all
prior results unaffected): an INTERACT at a stag arms a commitment held for
`commit_window` subsequent steps. While holding, the committer is frozen in
place but its messages still flow — the "waiting hunter" state. If the
partner INTERACTs on the same stag within the window: joint capture (correct
stag) or joint failure (wrong stag). Window expiry, or the partner taking a
hare, fails every open commitment and ends the episode — commitment stays
terminal, there are no free retries. Commitment countdowns are appended to
the global state for the critics (`global_state_size` 9→11 + targets).
Covered by five new tests (join-late, freeze, expiry, messages-while-holding,
PettingZoo API with window).

## Commitment-window sweep (`cw3_hare{10,20}_seed{0..2}`)

Launched 2026-07-23 (queued behind the proximity sweeps via
`launch_cw_sweeps.sh`). Factorial: `commit_window=3` × `hare_reward` ∈
{1.0, 2.0} × 3 seeds, keeping the risk curriculum (start 0.9 for hare=1.0,
1.8 for hare=2.0 — always below hare), the joint proximity bonus
(0.05 → 0 by update 1500), message entropy 0.05, PPO, 2000 updates.

The hare=1.0 arm lowers the cooperation belief threshold from p > 0.5 to
p > 0.25 (bootstrap condition); hare=2.0 is the canonical risk-dominant
target game. Predictions: window+hare=1.0 produces sustained joint stag; the
scientific readout is whether protocol and MI survive at hare=2.0, and later
whether they survive `observe_other_position=False`.

`analyze_language.py` now evaluates each run in its own environment
(reconstructed from the checkpoint's `env_config`), so hare=1.0 runs are
probed at hare=1.0 with the window active.

### First commit-window result and redesign (`cw3s_*`)

`cw3_hare10_seed0` (batch 32): joint-stag sampling jumped another ~7× (88 of
64k episodes, rate ~0.8% around update 200) but amplification still failed —
hare lock-in by update ~600 even at hare=1.0. At 0.8% a 32-episode batch
contains a joint stag only once every ~4 updates.

**Result (`cw3s_hare10_seed0`): cooperation emerges — through movement, not
symbols.** Joint stag grows monotonically from ~1% to ~22% of episodes and
*holds at ~21–23% after both anneals reach zero* (updates 1500–1600 are the
pure hare=1.0 game). 33,472 joint captures in 205k episodes. Training-phase
episode composition at the end: ~62% failed_stag (cheap failed attempts are
the exploration engine), ~22% joint stag, ~7% hare. Eval at the run's own
game: 21.6% joint stag, mean return ~0.9 per agent.

The language probe, however, is unambiguous: MI(clue; message) ≈ 0.001 bits
at every checkpoint, and muting or randomizing the channel leaves joint-stag
success *exactly* unchanged (108–109/500 in all three conditions). The
coordination is carried entirely by movement signaling — the commit window
makes "partner holding at a stag" a visible positional signal
(`observe_other_position=True`), which is cheaper to learn than symbols.
This cleanly separates the cooperation problem (solved) from the
symbol-grounding problem (untouched), and makes the
`observe_other_position=False` variant the decisive next experiment: with
positions hidden, the waiting-hunter state can only be communicated through
the symbolic channel.

**Replication and a methodological catch (`cw3s_hare10_seed1`).** Seed 1
reproduces cooperation almost exactly (32,736 joint captures; 22.4% final
rate; eval 104/500). Its interventions initially *looked* like causal
communication: muting dropped joint stag to 72/500 and returns 0.86 → 0.61.
But random messages left success untouched (107/500), and a direct probe of
the beacon hypothesis found MI(message; sender-or-partner-holding) ≈
0.0003–0.002 bits with near-identical message distributions in and out of
the holding state. Verdict: the muting effect is **input distribution
shift**, not communication — training exposes the listener to ~80%
non-silent messages, so forced constant silence is off-distribution for the
GRU and degrades the policy irrespective of content. Random symbols match
the training marginal and cost nothing. This is precisely why the
experimental contract pairs the constant-symbol control with the
shuffle/random control; either alone can mislead. The cooperation remains
movement-carried in both seeds.

**Seed 2 fails to bootstrap.** `cw3s_hare10_seed2`: 368 joint captures total,
rate ~0 by update 600, full hare lock-in at eval. The hare=1.0 arm therefore
ends 2/3 cooperative — escape from the risk-dominant basin under this design
is stochastic (bimodal across seeds), not guaranteed. Sample-size caveat:
3 seeds only bounds the escape probability loosely.

**hare=2.0 arm: 0/3 bootstrap.** All three full-risk seeds fail to escape
the hare basin (99–482 joint captures per 205k episodes; hare lock-in; null
probes). Factorial verdict (`cw3s_factorial_summary.png`): with the commit
window, slow anneals, and batch 128, the cooperation phase transition sits
between hare=1.0 (2/3 seeds escape) and hare=2.0 (0/3). Consistent with
risk-dominance: the bootstrap needs the low-opportunity-cost regime. Next
candidates when revisited: intermediate hare (1.25–1.5), a hare anneal
1.0 → 2.0 after cooperation establishes, or warm-starting hare=2.0 from a
cooperative hare=1.0 checkpoint. (Completed 2026-07-24.)

**REINFORCE arm (completed 2026-07-24).** All 5 seeds match the PPO
proximity arm: 14–35 joint captures per 64k, hare lock-in, MI ≈ 0.001–0.005
bits, null interventions (`prox_reinforce_sweep_summary.png`). The estimator
is not the binding constraint at batch 32 — the environment's
synchronization barrier and the payoff structure are. PPO vs REINFORCE is a
wash on this configuration.

Redesign (superseding the remaining `cw3_*` runs, which were cancelled):
`cw3s_hare{10,20}_seed{0..2}` with **batch 128** episodes/update (a joint
stag in nearly every batch at the observed sampling rate) and **slower
anneals** — risk over updates 0–1400 and proximity over 0–1500 of 1600
total updates, i.e. ~4× more shaped episodes than before (~205k
episodes/run). hare=1.0 seeds run first. The REINFORCE proximity arm
(unchanged, batch 32 for comparability with `prox_ppo_*`) is queued after
the factorial in `launch_slow_sweeps.sh`. Estimated ~2h per cw3s run on the
single-core allocation.
