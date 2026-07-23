# Project context

## Purpose

This project studies whether an ungrounded discrete channel causally helps two
embodied agents select the payoff-dominant cooperative equilibrium in a spatial
Stag Hunt.

The current priority is experimental legibility: a researcher should be able to
inspect the world, each agent's information, exchanged symbols, physical actions,
and reward outcome before evaluating the learning algorithm.

## Domain language

- **World** — the grid, agents, stag targets, hare targets, and current timestep.
- **Stag target** — a joint-reward location with public factorized attributes.
- **Correct stag** — the unique stag whose attributes match both private clues.
- **Hare** — a safe individual-reward location.
- **Private clue** — one component of the correct stag's identity, observed by
  only one agent.
- **Message** — an ungrounded discrete symbol chosen alongside a physical action
  and received by the other agent on the next timestep.
- **Commitment** — an `INTERACT` action at a reward target. It makes the strategic
  choice terminal rather than allowing cost-free retries.
- **Joint stag** — both agents commit to the correct stag at the same time.
- **Failed stag** — at least one agent commits to a stag without successful joint
  capture.
- **Movement signaling** — information conveyed through a visible trajectory
  rather than the symbolic channel.
- **Causal communication** — changing a message while holding relevant state
  fixed changes the listener's behavior or expected return.
- **Two-way communication** — both agents transmit private information. This is
  not called dialogue unless later turns conditionally depend on earlier ones.
- **Cross-play** — evaluation between agents that were not trained together.

## Experimental invariants

1. Neither agent can identify the correct stag from its private observation
   alone.
2. The symbolic channel has no predefined semantics.
3. Successful stag reward is greater than safe hare reward, which is greater
   than failed-stag reward.
4. Environment and policy code use the same definitions on MPS, CUDA, and CPU.
5. Task reward alone is not evidence of causal communication.

