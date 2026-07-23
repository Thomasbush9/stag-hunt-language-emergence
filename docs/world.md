# World design, version 0

This document is the main surface for judging the proposed experiment before
judging its training implementation.

## World

The world is a `7 × 7` open grid divided into two physically meaningful regions:
west and east. It contains:

- two embodied agents, `A0` and `A1`;
- four public stag targets, `S0`–`S3`;
- two public safe hare targets, `H0` and `H1`.

Each stag has two public attributes:

- colour: red or blue;
- region: west or east.

There is exactly one target for every `colour × region` combination. West targets
are physically placed in the western part of the map and east targets in the
eastern part. Positions change across episodes while the factorization remains.

## Private information

At reset, one of the four stags is selected as correct:

- `A0` privately observes its colour;
- `A1` privately observes its region.

Both agents publicly observe every target's attributes and position. Neither can
identify the correct target alone, but exchanging their clues identifies exactly
one stag.

In the tensor observation, `0` means an unknown clue and positive integers encode
observed categorical values. For example:

```text
A0 private clue: [2, 0]  → blue, unknown region
A1 private clue: [0, 1]  → unknown colour, west
```

These are observations, not message meanings. The discrete message tokens start
without semantics.

## Action and channel timing

At every timestep each agent simultaneously chooses:

- one physical action: stay, move north/east/south/west, or interact;
- one message: silence or one of four ungrounded symbols.

The partner receives that symbol in its next observation. This one-step delay
prevents an action at time `t` from responding to a message created at the same
time.

Partner positions are visible in the default world, so trajectories can function
as non-verbal signals. Setting `observe_other_position=False` removes that channel
for a causal comparison.

## Commitment and outcomes

`INTERACT` on a target is an irreversible strategic commitment:

- both agents interact on the correct stag during the same step: `(4, 4)`;
- an agent interacts with a hare: that agent gets `2`;
- an agent interacts with a stag without successful joint capture: `0`;
- interaction on an empty cell does nothing;
- any target commitment ends the episode;
- otherwise the episode truncates at 30 steps.

Terminating on commitment prevents agents from safely trying one choice and later
switching to another.

## What is intentionally absent

- No walls or path-planning puzzles.
- No natural-language supervision.
- No predefined token meanings.
- No dialogue claim: version 0 supports reciprocal messages, but not conditional
  question–answer structure.
- No evolutionary mechanism yet.
- No reward shaping beyond the game payoffs.

## Inspecting it

```bash
uv run stag-hunt-inspect --scenario reset --seed 7
uv run stag-hunt-inspect --scenario joint-stag --seed 7
uv run stag-hunt-inspect --scenario hare --seed 7
uv run stag-hunt-inspect --scenario failed-stag --seed 7
```

The scripted scenarios assign temporary token meanings only to make information
flow visible. Learned agents will not receive those meanings.

## Questions to judge before training

1. Should the agents see each other's positions, or should symbolic communication
   be isolated in the first study?
2. Is a terminal commitment the right embodied translation of the matrix game?
3. Should a lone stag commitment receive `0` or a negative payoff?
4. Should both agents receive individual rewards exactly as above, or should a
   team-return condition also be tested?
5. Is `colour × region` a sufficiently meaningful factorization for studying the
   first protocol, before adding more attributes?
6. Does one token per environmental step provide too much bandwidth because the
   agents can communicate while walking?

