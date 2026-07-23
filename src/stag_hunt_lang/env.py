"""A parallel, embodied Stag Hunt with an ungrounded discrete channel."""

from __future__ import annotations

from enum import IntEnum
from itertools import product
from typing import Any, ClassVar

import gymnasium.spaces as spaces
import numpy as np
from numpy.typing import NDArray
from pettingzoo import ParallelEnv

from stag_hunt_lang.config import EnvConfig

AgentID = str
Position = NDArray[np.int64]


class Move(IntEnum):
    """Physical actions available alongside every message action."""

    STAY = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    INTERACT = 5


_DELTAS: dict[Move, tuple[int, int]] = {
    Move.STAY: (0, 0),
    Move.UP: (0, -1),
    Move.RIGHT: (1, 0),
    Move.DOWN: (0, 1),
    Move.LEFT: (-1, 0),
}


class StagHuntLanguageEnv(ParallelEnv):
    """Two-agent Stag Hunt requiring complementary private information.

    `agent_0` knows the rewarding stag's colour and `agent_1` knows its region.
    Messages have no predefined meaning and arrive in the partner's next
    observation. Physical actions and messages are chosen simultaneously.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "name": "stag_hunt_language_v0",
        "render_modes": ["ansi"],
        "is_parallelizable": True,
    }

    possible_agents: ClassVar[list[AgentID]] = ["agent_0", "agent_1"]
    color_names: ClassVar[tuple[str, ...]] = ("red", "blue", "green", "amber")
    region_names: ClassVar[tuple[str, ...]] = ("west", "east")

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None) -> None:
        self.config = config or EnvConfig()
        if render_mode not in {None, "ansi"}:
            raise ValueError("render_mode must be None or 'ansi'")
        self.render_mode = render_mode

        self.agents: list[AgentID] = []
        self.np_random = np.random.default_rng()
        self._timestep = 0
        self._agent_positions: dict[AgentID, Position] = {}
        self._stag_positions = np.empty((self.config.n_stags, 2), dtype=np.int64)
        self._hare_positions = np.empty((self.config.n_hares, 2), dtype=np.int64)
        self._stag_features = np.asarray(
            list(product(range(self.config.n_colors), range(self.config.n_regions))),
            dtype=np.int64,
        )
        self._correct_color = 0
        self._correct_region = 0
        self._correct_target_index = 0
        self._last_messages: dict[AgentID, int] = {agent: 0 for agent in self.possible_agents}
        self._last_moves: dict[AgentID, Move] = {
            agent: Move.STAY for agent in self.possible_agents
        }
        self._outcome = "not_reset"
        self._observation_spaces = {
            agent: self._make_observation_space() for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: self._make_action_space() for agent in self.possible_agents
        }

    def observation_space(self, agent: AgentID) -> spaces.Space[Any]:
        self._validate_agent(agent)
        return self._observation_spaces[agent]

    def _make_observation_space(self) -> spaces.Space[Any]:
        max_feature = max(self.config.n_colors, self.config.n_regions)
        return spaces.Dict(
            {
                "agent_id": spaces.Discrete(len(self.possible_agents)),
                "self_position": spaces.Box(
                    low=0,
                    high=self.config.grid_size - 1,
                    shape=(2,),
                    dtype=np.int64,
                ),
                "other_position": spaces.Box(
                    low=-1,
                    high=self.config.grid_size - 1,
                    shape=(2,),
                    dtype=np.int64,
                ),
                "stag_positions": spaces.Box(
                    low=0,
                    high=self.config.grid_size - 1,
                    shape=(self.config.n_stags, 2),
                    dtype=np.int64,
                ),
                "stag_features": spaces.Box(
                    low=0,
                    high=max_feature - 1,
                    shape=(self.config.n_stags, 2),
                    dtype=np.int64,
                ),
                "hare_positions": spaces.Box(
                    low=0,
                    high=self.config.grid_size - 1,
                    shape=(self.config.n_hares, 2),
                    dtype=np.int64,
                ),
                "private_clue": spaces.Box(
                    low=0,
                    high=max_feature,
                    shape=(2,),
                    dtype=np.int64,
                ),
                "received_message": spaces.Discrete(self.config.vocab_size + 1),
                "timestep": spaces.Discrete(self.config.horizon + 1),
            }
        )

    def action_space(self, agent: AgentID) -> spaces.Space[Any]:
        self._validate_agent(agent)
        return self._action_spaces[agent]

    def _make_action_space(self) -> spaces.Space[Any]:
        return spaces.Dict(
            {
                "move": spaces.Discrete(len(Move)),
                "message": spaces.Discrete(self.config.vocab_size + 1),
            }
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[AgentID, dict[str, Any]], dict[AgentID, dict[str, Any]]]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        options = options or {}

        self.agents = self.possible_agents.copy()
        self._timestep = 0
        self._last_messages = {agent: 0 for agent in self.possible_agents}
        self._last_moves = {agent: Move.STAY for agent in self.possible_agents}
        self._outcome = "ongoing"

        self._correct_color = self._sample_feature(
            options.get("correct_color"), self.config.n_colors, "correct_color"
        )
        self._correct_region = self._sample_feature(
            options.get("correct_region"), self.config.n_regions, "correct_region"
        )
        matches = np.flatnonzero(
            (self._stag_features[:, 0] == self._correct_color)
            & (self._stag_features[:, 1] == self._correct_region)
        )
        self._correct_target_index = int(matches[0])

        self._stag_positions = self._sample_stag_positions()
        occupied = {tuple(map(int, position)) for position in self._stag_positions}
        available = np.asarray(
            [
                (x, y)
                for y in range(self.config.grid_size)
                for x in range(self.config.grid_size)
                if (x, y) not in occupied
            ],
            dtype=np.int64,
        )
        n_mobile_entities = len(self.possible_agents) + self.config.n_hares
        selected = self.np_random.choice(
            len(available), size=n_mobile_entities, replace=False
        )
        positions = available[selected]
        self._agent_positions = {
            agent: positions[index].copy() for index, agent in enumerate(self.possible_agents)
        }
        start = len(self.possible_agents)
        self._hare_positions = positions[start:].copy()

        observations = {agent: self._observation(agent) for agent in self.agents}
        infos = {agent: self._info(agent) for agent in self.agents}
        return observations, infos

    def step(
        self, actions: dict[AgentID, dict[str, int]]
    ) -> tuple[
        dict[AgentID, dict[str, Any]],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict[str, Any]],
    ]:
        if not self.agents:
            raise RuntimeError("step() called on a terminated environment; call reset()")
        if set(actions) != set(self.agents):
            raise ValueError(f"actions must have exactly these agents: {self.agents}")

        acting_agents = self.agents.copy()
        parsed_actions = {
            agent: self._parse_action(agent, actions[agent]) for agent in acting_agents
        }

        for agent, (move, _) in parsed_actions.items():
            if move != Move.INTERACT:
                self._agent_positions[agent] = self._moved_position(
                    self._agent_positions[agent], move
                )

        rewards = {agent: float(self.config.step_cost) for agent in acting_agents}
        terminations = {agent: False for agent in acting_agents}
        truncations = {agent: False for agent in acting_agents}

        interactors = {
            agent for agent, (move, _) in parsed_actions.items() if move == Move.INTERACT
        }
        episode_ended = self._resolve_interactions(interactors, rewards)

        self._last_messages = {
            agent: message for agent, (_, message) in parsed_actions.items()
        }
        self._last_moves = {agent: move for agent, (move, _) in parsed_actions.items()}
        self._timestep += 1

        if episode_ended:
            terminations = {agent: True for agent in acting_agents}
        elif self._timestep >= self.config.horizon:
            self._outcome = "timeout"
            truncations = {agent: True for agent in acting_agents}

        observations = {agent: self._observation(agent) for agent in acting_agents}
        infos = {agent: self._info(agent) for agent in acting_agents}

        if episode_ended or any(truncations.values()):
            self.agents = []

        if self.render_mode == "ansi":
            print(self.render())

        return observations, rewards, terminations, truncations, infos

    def state(self) -> NDArray[np.float32]:
        """Return global state suitable for a centralized critic."""

        agent_positions = np.concatenate(
            [self._agent_positions[agent] for agent in self.possible_agents]
        )
        components = [
            agent_positions,
            self._stag_positions.ravel(),
            self._stag_features.ravel(),
            self._hare_positions.ravel(),
            np.asarray([self._correct_color, self._correct_region], dtype=np.int64),
            np.asarray(
                [self._last_messages[agent] for agent in self.possible_agents],
                dtype=np.int64,
            ),
            np.asarray([self._timestep], dtype=np.int64),
        ]
        return np.concatenate(components).astype(np.float32)

    @property
    def correct_target_position(self) -> Position:
        return self._stag_positions[self._correct_target_index].copy()

    @property
    def correct_color(self) -> int:
        return self._correct_color

    @property
    def correct_region(self) -> int:
        return self._correct_region

    @property
    def agent_positions(self) -> dict[AgentID, Position]:
        return {agent: position.copy() for agent, position in self._agent_positions.items()}

    @property
    def hare_positions(self) -> NDArray[np.int64]:
        return self._hare_positions.copy()

    @property
    def stag_positions(self) -> NDArray[np.int64]:
        return self._stag_positions.copy()

    @property
    def stag_features(self) -> NDArray[np.int64]:
        return self._stag_features.copy()

    def color_name(self, color: int) -> str:
        return self._color_name(color)

    def region_name(self, region: int) -> str:
        return self._region_name(region)

    @property
    def outcome(self) -> str:
        return self._outcome

    def render(self) -> str:
        grid: list[list[list[str]]] = [
            [[] for _ in range(self.config.grid_size)] for _ in range(self.config.grid_size)
        ]
        for index, position in enumerate(self._hare_positions):
            x, y = map(int, position)
            grid[y][x].append(f"H{index}")
        for index, position in enumerate(self._stag_positions):
            x, y = map(int, position)
            grid[y][x].append(f"S{index}")
        for index, agent in enumerate(self.possible_agents):
            if agent not in self._agent_positions:
                continue
            x, y = map(int, self._agent_positions[agent])
            grid[y][x].append(f"A{index}")

        header = "    " + " ".join(f"x={x:<5}" for x in range(self.config.grid_size))
        rows = []
        for y, row in enumerate(grid):
            cells = ["+".join(cell) if cell else "." for cell in row]
            rows.append(f"y={y:<2} " + " ".join(f"{cell:<7}" for cell in cells))

        target_lines = []
        for index, (position, feature) in enumerate(
            zip(self._stag_positions, self._stag_features, strict=True)
        ):
            color, region = map(int, feature)
            correct = "  <-- correct (debug)" if index == self._correct_target_index else ""
            target_lines.append(
                f"  S{index} at {tuple(map(int, position))}: "
                f"color={self._color_name(color)}, region={self._region_name(region)}{correct}"
            )

        action_lines = [
            f"  {agent}: position={tuple(map(int, self._agent_positions[agent]))}, "
            f"last_move={self._last_moves[agent].name}, sent={self._last_messages[agent]}, "
            f"received={self._last_messages[self._other_agent(agent)]}"
            for agent in self.possible_agents
        ]
        clue_lines = [
            f"  agent_0 privately sees color={self._color_name(self._correct_color)}",
            f"  agent_1 privately sees region={self._region_name(self._correct_region)}",
        ]

        return "\n".join(
            [
                f"WORLD t={self._timestep}/{self.config.horizon} outcome={self._outcome}",
                header,
                *rows,
                "",
                "Public stag targets:",
                *target_lines,
                "Hares: "
                + ", ".join(
                    f"H{index} at {tuple(map(int, position))}"
                    for index, position in enumerate(self._hare_positions)
                ),
                "",
                "Private clues:",
                *clue_lines,
                "",
                "Agents and channel:",
                *action_lines,
            ]
        )

    def close(self) -> None:
        return None

    def _resolve_interactions(
        self, interactors: set[AgentID], rewards: dict[AgentID, float]
    ) -> bool:
        if not interactors:
            return False

        correct_position = self.correct_target_position
        joint_stag = all(
            agent in interactors
            and np.array_equal(self._agent_positions[agent], correct_position)
            for agent in self.possible_agents
        )
        if joint_stag:
            for agent in self.possible_agents:
                rewards[agent] += self.config.stag_reward
            self._outcome = "joint_stag"
            return True

        committed = False
        hare_hunters: set[AgentID] = set()
        stag_hunters: set[AgentID] = set()
        for agent in interactors:
            position = self._agent_positions[agent]
            if self._position_in(position, self._hare_positions):
                hare_hunters.add(agent)
                committed = True
            elif self._position_in(position, self._stag_positions):
                stag_hunters.add(agent)
                committed = True

        for agent in hare_hunters:
            rewards[agent] += self.config.hare_reward
        for agent in stag_hunters:
            rewards[agent] += self.config.failed_stag_reward

        if committed:
            if hare_hunters and stag_hunters:
                self._outcome = "hare_stag_mismatch"
            elif hare_hunters:
                self._outcome = "hare"
            else:
                self._outcome = "failed_stag"
        return committed

    def _observation(self, agent: AgentID) -> dict[str, Any]:
        other = self._other_agent(agent)
        other_position = (
            self._agent_positions[other].copy()
            if self.config.observe_other_position
            else np.asarray([-1, -1], dtype=np.int64)
        )
        if agent == "agent_0":
            private_clue = np.asarray([self._correct_color + 1, 0], dtype=np.int64)
        else:
            private_clue = np.asarray([0, self._correct_region + 1], dtype=np.int64)

        return {
            "agent_id": self.possible_agents.index(agent),
            "self_position": self._agent_positions[agent].copy(),
            "other_position": other_position,
            "stag_positions": self._stag_positions.copy(),
            "stag_features": self._stag_features.copy(),
            "hare_positions": self._hare_positions.copy(),
            "private_clue": private_clue,
            "received_message": self._last_messages[other],
            "timestep": self._timestep,
        }

    def _info(self, agent: AgentID) -> dict[str, Any]:
        return {"outcome": self._outcome, "agent": agent}

    def _parse_action(self, agent: AgentID, action: dict[str, int]) -> tuple[Move, int]:
        if not self.action_space(agent).contains(action):
            raise ValueError(f"invalid action for {agent}: {action}")
        return Move(int(action["move"])), int(action["message"])

    def _moved_position(self, position: Position, move: Move) -> Position:
        dx, dy = _DELTAS[move]
        moved = position + np.asarray([dx, dy], dtype=np.int64)
        return np.clip(moved, 0, self.config.grid_size - 1).astype(np.int64)

    def _sample_feature(self, value: Any, upper: int, name: str) -> int:
        if value is None:
            return int(self.np_random.integers(upper))
        sampled = int(value)
        if not 0 <= sampled < upper:
            raise ValueError(f"{name} must be in [0, {upper})")
        return sampled

    def _sample_stag_positions(self) -> NDArray[np.int64]:
        """Place every target inside the map region named by its public feature."""

        used: set[tuple[int, int]] = set()
        positions: list[tuple[int, int]] = []
        for _, region in self._stag_features:
            candidates = [
                (x, y)
                for y in range(self.config.grid_size)
                for x in range(self.config.grid_size)
                if self._region_for_x(x) == int(region) and (x, y) not in used
            ]
            if not candidates:
                raise RuntimeError(f"region {region} has too few cells for its stag targets")
            chosen = candidates[int(self.np_random.integers(len(candidates)))]
            used.add(chosen)
            positions.append(chosen)
        return np.asarray(positions, dtype=np.int64)

    def _region_for_x(self, x: int) -> int:
        return min(x * self.config.n_regions // self.config.grid_size, self.config.n_regions - 1)

    def _color_name(self, color: int) -> str:
        if color < len(self.color_names):
            return self.color_names[color]
        return f"color_{color}"

    def _region_name(self, region: int) -> str:
        if self.config.n_regions == 2 and region < len(self.region_names):
            return self.region_names[region]
        return f"region_{region}"

    def _other_agent(self, agent: AgentID) -> AgentID:
        self._validate_agent(agent)
        return "agent_1" if agent == "agent_0" else "agent_0"

    def _validate_agent(self, agent: AgentID) -> None:
        if agent not in self.possible_agents:
            raise ValueError(f"unknown agent: {agent}")

    @staticmethod
    def _position_in(position: Position, candidates: NDArray[np.int64]) -> bool:
        return bool(np.any(np.all(candidates == position, axis=1)))
