from __future__ import annotations

from typing import Any

import numpy as np
from pettingzoo.test import parallel_api_test

from stag_hunt_lang import EnvConfig, Move, StagHuntLanguageEnv


def action(move: Move, message: int = 0) -> dict[str, int]:
    return {"move": int(move), "message": message}


def assert_observations_equal(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]
) -> None:
    for agent in left:
        for key, value in left[agent].items():
            np.testing.assert_array_equal(value, right[agent][key])


def test_reset_is_deterministic_for_seed() -> None:
    env = StagHuntLanguageEnv()
    first, _ = env.reset(seed=17)
    second, _ = env.reset(seed=17)
    assert_observations_equal(first, second)


def test_agents_receive_complementary_private_clues() -> None:
    env = StagHuntLanguageEnv()
    observations, _ = env.reset(
        seed=1, options={"correct_color": 1, "correct_region": 0}
    )

    np.testing.assert_array_equal(observations["agent_0"]["private_clue"], [2, 0])
    np.testing.assert_array_equal(observations["agent_1"]["private_clue"], [0, 1])

    target_index = np.flatnonzero(
        np.all(observations["agent_0"]["stag_features"] == [1, 0], axis=1)
    )[0]
    np.testing.assert_array_equal(
        env.correct_target_position,
        observations["agent_0"]["stag_positions"][target_index],
    )


def test_stag_region_feature_matches_physical_map_region() -> None:
    env = StagHuntLanguageEnv()
    observations, _ = env.reset(seed=8)

    for position, feature in zip(
        observations["agent_0"]["stag_positions"],
        observations["agent_0"]["stag_features"],
        strict=True,
    ):
        x = int(position[0])
        region = int(feature[1])
        assert env._region_for_x(x) == region



def test_messages_are_delivered_to_partner_on_next_observation() -> None:
    env = StagHuntLanguageEnv()
    env.reset(seed=2)

    observations, _, _, _, _ = env.step(
        {
            "agent_0": action(Move.STAY, message=2),
            "agent_1": action(Move.STAY, message=3),
        }
    )

    assert observations["agent_0"]["received_message"] == 3
    assert observations["agent_1"]["received_message"] == 2


def test_joint_interaction_at_correct_stag_gets_cooperative_payoff() -> None:
    env = StagHuntLanguageEnv()
    env.reset(seed=3)
    target = env.correct_target_position
    env._agent_positions = {agent: target.copy() for agent in env.possible_agents}

    _, rewards, terminations, truncations, infos = env.step(
        {agent: action(Move.INTERACT) for agent in env.agents}
    )

    assert rewards == {"agent_0": 4.0, "agent_1": 4.0}
    assert all(terminations.values())
    assert not any(truncations.values())
    assert infos["agent_0"]["outcome"] == "joint_stag"
    assert env.agents == []


def test_solo_stag_commitment_fails_and_terminates() -> None:
    env = StagHuntLanguageEnv()
    env.reset(seed=4)
    env._agent_positions["agent_0"] = env.correct_target_position

    _, rewards, terminations, _, infos = env.step(
        {
            "agent_0": action(Move.INTERACT),
            "agent_1": action(Move.STAY),
        }
    )

    assert rewards == {"agent_0": 0.0, "agent_1": 0.0}
    assert all(terminations.values())
    assert infos["agent_0"]["outcome"] == "failed_stag"


def test_hare_is_safe_individual_payoff() -> None:
    env = StagHuntLanguageEnv()
    env.reset(seed=5)
    env._agent_positions["agent_0"] = env._hare_positions[0].copy()

    _, rewards, terminations, _, infos = env.step(
        {
            "agent_0": action(Move.INTERACT),
            "agent_1": action(Move.STAY),
        }
    )

    assert rewards == {"agent_0": 2.0, "agent_1": 0.0}
    assert all(terminations.values())
    assert infos["agent_0"]["outcome"] == "hare"


def test_hidden_partner_position_removes_movement_signal() -> None:
    env = StagHuntLanguageEnv(EnvConfig(observe_other_position=False))
    observations, _ = env.reset(seed=6)
    np.testing.assert_array_equal(observations["agent_0"]["other_position"], [-1, -1])
    np.testing.assert_array_equal(observations["agent_1"]["other_position"], [-1, -1])


def test_parallel_environment_api() -> None:
    parallel_api_test(StagHuntLanguageEnv(), num_cycles=100)
