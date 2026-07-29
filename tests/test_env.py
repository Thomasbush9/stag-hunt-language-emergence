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


def test_commit_window_lets_partner_join_late() -> None:
    env = StagHuntLanguageEnv(EnvConfig(commit_window=3))
    env.reset(seed=9)
    target = env.correct_target_position
    env._agent_positions = {agent: target.copy() for agent in env.possible_agents}

    _, rewards, terminations, _, infos = env.step(
        {"agent_0": action(Move.INTERACT), "agent_1": action(Move.STAY)}
    )
    assert rewards == {"agent_0": 0.0, "agent_1": 0.0}
    assert not any(terminations.values())
    assert infos["agent_0"]["outcome"] == "ongoing"

    _, rewards, terminations, _, infos = env.step(
        {"agent_0": action(Move.STAY), "agent_1": action(Move.INTERACT)}
    )
    assert rewards == {"agent_0": 4.0, "agent_1": 4.0}
    assert all(terminations.values())
    assert infos["agent_0"]["outcome"] == "joint_stag"


def test_commit_window_freezes_the_committer() -> None:
    env = StagHuntLanguageEnv(EnvConfig(commit_window=3))
    env.reset(seed=10)
    target = env.correct_target_position
    env._agent_positions["agent_0"] = target.copy()

    env.step({"agent_0": action(Move.INTERACT), "agent_1": action(Move.STAY)})
    env.step({"agent_0": action(Move.LEFT), "agent_1": action(Move.STAY)})
    np.testing.assert_array_equal(env.agent_positions["agent_0"], target)


def test_commit_window_expiry_fails_the_committer() -> None:
    env = StagHuntLanguageEnv(EnvConfig(commit_window=2))
    env.reset(seed=11)
    env._agent_positions["agent_0"] = env.correct_target_position

    _, _, terminations, _, _ = env.step(
        {"agent_0": action(Move.INTERACT), "agent_1": action(Move.STAY)}
    )
    assert not any(terminations.values())
    _, _, terminations, _, _ = env.step(
        {"agent_0": action(Move.STAY), "agent_1": action(Move.STAY)}
    )
    assert not any(terminations.values())

    _, rewards, terminations, _, infos = env.step(
        {"agent_0": action(Move.STAY), "agent_1": action(Move.STAY)}
    )
    assert rewards == {"agent_0": 0.0, "agent_1": 0.0}
    assert all(terminations.values())
    assert infos["agent_0"]["outcome"] == "failed_stag"


def test_commit_window_messages_still_flow_while_holding() -> None:
    env = StagHuntLanguageEnv(EnvConfig(commit_window=3))
    env.reset(seed=12)
    env._agent_positions["agent_0"] = env.correct_target_position

    env.step({"agent_0": action(Move.INTERACT), "agent_1": action(Move.STAY)})
    observations, _, _, _, _ = env.step(
        {"agent_0": action(Move.STAY, message=4), "agent_1": action(Move.STAY)}
    )
    assert observations["agent_1"]["received_message"] == 4


def test_presence_capture_at_correct_stag_is_terminal_and_cooperative() -> None:
    env = StagHuntLanguageEnv(EnvConfig(capture_mode="presence", n_hares=0))
    env.reset(seed=3)
    target = env.correct_target_position
    env._agent_positions = {agent: target.copy() for agent in env.possible_agents}

    _, rewards, terminations, _, infos = env.step(
        {agent: action(Move.STAY) for agent in env.agents}
    )

    assert rewards == {"agent_0": 4.0, "agent_1": 4.0}
    assert all(terminations.values())
    assert infos["agent_0"]["outcome"] == "joint_stag"
    assert infos["agent_0"]["first_joint_presence"] == "correct"
    assert env.agents == []


def test_presence_on_wrong_stag_is_non_terminal_and_tracked() -> None:
    env = StagHuntLanguageEnv(EnvConfig(capture_mode="presence", n_hares=0))
    env.reset(seed=4)
    wrong_index = next(
        index
        for index in range(env.config.n_stags)
        if index != env._correct_target_index
    )
    wrong = env.stag_positions[wrong_index]
    env._agent_positions = {agent: wrong.copy() for agent in env.possible_agents}

    for expected_steps in (1, 2):
        _, rewards, terminations, _, infos = env.step(
            {agent: action(Move.STAY) for agent in env.agents}
        )
        assert rewards == {"agent_0": 0.0, "agent_1": 0.0}
        assert not any(terminations.values())
        assert infos["agent_0"]["outcome"] == "ongoing"
        assert infos["agent_0"]["first_joint_presence"] == "wrong"
        assert infos["agent_0"]["wrong_presence_steps"] == expected_steps


def test_presence_solo_on_stag_does_nothing_and_interact_is_a_noop() -> None:
    env = StagHuntLanguageEnv(EnvConfig(capture_mode="presence", n_hares=0))
    env.reset(seed=5)
    env._agent_positions["agent_0"] = env.correct_target_position

    _, rewards, terminations, _, infos = env.step(
        {
            "agent_0": action(Move.INTERACT),
            "agent_1": action(Move.STAY),
        }
    )

    assert rewards == {"agent_0": 0.0, "agent_1": 0.0}
    assert not any(terminations.values())
    assert infos["agent_0"]["outcome"] == "ongoing"
    assert infos["agent_0"]["first_joint_presence"] is None


def test_presence_hare_capture_by_standing() -> None:
    env = StagHuntLanguageEnv(EnvConfig(capture_mode="presence"))
    env.reset(seed=6)
    env._agent_positions["agent_0"] = env._hare_positions[0].copy()

    _, rewards, terminations, _, infos = env.step(
        {agent: action(Move.STAY) for agent in env.agents}
    )

    assert rewards == {"agent_0": 2.0, "agent_1": 0.0}
    assert all(terminations.values())
    assert infos["agent_0"]["outcome"] == "hare"


def test_hare_free_world_has_empty_hare_observations() -> None:
    env = StagHuntLanguageEnv(EnvConfig(n_hares=0))
    observations, _ = env.reset(seed=7)
    assert observations["agent_0"]["hare_positions"].shape == (0, 2)
    assert env.state().shape == (11 + 4 * env.config.n_stags,)


def test_sticky_messages_persist_through_silence() -> None:
    env = StagHuntLanguageEnv(EnvConfig(sticky_messages=True))
    env.reset(seed=8)

    observations, *_ = env.step(
        {"agent_0": action(Move.STAY, message=3), "agent_1": action(Move.STAY)}
    )
    assert observations["agent_1"]["received_message"] == 3

    for _ in range(3):
        observations, *_ = env.step(
            {agent: action(Move.STAY, message=0) for agent in env.agents}
        )
        assert observations["agent_1"]["received_message"] == 3

    observations, *_ = env.step(
        {"agent_0": action(Move.STAY, message=1), "agent_1": action(Move.STAY)}
    )
    assert observations["agent_1"]["received_message"] == 1


def test_non_sticky_messages_reset_on_silence() -> None:
    env = StagHuntLanguageEnv()
    env.reset(seed=8)
    env.step({"agent_0": action(Move.STAY, message=3), "agent_1": action(Move.STAY)})
    observations, *_ = env.step(
        {agent: action(Move.STAY, message=0) for agent in env.agents}
    )
    assert observations["agent_1"]["received_message"] == 0


def test_talk_phase_freezes_movement_and_capture_but_not_messages() -> None:
    env = StagHuntLanguageEnv(
        EnvConfig(capture_mode="presence", n_hares=0, talk_phase_steps=2)
    )
    env.reset(seed=9)
    # Even standing on the correct stag must not resolve during the talk phase.
    env._agent_positions = {
        agent: env.correct_target_position for agent in env.possible_agents
    }

    observations, rewards, terminations, _, _ = env.step(
        {"agent_0": action(Move.UP, message=2), "agent_1": action(Move.LEFT)}
    )
    assert not any(terminations.values())
    assert rewards == {"agent_0": 0.0, "agent_1": 0.0}
    assert observations["agent_1"]["received_message"] == 2
    np.testing.assert_array_equal(
        env.agent_positions["agent_0"], env.correct_target_position
    )

    # Second talk step: still frozen.
    _, _, terminations, _, _ = env.step(
        {agent: action(Move.UP) for agent in env.agents}
    )
    assert not any(terminations.values())

    # Talk phase over: presence resolution fires on the next step.
    _, rewards, terminations, _, infos = env.step(
        {agent: action(Move.STAY) for agent in env.agents}
    )
    assert all(terminations.values())
    assert infos["agent_0"]["outcome"] == "joint_stag"
    assert rewards == {"agent_0": 4.0, "agent_1": 4.0}


def test_randomized_clue_assignment_is_complementary_and_varies() -> None:
    env = StagHuntLanguageEnv(EnvConfig(randomize_clue_assignment=True))
    holders = set()
    for seed in range(20):
        observations, infos = env.reset(seed=seed)
        holder = env.color_holder
        other = "agent_1" if holder == "agent_0" else "agent_0"
        holders.add(holder)
        assert infos["agent_0"]["color_holder"] == holder
        assert observations[holder]["private_clue"][0] == env.correct_color + 1
        assert observations[holder]["private_clue"][1] == 0
        assert observations[other]["private_clue"][0] == 0
        assert observations[other]["private_clue"][1] == env.correct_region + 1
    assert holders == {"agent_0", "agent_1"}
    # Assignment bit is appended to the global state.
    assert env.state().shape[0] == 12 + 4 * env.config.n_stags + 2 * env.config.n_hares


def test_fixed_clue_assignment_by_default() -> None:
    env = StagHuntLanguageEnv()
    for seed in range(5):
        env.reset(seed=seed)
        assert env.color_holder == "agent_0"


def test_parallel_environment_api() -> None:
    parallel_api_test(StagHuntLanguageEnv(), num_cycles=100)


def test_parallel_environment_api_with_commit_window() -> None:
    parallel_api_test(StagHuntLanguageEnv(EnvConfig(commit_window=3)), num_cycles=100)


def test_parallel_environment_api_presence_no_hares() -> None:
    parallel_api_test(
        StagHuntLanguageEnv(EnvConfig(capture_mode="presence", n_hares=0)),
        num_cycles=100,
    )


def test_co_observation_reveals_both_attributes_and_varies() -> None:
    env = StagHuntLanguageEnv(
        EnvConfig(randomize_clue_assignment=True, co_observation_prob=0.5)
    )
    seen = set()
    for seed in range(40):
        observations, infos = env.reset(seed=seed)
        co = infos["agent_0"]["co_observes"]
        for agent, observation in observations.items():
            color, region = observation["private_clue"]
            if co[agent]:
                # A co-observing agent sees both attributes itself.
                assert color == env.correct_color + 1
                assert region == env.correct_region + 1
            elif agent == env.color_holder:
                assert (color, region) == (env.correct_color + 1, 0)
            else:
                assert (color, region) == (0, env.correct_region + 1)
            seen.add(co[agent])
    # Both co-observing and blind episodes occur at p=0.5.
    assert seen == {True, False}
    # Two co-observation flags are appended to the global state.
    assert env.state().shape[0] == 14 + 4 * env.config.n_stags + 2 * env.config.n_hares


def test_co_observation_disabled_by_default() -> None:
    env = StagHuntLanguageEnv(EnvConfig())
    for seed in range(5):
        observations, infos = env.reset(seed=seed)
        assert infos["agent_0"]["co_observes"] == {"agent_0": False, "agent_1": False}
        assert observations["agent_0"]["private_clue"][1] == 0
        assert observations["agent_1"]["private_clue"][0] == 0
    assert env.state().shape[0] == 11 + 4 * env.config.n_stags + 2 * env.config.n_hares


def test_co_observation_prob_one_reveals_everything() -> None:
    env = StagHuntLanguageEnv(EnvConfig(co_observation_prob=1.0))
    observations, _ = env.reset(seed=0)
    for observation in observations.values():
        color, region = observation["private_clue"]
        assert (color, region) == (env.correct_color + 1, env.correct_region + 1)
