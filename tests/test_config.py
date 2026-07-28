import pytest

from stag_hunt_lang import EnvConfig


def test_config_requires_stag_hunt_payoff_ordering() -> None:
    with pytest.raises(ValueError, match="stag_reward > hare_reward"):
        EnvConfig(stag_reward=1.0, hare_reward=2.0)


def test_config_checks_grid_capacity() -> None:
    with pytest.raises(ValueError, match="reset requires"):
        EnvConfig(grid_size=3, n_colors=3, n_regions=3, n_hares=2)


def test_config_hare_ordering_waived_without_hares() -> None:
    config = EnvConfig(n_hares=0, hare_reward=0.0)
    assert config.n_hares == 0
    with pytest.raises(ValueError, match="stag_reward > hare_reward"):
        EnvConfig(n_hares=1, hare_reward=5.0)
    with pytest.raises(ValueError, match="stag_reward > failed_stag_reward"):
        EnvConfig(n_hares=0, failed_stag_reward=5.0)


def test_config_bounds_talk_phase() -> None:
    with pytest.raises(ValueError, match="talk_phase_steps"):
        EnvConfig(talk_phase_steps=30, horizon=30)
    with pytest.raises(ValueError, match="talk_phase_steps"):
        EnvConfig(talk_phase_steps=-1)
    assert EnvConfig(talk_phase_steps=5).talk_phase_steps == 5


def test_config_rejects_presence_mode_with_commit_window() -> None:
    with pytest.raises(ValueError, match="commit_window"):
        EnvConfig(capture_mode="presence", commit_window=2)
    with pytest.raises(ValueError, match="capture_mode"):
        EnvConfig(capture_mode="proximity")

