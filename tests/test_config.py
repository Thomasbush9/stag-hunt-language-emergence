import pytest

from stag_hunt_lang import EnvConfig


def test_config_requires_stag_hunt_payoff_ordering() -> None:
    with pytest.raises(ValueError, match="stag_reward > hare_reward"):
        EnvConfig(stag_reward=1.0, hare_reward=2.0)


def test_config_checks_grid_capacity() -> None:
    with pytest.raises(ValueError, match="reset requires"):
        EnvConfig(grid_size=3, n_colors=3, n_regions=3, n_hares=2)

