"""Configuration for the Stag Hunt environment."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """Parameters that define an environment instance.

    Message `0` is reserved for silence. `vocab_size` counts the non-silent
    symbols, so the message action space has `vocab_size + 1` actions.
    """

    grid_size: int = 7
    horizon: int = 30
    n_colors: int = 2
    n_regions: int = 2
    n_hares: int = 2
    vocab_size: int = 4
    stag_reward: float = 4.0
    hare_reward: float = 2.0
    failed_stag_reward: float = 0.0
    step_cost: float = 0.0
    observe_other_position: bool = True

    def __post_init__(self) -> None:
        if self.grid_size < 3:
            raise ValueError("grid_size must be at least 3")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.n_colors < 2 or self.n_regions < 2:
            raise ValueError("n_colors and n_regions must both be at least 2")
        if self.n_regions > self.grid_size:
            raise ValueError("n_regions cannot exceed grid_size")
        if self.n_hares < 1:
            raise ValueError("n_hares must be positive")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if not self.stag_reward > self.hare_reward > self.failed_stag_reward:
            raise ValueError(
                "payoffs must satisfy stag_reward > hare_reward > failed_stag_reward"
            )

        required_cells = 2 + self.n_stags + self.n_hares
        if required_cells > self.grid_size**2:
            raise ValueError(
                f"grid has {self.grid_size**2} cells but reset requires {required_cells}"
            )

    @property
    def n_stags(self) -> int:
        return self.n_colors * self.n_regions
