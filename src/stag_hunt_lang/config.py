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
    # A stag INTERACT arms a commitment that stays open for commit_window
    # subsequent steps, freezing the committer while the partner may join.
    # 0 restores instant resolution (joint capture requires the same step).
    commit_window: int = 0
    # "interact": captures require the INTERACT action (original semantics).
    # "presence": standing is enough — both agents on the correct stag cell
    # capture it (terminal); joint presence on a wrong stag is a non-terminal
    # no-op, and a single agent on a hare cell captures it. INTERACT becomes
    # a no-op and commit_window must stay 0.
    capture_mode: str = "interact"
    # Sticky channel: the partner's most recent NON-SILENT symbol persists in
    # received_message instead of being overwritten every step. Fights credit
    # dilution — one good symbol keeps informing the listener all episode.
    sticky_messages: bool = False
    # Talk-then-hunt: for the first talk_phase_steps steps, movement and all
    # capture resolution are frozen; only messages flow. Concentrates message
    # credit at the episode start. 0 disables.
    talk_phase_steps: int = 0
    # Randomize which agent holds the color clue each episode (the partner
    # always holds the region clue). The private_clue slot layout already
    # marks the held attribute, so no observation format change is needed.
    # Forces every agent to learn all four speak/decode mappings.
    randomize_clue_assignment: bool = False
    # Stochastic observability: per episode, each agent independently sees BOTH
    # attributes with this probability instead of only its own clue. Naturalistic
    # (perception is sometimes sufficient, sometimes not) and it supplies the
    # gradient the speaker otherwise lacks: in co-observed episodes a listener
    # can ground the partner's symbols against its own view rather than against
    # rare capture reward. 0 disables (the standard blind game).
    co_observation_prob: float = 0.0
    # "private": each agent draws co-observation independently, so an informed
    # agent cannot tell whether its partner is informed too. Measured 2026-07-29
    # to bootstrap nothing — private knowledge is not common knowledge, and in a
    # game that pays only for joint presence, acting on it is a losing gamble.
    # "shared": one draw per episode ("clear" vs "foggy"), so visibility is
    # identical for both agents and hence common knowledge.
    co_observation_mode: str = "private"
    # Small payoff for standing alone on the correct stag, once per episode per
    # agent. Gives "use the target information I hold" a gradient of its own,
    # which joint-only capture reward never provides. Must stay well below
    # stag_reward or solo scouting displaces cooperation.
    solo_presence_reward: float = 0.0

    def __post_init__(self) -> None:
        if self.grid_size < 3:
            raise ValueError("grid_size must be at least 3")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.n_colors < 2 or self.n_regions < 2:
            raise ValueError("n_colors and n_regions must both be at least 2")
        if self.n_regions > self.grid_size:
            raise ValueError("n_regions cannot exceed grid_size")
        if self.n_hares < 0:
            raise ValueError("n_hares must be non-negative")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.commit_window < 0:
            raise ValueError("commit_window must be non-negative")
        if self.capture_mode not in ("interact", "presence"):
            raise ValueError("capture_mode must be 'interact' or 'presence'")
        if self.capture_mode == "presence" and self.commit_window != 0:
            raise ValueError("commit_window is only meaningful with capture_mode='interact'")
        if not 0 <= self.talk_phase_steps < self.horizon:
            raise ValueError("talk_phase_steps must be in [0, horizon)")
        if not 0.0 <= self.co_observation_prob <= 1.0:
            raise ValueError("co_observation_prob must be in [0, 1]")
        if self.co_observation_mode not in ("private", "shared"):
            raise ValueError("co_observation_mode must be 'private' or 'shared'")
        if self.solo_presence_reward < 0:
            raise ValueError("solo_presence_reward must be non-negative")
        if self.solo_presence_reward >= self.stag_reward / 2:
            raise ValueError(
                "solo_presence_reward must stay well below stag_reward/2, "
                "otherwise solo scouting dominates cooperation"
            )
        if not self.stag_reward > self.failed_stag_reward:
            raise ValueError("payoffs must satisfy stag_reward > failed_stag_reward")
        if self.n_hares > 0 and not self.stag_reward > self.hare_reward > self.failed_stag_reward:
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
