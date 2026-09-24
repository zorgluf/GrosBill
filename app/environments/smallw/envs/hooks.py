"""Race and special-power **hook interface** for Small World (`smallw`).

The engine (`smallw.py`) implements only the *vanilla* rules: every race
behaves like the Ratmen (no special ability) and every power is a no-op. Every
race- or power-specific behaviour goes through one of the hooks defined here,
so the engine stays generic.

Two small class hierarchies:

* :class:`Hooks` — every hook the engine ever calls, all of them no-ops or
  vanilla defaults. :class:`RaceHooks` (races) and :class:`PowerHooks` (powers)
  derive from it; `PowerHooks` adds the end-of-turn sub-phase hooks
  (encampments / fortress / heroes / ally / stout decline), which only powers
  ever trigger.
* two registries, :data:`RACE_HOOKS` and :data:`POWER_HOOKS`, pre-filled with
  the vanilla classes, plus :func:`get_race_hooks` / :func:`get_power_hooks`
  (the latter maps ``None`` — a declined race has no power — to
  :data:`NO_POWER_HOOKS`).

T4 (`races.py`) and T5 (`powers.py`) subclass and register:

    class ElvesHooks(RaceHooks):
        def defender_loss(self, env, victim_rip, region):
            return 0            # the Elves never lose a token

    register_race_hooks(RaceId.ELVES, ElvesHooks())

Every hook receives the environment (except the pure flags, which only need the
race) and the acting :class:`~.classes.RaceInPlay`, so an implementation can
inspect the whole game state (`env.board`, `env.players`, `env.tray`, ...).
Hooks must be **stateless**: per-race state belongs in `RaceInPlay.extra`
(reset by nothing) or in the per-turn fields reset by `RaceInPlay.new_turn()`.
The registries hold shared singletons and are never deep-copied with the env.

How the engine combines the race hook and the power hook
--------------------------------------------------------
* `conquest_cost_modifier`: chained — the race hook first, then the power hook
  (each sees the running cost); the engine then clamps with ``max(1, cost)``
  and subtracts a Berserk die if one is in play (clamped to 1 again).
* `can_target` / `is_adjacent`: three-valued — any ``True`` wins, otherwise any
  ``False`` wins, otherwise the vanilla default applies (not water / adjacent
  to a region held by the acting race).
* boolean flags (`ignores_border_rule`, `die_before_each_conquest`,
  `keeps_tokens_in_decline`, `conquers_in_decline`, `losses_like_active`,
  `is_spirit_power`, `bonus_in_decline`, `dragon_available`): logical OR.
* `sorcerer_targets`: union.
* `defender_loss` / `redeployable_tokens`: minimum — the most protective hook
  wins (`redeployable_tokens` is additionally clamped to `tokens_in_hand`).
* `score_bonus`: sum.
* side-effect hooks (`on_*`): both are called, the race hook first.
* the sub-phase hooks are read from the **power** hook only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .classes import PowerId, RaceId

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .classes import RaceInPlay, Region
    from .smallw import SmallWorldEnv

__all__ = [
    'Hooks', 'RaceHooks', 'PowerHooks', 'NoPowerHooks',
    'RACE_HOOKS', 'POWER_HOOKS', 'NO_POWER_HOOKS',
    'get_race_hooks', 'get_power_hooks',
    'register_race_hooks', 'register_power_hooks',
]


# --------------------------------------------------------------------------- #
# The hook interface
# --------------------------------------------------------------------------- #

class Hooks:
    """Every hook the engine calls, with the vanilla (no-ability) behaviour.

    `rip` is always the :class:`~.classes.RaceInPlay` the hook belongs to — the
    acting race for the conquest / turn hooks, the *victim* for
    :meth:`on_region_lost` and :meth:`defender_loss`.
    """

    #: Human-readable name, only used by :meth:`__repr__` and the logs.
    name = 'vanilla'

    # -- costs and targets -------------------------------------------------- #

    def conquest_cost_modifier(self, env: 'SmallWorldEnv', rip: 'RaceInPlay',
                               region: 'Region', cost: int) -> int:
        """Return the conquest cost of `region` for `rip`, given `cost` so far.

        Called with `region.base_conquest_cost()` for the race hook, then with
        whatever the race hook returned for the power hook. The engine clamps
        the result to at least 1 (and then applies a Berserk die).
        Giants / Tritons / Commando / Mounted / Underworld live here.
        """
        return cost

    def can_target(self, env: 'SmallWorldEnv', rip: 'RaceInPlay',
                   region: 'Region') -> bool | None:
        """May `rip` attack `region` at all (terrain-wise)?

        Three-valued: ``None`` = no opinion (the default rule applies: every
        region but the seas and the lake), ``True`` = yes (Seafaring on water,
        Flying on any non-water region), ``False`` = never.
        """
        return None

    def is_adjacent(self, env: 'SmallWorldEnv', rip: 'RaceInPlay',
                    region: 'Region') -> bool | None:
        """Is `region` adjacent to `rip`'s territory (for a non-first conquest)?

        Three-valued like :meth:`can_target`. ``None`` = the printed adjacency
        applies. Flying returns True for everything, the Underworld returns
        True when `region` and one of the regions held are both Caverns.
        """
        return None

    def ignores_border_rule(self, rip: 'RaceInPlay') -> bool:
        """True if the *first* conquest may target any region, not only a border
        one (Halflings, Flying)."""
        return False

    def die_before_each_conquest(self, rip: 'RaceInPlay') -> bool:
        """True if the reinforcement die is rolled before **every** conquest
        decision and reduces the cost (Berserk)."""
        return False

    def sorcerer_targets(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> set[int]:
        """Region **ids** on which `rip` may perform a Sorcerer substitution.

        Vanilla: empty. The hook owns the whole legality check (lone active
        enemy token, adjacency, one substitution per opponent per turn —
        `rip.sorcerer_used_on` —, a token left in the tray).
        """
        return set()

    def dragon_available(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> bool:
        """True if a Dragon Master conquest is still available this turn."""
        return False

    # -- events ------------------------------------------------------------- #

    def on_conquered(self, env: 'SmallWorldEnv', rip: 'RaceInPlay',
                     region: 'Region', defender_info: dict) -> None:
        """`rip` has just taken `region` (tokens already placed).

        `defender_info` describes what was there *before* the conquest:
        ``kind`` (``'normal' | 'final' | 'dragon'``), ``lost_tribe``, ``owner``,
        ``race``, ``in_decline``, ``tokens``, ``was_empty``, ``victim`` (the
        defending `RaceInPlay` or None) and ``defender`` (a label for the log).
        Halfling holes and Troll lairs are placed here.
        """
        return None

    def on_region_lost(self, env: 'SmallWorldEnv', victim_rip: 'RaceInPlay',
                       region: 'Region') -> None:
        """`victim_rip` has just lost `region` (already cleared, attacker not in
        yet). Called with the **victim's** hooks."""
        return None

    def defender_loss(self, env: 'SmallWorldEnv', victim_rip: 'RaceInPlay',
                      region: 'Region') -> int:
        """Tokens `victim_rip` loses to the tray when `region` is conquered.

        Vanilla 1 (the rest goes back to the victim's hand for the end-of-turn
        redeployment); the Elves return 0. Called with the victim's hooks, and
        the engine takes the minimum of the race and power answers.
        """
        return 1

    def on_abandon(self, env: 'SmallWorldEnv', rip: 'RaceInPlay',
                   region: 'Region') -> None:
        """`rip` has just abandoned `region` (tokens back in hand, markers
        already returned to the supply)."""
        return None

    def on_redeploy_start(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> None:
        """Called once when `rip` starts its own redeployment (not for the
        end-of-turn redeployment of a victim). Skeletons grow here."""
        return None

    def redeployable_tokens(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> int:
        """How many of `rip.tokens_in_hand` must be placed on the board.

        Vanilla: all of them. The Amazons keep their 4 attack-only tokens in
        hand, so they return ``tokens_in_hand - 4``. The engine takes the
        minimum of the race and power answers, clamped to `tokens_in_hand`.
        """
        return rip.tokens_in_hand

    def on_turn_start(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> None:
        """Called once per turn for every race of the current player that acts
        (after `RaceInPlay.new_turn()`), and for a freshly picked race."""
        return None

    def on_turn_end(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> None:
        """Called at the very end of the current player's turn (after scoring
        and the optional Stout decline). Wealthy pays here."""
        return None

    def on_decline(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> None:
        """Called when `rip` goes in decline, *before* its power badge is
        discarded and its flags flipped."""
        return None

    def keeps_tokens_in_decline(self, rip: 'RaceInPlay') -> bool:
        """True if the race keeps **all** its tokens when it declines (Ghouls)
        instead of one per region."""
        return False

    def conquers_in_decline(self, rip: 'RaceInPlay') -> bool:
        """True if the race still conquers while in decline (Ghouls), before
        its owner's active race plays."""
        return False

    def losses_like_active(self, rip: 'RaceInPlay') -> bool:
        """True if, while in decline, the race loses tokens like an active race
        (one to the tray, the rest back in hand + redeployment) instead of
        being wiped out region by region (declined Ghouls)."""
        return False

    def is_spirit_power(self, rip: 'RaceInPlay') -> bool:
        """True if the race declines with the Spirit power, i.e. it does not
        count toward the one-declined-race-per-player limit."""
        return False

    # -- scoring ------------------------------------------------------------ #

    def score_bonus(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> int:
        """Extra coins for `rip` at its owner's scoring step.

        Called for an active race, and for a declined race only when
        :meth:`bonus_in_decline` is True. The engine adds the region count
        itself; this is the race/power bonus on top (Dwarves' mines, Humans'
        farmlands, Alchemist, Merchant, Fortified, ...).
        """
        return 0

    def bonus_in_decline(self, rip: 'RaceInPlay') -> bool:
        """True if :meth:`score_bonus` still applies once the race declined
        (Dwarves)."""
        return False

    def __repr__(self) -> str:
        return f'<{type(self).__name__} {self.name}>'


class RaceHooks(Hooks):
    """Hooks attached to a race banner (`RACE_HOOKS[race_id]`)."""


class PowerHooks(Hooks):
    """Hooks attached to a special power badge (`POWER_HOOKS[power_id]`).

    Adds the optional end-of-turn sub-phases: the engine walks
    ENCAMPMENTS → FORTRESS → HEROES → ALLY after the redeployment and enters
    each phase only if the corresponding hook asks for it (and there is at
    least one legal placement), then scores and offers STOUT_DECLINE.
    """

    def encampments_to_place(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> int:
        """How many encampments `rip` places this turn (Bivouacking: 5).

        The engine first takes every encampment of that race back from the
        board into the supply, so the player re-places all of them each turn.
        """
        return 0

    def can_place_fortress(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> bool:
        """True if `rip` may place a fortress this turn (Fortified, once per
        turn — the engine also checks `rip.fortress_used` and the supply)."""
        return False

    def heroes_to_place(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> int:
        """How many heroes `rip` places this turn (Heroic: 2).

        The engine first removes that race's heroes from the board.
        """
        return 0

    def has_ally_choice(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> bool:
        """True if `rip` ends its turn by choosing an ally (Diplomat)."""
        return False

    def can_stout_decline(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> bool:
        """True if `rip` may go in decline after having played a normal turn
        (Stout)."""
        return False


class NoPowerHooks(PowerHooks):
    """The hooks of a race that has no power any more (declined): all vanilla."""

    name = 'no power'


# --------------------------------------------------------------------------- #
# Registries
# --------------------------------------------------------------------------- #

#: Hooks of the 14 races, keyed by :class:`~.classes.RaceId`. Pre-filled with
#: the vanilla :class:`RaceHooks`; T4 replaces the entries it implements.
RACE_HOOKS: dict[RaceId, RaceHooks] = {rid: RaceHooks() for rid in RaceId}

#: Hooks of the 20 special powers, keyed by :class:`~.classes.PowerId`.
#: Pre-filled with the vanilla :class:`PowerHooks`; T5 replaces them.
POWER_HOOKS: dict[PowerId, PowerHooks] = {pid: PowerHooks() for pid in PowerId}

#: Hooks used when a race has no power (declined).
NO_POWER_HOOKS: NoPowerHooks = NoPowerHooks()


def get_race_hooks(race_id: RaceId) -> RaceHooks:
    """Hooks of `race_id` (vanilla ones if nothing was registered)."""
    return RACE_HOOKS[RaceId(race_id)]


def get_power_hooks(power_id: PowerId | None) -> PowerHooks:
    """Hooks of `power_id`, or :data:`NO_POWER_HOOKS` when it is ``None``."""
    if power_id is None:
        return NO_POWER_HOOKS
    return POWER_HOOKS[PowerId(power_id)]


def register_race_hooks(race_id: RaceId, hooks: RaceHooks) -> None:
    """Register the hooks of one race (called from `races.py` at import time)."""
    if not isinstance(hooks, RaceHooks):
        raise TypeError(f'{hooks!r} is not a RaceHooks instance')
    RACE_HOOKS[RaceId(race_id)] = hooks


def register_power_hooks(power_id: PowerId, hooks: PowerHooks) -> None:
    """Register the hooks of one power (called from `powers.py` at import time)."""
    if not isinstance(hooks, PowerHooks):
        raise TypeError(f'{hooks!r} is not a PowerHooks instance')
    POWER_HOOKS[PowerId(power_id)] = hooks
