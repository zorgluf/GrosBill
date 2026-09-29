"""Special-power behaviours of Small World (`smallw`) — task **T5**.

The engine (`smallw.py`) only knows the vanilla rules: every power is a no-op.
Each of the 20 base-game badges gets a :class:`~.hooks.PowerHooks` subclass
here, registered at import time with :func:`~.hooks.register_power_hooks`;
`smallw.py` imports this module for that side effect, so a power becomes
active as soon as it is registered.

Hooks are **stateless singletons** (never deep-copied with the env): every bit
of per-race state lives on the acting :class:`~.classes.RaceInPlay`
(`dragon_used`, `fortress_used`, `wealthy_paid`, `nonempty_conquests`,
`turns_played`, ... — see `classes.py`) or on the board.

"Active only"
-------------
Every power is discarded when its race goes in decline (`_decline` sets
`RaceInPlay.power = None` and pushes the badge on the discard pile), and
`get_power_hooks(None)` then answers with the vanilla
:class:`~.hooks.NoPowerHooks`. **A power hook is therefore never called for a
declined race** and none of the classes below needs an `in_decline` guard.
The two exceptions are called *before* the badge is discarded, while
`rip.in_decline` is still False: :meth:`~.hooks.Hooks.is_spirit_power` (Spirit)
and :meth:`~.hooks.Hooks.on_decline`.

Rules implemented (plan 2.1, `docs/SmallWorld_v3.pdf` p.2-4)
------------------------------------------------------------
=============  ==============================================================
Alchemist      +2 coins per turn
Berserk        the reinforcement die is rolled before **every** conquest and
               reduces its cost (min 1)
Bivouacking    5 Encampments re-deployed every turn, +1 defence each, they
               protect a lone token from the Sorcerers
Commando       -1 on every conquest
Diplomat       at end of turn, ally with an opponent not attacked this turn;
               he may not attack the Diplomat's *active* race until the
               Diplomat's next turn
Dragon Master  once per turn, conquer any attackable region with 1 token
               whatever its defence; the region is immune while the Dragon
               stays there
Flying         conquer any region except the seas and the lake, adjacency and
               the first-conquest border rule waived
Forest         +1 coin per Forest region
Fortified      once per turn, 1 Fortress in a region held (max 1 per region,
               6 on the map): +1 defence (even in decline), +1 coin each
Heroic         the 2 Heroes are re-placed every turn in 2 distinct regions
               held; a Hero makes its region immune
Hill           +1 coin per Hill region
Merchant       +1 coin per region held
Mounted        -1 on Hill and Farmland regions
Pillaging      +1 coin per non-empty region conquered this turn
Seafaring      the seas and the lake may be conquered like empty regions and
               are kept in decline
Spirit         the declined race does not count toward the one-declined-race
               limit
Stout          may go in decline at the end of a normal turn, after scoring
Swamp          +1 coin per Swamp region
Underworld     -1 on Cavern regions; every Cavern is adjacent to every Cavern
Wealthy        +7 coins at the end of the first turn with this power
=============  ==============================================================

Every cost modifier is chained after the race's and clamped to at least 1 by
the engine, every `score_bonus` is summed with the race's, `can_target` /
`is_adjacent` are three-valued and the boolean flags are OR-ed — see the
combination rules in `hooks.py`.

Simplifications and rulings (also listed in the plan's T5 log entry)
--------------------------------------------------------------------
* **Berserk** never makes a "final attempt": its die is already part of the
  cost, so `_conquest_options` never offers the short-by-3 gamble (plan
  section 5 item 8). The die is not applied to a Sorcerer substitution nor to
  a Dragon Master conquest (neither pays a cost).
* **Flying / Underworld and the Sorcerers**: a substitution needs the target
  to be "adjacent to the Sorcerers", so `SorcerersHooks.sorcerer_targets`
  resolves that reach through the engine's three-valued `is_adjacent` hook
  (T4 left this as an open point). Flying Sorcerers may therefore substitute
  anywhere and Underworld Sorcerers through the cavern links; an immune or
  encamped region is still never a target.
* **Seafaring** needs no cost modifier: a water region carries no Mountain, no
  Lost Tribe and no symbol, so the generic `2 + defence` formula already gives
  2 for an empty sea or lake, and enemy tokens / markers on the water defend
  normally, exactly like on land.
* **Spirit** twice: a player may end up with more than two declined races.
  They are all kept in the state (and all score), but the `players`
  observation only has two declined slots and shows the first two.
* **Wealthy** pays at the end of the *first* turn of the race
  (`turns_played == 1`, which is the turn the combo was picked). If the race
  declines during that turn the badge is already discarded when
  `on_turn_end` fires, so nothing is paid (plan: no bonus on the decline turn).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .classes import N_ENCAMPMENTS, N_HEROES, PowerId, Symbol, Terrain
from .hooks import PowerHooks, register_power_hooks
# `_terrain_bonus` counts the regions of one terrain held by a race; the plan
# (T4 progress log) explicitly offers it to the T5 terrain powers.
from .races import _terrain_bonus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .classes import RaceInPlay, Region
    from .smallw import SmallWorldEnv

__all__ = [
    'AlchemistHooks', 'BerserkHooks', 'BivouackingHooks', 'CommandoHooks',
    'DiplomatHooks', 'DragonMasterHooks', 'FlyingHooks', 'ForestHooks',
    'FortifiedHooks', 'HeroicHooks', 'HillHooks', 'MerchantHooks',
    'MountedHooks', 'PillagingHooks', 'SeafaringHooks', 'SpiritHooks',
    'StoutHooks', 'SwampHooks', 'UnderworldHooks', 'WealthyHooks',
    'ALCHEMIST_BONUS', 'WEALTHY_BONUS', 'POWER_HOOK_INSTANCES',
]

#: Coins the Alchemist adds at every scoring step of its race.
ALCHEMIST_BONUS = 2

#: Coins the Wealthy power pays once, at the end of the race's first turn.
WEALTHY_BONUS = 7


# --------------------------------------------------------------------------- #
# The 20 special powers
# --------------------------------------------------------------------------- #

class AlchemistHooks(PowerHooks):
    """Alchemist (4): *"+2 coins per turn while active."*

    The bonus does not depend on the regions held: an Alchemist race that was
    wiped off the map still collects its 2 coins. The badge is discarded at
    decline, so the turn the race declines pays nothing extra.
    """

    name = 'Alchemist'

    def score_bonus(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> int:
        return ALCHEMIST_BONUS


class BerserkHooks(PowerHooks):
    """Berserk (4): *"roll the die before each conquest; it reduces the cost."*

    `die_before_each_conquest` makes the engine draw `rip.die` when the
    conquest phase opens and again after every resolved conquest
    (`_roll_berserk`), and `_conquest_cost` subtracts it with a floor of 1. The
    masks are therefore built with the die already known, which is why a
    Berserk race never gets the "final attempt" gamble (plan section 5 item 8):
    `_conquest_options` only offers it while `rip.die is None`.

    The die is *not* spent on a Sorcerer substitution (it stays in play for
    the next conquest) nor on a Dragon Master conquest (1 token, no cost) —
    neither of them goes through `_conquest_cost`.
    """

    name = 'Berserk'

    def die_before_each_conquest(self, rip: 'RaceInPlay') -> bool:
        return True


class BivouackingHooks(PowerHooks):
    """Bivouacking (5): *"5 Encampments deployed during redeployment."*

    `encampments_to_place` opens the ENCAMPMENTS sub-phase with the whole
    supply: the engine first takes the race's encampments back from the board
    into the supply and then makes the player place all 5 again (any split,
    one action per encampment), so they may be moved every turn.

    Each encampment is worth +1 defence (`Region.defence`) and protects a lone
    token from a Sorcerer substitution (`SorcerersHooks.sorcerer_targets`
    skips an encamped region). They are never *lost*: when the region is
    conquered, abandoned or the race declines, the engine's `_clear_markers`
    returns every encampment to the supply, and the Bivouacking race re-places
    all of them on its next turn.
    """

    name = 'Bivouacking'

    def encampments_to_place(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> int:
        return N_ENCAMPMENTS


class CommandoHooks(PowerHooks):
    """Commando (4): *"-1 (min 1) on every conquest."*

    Chained after the race's modifier, so a Commando Triton pays 2 less on a
    coastal region; the engine clamps the result to at least 1.
    """

    name = 'Commando'

    def conquest_cost_modifier(self, env: 'SmallWorldEnv', rip: 'RaceInPlay',
                               region: 'Region', cost: int) -> int:
        return cost - 1


class DiplomatHooks(PowerHooks):
    """Diplomat (5): *"choose an opponent you did not attack this turn."*

    `has_ally_choice` opens the ALLY sub-phase at the end of the turn; the
    engine offers every opponent absent from the union of `attacked_players`
    over **all** the player's races (a declined Ghoul attack also forbids the
    alliance), plus `A_PASS` for "no ally".

    The chosen seat is stored in `PlayerState.ally` and `_can_attack` then
    refuses any attack of that seat on a region held by the Diplomat's owner
    whose tokens are **not** in decline — i.e. only his active race is
    protected, declined tokens (declined Ghouls included) are fair game. A
    Sorcerer substitution is an attack too and is blocked as well. The
    protection expires when the Diplomat's own next turn starts
    (`_begin_turn` clears `player.ally`).
    """

    name = 'Diplomat'

    def has_ally_choice(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> bool:
        return True


class DragonMasterHooks(PowerHooks):
    """Dragon Master (5): *"once per turn, conquer a region with one token."*

    `dragon_available` opens the `A_DRAGON` action range while the race still
    has a dragon conquest this turn (`rip.dragon_used`, reset by `new_turn()`)
    and at least one token in hand. The engine's `_resolve_dragon` takes the
    region with a single token whatever its defence — the target must still be
    *legal* (`_can_attack`: adjacency or the first-conquest border rule, not
    water, not immune, not protected by a Diplomat alliance) — moves the Dragon
    marker there from wherever it was (the previous region loses its immunity)
    and makes the region immune (`Region.is_immune`) while it stays.

    The Dragon goes back to the supply when the region is abandoned and when
    the race declines (`_clear_markers`), and the marker supply is 1, so a
    second dragon conquest simply moves it.
    """

    name = 'Dragon Master'

    def dragon_available(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> bool:
        return not rip.dragon_used


class FlyingHooks(PowerHooks):
    """Flying (5): *"conquer any region except the seas and the lake."*

    Three hooks: `can_target` says yes for every non-water region (water is
    left to the default rule, so only Seafaring ever opens it), `is_adjacent`
    says yes for everything (no adjacency needed) and `ignores_border_rule`
    lifts the border restriction of a first conquest.

    Because a Sorcerer substitution requires the target to be "adjacent to the
    Sorcerers" and `SorcerersHooks.sorcerer_targets` resolves that reach
    through the engine's `is_adjacent` hook, **Flying Sorcerers may substitute
    anywhere on the map** (T4 open point). Immunity, encampments, the
    once-per-opponent limit and the Diplomat alliance still apply.
    """

    name = 'Flying'

    def can_target(self, env: 'SmallWorldEnv', rip: 'RaceInPlay',
                   region: 'Region') -> bool | None:
        return True if not region.is_water else None

    def is_adjacent(self, env: 'SmallWorldEnv', rip: 'RaceInPlay',
                    region: 'Region') -> bool | None:
        return True

    def ignores_border_rule(self, rip: 'RaceInPlay') -> bool:
        return True


class ForestHooks(PowerHooks):
    """Forest (4): +1 coin per Forest region held (while active)."""

    name = 'Forest'

    def score_bonus(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> int:
        return _terrain_bonus(env, rip, Terrain.FOREST)


class FortifiedHooks(PowerHooks):
    """Fortified (3): *"one Fortress per turn, +1 defence, +1 coin each."*

    `can_place_fortress` opens the FORTRESS sub-phase once per turn
    (`rip.fortress_used`); the engine masks out the regions that already have
    one (max 1 per region) and skips the phase when the supply of 6 is
    exhausted or every region held is fortified. `A_PASS` is always available:
    building is optional.

    A fortress is worth +1 defence through `Region.defence`, **even once the
    race declined** (the marker stays on the map, user ruling — plan section 5
    item 7), and it is removed when the region is abandoned or conquered. The
    +1 coin per fortress only counts while the badge is held, so only the
    fortresses on a region of the *active* Fortified race score.
    """

    name = 'Fortified'

    def can_place_fortress(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> bool:
        return not rip.fortress_used

    def score_bonus(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> int:
        return sum(1 for r in env.board.regions_of_race(rip) if r.fortress)


class HeroicHooks(PowerHooks):
    """Heroic (5): *"place the 2 Heroes in 2 occupied regions."*

    `heroes_to_place` opens the HEROES sub-phase with both markers: the engine
    first takes the race's Heroes back from the board (they may be moved every
    turn), then asks for one region per Hero, distinct (the mask excludes a
    region that already has one) and fewer than 2 when the race holds fewer
    regions. A Hero makes its region immune to conquests and to the racial /
    special powers (`Region.is_immune`), and the markers go back to the supply
    when the region is abandoned or the race declines.
    """

    name = 'Heroic'

    def heroes_to_place(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> int:
        return N_HEROES


class HillHooks(PowerHooks):
    """Hill (4): +1 coin per Hill region held (while active)."""

    name = 'Hill'

    def score_bonus(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> int:
        return _terrain_bonus(env, rip, Terrain.HILL)


class MerchantHooks(PowerHooks):
    """Merchant (2): +1 coin per region occupied (so 2 coins per region)."""

    name = 'Merchant'

    def score_bonus(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> int:
        return len(env.board.regions_of_race(rip))


class MountedHooks(PowerHooks):
    """Mounted (5): *"-1 (min 1) on Hill and Farmland regions."*"""

    name = 'Mounted'

    def conquest_cost_modifier(self, env: 'SmallWorldEnv', rip: 'RaceInPlay',
                               region: 'Region', cost: int) -> int:
        if region.terrain in (Terrain.HILL, Terrain.FARMLAND):
            return cost - 1
        return cost


class PillagingHooks(PowerHooks):
    """Pillaging (5): +1 coin per non-empty region conquered this turn.

    `RaceInPlay.nonempty_conquests` is maintained by the engine (a region is
    non-empty when it held a Lost Tribe or at least one race token) and reset
    by `new_turn()`, so the bonus is read at the scoring step of the same turn.
    A Sorcerer substitution is not a conquest and does not count. Pillaging
    Orcs collect the bonus twice, once per rule.
    """

    name = 'Pillaging'

    def score_bonus(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> int:
        return rip.nonempty_conquests


class SeafaringHooks(PowerHooks):
    """Seafaring (5): *"seas and the lake count as empty regions (cost 2)."*

    `can_target` opens the two seas and the lake; everything else is
    unchanged, which is exactly the rule:

    * no cost modifier is needed — a water region carries no Mountain token,
      no Lost Tribe and no symbol, so `2 + defence` is already 2 for an empty
      sea or lake, while enemy tokens and markers on the water defend like on
      land;
    * adjacency (or the border rule for a first conquest: seas 1 and 30 touch
      the edge, the lake 15 does not) still applies;
    * the water regions are kept when the race declines, like any other
      region (the engine's `_decline` keeps one token per region and has no
      terrain test), and they then become untouchable: no other race can ever
      target water without this badge.
    """

    name = 'Seafaring'

    def can_target(self, env: 'SmallWorldEnv', rip: 'RaceInPlay',
                   region: 'Region') -> bool | None:
        return True if region.is_water else None


class SpiritHooks(PowerHooks):
    """Spirit (5): *"the declined Spirit race never leaves the map."*

    `is_spirit_power` is read by `_decline` while the badge is still held and
    stored on the declined race (`RaceInPlay.is_spirit`); the engine then never
    removes it to make room for a newer declined race.

    A player who picks the Spirit badge twice (it is reshuffled from the
    discard pile when the power stack runs out) can therefore hold three or
    more declined races. They are all kept in the state and all score, but the
    `players` observation only has two declined slots and shows the first two
    (documented deviation, plan T5).
    """

    name = 'Spirit'

    def is_spirit_power(self, rip: 'RaceInPlay') -> bool:
        return True


class StoutHooks(PowerHooks):
    """Stout (4): *"may go in decline at the end of a normal turn."*

    `can_stout_decline` makes the engine enter STOUT_DECLINE **after** the
    scoring step; `A_DECLINE` there resolves the decline (tokens to the tray,
    badge discarded, previous declined race removed) with no second scoring,
    and `A_PASS` stays active.
    """

    name = 'Stout'

    def can_stout_decline(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> bool:
        return True


class SwampHooks(PowerHooks):
    """Swamp (4): +1 coin per Swamp region held (while active)."""

    name = 'Swamp'

    def score_bonus(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> int:
        return _terrain_bonus(env, rip, Terrain.SWAMP)


class UnderworldHooks(PowerHooks):
    """Underworld (5): *"-1 on Caverns; all Caverns are adjacent."*

    The discount applies to every region carrying a Cavern symbol (5, 13, 16,
    20 and 24 on the 3-player map) and the engine clamps the cost to at least
    1. `is_adjacent` links the caverns together: a Cavern region counts as
    adjacent as soon as the race holds at least one Cavern region — and
    returns `None` (no opinion) otherwise, so the printed adjacency still
    decides for every other target.

    Underworld Sorcerers reach a lone enemy token through the cavern links
    too: `SorcerersHooks.sorcerer_targets` resolves its reach with this same
    hook (T4 open point).
    """

    name = 'Underworld'

    def conquest_cost_modifier(self, env: 'SmallWorldEnv', rip: 'RaceInPlay',
                               region: 'Region', cost: int) -> int:
        return cost - 1 if region.has(Symbol.CAVERN) else cost

    def is_adjacent(self, env: 'SmallWorldEnv', rip: 'RaceInPlay',
                    region: 'Region') -> bool | None:
        if not region.has(Symbol.CAVERN):
            return None
        for held in env.board.regions_of_race(rip):
            if held.has(Symbol.CAVERN):
                return True
        return None


class WealthyHooks(PowerHooks):
    """Wealthy (4): *"+7 coins at the end of the first turn with this power."*

    Paid by `on_turn_end` when `rip.turns_played == 1` (the turn the combo was
    picked: `_step_pick_combo` calls `new_turn()`), guarded by the persistent
    `rip.wealthy_paid` flag so it can never be paid twice. The gain is fed to
    the dense shaping reward like a scoring event (`env._shape`), so the 7
    coins are as visible to a learner as the coins of the scoring step.

    If the race declines during that first turn its badge is discarded before
    `on_turn_end` runs, so nothing is paid — the plan's ruling ("no bonus in
    the decline turn").
    """

    name = 'Wealthy'

    def on_turn_end(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> None:
        if rip.wealthy_paid or rip.turns_played != 1:
            return
        rip.wealthy_paid = True
        player = env.players[rip.owner]
        player.coins += WEALTHY_BONUS
        gains = [0] * env.n_players
        gains[rip.owner] = WEALTHY_BONUS
        env._shape(gains)
        env._log(f'{env._label(rip)} is Wealthy: +{WEALTHY_BONUS} coins for its '
                 f'first turn -> {player.coins} coins in '
                 f'{player.coin_count} token(s)')


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

#: The hook singleton of every special power, in `PowerId` order.
POWER_HOOK_INSTANCES: dict[PowerId, PowerHooks] = {
    PowerId.ALCHEMIST: AlchemistHooks(),
    PowerId.BERSERK: BerserkHooks(),
    PowerId.BIVOUACKING: BivouackingHooks(),
    PowerId.COMMANDO: CommandoHooks(),
    PowerId.DIPLOMAT: DiplomatHooks(),
    PowerId.DRAGON_MASTER: DragonMasterHooks(),
    PowerId.FLYING: FlyingHooks(),
    PowerId.FOREST: ForestHooks(),
    PowerId.FORTIFIED: FortifiedHooks(),
    PowerId.HEROIC: HeroicHooks(),
    PowerId.HILL: HillHooks(),
    PowerId.MERCHANT: MerchantHooks(),
    PowerId.MOUNTED: MountedHooks(),
    PowerId.PILLAGING: PillagingHooks(),
    PowerId.SEAFARING: SeafaringHooks(),
    PowerId.SPIRIT: SpiritHooks(),
    PowerId.STOUT: StoutHooks(),
    PowerId.SWAMP: SwampHooks(),
    PowerId.UNDERWORLD: UnderworldHooks(),
    PowerId.WEALTHY: WealthyHooks(),
}

for _power_id, _hooks in POWER_HOOK_INSTANCES.items():
    register_power_hooks(_power_id, _hooks)
del _power_id, _hooks
