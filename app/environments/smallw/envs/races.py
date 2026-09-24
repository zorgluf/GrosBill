"""Race behaviours of Small World (`smallw`) — task **T4**.

The engine (`smallw.py`) only knows the vanilla rules: every race plays like
the Ratmen. Each of the 14 base-game races gets a
:class:`~.hooks.RaceHooks` subclass here, registered at import time with
:func:`~.hooks.register_race_hooks`; `smallw.py` imports this module for that
side effect, so a race becomes active as soon as it is registered.

Hooks are **stateless singletons** (never deep-copied with the env): every bit
of per-race state lives on the acting :class:`~.classes.RaceInPlay`
(`holes_placed`, `nonempty_conquests`, `sorcerer_used_on`, ... — see
`classes.py`) or on the board.

Rules implemented (plan 2.1, `docs/SmallWorld_v3.pdf` p.2-4)
------------------------------------------------------------
=========  ==================================================================
Amazons    +4 attack-only tokens: they stay in hand at the end of every
           redeployment (`redeployable_tokens`)
Dwarves    +1 coin per Mine region, **even in decline**
Elves      lose no token when a region is conquered (`defender_loss` = 0)
Ghouls     keep every token in decline, still conquer while in decline (before
           the owner's active race) and take losses like an active race
Giants     -1 to conquer a region adjacent to a Mountain region they occupy
Halflings  enter anywhere (no border rule) + a Hole-in-the-Ground on their
           first 2 conquests (immune region), removed on decline / abandon
Humans     +1 coin per Farmland region
Orcs       +1 coin per non-empty region conquered this turn
Ratmen     nothing (the vanilla behaviour)
Skeletons  +1 token from the tray per 2 non-empty regions conquered this turn,
           taken when their redeployment starts
Sorcerers  replace a lone *active* enemy token in an adjacent region by a new
           Sorcerer from the tray, once per turn per opponent
Tritons    -1 on Coastal regions
Trolls     a Troll's Lair (+1 defence) in every region they conquer; it stays
           after a decline, goes when the region is abandoned or conquered
Wizards    +1 coin per Magic Source region
=========  ==================================================================

Every cost modifier is clamped to at least 1 by the engine, every `score_bonus`
is summed with the power's, and a bonus only applies in decline when
`bonus_in_decline` says so (Dwarves) — see the combination rules in `hooks.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .classes import N_HOLES, RaceId, Symbol, Terrain
from .hooks import RaceHooks, register_race_hooks

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .classes import RaceInPlay, Region
    from .smallw import SmallWorldEnv

__all__ = [
    'AmazonsHooks', 'DwarvesHooks', 'ElvesHooks', 'GhoulsHooks', 'GiantsHooks',
    'HalflingsHooks', 'HumansHooks', 'OrcsHooks', 'RatmenHooks',
    'SkeletonsHooks', 'SorcerersHooks', 'TritonsHooks', 'TrollsHooks',
    'WizardsHooks', 'RACE_HOOK_INSTANCES',
]


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #

def _terrain_bonus(env: 'SmallWorldEnv', rip: 'RaceInPlay',
                   terrain: Terrain) -> int:
    """Coins for a race scoring +1 per region of `terrain` it holds."""
    if rip.in_decline:                     # no racial bonus once declined
        return 0
    return sum(1 for r in env.board.regions_of_race(rip) if r.terrain == terrain)


def _symbol_bonus(env: 'SmallWorldEnv', rip: 'RaceInPlay',
                  symbol: Symbol) -> int:
    """Coins for a race scoring +1 per region carrying `symbol`."""
    return sum(1 for r in env.board.regions_of_race(rip) if r.has(symbol))


# --------------------------------------------------------------------------- #
# The 14 races
# --------------------------------------------------------------------------- #


class AmazonsHooks(RaceHooks):
    """Amazons (6, +4 attack-only) [15].

    *"The 4 extra tokens are removed from the map at the end of each
    redeployment and come back in hand next turn."*

    The +4 are already taken from the tray when the combo is picked
    (`RaceDef.attack_bonus_tokens` → `RaceInPlay.initial_tokens()`), so the
    only thing left is to stop the player from putting them on the board:
    :meth:`redeployable_tokens` reserves them in the hand (plan section 5
    item 3 — the equivalent formulation of the physical removal, which also
    keeps them out of reach of the "≥ 1 token per region" rule).

    This applies to *every* redeployment of the race, its own and the
    end-of-turn one of a victim who lost regions (`pending_redeploy`), so the
    4 tokens are never deployed. They stay counted in the conservation
    invariant (they are in `tokens_in_hand`) and the engine puts them back in
    the tray with the rest of the hand when the Amazons decline.
    """

    name = 'Amazons'

    def redeployable_tokens(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> int:
        if rip.in_decline:
            # a declined race has no attack-only token any more (and the
            # engine already emptied its hand into the tray at decline time)
            return rip.tokens_in_hand
        # negative results are clamped to 0 by the engine
        return rip.tokens_in_hand - rip.race_def.attack_bonus_tokens


class DwarvesHooks(RaceHooks):
    """Dwarves (3) [8]: +1 coin per Mine region, **even in decline**."""

    name = 'Dwarves'

    def score_bonus(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> int:
        return _symbol_bonus(env, rip, Symbol.MINE)

    def bonus_in_decline(self, rip: 'RaceInPlay') -> bool:
        return True


class ElvesHooks(RaceHooks):
    """Elves (6) [11]: they lose no token when one of their regions is taken.

    Every token of the region goes back to the hand (`defender_loss` = 0) and
    the race redeploys them at the end of the attacker's turn
    (`pending_redeploy`, handled by the engine). A Sorcerer substitution still
    removes the token (it goes to the tray, rules p.4) — the engine's
    `_resolve_sorcerer` never asks this hook.

    While the Elves are *in decline* the ability is gone: the engine wipes a
    declined region like any other (this hook is not consulted then, because
    `losses_like_active` is False for the Elves).
    """

    name = 'Elves'

    def defender_loss(self, env: 'SmallWorldEnv', victim_rip: 'RaceInPlay',
                      region: 'Region') -> int:
        return 0


class GhoulsHooks(RaceHooks):
    """Ghouls (5) [10]: they keep all their tokens in decline and still fight.

    Three flags drive the engine:

    * `keeps_tokens_in_decline` — the decline keeps every token (the hand
      included, redeployed right away in a GHOUL_REDEPLOY phase);
    * `conquers_in_decline` — the owner plays a GHOUL_CONQUER /
      GHOUL_REDEPLOY pair with the declined Ghouls *before* his active race,
      with the usual *ready your troops* (they may attack anybody, their
      owner's active race included);
    * `losses_like_active` — when a Ghoul region is conquered they lose 1
      token to the tray and keep the rest in hand for the end-of-turn
      redeployment, like an active race.

    No scoring bonus: declined Ghoul regions are worth 1 coin each like any
    declined region. A declined Ghoul race with tokens in hand but no region
    stays on the roster and re-enters through a border region at its next
    Ghoul phase (engine: `_after_loss` / `_begin_turn`).
    """

    name = 'Ghouls'

    def keeps_tokens_in_decline(self, rip: 'RaceInPlay') -> bool:
        return True

    def conquers_in_decline(self, rip: 'RaceInPlay') -> bool:
        return True

    def losses_like_active(self, rip: 'RaceInPlay') -> bool:
        return True


class GiantsHooks(RaceHooks):
    """Giants (6) [11]: -1 token to conquer a region next to their Mountains.

    The discount applies when the target touches (printed adjacency) a
    Mountain region the *active* Giants occupy; the engine clamps the cost to
    at least 1.
    """

    name = 'Giants'

    def conquest_cost_modifier(self, env: 'SmallWorldEnv', rip: 'RaceInPlay',
                               region: 'Region', cost: int) -> int:
        if rip.in_decline:
            return cost
        mountains = {r.id for r in env.board.regions_of_race(rip) if r.is_mountain}
        if mountains and mountains.intersection(region.adjacent):
            return cost - 1
        return cost


class HalflingsHooks(RaceHooks):
    """Halflings (6) [11]: they enter anywhere and dig two Holes.

    * `ignores_border_rule` — their first conquest may target any region, not
      only a border one (the later ones still need the usual adjacency).
    * `on_conquered` — a Hole-in-the-Ground is placed on each of the first
      `N_HOLES` (2) regions the race conquers (`RaceInPlay.holes_placed`
      counts them for the whole life of the race, so a hole removed later is
      not dug again). A hole makes the region immune to conquests and to the
      racial / special powers (`Region.is_immune`, checked by the engine's
      `_can_attack` and by `SorcerersHooks.sorcerer_targets`).
    * the holes are returned to the supply when the race declines
      (`on_decline`) or abandons the region (`on_abandon`) — the engine's
      `_clear_markers` already does it in both cases, these hooks are the
      belt-and-braces version and stay idempotent. `on_region_lost` is there
      for the same reason: an immune region cannot be lost, but if a future
      power ever takes one, its hole must not stay behind.
    """

    name = 'Halflings'

    def ignores_border_rule(self, rip: 'RaceInPlay') -> bool:
        return not rip.in_decline

    def on_conquered(self, env: 'SmallWorldEnv', rip: 'RaceInPlay',
                     region: 'Region', defender_info: dict) -> None:
        if rip.in_decline or region.hole or rip.holes_placed >= N_HOLES:
            return
        if not env.tray.take_marker('hole'):       # supply exhausted
            return
        region.hole = True
        rip.holes_placed += 1
        env._log(f'{env._label(rip)} digs a Hole-in-the-Ground in region '
                 f'{region.id} ({rip.holes_placed}/{N_HOLES}): the region is '
                 f'now immune')

    def on_decline(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> None:
        for region in env.board.regions_of_race(rip):
            self._remove_hole(env, region)

    def on_abandon(self, env: 'SmallWorldEnv', rip: 'RaceInPlay',
                   region: 'Region') -> None:
        self._remove_hole(env, region)

    def on_region_lost(self, env: 'SmallWorldEnv', victim_rip: 'RaceInPlay',
                       region: 'Region') -> None:
        self._remove_hole(env, region)

    @staticmethod
    def _remove_hole(env: 'SmallWorldEnv', region: 'Region') -> None:
        """Return the hole of `region` to the supply, if it still has one."""
        if region.hole:
            region.hole = False
            env.tray.put_marker('hole')


class HumansHooks(RaceHooks):
    """Humans (5) [10]: +1 coin per Farmland region (while active)."""

    name = 'Humans'

    def score_bonus(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> int:
        return _terrain_bonus(env, rip, Terrain.FARMLAND)


class OrcsHooks(RaceHooks):
    """Orcs (5) [10]: +1 coin per non-empty region conquered this turn.

    `RaceInPlay.nonempty_conquests` is maintained by the engine (a region is
    non-empty when it held a Lost Tribe or at least one race token) and reset
    by `new_turn()`, so the bonus is read at the scoring step of the same
    turn. A Sorcerer substitution is not a conquest and does not count.
    """

    name = 'Orcs'

    def score_bonus(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> int:
        if rip.in_decline:
            return 0
        return rip.nonempty_conquests


class RatmenHooks(RaceHooks):
    """Ratmen (8) [13]: no special ability at all (the vanilla behaviour).

    Registered anyway so that the 14 banners all have their own hooks (and so
    that the registry never falls back to the plain `RaceHooks`).
    """

    name = 'Ratmen'


class SkeletonsHooks(RaceHooks):
    """Skeletons (6) [20]: they multiply on the corpses of their victims.

    *"During redeployment, +1 token from the tray per 2 non-empty regions
    conquered this turn (while the tray has some)."* The tokens are taken when
    the race's own redeployment starts (`on_redeploy_start`, which the engine
    calls for the active race and for a Ghoul-style declined race, but never
    for a victim's end-of-turn redeployment) and join `tokens_in_hand`, so
    they have to be deployed like the rest.
    """

    name = 'Skeletons'

    def on_redeploy_start(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> None:
        if rip.in_decline:
            return
        wanted = rip.nonempty_conquests // 2
        if wanted <= 0:
            return
        taken = env.tray.take(rip.race, wanted)
        rip.tokens_in_hand += taken
        short = '' if taken == wanted else f' ({wanted - taken} short, tray empty)'
        env._log(f'{env._label(rip)} multiplies: +{taken} token(s) from the tray '
                 f'for {rip.nonempty_conquests} non-empty region(s) conquered'
                 f'{short} -> {rip.tokens_in_hand} in hand')


class SorcerersHooks(RaceHooks):
    """Sorcerers (5) [18]: they turn a lone enemy into one of their own.

    Legality of a substitution on region `r` (the whole check lives here, the
    engine only owns the mechanics — `_resolve_sorcerer`):

    * the Sorcerers are **active** and have at least one token left in the
      tray (the new token comes from there);
    * `r` is *adjacent to the Sorcerers*, resolved exactly like a conquest
      target through the engine's three-valued `is_adjacent` hook (T5): the
      printed adjacency by default, but Flying Sorcerers reach the whole map
      and Underworld Sorcerers reach every Cavern region as soon as they hold
      one;
    * `r` holds **exactly one** token of an **opponent's active** race (not a
      Lost Tribe, not a declined race, not one of the owner's own races);
    * a Lair, a Fortress or a Mountain give no protection, but an Encampment,
      a Hole, a Hero and the Dragon do (`Region.is_immune`);
    * the victim did not protect his active race with a Diplomat alliance
      with the Sorcerers' owner;
    * that opponent has not been hit by a substitution yet this turn
      (`RaceInPlay.sorcerer_used_on`);
    * the Sorcerers could occupy that terrain at all (seas and the lake are
      out of reach without Seafaring — resolved exactly like a conquest
      target through the engine's `can_target` hook).

    The engine then sends the victim's token to the tray (an Elf too), takes a
    fresh Sorcerer from the tray, clears the region's fortress / lair /
    encampments like a conquest would (plan section 5 item 7), marks the
    opponent as attacked (Diplomat) and remembers him in `sorcerer_used_on`.
    A substitution is *not* a conquest: it feeds no Orc / Pillaging bonus.
    """

    name = 'Sorcerers'

    def sorcerer_targets(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> set[int]:
        if rip.in_decline or env.tray.available(rip.race) < 1:
            return set()
        held = env.board.regions_of_race(rip)
        if not held:
            return set()
        held_ids = {r.id for r in held}

        targets: set[int] = set()
        for region in env.board.regions:
            owner = region.owner
            if owner is None or owner == rip.owner:
                continue                       # empty, Lost Tribe, or his own
            if region.in_decline or region.tokens != 1:
                continue                       # active race, lone token only
            if region.encampments or region.is_immune:
                continue                       # protected piece
            if owner in rip.sorcerer_used_on:
                continue                       # once per opponent per turn
            if env.players[owner].ally == rip.owner:
                continue                       # Diplomat alliance
            if not env._three_valued('can_target', rip, region,
                                     not region.is_water):
                continue                       # unreachable terrain (seas)
            # "adjacent to the Sorcerers", resolved like a conquest target:
            # printed adjacency unless a power extends it (Flying, Underworld)
            printed = bool(set(region.adjacent) & held_ids)
            if not env._three_valued('is_adjacent', rip, region, printed):
                continue
            targets.add(region.id)
        return targets


class TritonsHooks(RaceHooks):
    """Tritons (6) [11]: -1 token on Coastal regions (next to a sea or lake).

    `Board.is_coastal` holds the pre-computed list; the engine clamps the cost
    to at least 1.
    """

    name = 'Tritons'

    def conquest_cost_modifier(self, env: 'SmallWorldEnv', rip: 'RaceInPlay',
                               region: 'Region', cost: int) -> int:
        if rip.in_decline:
            return cost
        return cost - 1 if env.board.is_coastal(region) else cost


class TrollsHooks(RaceHooks):
    """Trolls (5) [10]: a Troll's Lair in every region they occupy.

    A Lair is worth +1 defence (`Region.defence`) and is not a limited
    component (`Region.lair` is a flag, the `Tray` keeps no Lair supply). The
    Trolls only ever occupy regions they conquered, so placing it in
    `on_conquered` covers every case.

    Life cycle (plan section 5 item 7, user ruling): the Lair **stays** when
    the Trolls decline and when the declined Trolls leave the map (it then
    defends an empty region), and it is removed when the region is abandoned
    or conquered by an enemy — both already done by the engine's
    `_clear_markers`; `on_abandon` is the explicit, idempotent version.
    """

    name = 'Trolls'

    def on_conquered(self, env: 'SmallWorldEnv', rip: 'RaceInPlay',
                     region: 'Region', defender_info: dict) -> None:
        if rip.in_decline or region.lair:
            return
        region.lair = True
        env._log(f"{env._label(rip)} digs a Troll's Lair in region {region.id} "
                 f'(+1 defence)')

    def on_abandon(self, env: 'SmallWorldEnv', rip: 'RaceInPlay',
                   region: 'Region') -> None:
        region.lair = False


class WizardsHooks(RaceHooks):
    """Wizards (5) [10]: +1 coin per Magic Source region (while active)."""

    name = 'Wizards'

    def score_bonus(self, env: 'SmallWorldEnv', rip: 'RaceInPlay') -> int:
        if rip.in_decline:
            return 0
        return _symbol_bonus(env, rip, Symbol.MAGIC)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

#: The hook singleton of every race, in `RaceId` order.
RACE_HOOK_INSTANCES: dict[RaceId, RaceHooks] = {
    RaceId.AMAZONS: AmazonsHooks(),
    RaceId.DWARVES: DwarvesHooks(),
    RaceId.ELVES: ElvesHooks(),
    RaceId.GHOULS: GhoulsHooks(),
    RaceId.GIANTS: GiantsHooks(),
    RaceId.HALFLINGS: HalflingsHooks(),
    RaceId.HUMANS: HumansHooks(),
    RaceId.ORCS: OrcsHooks(),
    RaceId.RATMEN: RatmenHooks(),
    RaceId.SKELETONS: SkeletonsHooks(),
    RaceId.SORCERERS: SorcerersHooks(),
    RaceId.TRITONS: TritonsHooks(),
    RaceId.TROLLS: TrollsHooks(),
    RaceId.WIZARDS: WizardsHooks(),
}

for _race_id, _hooks in RACE_HOOK_INSTANCES.items():
    register_race_hooks(_race_id, _hooks)
del _race_id, _hooks
