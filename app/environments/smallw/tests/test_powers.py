"""Tests for the 20 special powers of Small World (`envs/powers.py`, task T5).

Plain `assert`-based functions, usable either with pytest or through the
bundled runners:

    cd app
    python -m environments.smallw.tests.run_all        # every test module
    python -m environments.smallw.tests.test_powers    # this module only

One or more **scenario** tests per badge prove the rule on a hand-built state
(the helpers `_env` / `_setup_turn` / `_occupy` / `_legal` of `test_engine.py`
are reused, like `test_races.py` does), plus a **fuzz** run of `FUZZ_GAMES`
random games with all the race *and* power hooks registered, re-checking every
T3/T4 invariant (plan 3.10) plus the power-specific ones
(`_check_powers_invariants`) and reporting coverage counters.

Unless a scenario is explicitly about a race/power combination, the tested
race is the **Ratmen** (no ability at all), so the numbers are those of the
power alone.
"""

from __future__ import annotations

import time
import traceback
from collections import Counter

import numpy as np

from ..envs.classes import (
    MARKER_SUPPLY,
    N_ENCAMPMENTS,
    N_HEROES,
    POWERS,
    PowerId,
    RaceId,
    Symbol,
    Terrain,
    turns_for,
)
from ..envs.hooks import NO_POWER_HOOKS, PowerHooks, get_power_hooks
from ..envs.powers import (
    ALCHEMIST_BONUS,
    POWER_HOOK_INSTANCES,
    WEALTHY_BONUS,
)
from ..envs.smallw import (
    A_ALLY,
    A_DECLINE,
    A_DRAGON,
    A_PASS,
    A_REGION,
    A_REGION_ALL,
    Phase,
    SmallWorldEnv,
    action_kind,
    ally_action,
    dragon_action,
    region_action,
    region_all_action,
    sorcerer_action,
)
from .test_engine import _check_invariants, _env, _force_dice, _legal, _occupy, _setup_turn

#: Games played by the power fuzz test.
FUZZ_GAMES = 150
#: Hard step budget per game.
STEP_BUDGET = 3000

#: Statistics of the last fuzz run, printed by `run_all()`.
FUZZ_STATS: dict = {}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _index(region_id: int) -> int:
    """0-based index of the region printed `region_id`."""
    return region_id - 1


def _conquest_targets(env: SmallWorldEnv) -> set[int]:
    """Region **ids** the legal REGION actions point at."""
    return {a - A_REGION + 1 for a in _legal(env) if A_REGION <= a < A_REGION_ALL}


def _dragon_targets(env: SmallWorldEnv) -> set[int]:
    """Region **ids** the legal DRAGON actions point at."""
    return {a - A_DRAGON + 1 for a in _legal(env) if A_DRAGON <= a < A_ALLY}


def _player_row(env: SmallWorldEnv, seat: int) -> np.ndarray:
    """The `players` observation row of `seat` (rows rotate with the POV)."""
    offset = (seat - env.current_player) % env.n_players
    return env.observation['players'][offset]


def _markers(env: SmallWorldEnv) -> dict[str, int]:
    """Markers available in the supply, per kind."""
    return {kind: env.tray.available_marker(kind) for kind in MARKER_SUPPLY}


def _check_powers_invariants(env: SmallWorldEnv, ctx: str) -> None:
    """Every T3/T4 invariant plus the ones the 20 powers add.

    The power-specific ones: an encampment / a Hero / the Dragon only ever sit
    on a region held by the *active* race that owns the matching badge, only a
    Seafaring race holds water while active, and the once-per-turn / once-per-
    race flags belong to the right badge.
    """
    _check_invariants(env, ctx)

    for region in env.board.regions:
        holder = (None if region.owner is None else
                  env.players[region.owner].race_by(region.race,
                                                    in_decline=region.in_decline))
        power = None if holder is None else holder.power
        if region.encampments:
            assert power == PowerId.BIVOUACKING, (
                f'{ctx}: {region.encampments} encampment(s) on region '
                f'{region.id} held by {holder!r}')
        if region.hero:
            assert power == PowerId.HEROIC, (
                f'{ctx}: a Hero on region {region.id} held by {holder!r}')
        if region.dragon:
            assert power == PowerId.DRAGON_MASTER, (
                f'{ctx}: the Dragon on region {region.id} held by {holder!r}')
        if region.is_water and region.owner is not None and not region.in_decline:
            assert power == PowerId.SEAFARING, (
                f'{ctx}: water region {region.id} held by {holder!r}')

    for player in env.players:
        for rip in player.all_races():
            if rip.power is None:           # declined: the badge is gone
                continue
            if rip.dragon_used:
                assert rip.power == PowerId.DRAGON_MASTER, f'{ctx}: {rip!r}'
            if rip.fortress_used:
                assert rip.power == PowerId.FORTIFIED, f'{ctx}: {rip!r}'
            if rip.wealthy_paid:
                assert rip.power == PowerId.WEALTHY, f'{ctx}: {rip!r}'
        # more than two declined races is only possible with several Spirits
        if len(player.declined) > 2:
            spirits = sum(1 for d in player.declined if d.is_spirit)
            assert spirits >= len(player.declined) - 1, (
                f'{ctx}: {player.name} has {len(player.declined)} declined '
                f'races but only {spirits} Spirit(s)')


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

def test_every_power_has_its_own_hooks():
    """The 20 badges are registered, each with a distinct hook class."""
    assert set(POWER_HOOK_INSTANCES) == set(PowerId)
    classes = set()
    for power_id in PowerId:
        hooks = get_power_hooks(power_id)
        assert hooks is POWER_HOOK_INSTANCES[power_id]
        assert isinstance(hooks, PowerHooks)
        assert type(hooks) is not PowerHooks, f'{power_id.name} is still vanilla'
        assert hooks.name == POWERS[power_id].name, power_id.name
        classes.add(type(hooks))
    assert len(classes) == len(PowerId)


def test_a_declined_race_has_no_power_hooks_at_all():
    """The badge is discarded at decline, so no power hook can fire again."""
    env = _env(300)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.ALCHEMIST, hand=0,
                      held={9: 1, 10: 1}, turns_played=2)
    assert env._score_bonus(rip) == ALCHEMIST_BONUS
    env.step(A_DECLINE)
    assert rip.in_decline and rip.power is None
    assert get_power_hooks(rip.power) is NO_POWER_HOOKS
    assert env._score_bonus(rip) == 0, 'the Alchemist bonus stops at decline'
    assert env.players[0].coins == 5 + 2, '1 coin per region, no bonus'
    _check_powers_invariants(env, 'declined power')


# --------------------------------------------------------------------------- #
# Alchemist
# --------------------------------------------------------------------------- #

def test_alchemist_scores_two_extra_coins_every_turn():
    """+2 coins per turn, whatever the race holds."""
    env = _env(301)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.ALCHEMIST, hand=0,
                      held={9: 1, 10: 1}, turns_played=2)
    assert env._score_bonus(rip) == 2
    env.step(A_PASS)                                  # nothing to redeploy
    assert env.players[0].coins == 5 + 2 + ALCHEMIST_BONUS
    assert env._turn_seat == 1, 'the turn is over'
    _check_powers_invariants(env, 'alchemist')

    # even with no region left the 2 coins are paid
    env2 = _env(302)
    empty = _setup_turn(env2, 0, RaceId.RATMEN, PowerId.ALCHEMIST, hand=3)
    assert env2.board.regions_of_race(empty) == []
    assert env2._score_bonus(empty) == ALCHEMIST_BONUS
    env2._score(env2.players[0])
    assert env2.players[0].coins == 5 + ALCHEMIST_BONUS


# --------------------------------------------------------------------------- #
# Berserk
# --------------------------------------------------------------------------- #

def test_berserk_rolls_the_die_before_every_conquest():
    """The die is drawn at CONQUER entry, subtracted (min 1), then re-drawn."""
    env = _env(303)
    _force_dice(env, [3, 1, 0])
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.BERSERK, hand=4,
                      held={14: 1})
    assert rip.die == 3, 'the die is rolled when the conquest phase opens'
    board = env.board
    assert board.by_id(10).lost_tribe
    assert board.by_id(10).base_conquest_cost() == 3
    assert env._conquest_cost(rip, board.by_id(10)) == 1, '3 - 3, floored at 1'
    # a mountain + the die: the floor of 1 applies too
    assert board.by_id(13).base_conquest_cost() == 3
    assert env._conquest_cost(rip, board.by_id(13)) == 1
    # a heavily defended region is only discounted by the die
    _occupy(env, 1, RaceId.HUMANS, 9, 4)
    assert env._conquest_cost(rip, board.by_id(9)) == 6 - 3
    # with a die in play there is no "final attempt" (plan section 5 item 8)
    options = env._conquest_options(rip)
    assert options, options
    assert all(not final for _, final in options.values()), options
    assert env.observation['global'][3] == 3 + 1, 'the die is observable'

    env.step(region_action(_index(10)))
    assert board.by_id(10).tokens == 1 and rip.tokens_in_hand == 3
    assert rip.die == 1, 'a fresh die is rolled for the next conquest'
    assert env._conquest_cost(rip, board.by_id(9)) == 6 - 1
    _check_powers_invariants(env, 'berserk')


def test_the_reinforcement_die_is_only_rolled_for_berserk():
    """`_roll_berserk` clears `rip.die` for every other power."""
    env = _env(349)
    plain = _setup_turn(env, 0, RaceId.RATMEN, PowerId.COMMANDO, hand=4,
                        held={14: 1})
    assert plain.die is None, 'no die without the Berserk badge'
    assert env.observation['global'][3] == 0
    # and without a die in play the "final attempt" gamble is offered again
    _occupy(env, 1, RaceId.HUMANS, 9, 5)
    assert env._conquest_options(plain)[_index(9)] == (6, True), '2 + 5 - 1'
    _check_powers_invariants(env, 'no berserk')


def test_berserk_die_is_not_spent_by_a_sorcerer_substitution():
    """A substitution pays no cost, so it neither uses nor re-rolls the die."""
    env = _env(304)
    _force_dice(env, [2, 0, 0])
    rip = _setup_turn(env, 0, RaceId.SORCERERS, PowerId.BERSERK, hand=2,
                      held={14: 1})
    assert rip.die == 2
    _occupy(env, 1, RaceId.HUMANS, 10, 1)
    assert sorcerer_action(_index(10)) in _legal(env)
    env.step(sorcerer_action(_index(10)))
    assert env.board.by_id(10).race == RaceId.SORCERERS
    assert rip.die == 2, 'the die is untouched by a substitution'
    _check_powers_invariants(env, 'berserk sorcerer')


# --------------------------------------------------------------------------- #
# Bivouacking
# --------------------------------------------------------------------------- #

def test_bivouacking_places_five_encampments_and_they_defend():
    """All 5 encampments are placed (any split) and each is worth +1 defence."""
    env = _env(305)
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.BIVOUACKING, hand=0,
                held={9: 1, 10: 1}, turns_played=2)
    env.step(A_PASS)
    assert env.phase == Phase.ENCAMPMENTS and env._encampments_left == N_ENCAMPMENTS
    assert set(_legal(env)) == {region_action(_index(9)), region_action(_index(10))}
    assert A_PASS not in _legal(env), 'every encampment must be placed'
    assert env.observation['global'][14] == N_ENCAMPMENTS
    for _ in range(3):
        env.step(region_action(_index(9)))
    for _ in range(2):
        env.step(region_action(_index(10)))
    board = env.board
    assert board.by_id(9).encampments == 3 and board.by_id(10).encampments == 2
    assert env.tray.available_marker('encampment') == 0
    assert board.by_id(9).defence == 3 + 1, '3 encampments + 1 token'
    assert board.by_id(9).base_conquest_cost() == 6
    assert board.by_id(10).base_conquest_cost() == 5
    assert env.players[0].coins == 5 + 2, 'Bivouacking pays no coin'
    _check_powers_invariants(env, 'bivouacking placement')


def test_bivouacking_encampments_are_re_placed_every_turn():
    """The engine takes them back into the pool before every ENCAMPMENTS phase."""
    env = _env(306)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.BIVOUACKING, hand=0,
                      held={9: 1, 10: 1}, turns_played=2)
    env.step(A_PASS)
    for _ in range(N_ENCAMPMENTS):
        env.step(region_action(_index(9)))
    assert env.board.by_id(9).encampments == 5

    env._begin_turn(0)                                # his next turn
    assert env.phase == Phase.CONQUER and env.current_race is rip
    assert env.board.by_id(9).encampments == 5, 'still there during the turn'
    env.step(A_PASS)
    assert env.phase == Phase.ENCAMPMENTS and env._encampments_left == N_ENCAMPMENTS
    assert env.board.by_id(9).encampments == 0, 'taken back into the pool'
    assert env.tray.available_marker('encampment') == N_ENCAMPMENTS
    for _ in range(3):
        env.step(region_action(_index(10)))
    for _ in range(2):
        env.step(region_action(_index(9)))
    assert env.board.by_id(10).encampments == 3
    _check_powers_invariants(env, 'bivouacking re-placement')


def test_bivouacking_encampment_protects_a_lone_token_from_the_sorcerers():
    """An encampment is one of the pieces that stop a substitution."""
    env = _env(307)
    sorcerers = _setup_turn(env, 0, RaceId.SORCERERS, None, hand=0, held={14: 1})
    _occupy(env, 1, RaceId.RATMEN, 10, 1, power=PowerId.BIVOUACKING)
    region = env.board.by_id(10)
    assert env._sorcerer_targets(sorcerers) == {10}
    region.encampments = 1
    assert env.tray.take_marker('encampment')
    assert env._sorcerer_targets(sorcerers) == set(), 'the encampment protects'
    assert sorcerer_action(_index(10)) not in _legal(env)
    _check_powers_invariants(env, 'bivouacking vs sorcerers')


def test_bivouacking_encampments_return_to_the_pool_when_the_region_falls():
    """They are never lost: a conquest sends every encampment back to the supply."""
    env = _env(308)
    rip = _setup_turn(env, 0, RaceId.RATMEN, None, hand=6, held={14: 1})
    victim = _occupy(env, 1, RaceId.HUMANS, 10, 1, power=PowerId.BIVOUACKING)
    region = env.board.by_id(10)
    region.encampments = 2
    for _ in range(2):
        assert env.tray.take_marker('encampment')
    assert env.tray.available_marker('encampment') == 3
    assert env._conquest_cost(rip, region) == 2 + 1 + 2, 'tokens + encampments'
    env.step(region_action(_index(10)))
    assert region.owner == 0 and region.encampments == 0
    assert env.tray.available_marker('encampment') == N_ENCAMPMENTS
    assert victim.tokens_in_hand == 0, 'its single token went to the tray'
    _check_powers_invariants(env, 'bivouacking conquered')


def test_bivouacking_encampments_are_removed_on_decline():
    """The badge goes, and so do the encampments."""
    env = _env(309)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.BIVOUACKING, hand=0,
                      held={9: 1, 10: 1}, turns_played=2)
    env.board.by_id(9).encampments = 4
    env.board.by_id(10).encampments = 1
    for _ in range(N_ENCAMPMENTS):
        assert env.tray.take_marker('encampment')
    env.step(A_DECLINE)
    assert rip.in_decline and rip.power is None
    assert env.board.by_id(9).encampments == 0
    assert env.board.by_id(10).encampments == 0
    assert env.tray.available_marker('encampment') == N_ENCAMPMENTS
    _check_powers_invariants(env, 'bivouacking decline')


# --------------------------------------------------------------------------- #
# Commando
# --------------------------------------------------------------------------- #

def test_commando_pays_one_token_less_everywhere():
    """-1 on every conquest, whatever the terrain."""
    env = _env(310)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.COMMANDO, hand=8,
                      held={9: 1})
    board = env.board
    assert env._conquest_cost(rip, board.by_id(2)) == 1, 'empty forest: 2 - 1'
    assert env._conquest_cost(rip, board.by_id(3)) == 2, 'mountain: 3 - 1'
    assert env._conquest_cost(rip, board.by_id(10)) == 2, 'hill + lost tribe'
    env.step(region_action(_index(2)))
    assert board.by_id(2).tokens == 1 and rip.tokens_in_hand == 7
    _check_powers_invariants(env, 'commando')


def test_commando_tritons_are_clamped_at_one_token():
    """Race and power modifiers chain, then the engine clamps to at least 1."""
    env = _env(311)
    rip = _setup_turn(env, 0, RaceId.TRITONS, PowerId.COMMANDO, hand=8,
                      held={9: 1})
    board = env.board
    assert board.is_coastal(board.by_id(2)) and not board.is_coastal(board.by_id(3))
    assert board.by_id(2).base_conquest_cost() == 2
    assert env._conquest_cost(rip, board.by_id(2)) == 1, '2 - 1 - 1 = 0 -> 1'
    assert env._conquest_cost(rip, board.by_id(10)) == 1, 'coastal + lost tribe'
    assert env._conquest_cost(rip, board.by_id(3)) == 2, 'inland mountain: 3 - 1'
    env.step(region_action(_index(2)))
    assert board.by_id(2).tokens == 1 and rip.tokens_in_hand == 7
    _check_powers_invariants(env, 'commando tritons')


# --------------------------------------------------------------------------- #
# Diplomat
# --------------------------------------------------------------------------- #

def test_diplomat_ally_phase_offers_the_opponents_not_attacked():
    """The ALLY sub-phase lists every un-attacked opponent, plus PASS."""
    env = _env(312)
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.DIPLOMAT, hand=0, held={9: 1},
                turns_played=2)
    env.step(A_PASS)
    assert env.phase == Phase.ALLY
    assert set(_legal(env)) == {A_PASS, ally_action(1), ally_action(2)}
    env.step(ally_action(1))
    assert env.players[0].ally == 1
    assert env.players[0].coins == 5 + 1
    assert env._turn_seat == 1
    # the turn is over, so the observation is player 1's: the Diplomat's row
    # holds the ally as a *relative* offset + 1 (seat 1 = the observer itself)
    assert _player_row(env, 0)[10] == 0 + 1, 'ally relative offset + 1'
    _check_powers_invariants(env, 'diplomat ally')


def test_diplomat_cannot_ally_with_a_player_he_attacked():
    """An attacked opponent leaves the ALLY options."""
    env = _env(313)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.DIPLOMAT, hand=8,
                      held={9: 1})
    _occupy(env, 1, RaceId.HUMANS, 10, 1)
    env.step(region_action(_index(10)))              # attack player 1
    assert rip.attacked_players == {1}
    env.step(A_PASS)
    assert env.phase == Phase.REDEPLOY
    env.step(region_all_action(_index(9)))
    assert env.phase == Phase.ALLY
    assert set(_legal(env)) == {A_PASS, ally_action(2)}
    env.step(A_PASS)                                  # no ally at all
    assert env.players[0].ally is None
    _check_powers_invariants(env, 'diplomat attacked')


def test_diplomat_protects_only_his_active_race():
    """The ally may not touch the active race, but declined tokens are fair game."""
    env = _env(314)
    _occupy(env, 0, RaceId.RATMEN, 10, 1, power=PowerId.DIPLOMAT)
    ghouls = _occupy(env, 0, RaceId.GHOULS, 13, 2, in_decline=True)
    assert ghouls.in_decline
    attacker = _setup_turn(env, 1, RaceId.HUMANS, None, hand=8, held={14: 1})
    board = env.board
    assert env._can_attack(attacker, board.by_id(10))
    env.players[0].ally = 1                           # player 0 protects player 1
    assert not env._can_attack(attacker, board.by_id(10)), 'active race protected'
    assert env._can_attack(attacker, board.by_id(13)), 'declined Ghouls are not'
    assert 10 not in _conquest_targets(env)
    assert 13 in _conquest_targets(env)
    _check_powers_invariants(env, 'diplomat protection')


def test_diplomat_protection_also_stops_a_sorcerer_substitution():
    """A substitution is an attack, so the alliance blocks it too."""
    env = _env(315)
    sorcerers = _setup_turn(env, 1, RaceId.SORCERERS, None, hand=0, held={14: 1})
    _occupy(env, 0, RaceId.RATMEN, 10, 1, power=PowerId.DIPLOMAT)
    assert env._sorcerer_targets(sorcerers) == {10}
    env.players[0].ally = 1
    assert env._sorcerer_targets(sorcerers) == set()
    _check_powers_invariants(env, 'diplomat vs sorcerers')


def test_diplomat_protection_expires_at_his_next_turn():
    """`player.ally` is cleared when the protector's own turn starts."""
    env = _env(316)
    _occupy(env, 0, RaceId.RATMEN, 10, 1, power=PowerId.DIPLOMAT)
    env.players[0].ally = 1
    env._begin_turn(1)
    assert env.players[0].ally == 1, 'still protecting during player 1\'s turn'
    env._begin_turn(0)
    assert env.players[0].ally is None, 'the protection expires'
    _check_powers_invariants(env, 'diplomat expiry')


# --------------------------------------------------------------------------- #
# Dragon Master
# --------------------------------------------------------------------------- #

def test_dragon_master_conquers_any_region_with_one_token():
    """One token, any defence; the region then becomes immune."""
    env = _env(317)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.DRAGON_MASTER, hand=2,
                      held={14: 1})
    victim = _occupy(env, 1, RaceId.HUMANS, 10, 5)
    board = env.board
    target = board.by_id(10)
    assert env._conquest_cost(rip, target) == 7
    assert target.id not in _conquest_targets(env), 'far too strong normally'
    assert 10 in _dragon_targets(env)
    assert 15 not in _dragon_targets(env), 'the lake is still out of reach'
    env.step(dragon_action(_index(10)))
    assert target.owner == 0 and target.tokens == 1 and target.dragon
    assert target.is_immune and rip.tokens_in_hand == 1
    assert rip.dragon_used and env.tray.available_marker('dragon') == 0
    assert victim.tokens_in_hand == 4 and victim.pending_redeploy
    assert _dragon_targets(env) == set(), 'once per turn'
    assert env.observation['global'][6] == 1
    _check_powers_invariants(env, 'dragon conquest')


def test_dragon_needs_an_otherwise_legal_target():
    """The Dragon ignores the defence, not the targeting rules."""
    env = _env(348)
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.DRAGON_MASTER, hand=3)
    assert env.players[0].must_first_conquest
    border = {r.id for r in env.board.regions if r.border and not r.is_water}
    assert _dragon_targets(env) == border, 'the first-conquest rule still holds'
    env.step(region_action(_index(25)))              # normal conquest, cost 2
    assert env.board.by_id(25).owner == 0
    # an immune region stays out of reach
    env.board.by_id(19).hole = True
    assert env.tray.take_marker('hole')
    assert 19 not in _dragon_targets(env)
    assert 26 in _dragon_targets(env)
    # and so does the active race of a player who allied with the Dragon Master
    _occupy(env, 1, RaceId.HUMANS, 26, 1)
    assert 26 in _dragon_targets(env)
    env.players[1].ally = 0
    assert 26 not in _dragon_targets(env)
    _check_powers_invariants(env, 'dragon legality')


def test_dragon_moves_with_the_next_dragon_conquest():
    """The marker follows the Dragon Master; the old region loses its immunity."""
    env = _env(318)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.DRAGON_MASTER, hand=3,
                      held={14: 1})
    env.step(dragon_action(_index(10)))
    first = env.board.by_id(10)
    assert first.dragon and first.is_immune
    rip.dragon_used = False                            # as a new turn would
    assert 4 in _dragon_targets(env)
    env.step(dragon_action(_index(4)))
    second = env.board.by_id(4)
    assert second.dragon and not first.dragon
    assert first.owner == 0 and not first.is_immune, 'immunity moved with it'
    assert env.tray.available_marker('dragon') == 0
    assert _markers(env)['dragon'] == 0
    _check_powers_invariants(env, 'dragon move')


def test_dragon_is_returned_to_the_supply_on_decline():
    """The Dragon leaves the map with the badge."""
    env = _env(319)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.DRAGON_MASTER, hand=0,
                      held={9: 1, 10: 1}, turns_played=2)
    region = env.board.by_id(10)
    region.dragon = True
    assert env.tray.take_marker('dragon')
    env.step(A_DECLINE)
    assert rip.in_decline and not region.dragon
    assert env.tray.available_marker('dragon') == 1
    assert not region.is_immune
    _check_powers_invariants(env, 'dragon decline')


# --------------------------------------------------------------------------- #
# Flying
# --------------------------------------------------------------------------- #

def test_flying_conquers_anywhere_but_the_water():
    """No border rule, no adjacency; the seas and the lake stay out of reach."""
    env = _env(320)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.FLYING, hand=10)
    assert env.players[0].must_first_conquest
    land = {r.id for r in env.board.regions if not r.is_water}
    assert _conquest_targets(env) == land, 'the first conquest may go anywhere'
    assert 21 in land and not env.board.by_id(21).border
    env.step(region_action(_index(21)))                # a non-border hill
    assert env.board.by_id(21).owner == 0
    # and the next conquests need no adjacency either
    assert _conquest_targets(env) == land - {21}
    assert env._can_attack(rip, env.board.by_id(6)), 'far away, still legal'
    assert not env._can_attack(rip, env.board.by_id(15)), 'the lake'
    _check_powers_invariants(env, 'flying')


def test_flying_sorcerers_substitute_anywhere():
    """`sorcerer_targets` resolves its reach through the `is_adjacent` hook."""
    env = _env(321)
    sorcerers = _setup_turn(env, 0, RaceId.SORCERERS, PowerId.FLYING, hand=0,
                            held={25: 1})
    _occupy(env, 1, RaceId.HUMANS, 5, 1)               # far from region 25
    _occupy(env, 2, RaceId.DWARVES, 29, 1)             # also far away
    assert 5 not in env.board.by_id(25).adjacent
    assert env._sorcerer_targets(sorcerers) == {5, 29}
    assert sorcerer_action(_index(5)) in _legal(env)
    # immunity and encampments still protect, wherever the target is
    env.board.by_id(5).hero = True
    assert env.tray.take_marker('hero')
    env.board.by_id(29).encampments = 1
    assert env.tray.take_marker('encampment')
    assert env._sorcerer_targets(sorcerers) == set()
    env.board.by_id(5).hero = False
    env.tray.put_marker('hero')
    assert env._sorcerer_targets(sorcerers) == {5}
    env.step(sorcerer_action(_index(5)))
    assert env.board.by_id(5).race == RaceId.SORCERERS
    env.board.by_id(29).encampments = 0
    env.tray.put_marker('encampment')
    _check_powers_invariants(env, 'flying sorcerers')


# --------------------------------------------------------------------------- #
# Forest / Hill / Swamp (terrain powers)
# --------------------------------------------------------------------------- #

def test_forest_scores_one_coin_per_forest_region():
    """+1 coin per Forest region held."""
    env = _env(322)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.FOREST, hand=0,
                      held={2: 1, 28: 1, 9: 1}, turns_played=2)
    forests = [r.id for r in env.board.regions_of_race(rip)
               if r.terrain == Terrain.FOREST]
    assert sorted(forests) == [2, 28]
    assert env._score_bonus(rip) == 2
    env.step(A_PASS)
    assert env.players[0].coins == 5 + 3 + 2
    _check_powers_invariants(env, 'forest')


def test_hill_scores_one_coin_per_hill_region():
    """+1 coin per Hill region held."""
    env = _env(323)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.HILL, hand=0,
                      held={10: 1, 21: 1, 9: 1}, turns_played=2)
    assert env._score_bonus(rip) == 2
    env.step(A_PASS)
    assert env.players[0].coins == 5 + 3 + 2
    _check_powers_invariants(env, 'hill')


def test_swamp_scores_one_coin_per_swamp_region():
    """+1 coin per Swamp region held."""
    env = _env(324)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.SWAMP, hand=0,
                      held={9: 1, 14: 1, 10: 1}, turns_played=2)
    assert env._score_bonus(rip) == 2
    env.step(A_PASS)
    assert env.players[0].coins == 5 + 3 + 2
    _check_powers_invariants(env, 'swamp')


# --------------------------------------------------------------------------- #
# Fortified
# --------------------------------------------------------------------------- #

def test_fortified_builds_one_fortress_per_turn_and_scores_it():
    """The FORTRESS phase is optional, once per turn, max 1 per region."""
    env = _env(325)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.FORTIFIED, hand=0,
                      held={9: 1, 10: 1}, turns_played=2)
    env.step(A_PASS)
    assert env.phase == Phase.FORTRESS
    assert set(_legal(env)) == {A_PASS, region_action(_index(9)),
                                region_action(_index(10))}
    env.step(region_action(_index(9)))
    board = env.board
    assert board.by_id(9).fortress and rip.fortress_used
    assert env.tray.available_marker('fortress') == MARKER_SUPPLY['fortress'] - 1
    assert env.phase != Phase.FORTRESS, 'only one fortress per turn'
    assert env.players[0].coins == 5 + 2 + 1, '2 regions + 1 fortress'
    assert board.by_id(9).defence == 1 + 1, 'fortress + token'
    assert board.by_id(9).base_conquest_cost() == 4
    _check_powers_invariants(env, 'fortified')

    # next turn: the region that already has one is not offered again
    env._begin_turn(0)
    assert not rip.fortress_used, 'the flag is reset by new_turn()'
    env.step(A_PASS)
    assert env.phase == Phase.FORTRESS
    assert set(_legal(env)) == {A_PASS, region_action(_index(10))}
    env.step(A_PASS)
    assert env.players[0].coins == 5 + 3 + 2 + 1


def test_fortified_is_skipped_when_the_six_fortresses_are_out():
    """The supply of 6 caps the power; the sub-phase is then skipped."""
    env = _env(326)
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.FORTIFIED, hand=0,
                held={9: 1}, turns_played=2)
    for region_id in (4, 25, 26, 27, 28, 29):          # 6 fortresses elsewhere
        env.board.by_id(region_id).fortress = True
        assert env.tray.take_marker('fortress')
    assert env.tray.available_marker('fortress') == 0
    env.step(A_PASS)
    assert env.phase != Phase.FORTRESS, 'no fortress left to build'
    assert env.players[0].coins == 5 + 1, 'no fortress of his own to score'
    _check_powers_invariants(env, 'fortified supply')


def test_fortress_defends_even_in_decline_but_scores_only_while_active():
    """The marker stays on the map at decline (user ruling) and keeps its +1."""
    env = _env(327)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.FORTIFIED, hand=0,
                      held={9: 1}, turns_played=2)
    region = env.board.by_id(9)
    region.fortress = True
    assert env.tray.take_marker('fortress')
    assert env._score_bonus(rip) == 1
    env.step(A_DECLINE)
    assert rip.in_decline and region.fortress, 'the fortress stays'
    assert region.defence == 1 + 1 and region.base_conquest_cost() == 4
    assert env.players[0].coins == 5 + 1, 'no fortress coin once declined'
    attacker = _setup_turn(env, 1, RaceId.HUMANS, None, hand=6, held={14: 1})
    assert env._conquest_cost(attacker, region) == 4
    env.step(region_action(_index(9)))
    assert not region.fortress, 'a conquest removes it'
    assert env.tray.available_marker('fortress') == MARKER_SUPPLY['fortress']
    _check_powers_invariants(env, 'fortress decline')


def test_fortress_is_removed_when_the_region_is_abandoned():
    """Abandoning a region returns its fortress to the supply."""
    env = _env(328)
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.FORTIFIED, hand=0,
                held={9: 2, 10: 1}, turns_played=2)
    region = env.board.by_id(9)
    region.fortress = True
    assert env.tray.take_marker('fortress')
    env.step(region_action(_index(9)))                 # abandon region 9
    assert region.owner is None and not region.fortress
    assert env.tray.available_marker('fortress') == MARKER_SUPPLY['fortress']
    _check_powers_invariants(env, 'fortress abandon')


# --------------------------------------------------------------------------- #
# Heroic
# --------------------------------------------------------------------------- #

def test_heroic_places_two_heroes_in_distinct_regions():
    """Both Heroes are placed, on 2 different regions, and make them immune."""
    env = _env(329)
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.HEROIC, hand=0,
                held={9: 1, 10: 1}, turns_played=2)
    env.step(A_PASS)
    assert env.phase == Phase.HEROES and env._heroes_left == N_HEROES
    assert set(_legal(env)) == {region_action(_index(9)), region_action(_index(10))}
    assert A_PASS not in _legal(env), 'the Heroes must be placed'
    env.step(region_action(_index(9)))
    assert env.board.by_id(9).hero and env._heroes_left == 1
    assert set(_legal(env)) == {region_action(_index(10))}, 'distinct regions'
    env.step(region_action(_index(10)))
    assert env.board.by_id(10).hero
    assert env.tray.available_marker('hero') == 0
    assert env.board.by_id(9).is_immune and env.board.by_id(10).is_immune
    assert env.players[0].coins == 5 + 2, 'Heroic pays no coin'
    # an immune region cannot be attacked
    attacker = _setup_turn(env, 1, RaceId.HUMANS, None, hand=8, held={14: 1})
    assert not env._can_attack(attacker, env.board.by_id(10))
    assert 10 not in _conquest_targets(env)
    _check_powers_invariants(env, 'heroic')


def test_heroic_places_fewer_heroes_with_fewer_regions_and_moves_them():
    """One region = one Hero; the Heroes are taken back every turn."""
    env = _env(330)
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.HEROIC, hand=0, held={9: 1},
                turns_played=2)
    env.step(A_PASS)
    assert env.phase == Phase.HEROES and env._heroes_left == 1
    env.step(region_action(_index(9)))
    assert env.board.by_id(9).hero
    assert env.tray.available_marker('hero') == 1, 'the second Hero stays home'
    assert env._turn_seat == 1

    # next turn, with a second region: both Heroes are re-placed
    _occupy(env, 0, RaceId.RATMEN, 10, 1)
    env._begin_turn(0)
    env.step(A_PASS)
    assert env.phase == Phase.HEROES and env._heroes_left == N_HEROES
    assert not env.board.by_id(9).hero, 'taken back before being re-placed'
    env.step(region_action(_index(10)))
    env.step(region_action(_index(9)))
    assert env.board.by_id(9).hero and env.board.by_id(10).hero
    _check_powers_invariants(env, 'heroic moves')


def test_heroes_are_returned_to_the_supply_on_decline():
    """The Heroes leave the map with the badge."""
    env = _env(331)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.HEROIC, hand=0,
                      held={9: 1, 10: 1}, turns_played=2)
    for region_id in (9, 10):
        env.board.by_id(region_id).hero = True
        assert env.tray.take_marker('hero')
    env.step(A_DECLINE)
    assert rip.in_decline
    assert not env.board.by_id(9).hero and not env.board.by_id(10).hero
    assert env.tray.available_marker('hero') == N_HEROES
    _check_powers_invariants(env, 'heroic decline')


# --------------------------------------------------------------------------- #
# Merchant
# --------------------------------------------------------------------------- #

def test_merchant_scores_one_extra_coin_per_region():
    """+1 coin per region held, so 2 coins per region in total."""
    env = _env(332)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.MERCHANT, hand=0,
                      held={9: 1, 10: 1, 14: 1}, turns_played=2)
    assert env._score_bonus(rip) == 3
    env.step(A_PASS)
    assert env.players[0].coins == 5 + 3 + 3
    _check_powers_invariants(env, 'merchant')


# --------------------------------------------------------------------------- #
# Mounted
# --------------------------------------------------------------------------- #

def test_mounted_pays_one_less_on_hills_and_farmlands():
    """-1 on Hill and Farmland regions only, min 1."""
    env = _env(333)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.MOUNTED, hand=8,
                      held={9: 1})
    board = env.board
    assert env._conquest_cost(rip, board.by_id(10)) == 2, 'hill + lost tribe'
    assert env._conquest_cost(rip, board.by_id(8)) == 2, 'farmland + lost tribe'
    assert env._conquest_cost(rip, board.by_id(4)) == 1, 'empty farmland: 2 - 1'
    assert env._conquest_cost(rip, board.by_id(3)) == 3, 'mountain: no discount'
    assert env._conquest_cost(rip, board.by_id(2)) == 2, 'forest: no discount'
    env.step(region_action(_index(10)))
    assert board.by_id(10).tokens == 2 and rip.tokens_in_hand == 6
    _check_powers_invariants(env, 'mounted')


# --------------------------------------------------------------------------- #
# Pillaging
# --------------------------------------------------------------------------- #

def test_pillaging_scores_the_non_empty_regions_conquered_this_turn():
    """+1 coin per non-empty conquest; empty regions and substitutions pay nothing."""
    env = _env(334)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.PILLAGING, hand=8,
                      held={9: 1})
    env.step(region_action(_index(10)))                # hill + lost tribe: 3
    assert rip.nonempty_conquests == 1
    env.step(region_action(_index(3)))                 # empty mountain: 3
    assert rip.nonempty_conquests == 1 and len(rip.conquests_this_turn) == 2
    assert env._score_bonus(rip) == 1
    env.step(A_PASS)
    env.step(region_all_action(_index(9)))             # place the last 2 tokens
    assert env.players[0].coins == 5 + 3 + 1
    _check_powers_invariants(env, 'pillaging')


def test_pillaging_ignores_a_sorcerer_substitution():
    """A substitution is not a conquest: no Pillaging coin."""
    env = _env(335)
    rip = _setup_turn(env, 0, RaceId.SORCERERS, PowerId.PILLAGING, hand=0,
                      held={14: 1})
    _occupy(env, 1, RaceId.HUMANS, 10, 1)
    env.step(sorcerer_action(_index(10)))
    assert rip.nonempty_conquests == 0 and rip.conquests_this_turn == []
    assert env._score_bonus(rip) == 0
    _check_powers_invariants(env, 'pillaging sorcerer')


# --------------------------------------------------------------------------- #
# Seafaring
# --------------------------------------------------------------------------- #

def test_seafaring_conquers_the_seas_and_the_lake_like_empty_regions():
    """Water costs 2 (nothing defends it) but adjacency still applies."""
    env = _env(336)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.SEAFARING, hand=8,
                      held={14: 1})
    board = env.board
    lake = board.by_id(15)
    assert lake.is_water and lake.base_conquest_cost() == 2
    assert env._conquest_cost(rip, lake) == 2, 'the normal empty-region cost'
    assert 15 in _conquest_targets(env)
    assert 1 not in _conquest_targets(env), 'sea 1 is not adjacent to region 14'
    env.step(region_action(_index(15)))
    assert lake.owner == 0 and lake.tokens == 2 and lake.race == RaceId.RATMEN
    assert env._score_bonus(rip) == 0, 'Seafaring pays no bonus'
    _check_powers_invariants(env, 'seafaring lake')


def test_seafaring_may_enter_through_a_border_sea():
    """The first conquest may be sea 1 or 30 (they touch the edge), not the lake."""
    env = _env(337)
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.SEAFARING, hand=8)
    targets = _conquest_targets(env)
    assert {1, 30} <= targets, 'both seas are border regions'
    assert 15 not in targets, 'the lake is not a border region'
    env.step(region_action(_index(30)))
    assert env.board.by_id(30).owner == 0 and env.board.by_id(30).tokens == 2
    _check_powers_invariants(env, 'seafaring border sea')


def test_seafaring_keeps_the_water_in_decline_and_nobody_can_take_it():
    """Water regions are kept like any other, and stay unreachable afterwards."""
    env = _env(338)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.SEAFARING, hand=0,
                      held={15: 3, 14: 1}, turns_played=2)
    env.step(A_DECLINE)
    lake = env.board.by_id(15)
    assert rip.in_decline and lake.owner == 0 and lake.in_decline
    assert lake.tokens == 1, 'one token per region, water included'
    assert env.players[0].coins == 5 + 2, 'the lake scores a coin'
    # a race without the badge can never target it, even in decline
    attacker = _setup_turn(env, 1, RaceId.HUMANS, None, hand=8, held={10: 1})
    assert 15 in lake.adjacent or 15 in env.board.by_id(10).adjacent
    assert not env._can_attack(attacker, lake)
    assert 15 not in _conquest_targets(env)
    _check_powers_invariants(env, 'seafaring decline')


# --------------------------------------------------------------------------- #
# Spirit
# --------------------------------------------------------------------------- #

def test_spirit_declined_race_survives_the_next_decline():
    """`is_spirit_power` exempts the race from the one-declined-race limit."""
    env = _env(339)
    first = _setup_turn(env, 0, RaceId.DWARVES, PowerId.SPIRIT, hand=0,
                        held={25: 1}, turns_played=2)
    env.step(A_DECLINE)
    assert first.is_spirit and first.power is None
    assert env.players[0].declined == [first]
    assert any('(Spirit)' in line for line in env.event_log)

    second = _setup_turn(env, 0, RaceId.RATMEN, PowerId.MERCHANT, hand=0,
                         held={9: 1}, turns_played=2)
    env.step(A_DECLINE)
    assert not second.is_spirit
    assert [d.race for d in env.players[0].declined] == [RaceId.DWARVES,
                                                         RaceId.RATMEN]
    assert env.board.by_id(25).owner == 0, 'the Spirit race stays on the map'
    _check_powers_invariants(env, 'spirit')


def test_two_spirits_give_three_declined_races_and_the_first_two_are_observed():
    """More than 2 declined races is legal; the observation shows the first two."""
    env = _env(340)
    for race, power, region_id in ((RaceId.DWARVES, PowerId.SPIRIT, 25),
                                   (RaceId.HUMANS, PowerId.SPIRIT, 26),
                                   (RaceId.RATMEN, PowerId.MERCHANT, 9)):
        _setup_turn(env, 0, race, power, hand=0, held={region_id: 1},
                    turns_played=2)
        env.step(A_DECLINE)
    player = env.players[0]
    assert [d.race for d in player.declined] == [RaceId.DWARVES, RaceId.HUMANS,
                                                 RaceId.RATMEN]
    assert [d.is_spirit for d in player.declined] == [True, True, False]
    row = _player_row(env, 0)
    assert row[6] == int(RaceId.DWARVES) + 1, 'first declined slot'
    assert row[7] == int(RaceId.HUMANS) + 1, 'second declined slot'
    assert row[9] == 3, 'every declined region is counted'
    assert env.observation_space.contains(env.observation)
    # all three score, and a fourth decline only evicts the non-Spirit one
    coins = player.coins
    env._score(player)
    assert player.coins == coins + 3
    fourth = _setup_turn(env, 0, RaceId.ORCS, PowerId.MERCHANT, hand=0,
                         held={10: 1}, turns_played=2)
    env.step(A_DECLINE)
    assert [d.race for d in player.declined] == [RaceId.DWARVES, RaceId.HUMANS,
                                                 RaceId.ORCS]
    assert fourth in player.declined and env.board.by_id(9).owner is None
    _check_powers_invariants(env, 'two spirits')


# --------------------------------------------------------------------------- #
# Stout
# --------------------------------------------------------------------------- #

def test_stout_may_decline_after_scoring():
    """STOUT_DECLINE comes after the scoring step and scores nothing again."""
    env = _env(341)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.STOUT, hand=0,
                      held={9: 1, 10: 1}, turns_played=2)
    env.step(A_PASS)
    assert env.phase == Phase.STOUT_DECLINE
    assert set(_legal(env)) == {A_DECLINE, A_PASS}
    assert env.players[0].coins == 5 + 2, 'already scored'
    env.step(A_DECLINE)
    assert rip.in_decline and rip.power is None
    assert env.players[0].coins == 5 + 2, 'no second scoring'
    assert [r.tokens for r in env.board.regions_of_race(rip)] == [1, 1]
    assert env._turn_seat == 1
    _check_powers_invariants(env, 'stout decline')


def test_stout_may_also_stay_active():
    """PASS in STOUT_DECLINE simply ends the turn."""
    env = _env(342)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.STOUT, hand=0,
                      held={9: 1}, turns_played=2)
    env.step(A_PASS)
    assert env.phase == Phase.STOUT_DECLINE
    env.step(A_PASS)
    assert not rip.in_decline and env.players[0].active is rip
    assert env._turn_seat == 1
    _check_powers_invariants(env, 'stout stays')


# --------------------------------------------------------------------------- #
# Underworld
# --------------------------------------------------------------------------- #

def test_underworld_links_the_caverns_and_discounts_them():
    """-1 on a Cavern region, and every Cavern is adjacent to every Cavern."""
    env = _env(343)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.UNDERWORLD, hand=8,
                      held={13: 1})
    board = env.board
    assert board.by_id(13).has(Symbol.CAVERN)
    caverns = {r.id for r in board.caverns()}
    assert caverns == {5, 13, 16, 20, 24}
    # the other caverns are not printed as neighbours of 13 but are reachable
    for cavern_id in (5, 16, 20, 24):
        assert cavern_id not in board.by_id(13).adjacent
        assert env._can_attack(rip, board.by_id(cavern_id)), cavern_id
    assert env._conquest_cost(rip, board.by_id(16)) == 2, 'mountain cavern: 3 - 1'
    assert env._conquest_cost(rip, board.by_id(20)) == 2, 'cavern + lost tribe'
    assert env._conquest_cost(rip, board.by_id(5)) == 1, 'empty cavern: 2 - 1'
    assert env._conquest_cost(rip, board.by_id(9)) == 2, 'no cavern, no discount'
    # a region that is neither a cavern nor adjacent stays out of reach
    assert not env._can_attack(rip, board.by_id(10))
    # the printed neighbours of 13 (bar the sea 1) plus the 4 other caverns;
    # region 13 itself is only there as the "abandon" action
    assert _conquest_targets(env) - {13} == {8, 9, 14, 19} | {5, 16, 20, 24}
    env.step(region_action(_index(20)))
    assert board.by_id(20).tokens == 2 and rip.tokens_in_hand == 6
    _check_powers_invariants(env, 'underworld')


def test_underworld_needs_a_cavern_of_its_own():
    """Without a Cavern region the links do not exist (the discount still does)."""
    env = _env(344)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.UNDERWORLD, hand=8,
                      held={9: 1})
    board = env.board
    assert not any(r.has(Symbol.CAVERN) for r in board.regions_of_race(rip))
    assert not env._can_attack(rip, board.by_id(16)), 'no cavern held'
    assert not env._can_attack(rip, board.by_id(5))
    assert env._can_attack(rip, board.by_id(13)), 'printed adjacency still works'
    assert env._conquest_cost(rip, board.by_id(16)) == 2, 'the discount applies'
    assert env._conquest_cost(rip, board.by_id(13)) == 2, 'cavern mountain: 3 - 1'
    _check_powers_invariants(env, 'underworld no cavern')


def test_underworld_sorcerers_reach_through_the_caverns():
    """The Sorcerers' reach uses the same `is_adjacent` hook."""
    env = _env(345)
    sorcerers = _setup_turn(env, 0, RaceId.SORCERERS, PowerId.UNDERWORLD,
                            hand=0, held={13: 1})
    _occupy(env, 1, RaceId.HUMANS, 20, 1)              # lone token, far cavern
    _occupy(env, 2, RaceId.DWARVES, 10, 1)             # lone token, no cavern
    assert 20 not in env.board.by_id(13).adjacent
    assert env._sorcerer_targets(sorcerers) == {20}
    env.step(sorcerer_action(_index(20)))
    assert env.board.by_id(20).race == RaceId.SORCERERS
    _check_powers_invariants(env, 'underworld sorcerers')


# --------------------------------------------------------------------------- #
# Wealthy
# --------------------------------------------------------------------------- #

def test_wealthy_pays_seven_coins_at_the_end_of_its_first_turn():
    """+7 once, on the turn the combo was picked (`turns_played == 1`)."""
    env = _env(346)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.WEALTHY, hand=0,
                      held={9: 1}, turns_played=1)
    assert rip.turns_played == 1 and not rip.wealthy_paid
    _, rewards, *_ = env.step(A_PASS)
    assert rip.wealthy_paid
    assert env.players[0].coins == 5 + 1 + WEALTHY_BONUS
    assert any('is Wealthy' in line for line in env.event_log)
    # the gain is shaped like a scoring event, and stays zero-sum
    assert abs(sum(rewards)) < 1e-9
    assert rewards[0] == (1 + WEALTHY_BONUS) / env.SHAPING_SCALE

    # second turn: nothing more
    env._begin_turn(0)
    assert rip.turns_played == 2
    env.step(A_PASS)
    assert env.players[0].coins == 5 + 1 + WEALTHY_BONUS + 1
    assert sum(1 for line in env.event_log if 'is Wealthy' in line) == 1
    _check_powers_invariants(env, 'wealthy')


def test_wealthy_pays_nothing_if_the_race_declines_on_its_first_turn():
    """The badge is discarded before `on_turn_end`, so no bonus (plan ruling)."""
    env = _env(347)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.WEALTHY, hand=0,
                      held={9: 1}, turns_played=1)
    env.step(A_DECLINE)
    assert rip.in_decline and not rip.wealthy_paid
    assert env.players[0].coins == 5 + 1, 'only the region coin'
    assert not any('is Wealthy' in line for line in env.event_log)
    _check_powers_invariants(env, 'wealthy decline')


# --------------------------------------------------------------------------- #
# Race + power combinations
# --------------------------------------------------------------------------- #

def test_race_and_power_score_bonuses_add_up():
    """`score_bonus` is the **sum** of the race's and the power's (hooks.py)."""
    env = _env(350)
    rip = _setup_turn(env, 0, RaceId.DWARVES, PowerId.MERCHANT, hand=0,
                      held={13: 1, 11: 1, 9: 1}, turns_played=2)
    mines = sorted(r.id for r in env.board.regions_of_race(rip)
                   if r.has(Symbol.MINE))
    assert mines == [11, 13]
    assert env._score_bonus(rip) == 2 + 3, "Dwarves' mines + Merchant's regions"
    env.step(A_PASS)
    assert env.players[0].coins == 5 + 3 + 5
    _check_powers_invariants(env, 'dwarves merchant')


# --------------------------------------------------------------------------- #
# Fuzz
# --------------------------------------------------------------------------- #

def test_fuzz_random_games_with_every_race_and_power():
    """`FUZZ_GAMES` random games with all 14 races and all 20 powers live.

    Re-checks every T3/T4 invariant (plan 3.10) plus the power-specific ones at
    **every** step, and asserts that random play really does exercise the power
    mechanics (every badge picked, dragon conquests, substitutions, alliances,
    Stout declines, encampments, fortresses, Heroes, water conquests, Berserk
    rolls, Wealthy payments, Spirit declines).
    """
    env = SmallWorldEnv(3)
    counters: Counter = Counter()
    powers_picked: set = set()
    races_picked: set = set()
    phases_seen: set = set()
    steps_per_game = []
    coins = []
    started = time.time()

    for seed in range(FUZZ_GAMES):
        obs, info = env.reset(seed=2000 + seed)
        assert env.observation_space.contains(obs)
        assert info == {'next_step_no_action': False}
        _check_powers_invariants(env, f'seed {seed} reset')
        rng = np.random.default_rng(seed)
        steps = 0
        while not env.done:
            assert steps < STEP_BUDGET, f'seed {seed}: over {STEP_BUDGET} steps'
            phase = env.phase
            phases_seen.add(phase)
            if env.phase == Phase.TURN_PAUSE:            # no-action step
                assert env.current_player == -1 and not env.action_masks().any()
                obs, rewards, terminated, truncated, info = env.step(-1)
                steps += 1
                assert rewards == [0.0] * env.n_players
                assert info == {'next_step_no_action': env.phase == Phase.TURN_PAUSE}
                continue
            legal = np.flatnonzero(env.action_masks())
            assert legal.size, f'seed {seed}: no legal action in {phase.name}'
            action = int(rng.choice(legal))
            kind, arg = action_kind(action)
            counters[kind] += 1
            if kind == 'combo':
                combo = env.combo_column.visible[arg]
                powers_picked.add(combo.power)
                races_picked.add(combo.race)
            elif kind == 'dragon':
                counters['dragon_conquests'] += 1
            elif kind == 'sorcerer':
                counters['substitutions'] += 1
            elif kind == 'ally':
                counters['allies'] += 1
            elif kind == 'region':
                if phase == Phase.ENCAMPMENTS:
                    counters['encampments_placed'] += 1
                elif phase == Phase.FORTRESS:
                    counters['fortresses_built'] += 1
                elif phase == Phase.HEROES:
                    counters['heroes_placed'] += 1
                elif phase in (Phase.CONQUER, Phase.GHOUL_CONQUER):
                    region = env.board.by_index(arg)
                    if region.is_water and not env._holds(env.current_race, region):
                        counters['water_conquests'] += 1
            if phase == Phase.STOUT_DECLINE:
                counters['stout_offers'] += 1
                if kind == 'decline':
                    counters['stout_declines'] += 1
            assert env.describe_action(action), phase.name
            obs, rewards, terminated, truncated, info = env.step(action)
            steps += 1
            ctx = f'seed {seed} step {steps} ({env.phase.name})'
            assert truncated is False and info == {
                'next_step_no_action': env.phase == Phase.TURN_PAUSE}, ctx
            assert len(rewards) == env.n_players, ctx
            assert abs(sum(rewards)) < 1e-9, f'{ctx}: {rewards} not zero-sum'
            assert terminated == env.done, ctx
            _check_powers_invariants(env, ctx)

        # -- terminal state ------------------------------------------------- #
        assert env.turns_taken == turns_for(3), f'seed {seed}: {env.turns_taken}'
        best = max(p.coins for p in env.players)
        assert env.players[env.winner_player].coins == best, f'seed {seed}'
        assert abs(sum(env.terminal_rewards)) < 1e-9
        assert not env.action_masks().any(), f'seed {seed}: mask after the end'
        for line in env.event_log:
            if 'rolls the reinforcement die' in line:
                counters['berserk_rolls'] += 1
            elif 'is Wealthy' in line:
                counters['wealthy_payments'] += 1
            elif '(Spirit)' in line:
                counters['spirit_declines'] += 1
            elif 'the dragon settles' in line:
                counters['dragon_settles'] += 1
        steps_per_game.append(steps)
        coins.append([p.coins for p in env.players])

    FUZZ_STATS.update(
        games=FUZZ_GAMES,
        steps_total=int(np.sum(steps_per_game)),
        steps_mean=round(float(np.mean(steps_per_game)), 1),
        steps_min=int(np.min(steps_per_game)),
        steps_max=int(np.max(steps_per_game)),
        coins_mean=round(float(np.mean(coins)), 1),
        coins_max=int(np.max(coins)),
        races_picked=len(races_picked),
        powers_picked=len(powers_picked),
        phases=sorted(p.name for p in phases_seen),
        seconds=round(time.time() - started, 1),
        **{k: int(v) for k, v in sorted(counters.items())},
    )

    assert powers_picked == set(PowerId), sorted(
        p.name for p in set(PowerId) - powers_picked)
    assert races_picked == set(RaceId), sorted(
        r.name for r in set(RaceId) - races_picked)
    assert {Phase.ENCAMPMENTS, Phase.FORTRESS, Phase.HEROES, Phase.ALLY,
            Phase.STOUT_DECLINE} <= phases_seen, sorted(p.name for p in phases_seen)
    for key in ('dragon_conquests', 'dragon_settles', 'substitutions', 'allies',
                'stout_declines', 'encampments_placed', 'fortresses_built',
                'heroes_placed', 'water_conquests', 'berserk_rolls',
                'wealthy_payments', 'spirit_declines'):
        assert counters[key] > 0, f'{key} never happened in {FUZZ_GAMES} games'


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def run_all() -> bool:
    """Run every `test_*` function of this module; print a summary."""
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failed = []
    for name, fn in tests:
        started = time.time()
        try:
            fn()
        except Exception:                                   # noqa: BLE001
            failed.append(name)
            print(f'FAIL {name}')
            print(''.join('    ' + line for line in
                          traceback.format_exc().splitlines(keepends=True)))
        else:
            print(f'ok   {name} ({time.time() - started:.1f}s)')
    if FUZZ_STATS:
        print(f'\nfuzz statistics: {FUZZ_STATS}')
    print(f'\n{__name__}: {len(tests) - len(failed)}/{len(tests)} passed'
          + (f', FAILED: {", ".join(failed)}' if failed else ''))
    return not failed


if __name__ == '__main__':
    import sys

    sys.exit(0 if run_all() else 1)
