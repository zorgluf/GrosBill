"""Tests for the 14 races of Small World (`envs/races.py`, task T4).

Plain `assert`-based functions, usable either with pytest or through the
bundled runners:

    cd app
    python -m environments.smallw.tests.run_all        # every test module
    python -m environments.smallw.tests.test_races     # this module only

One or more **scenario** tests per race prove the rule on a hand-built state
(the helpers `_env` / `_setup_turn` / `_occupy` / `_legal` of `test_engine.py`
are reused), plus a **fuzz** run of `FUZZ_GAMES` random games with the real
race hooks registered, re-checking every T3 invariant (plan 3.10) and the
race-specific ones (`_check_races_invariants`).

The scenarios give the active race **no special power** (`power=None`), so that
T5 cannot perturb them: the tested numbers are those of the race alone.
"""

from __future__ import annotations

import time
import traceback
from collections import Counter

import numpy as np

from ..envs.classes import (
    MARKER_SUPPLY,
    N_HOLES,
    RACES,
    PowerId,
    RaceId,
    RaceInPlay,
    turns_for,
)
from ..envs.hooks import RaceHooks, get_race_hooks
from ..envs.races import (
    RACE_HOOK_INSTANCES,
    AmazonsHooks,
    GiantsHooks,
    HumansHooks,
    TritonsHooks,
)
from ..envs.smallw import (
    A_DECLINE,
    A_PASS,
    A_REGION,
    A_REGION_ALL,
    Phase,
    SmallWorldEnv,
    action_kind,
    region_action,
    region_all_action,
    sorcerer_action,
)
from .test_engine import _check_invariants, _env, _legal, _occupy, _setup_turn

#: Games played by the race fuzz test.
FUZZ_GAMES = 150
#: Hard step budget per game.
STEP_BUDGET = 3000

#: Statistics of the last fuzz run, printed by `run_all()`.
FUZZ_STATS: dict = {}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _conquest_targets(env: SmallWorldEnv) -> set[int]:
    """0-based indices of the regions the legal REGION actions point at."""
    return {a - A_REGION for a in _legal(env) if A_REGION <= a < A_REGION_ALL}


def _index(region_id: int) -> int:
    """0-based index of the region printed `region_id`."""
    return region_id - 1


def _check_races_invariants(env: SmallWorldEnv, ctx: str) -> None:
    """Every T3 invariant plus the ones the 14 races add.

    `_check_invariants` is hook-aware for the declined tokens (the Ghouls keep
    all of theirs), so it runs with its full strength here.
    """
    _check_invariants(env, ctx)

    for region in env.board.regions:
        if region.hole:
            # only the *active* Halflings ever own a hole (removed at decline)
            assert region.race == RaceId.HALFLINGS and not region.in_decline, (
                f'{ctx}: hole on region {region.id} ({region!r})')
        if region.lair:
            # a Lair sits on a Troll region, or on the region a declined Troll
            # race left behind (user ruling: it stays there)
            assert region.race == RaceId.TROLLS or region.owner is None, (
                f'{ctx}: Lair on region {region.id} ({region!r})')
        if region.in_decline and region.tokens > 1:
            assert region.race == RaceId.GHOULS, (
                f'{ctx}: declined region {region.id} holds {region.tokens} tokens')

    for player in env.players:
        for rip in player.all_races():
            assert rip.holes_placed <= N_HOLES, f'{ctx}: {rip!r}'
            if rip.race != RaceId.HALFLINGS:
                assert rip.holes_placed == 0, f'{ctx}: {rip!r} dug a hole'
            if rip.race != RaceId.SORCERERS:
                assert not rip.sorcerer_used_on, f'{ctx}: {rip!r} substituted'
            assert rip.owner not in rip.sorcerer_used_on, f'{ctx}: {rip!r}'


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

def test_every_race_has_its_own_hooks():
    """The 14 banners are registered, each with a distinct hook class."""
    assert set(RACE_HOOK_INSTANCES) == set(RaceId)
    classes = set()
    for race_id in RaceId:
        hooks = get_race_hooks(race_id)
        assert hooks is RACE_HOOK_INSTANCES[race_id]
        assert isinstance(hooks, RaceHooks)
        assert type(hooks) is not RaceHooks, f'{race_id.name} is still vanilla'
        assert hooks.name == RACES[race_id].name, race_id.name
        classes.add(type(hooks))
    assert len(classes) == len(RaceId)


# --------------------------------------------------------------------------- #
# Amazons
# --------------------------------------------------------------------------- #

def test_amazons_keep_their_four_attack_tokens_in_hand():
    """The 4 attack-only tokens are never deployed (plan section 5 item 3)."""
    env = _env(201)
    # 6 (banner) + 4 (attack only) = 10 without a power
    rip = _setup_turn(env, 0, RaceId.AMAZONS, None, hand=10, held={9: 1})
    assert rip.race_def.attack_bonus_tokens == 4
    assert env._redeployable(rip) == 6, 'only 10 - 4 tokens may be placed'
    env.step(A_PASS)                                   # end the conquests
    assert env.phase == Phase.REDEPLOY
    assert 'all 6 remaining' in env.describe_action(region_all_action(_index(9)))
    env.step(region_all_action(_index(9)))
    assert env.board.by_id(9).tokens == 7
    assert rip.tokens_in_hand == 4, 'the 4 attack tokens stay in hand'
    assert env.phase != Phase.REDEPLOY, 'nothing left to place'
    assert env._turn_seat == 1, 'the turn is over'
    # conservation: the 4 tokens in hand are still counted
    assert (env.tray.available(RaceId.AMAZONS) + rip.tokens_in_hand
            + env.board.by_id(9).tokens) == RACES[RaceId.AMAZONS].total_tokens
    _check_races_invariants(env, 'amazons redeploy')


def test_amazons_attack_tokens_come_back_next_turn():
    """They are usable again on the next turn (they never left the hand)."""
    env = _env(202)
    rip = _setup_turn(env, 0, RaceId.AMAZONS, None, hand=4, held={9: 3})
    assert env._redeployable(rip) == 0
    env.step(A_PASS)                       # the 2 extra tokens of 9 are lifted
    assert env.phase == Phase.REDEPLOY
    assert rip.tokens_in_hand == 4 + 2 and env._redeployable(rip) == 2
    env.step(region_all_action(_index(9)))  # put them back, turn over
    assert rip.tokens_in_hand == 4 and env.board.by_id(9).tokens == 3
    env._begin_turn(0)                     # his next turn
    assert env.phase == Phase.CONQUER
    assert rip.tokens_in_hand == 4 + 2, 'ready your troops + the attack tokens'
    assert env.board.by_id(9).tokens == 1
    # and they may be spent on a conquest
    target = env.board.by_id(10)           # hill + lost tribe, adjacent to 9
    assert env._conquest_cost(rip, target) == 3
    env.step(region_action(_index(10)))
    assert rip.tokens_in_hand == 3 and target.tokens == 3
    _check_races_invariants(env, 'amazons next turn')


def test_amazons_attack_tokens_go_back_to_the_tray_on_decline():
    """Decline: the whole hand returns to the tray, attack tokens included."""
    env = _env(203)
    rip = _setup_turn(env, 0, RaceId.AMAZONS, None, hand=4, held={9: 3, 10: 1})
    total = RACES[RaceId.AMAZONS].total_tokens
    env.step(A_DECLINE)
    assert rip.in_decline and rip.tokens_in_hand == 0
    assert env.board.by_id(9).tokens == 1 and env.board.by_id(10).tokens == 1
    assert env.tray.available(RaceId.AMAZONS) == total - 2
    assert env.players[0].coins == 5 + 2, '1 coin per region, no bonus'
    _check_races_invariants(env, 'amazons decline')


def test_amazons_victim_redeploy_uses_the_same_rule():
    """After losses the Amazons redeploy their hand minus the 4 attack tokens."""
    env = _env(204)
    # seat 2 so that the next seat to play is not the victim (its own turn
    # would immediately take the tokens back with *ready your troops*)
    amazons = _occupy(env, 2, RaceId.AMAZONS, 20, 6)
    _occupy(env, 2, RaceId.AMAZONS, 21, 1)
    _occupy(env, 2, RaceId.AMAZONS, 22, 1)
    rip = _setup_turn(env, 0, RaceId.RATMEN, None, hand=10, held={14: 1})
    assert env._conquest_cost(rip, env.board.by_id(20)) == 2 + 6
    env.step(region_action(_index(20)))
    assert amazons.tokens_in_hand == 5 and amazons.pending_redeploy
    env.step(A_PASS)
    while env.phase == Phase.REDEPLOY:
        env.step(region_all_action(_index(14)))
    assert env.phase == Phase.VICTIM_REDEPLOY and env.current_player == 2
    assert env.current_race is amazons
    assert env._redeployable(amazons) == 1, '5 in hand - 4 attack tokens'
    assert set(_legal(env)) == {region_action(_index(21)), region_action(_index(22)),
                                region_all_action(_index(21)),
                                region_all_action(_index(22))}
    env.step(region_action(_index(21)))
    assert amazons.tokens_in_hand == 4 and not amazons.pending_redeploy
    assert env.board.by_id(21).tokens == 2
    _check_races_invariants(env, 'amazons victim')


def test_amazons_hook_is_off_in_decline():
    """A declined race has no attack-only token to reserve."""
    env = _env(205)
    rip = RaceInPlay(RaceId.AMAZONS, None, 0, in_decline=True, tokens_in_hand=3)
    assert AmazonsHooks().redeployable_tokens(env, rip) == 3


# --------------------------------------------------------------------------- #
# Dwarves
# --------------------------------------------------------------------------- #

def test_dwarves_score_their_mines_even_in_decline():
    """+1 coin per Mine region, active and in decline (`bonus_in_decline`)."""
    env = _env(206)
    # mines are printed in regions 2, 7, 11, 13 and 29
    rip = _setup_turn(env, 0, RaceId.DWARVES, None, hand=0, held={2: 1, 3: 1, 13: 1})
    assert env._score_bonus(rip) == 2, 'regions 2 and 13 hold a mine'
    assert env._rip_flag('bonus_in_decline', rip)
    env.step(A_DECLINE)
    assert env.players[0].coins == 5 + 3 + 2, '3 regions + 2 mines'
    declined = env.players[0].declined[0]
    assert declined is rip and declined.power is None
    env._score(env.players[0])             # the following turns keep the bonus
    assert env.players[0].coins == 5 + 3 + 2 + 3 + 2
    _check_races_invariants(env, 'dwarves')


# --------------------------------------------------------------------------- #
# Elves
# --------------------------------------------------------------------------- #

def test_elves_lose_no_token_when_a_region_is_conquered():
    """`defender_loss` = 0: every token goes back in hand for redeployment."""
    env = _env(207)
    elves = _occupy(env, 2, RaceId.ELVES, 20, 4)     # seat 2: it plays last
    _occupy(env, 2, RaceId.ELVES, 21, 1)
    rip = _setup_turn(env, 0, RaceId.RATMEN, None, hand=8, held={14: 1})
    before = env.tray.available(RaceId.ELVES)
    assert env._defender_loss(elves, env.board.by_id(20)) == 0
    env.step(region_action(_index(20)))                # cost 2 + 4
    assert elves.tokens_in_hand == 4, 'all four tokens survive'
    assert env.tray.available(RaceId.ELVES) == before, 'nothing goes to the tray'
    assert elves.pending_redeploy
    env.step(A_PASS)
    while env.phase == Phase.REDEPLOY:
        env.step(region_all_action(_index(14)))
    # one region left: the engine redeploys the four tokens there
    assert env.board.by_id(21).tokens == 5 and elves.tokens_in_hand == 0
    _check_races_invariants(env, 'elves losses')


def test_elves_token_is_still_removed_by_a_sorcerer():
    """A substitution sends the Elf token to the tray (rules p.4)."""
    env = _env(208)
    rip = _setup_turn(env, 0, RaceId.SORCERERS, None, hand=0, held={14: 1})
    elves = _occupy(env, 1, RaceId.ELVES, 10, 1)
    _occupy(env, 1, RaceId.ELVES, 11, 1)
    before = env.tray.available(RaceId.ELVES)
    assert sorcerer_action(_index(10)) in _legal(env)
    env.step(sorcerer_action(_index(10)))
    assert env.tray.available(RaceId.ELVES) == before + 1
    assert elves.tokens_in_hand == 0 and not elves.pending_redeploy
    assert env.board.by_id(10).race == RaceId.SORCERERS
    _check_races_invariants(env, 'elves vs sorcerer')


# --------------------------------------------------------------------------- #
# Ghouls
# --------------------------------------------------------------------------- #

def test_ghouls_keep_every_token_in_decline_and_redeploy_them():
    """Decline: nothing goes back to the tray, the hand is redeployed."""
    env = _env(209)
    rip = _setup_turn(env, 0, RaceId.GHOULS, None, hand=4, held={9: 2, 10: 1})
    tray_before = env.tray.available(RaceId.GHOULS)
    env.step(A_DECLINE)
    assert rip.in_decline and env.phase == Phase.GHOUL_REDEPLOY
    assert rip.tokens_in_hand == 4, 'the hand is kept'
    assert env.tray.available(RaceId.GHOULS) == tray_before, 'nothing returned'
    assert env.board.by_id(9).tokens == 2, 'the region keeps all its tokens'
    env.step(region_all_action(_index(10)))
    assert env.board.by_id(10).tokens == 5 and rip.tokens_in_hand == 0
    assert env.players[0].coins == 5 + 2, '1 coin per region, no Ghoul bonus'
    assert env._score_bonus(rip) == 0
    assert env._turn_seat == 1
    _check_races_invariants(env, 'ghoul decline')


def test_declined_ghouls_play_before_the_active_race_and_may_attack_it():
    """GHOUL_CONQUER runs first and the owner's active race is a legal target."""
    env = _env(210)
    ghouls = _occupy(env, 0, RaceId.GHOULS, 25, 6, in_decline=True)
    _occupy(env, 0, RaceId.GHOULS, 26, 1, in_decline=True)
    ratmen = _occupy(env, 0, RaceId.RATMEN, 19, 2)        # adjacent to 25 and 26
    env._begin_turn(0)
    assert env.phase == Phase.GHOUL_CONQUER and env.current_race is ghouls
    assert ghouls.tokens_in_hand == 5, 'ready your troops, 1 token per region'
    assert A_DECLINE not in _legal(env), 'a declined race cannot decline'
    assert env._conquest_cost(ghouls, env.board.by_id(19)) == 2 + 2
    assert _index(19) in _conquest_targets(env), 'his own active race is a target'
    env.step(region_action(_index(19)))
    region = env.board.by_id(19)
    assert region.race == RaceId.GHOULS and region.in_decline
    assert region.tokens == 4 and ghouls.tokens_in_hand == 1
    assert ratmen.tokens_in_hand == 1 and ratmen.pending_redeploy
    assert ghouls.attacked_players == set(), 'his own player is not "attacked"'
    assert env.players[0].must_first_conquest, 'the Ratmen lost everything'
    env.step(A_PASS)
    assert env.phase == Phase.GHOUL_REDEPLOY
    env.step(region_all_action(_index(25)))
    # the active race plays next, with its withdrawn token back in hand
    assert env.phase == Phase.CONQUER and env.current_race is ratmen
    assert ratmen.tokens_in_hand == 1 and not ratmen.pending_redeploy
    targets = _conquest_targets(env)
    assert targets and all(env.board.by_index(i).border for i in targets)
    _check_races_invariants(env, 'ghoul phase')


def test_declined_ghouls_take_losses_like_an_active_race():
    """`losses_like_active`: 1 token to the tray, the rest in hand + redeploy."""
    env = _env(211)
    ghouls = _occupy(env, 2, RaceId.GHOULS, 20, 4, in_decline=True)
    _occupy(env, 2, RaceId.GHOULS, 21, 2, in_decline=True)
    rip = _setup_turn(env, 0, RaceId.RATMEN, None, hand=8, held={14: 1})
    before = env.tray.available(RaceId.GHOULS)
    env.step(region_action(_index(20)))                   # cost 2 + 4
    assert env.tray.available(RaceId.GHOULS) == before + 1, 'one token lost'
    assert ghouls.tokens_in_hand == 3 and ghouls.pending_redeploy
    env.step(A_PASS)
    while env.phase == Phase.REDEPLOY:
        env.step(region_all_action(_index(14)))
    # only region 21 left: the engine redeploys the three tokens there
    assert env.board.by_id(21).tokens == 5 and not ghouls.pending_redeploy
    _check_races_invariants(env, 'ghoul losses')


def test_declined_ghouls_with_no_region_reenter_through_a_border():
    """Tokens in hand but no region: the race survives and starts over."""
    env = _env(212)
    ghouls = RaceInPlay(RaceId.GHOULS, None, 0, in_decline=True)
    ghouls.tokens_in_hand = env.tray.take(RaceId.GHOULS, 4)
    env.players[0].declined.append(ghouls)
    _occupy(env, 0, RaceId.RATMEN, 9, 1)                  # an active race too
    env._begin_turn(0)
    assert env.phase == Phase.GHOUL_CONQUER and env.current_race is ghouls
    expected = {r.index for r in env.board.regions if r.border and not r.is_water}
    assert _conquest_targets(env) == expected, 'first conquest: border only'
    env.step(region_action(_index(25)))                   # empty border hill
    region = env.board.by_id(25)
    assert region.race == RaceId.GHOULS and region.in_decline
    assert region.tokens == 2 and ghouls.tokens_in_hand == 2
    _check_races_invariants(env, 'ghoul re-entry')


def test_ghoul_regions_score_one_coin_each():
    """No bonus: a declined Ghoul region is worth 1 coin like any other."""
    env = _env(213)
    ghouls = _occupy(env, 0, RaceId.GHOULS, 25, 4, in_decline=True)
    _occupy(env, 0, RaceId.GHOULS, 26, 2, in_decline=True)
    _occupy(env, 0, RaceId.GHOULS, 27, 1, in_decline=True)
    assert env.players[0].active is None
    env._score(env.players[0])
    assert env.players[0].coins == 5 + 3, '3 regions, tokens do not count'
    assert env._score_bonus(ghouls) == 0
    _check_races_invariants(env, 'ghoul scoring')


# --------------------------------------------------------------------------- #
# Giants
# --------------------------------------------------------------------------- #

def test_giants_pay_one_less_next_to_their_mountains():
    """-1 when the target touches a Mountain region the Giants occupy."""
    env = _env(214)
    # 26 is a Mountain, 19 a Farmland; both border regions, both adjacent to 25
    rip = _setup_turn(env, 0, RaceId.GIANTS, None, hand=8, held={26: 1, 19: 1})
    board = env.board
    assert env._conquest_cost(rip, board.by_id(27)) == 2, 'mountain 3 - 1'
    assert env._conquest_cost(rip, board.by_id(25)) == 1, 'empty hill 2 - 1, min 1'
    assert env._conquest_cost(rip, board.by_id(20)) == 2, 'lost tribe 3 - 1'
    assert env._conquest_cost(rip, board.by_id(13)) == 3, (
        'region 13 only touches their Farmland, no discount')
    assert env._conquest_cost(rip, board.by_id(14)) == 3, 'lost tribe, no mountain'
    # the discount really is applied when conquering
    env.step(region_action(_index(27)))
    assert board.by_id(27).tokens == 2 and rip.tokens_in_hand == 6
    # ... and it is gone once the race declines
    rip.in_decline = True
    assert GiantsHooks().conquest_cost_modifier(env, rip, board.by_id(25), 2) == 2
    rip.in_decline = False
    _check_races_invariants(env, 'giants')


# --------------------------------------------------------------------------- #
# Halflings
# --------------------------------------------------------------------------- #

def test_halflings_enter_anywhere_and_dig_two_holes():
    """No border rule, and the first 2 conquests become immune regions."""
    env = _env(215)
    rip = _setup_turn(env, 0, RaceId.HALFLINGS, None, hand=8)
    assert env.players[0].must_first_conquest
    assert env._rip_flag('ignores_border_rule', rip)
    assert _conquest_targets(env) == {r.index for r in env.board.regions
                                      if not r.is_water}, 'anywhere but the water'
    env.step(region_action(_index(9)))            # interior swamp, cost 2
    assert env.board.by_id(9).hole and rip.holes_placed == 1
    assert env.tray.available_marker('hole') == 1
    env.step(region_action(_index(10)))           # lost tribe, cost 3
    assert env.board.by_id(10).hole and rip.holes_placed == 2
    assert env.tray.available_marker('hole') == 0
    env.step(region_action(_index(14)))           # lost tribe, cost 3
    assert not env.board.by_id(14).hole, 'only the first two regions get a hole'
    assert rip.holes_placed == 2 and rip.tokens_in_hand == 0
    # the holes make the regions immune to conquests
    enemy = _occupy(env, 1, RaceId.RATMEN, 3, 5)
    assert not env._can_attack(enemy, env.board.by_id(9))
    assert not env._can_attack(enemy, env.board.by_id(10))
    assert env._can_attack(enemy, env.board.by_id(4)), 'a normal neighbour'
    # ... and to a Sorcerer substitution
    sorcerers = _occupy(env, 2, RaceId.SORCERERS, 11, 1)
    env.board.by_id(10).tokens = 1                # a lone Halfling token
    env.tray.put(RaceId.HALFLINGS, 2)
    assert env._sorcerer_targets(sorcerers) == set(), 'the hole protects it'
    env.board.by_id(10).hole = False
    assert env._sorcerer_targets(sorcerers) == {10}, 'without the hole it works'
    env.board.by_id(10).hole = True
    _check_races_invariants(env, 'halfling holes')


def test_halfling_holes_are_removed_on_decline():
    """Decline: the holes go back to the supply, the regions become normal."""
    env = _env(216)
    rip = _setup_turn(env, 0, RaceId.HALFLINGS, None, hand=0, held={9: 1, 10: 1})
    for region_id in (9, 10):
        env.board.by_id(region_id).hole = True
        assert env.tray.take_marker('hole')
    rip.holes_placed = 2
    env.step(A_DECLINE)
    assert not env.board.by_id(9).hole and not env.board.by_id(10).hole
    assert env.tray.available_marker('hole') == MARKER_SUPPLY['hole']
    assert env.board.by_id(9).in_decline and env.board.by_id(9).tokens == 1
    assert rip.holes_placed == 2, 'the race never digs again'
    _check_races_invariants(env, 'halfling decline')


def test_halfling_hole_is_removed_when_the_region_is_abandoned():
    """Abandoning a hole region returns the marker to the supply."""
    env = _env(217)
    rip = _setup_turn(env, 0, RaceId.HALFLINGS, None, hand=0, held={9: 1, 10: 1})
    env.board.by_id(9).hole = True
    assert env.tray.take_marker('hole')
    rip.holes_placed = 1
    env.step(region_action(_index(9)))                    # abandon region 9
    assert env.board.by_id(9).owner is None
    assert not env.board.by_id(9).hole
    assert env.tray.available_marker('hole') == MARKER_SUPPLY['hole']
    assert rip.tokens_in_hand == 1
    _check_races_invariants(env, 'halfling abandon')


# --------------------------------------------------------------------------- #
# Humans
# --------------------------------------------------------------------------- #

def test_humans_score_their_farmlands():
    """+1 coin per Farmland region, while active only."""
    env = _env(218)
    # farmlands are regions 4, 8, 12, 19 and 23
    rip = _setup_turn(env, 0, RaceId.HUMANS, None, hand=0, held={4: 1, 19: 1, 3: 1})
    assert env._score_bonus(rip) == 2, 'regions 4 and 19 are farmlands'
    env.step(A_PASS)
    assert env.players[0].coins == 5 + 3 + 2, '3 regions + 2 farmlands'
    assert not env._rip_flag('bonus_in_decline', rip)
    _check_races_invariants(env, 'humans')

    env = _env(219)
    rip = _setup_turn(env, 0, RaceId.HUMANS, None, hand=0, held={4: 1, 19: 1})
    env.step(A_DECLINE)
    assert env.players[0].coins == 5 + 2, 'no farmland bonus once declined'
    assert HumansHooks().score_bonus(env, rip) == 0


# --------------------------------------------------------------------------- #
# Orcs
# --------------------------------------------------------------------------- #

def test_orcs_score_their_non_empty_conquests():
    """+1 coin per non-empty region conquered this turn (empty ones do not)."""
    env = _env(220)
    rip = _setup_turn(env, 0, RaceId.ORCS, None, hand=9, held={9: 1})
    env.step(region_action(_index(10)))          # hill + lost tribe: non-empty
    env.step(region_action(_index(14)))          # swamp + lost tribe: non-empty
    env.step(region_action(_index(3)))           # empty mountain
    assert len(rip.conquests_this_turn) == 3 and rip.nonempty_conquests == 2
    assert env._score_bonus(rip) == 2
    env.step(A_PASS)
    assert env.phase == Phase.REDEPLOY, 'the conquering tokens may move'
    env.step(region_all_action(_index(10)))
    assert env.players[0].coins == 5 + 4 + 2, '4 regions + 2 non-empty conquests'
    assert not env._rip_flag('bonus_in_decline', rip)
    _check_races_invariants(env, 'orcs')


def test_orcs_get_nothing_for_a_sorcerer_substitution():
    """A substitution is not a conquest, so it feeds no Orc coin."""
    env = _env(221)
    rip = _setup_turn(env, 0, RaceId.ORCS, None, hand=0, held={14: 1})
    _occupy(env, 1, RaceId.TRITONS, 10, 1)
    # the Orcs have no substitution of their own; check the engine's counters
    # through the Sorcerers' hook on the same state
    sorcerers = _occupy(env, 2, RaceId.SORCERERS, 11, 1)
    assert env._sorcerer_targets(sorcerers) == {10}
    env._resolve_sorcerer(sorcerers, env.board.by_id(10))
    assert sorcerers.conquests_this_turn == [] and sorcerers.nonempty_conquests == 0
    assert env._score_bonus(rip) == 0
    _check_races_invariants(env, 'orcs vs substitution')


# --------------------------------------------------------------------------- #
# Ratmen
# --------------------------------------------------------------------------- #

def test_ratmen_have_no_ability_at_all():
    """Every Ratmen hook answers exactly like the vanilla `RaceHooks`."""
    env = _env(222)
    rip = _setup_turn(env, 0, RaceId.RATMEN, None, hand=5, held={9: 1})
    hooks = get_race_hooks(RaceId.RATMEN)
    vanilla = RaceHooks()
    region = env.board.by_id(10)
    assert hooks.conquest_cost_modifier(env, rip, region, 7) == 7
    assert hooks.redeployable_tokens(env, rip) == rip.tokens_in_hand == 5
    assert hooks.defender_loss(env, rip, region) == 1
    assert hooks.score_bonus(env, rip) == 0
    assert hooks.sorcerer_targets(env, rip) == set()
    assert hooks.can_target(env, rip, region) is None
    assert hooks.is_adjacent(env, rip, region) is None
    assert not hooks.dragon_available(env, rip)
    for flag in ('ignores_border_rule', 'die_before_each_conquest',
                 'keeps_tokens_in_decline', 'conquers_in_decline',
                 'losses_like_active', 'is_spirit_power', 'bonus_in_decline'):
        assert getattr(hooks, flag)(rip) is getattr(vanilla, flag)(rip) is False
    # the side-effect hooks are no-ops: nothing on the board moves
    before = repr(env.board.regions) + repr(rip.__dict__)
    hooks.on_conquered(env, rip, region, {'was_empty': True})
    hooks.on_region_lost(env, rip, region)
    hooks.on_abandon(env, rip, region)
    hooks.on_redeploy_start(env, rip)
    hooks.on_turn_start(env, rip)
    hooks.on_turn_end(env, rip)
    hooks.on_decline(env, rip)
    assert repr(env.board.regions) + repr(rip.__dict__) == before
    _check_races_invariants(env, 'ratmen')


# --------------------------------------------------------------------------- #
# Skeletons
# --------------------------------------------------------------------------- #

def test_skeletons_multiply_on_their_conquests():
    """+1 token from the tray per 2 non-empty regions conquered this turn."""
    env = _env(223)
    rip = _setup_turn(env, 0, RaceId.SKELETONS, None, hand=9, held={9: 1})
    env.step(region_action(_index(10)))              # lost tribe
    env.step(region_action(_index(14)))              # lost tribe
    env.step(region_action(_index(20)))              # lost tribe
    assert rip.nonempty_conquests == 3 and rip.tokens_in_hand == 0
    tray_before = env.tray.available(RaceId.SKELETONS)
    on_board = sum(r.tokens for r in env.board.regions_of_race(rip))
    env.step(A_PASS)
    # every token but one per region is lifted, plus 3 // 2 = 1 new token
    lifted = on_board - len(env.board.regions_of_race(rip))
    assert rip.tokens_in_hand == lifted + 1, '3 // 2 = 1 new token'
    assert env.tray.available(RaceId.SKELETONS) == tray_before - 1
    assert env.phase == Phase.REDEPLOY, 'the new token has to be deployed'
    assert any('multiplies' in line for line in env.turn_events)
    env.step(region_all_action(_index(9)))
    assert env.board.by_id(9).tokens == 2 + lifted and rip.tokens_in_hand == 0
    assert sum(r.tokens for r in env.board.regions_of_race(rip)) == on_board + 1
    _check_races_invariants(env, 'skeletons growth')


def test_skeletons_growth_is_limited_by_the_tray():
    """The tray is the hard limit (rules: "while the tray has some")."""
    total = RACES[RaceId.SKELETONS].total_tokens
    # one token left in the tray, two wanted
    env = _env(224)
    rip = _setup_turn(env, 0, RaceId.SKELETONS, None, hand=total - 2, held={9: 1})
    assert env.tray.available(RaceId.SKELETONS) == 1
    rip.nonempty_conquests = 5
    env.step(A_PASS)
    assert rip.tokens_in_hand == total - 1
    assert env.tray.available(RaceId.SKELETONS) == 0
    assert any('1 short, tray empty' in line for line in env.turn_events)
    _check_races_invariants(env, 'skeletons partial growth')

    # nothing left at all
    env = _env(225)
    rip = _setup_turn(env, 0, RaceId.SKELETONS, None, hand=total - 1, held={9: 1})
    assert env.tray.available(RaceId.SKELETONS) == 0
    rip.nonempty_conquests = 4
    env.step(A_PASS)
    assert rip.tokens_in_hand == total - 1, 'no token to take'
    assert any('2 short, tray empty' in line for line in env.turn_events)
    _check_races_invariants(env, 'skeletons no growth')


def test_skeletons_do_not_grow_without_non_empty_conquests():
    """An empty region feeds nobody."""
    env = _env(226)
    rip = _setup_turn(env, 0, RaceId.SKELETONS, None, hand=6, held={19: 1})
    env.step(region_action(_index(25)))              # empty hill
    assert rip.nonempty_conquests == 0
    tray_before = env.tray.available(RaceId.SKELETONS)
    env.step(A_PASS)
    assert env.tray.available(RaceId.SKELETONS) == tray_before
    assert not any('multiplies' in line for line in env.turn_events)


# --------------------------------------------------------------------------- #
# Sorcerers
# --------------------------------------------------------------------------- #

def test_sorcerer_targets_and_every_exclusion():
    """`sorcerer_targets` implements the whole legality check of the rules."""
    env = _env(227)
    # the Sorcerers hold region 14, whose neighbours are 9, 10, 13, 15, 19, 20
    rip = _setup_turn(env, 0, RaceId.SORCERERS, None, hand=0, held={14: 1})
    _occupy(env, 1, RaceId.ELVES, 10, 1)             # lone active token: legal
    _occupy(env, 1, RaceId.ELVES, 9, 2)              # two tokens: illegal
    _occupy(env, 2, RaceId.DWARVES, 13, 1, in_decline=True)   # declined: illegal
    _occupy(env, 0, RaceId.TROLLS, 19, 1, in_decline=True)    # his own: illegal
    assert env.board.by_id(20).lost_tribe, 'a Lost Tribe is not a race token'
    assert env.board.by_id(15).is_water, 'the lake cannot be reached'
    assert env._sorcerer_targets(rip) == {10}
    region = env.board.by_id(10)

    # an encampment protects the lone token
    region.encampments = 1
    assert env.tray.take_marker('encampment')
    assert env._sorcerer_targets(rip) == set()
    region.encampments = 0
    env.tray.put_marker('encampment')

    # so does an immune marker (hole / hero / dragon)
    region.hero = True
    assert env.tray.take_marker('hero')
    assert env._sorcerer_targets(rip) == set()
    region.hero = False
    env.tray.put_marker('hero')

    # a Diplomat alliance with the Sorcerers' owner protects the active race
    env.players[1].ally = 0
    assert env._sorcerer_targets(rip) == set()
    env.players[1].ally = None

    # once per opponent per turn
    rip.sorcerer_used_on.add(1)
    assert env._sorcerer_targets(rip) == set()
    rip.sorcerer_used_on.clear()

    # no Sorcerer token left in the tray
    left = env.tray.take(RaceId.SORCERERS, env.tray.available(RaceId.SORCERERS))
    assert env._sorcerer_targets(rip) == set()
    env.tray.put(RaceId.SORCERERS, left)

    # a lone enemy token that is not adjacent is out of reach
    _occupy(env, 1, RaceId.ELVES, 4, 1)
    assert env._sorcerer_targets(rip) == {10}

    # ... and with no region at all there is nothing adjacent
    assert env._sorcerer_targets(rip) == {10}
    held = env.board.by_id(14)
    env.tray.put(RaceId.SORCERERS, held.tokens)
    held.clear_race()
    assert env._sorcerer_targets(rip) == set()
    _check_races_invariants(env, 'sorcerer targets')


def test_sorcerer_substitution_mechanics_and_markers():
    """The substitution takes a tray token, clears the markers, attacks once."""
    env = _env(228)
    rip = _setup_turn(env, 0, RaceId.SORCERERS, None, hand=0, held={14: 1})
    victim = _occupy(env, 1, RaceId.TRITONS, 10, 1)
    _occupy(env, 1, RaceId.TRITONS, 11, 1)
    region = env.board.by_id(10)
    region.fortress = True
    assert env.tray.take_marker('fortress')
    region.lair = True              # artificial, to check the marker clearing
    tritons_before = env.tray.available(RaceId.TRITONS)
    sorcerers_before = env.tray.available(RaceId.SORCERERS)
    assert sorcerer_action(_index(10)) in _legal(env)
    env.step(sorcerer_action(_index(10)))
    assert region.owner == 0 and region.race == RaceId.SORCERERS
    assert region.tokens == 1 and not region.in_decline
    assert env.tray.available(RaceId.TRITONS) == tritons_before + 1
    assert env.tray.available(RaceId.SORCERERS) == sorcerers_before - 1
    assert not region.fortress and not region.lair, 'both markers are removed'
    assert env.tray.available_marker('fortress') == MARKER_SUPPLY['fortress']
    assert victim.tokens_in_hand == 0 and not victim.pending_redeploy
    assert rip.sorcerer_used_on == {1} and rip.attacked_players == {1}
    assert rip.conquests_this_turn == [], 'a substitution is not a conquest'
    assert env._sorcerer_targets(rip) == set(), 'once per opponent per turn'
    # the region now feeds the Sorcerers' own adjacency
    assert env._can_attack(rip, env.board.by_id(4))
    _check_races_invariants(env, 'sorcerer substitution')


def test_sorcerer_substitution_can_wipe_the_victim_out():
    """Losing his last token re-arms the victim's first-conquest rule."""
    env = _env(229)
    rip = _setup_turn(env, 0, RaceId.SORCERERS, None, hand=0, held={14: 1})
    victim = _occupy(env, 1, RaceId.HUMANS, 10, 1)
    env.step(sorcerer_action(_index(10)))
    assert env.players[1].must_first_conquest
    assert victim.tokens_in_hand == 0
    assert env.board.regions_of_race(victim) == []
    _check_races_invariants(env, 'sorcerer wipe-out')


# --------------------------------------------------------------------------- #
# Tritons
# --------------------------------------------------------------------------- #

def test_tritons_pay_one_less_on_coastal_regions():
    """-1 on every Coastal region (next to a sea or the lake), min 1."""
    env = _env(230)
    rip = _setup_turn(env, 0, RaceId.TRITONS, None, hand=8, held={9: 1})
    board = env.board
    assert not board.is_coastal(board.by_id(9))
    assert env._conquest_cost(rip, board.by_id(2)) == 1, 'coastal, empty: 2 - 1'
    assert env._conquest_cost(rip, board.by_id(10)) == 2, 'coastal + lost tribe'
    assert env._conquest_cost(rip, board.by_id(13)) == 2, 'coastal mountain'
    assert env._conquest_cost(rip, board.by_id(3)) == 3, 'inland mountain'
    assert env._conquest_cost(rip, board.by_id(4)) == 2, 'inland farmland'
    env.step(region_action(_index(2)))
    assert board.by_id(2).tokens == 1 and rip.tokens_in_hand == 7
    rip.in_decline = True
    assert TritonsHooks().conquest_cost_modifier(env, rip, board.by_id(10), 3) == 3
    rip.in_decline = False
    _check_races_invariants(env, 'tritons')


# --------------------------------------------------------------------------- #
# Trolls
# --------------------------------------------------------------------------- #

def test_trolls_raise_a_lair_in_every_conquered_region():
    """A Lair is placed on conquest and adds +1 to the defence."""
    env = _env(231)
    rip = _setup_turn(env, 0, RaceId.TROLLS, None, hand=6)
    env.step(region_action(_index(25)))               # empty border hill, cost 2
    lair_region = env.board.by_id(25)
    assert lair_region.lair and lair_region.tokens == 2
    assert lair_region.defence == 1 + 2, 'lair + 2 tokens'
    assert any("Lair" in line for line in env.turn_events)
    env.step(region_action(_index(19)))               # farmland, cost 2
    assert env.board.by_id(19).lair
    # an enemy pays the extra token
    enemy = _occupy(env, 1, RaceId.RATMEN, 26, 3)
    assert env._conquest_cost(enemy, lair_region) == 2 + 1 + 2
    _check_races_invariants(env, 'troll lairs')


def test_troll_lairs_survive_decline_and_the_banner_leaving_the_map():
    """User ruling (plan section 5 item 7): the Lair stays on the board."""
    env = _env(232)
    rip = _setup_turn(env, 0, RaceId.TROLLS, None, hand=2, held={25: 1, 19: 1})
    env.board.by_id(25).lair = True
    env.board.by_id(19).lair = True
    env.step(A_DECLINE)
    assert env.board.by_id(25).lair and env.board.by_id(19).lair
    assert env.board.by_id(25).in_decline
    declined = env.players[0].declined[0]
    assert declined is rip
    env._remove_declined_race(declined)
    assert env.board.by_id(25).owner is None and env.board.by_id(25).lair
    assert env.board.by_id(19).lair, 'the Lair defends the empty region'
    assert env._conquest_cost(_occupy(env, 1, RaceId.RATMEN, 26, 1),
                              env.board.by_id(25)) == 2 + 1
    _check_races_invariants(env, 'troll decline')


def test_troll_lair_goes_when_the_region_is_abandoned_or_conquered():
    """Abandon and enemy conquest both take the Lair away."""
    env = _env(233)
    rip = _setup_turn(env, 0, RaceId.TROLLS, None, hand=0, held={25: 1, 19: 1})
    env.board.by_id(25).lair = True
    env.board.by_id(19).lair = True
    env.step(region_action(_index(25)))               # abandon region 25
    assert env.board.by_id(25).owner is None and not env.board.by_id(25).lair
    assert env.board.by_id(19).lair

    env = _env(234)
    trolls = _occupy(env, 2, RaceId.TROLLS, 25, 1)
    env.board.by_id(25).lair = True
    rip = _setup_turn(env, 0, RaceId.RATMEN, None, hand=8, held={19: 1})
    assert env._conquest_cost(rip, env.board.by_id(25)) == 2 + 1 + 1
    env.step(region_action(_index(25)))
    assert env.board.by_id(25).race == RaceId.RATMEN
    assert not env.board.by_id(25).lair, 'an enemy conquest removes it'
    _check_races_invariants(env, 'troll lair lost')


# --------------------------------------------------------------------------- #
# Wizards
# --------------------------------------------------------------------------- #

def test_wizards_score_their_magic_sources():
    """+1 coin per Magic Source region, while active only."""
    env = _env(235)
    # magic sources are regions 8, 10, 17, 19 and 22
    rip = _setup_turn(env, 0, RaceId.WIZARDS, None, hand=0, held={10: 1, 19: 1, 9: 1})
    assert env._score_bonus(rip) == 2
    env.step(A_PASS)
    assert env.players[0].coins == 5 + 3 + 2
    _check_races_invariants(env, 'wizards')

    env = _env(236)
    _setup_turn(env, 0, RaceId.WIZARDS, None, hand=0, held={10: 1, 19: 1})
    env.step(A_DECLINE)
    assert env.players[0].coins == 5 + 2, 'no bonus once declined'


# --------------------------------------------------------------------------- #
# Fuzz
# --------------------------------------------------------------------------- #

def test_fuzz_random_games_with_the_real_races():
    """`FUZZ_GAMES` random games with the 14 races registered.

    Re-checks every T3 invariant (plan 3.10) plus the race-specific ones at
    **every** step, and asserts that random play really does exercise the race
    mechanics (every banner picked, holes dug, Lairs raised, Skeletons grown,
    substitutions made, Ghoul phases played).
    """
    env = SmallWorldEnv(3)
    counters: Counter = Counter()
    races_picked: set = set()
    phases_seen: set = set()
    steps_per_game = []
    coins = []
    started = time.time()

    for seed in range(FUZZ_GAMES):
        obs, info = env.reset(seed=1000 + seed)
        assert env.observation_space.contains(obs)
        assert info == {'next_step_no_action': False}
        _check_races_invariants(env, f'seed {seed} reset')
        rng = np.random.default_rng(seed)
        steps = 0
        while not env.done:
            assert steps < STEP_BUDGET, f'seed {seed}: over {STEP_BUDGET} steps'
            phases_seen.add(env.phase)
            if env.phase == Phase.TURN_PAUSE:            # no-action step
                assert env.current_player == -1 and not env.action_masks().any()
                obs, rewards, terminated, truncated, info = env.step(-1)
                steps += 1
                assert rewards == [0.0] * env.n_players
                assert info == {'next_step_no_action': env.phase == Phase.TURN_PAUSE}
                continue
            legal = np.flatnonzero(env.action_masks())
            assert legal.size, f'seed {seed}: no legal action in {env.phase.name}'
            action = int(rng.choice(legal))
            kind, arg = action_kind(action)
            counters[kind] += 1
            if kind == 'combo':
                races_picked.add(env.combo_column.visible[arg].race)
            assert env.describe_action(action), env.phase.name
            obs, rewards, terminated, truncated, info = env.step(action)
            steps += 1
            ctx = f'seed {seed} step {steps} ({env.phase.name})'
            assert truncated is False and info == {
                'next_step_no_action': env.phase == Phase.TURN_PAUSE}, ctx
            assert len(rewards) == env.n_players, ctx
            assert abs(sum(rewards)) < 1e-9, f'{ctx}: {rewards} not zero-sum'
            assert terminated == env.done, ctx
            _check_races_invariants(env, ctx)

        # -- terminal state ------------------------------------------------- #
        assert env.turns_taken == turns_for(3), f'seed {seed}: {env.turns_taken}'
        best = max(p.coins for p in env.players)
        assert env.players[env.winner_player].coins == best, f'seed {seed}'
        assert abs(sum(env.terminal_rewards)) < 1e-9
        assert not env.action_masks().any(), f'seed {seed}: mask after the end'
        # every pending redeployment is settled, so the Amazons hold at most
        # their 4 attack-only tokens as long as they own a region
        for player in env.players:
            for rip in player.all_races():
                if (rip.race == RaceId.AMAZONS and not rip.in_decline
                        and env.board.regions_of_race(rip)):
                    assert rip.tokens_in_hand <= 4, f'seed {seed}: {rip!r}'
        for line in env.event_log:
            if 'Hole-in-the-Ground' in line:
                counters['holes_dug'] += 1
            elif "Troll's Lair" in line:
                counters['lairs_raised'] += 1
            elif 'multiplies' in line:
                counters['skeletons_grown'] += 1
            elif '[sorcerer]' in line:
                counters['substitutions'] += 1
            elif 'stay in hand' in line:
                counters['tokens_kept_in_hand'] += 1
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
        phases=sorted(p.name for p in phases_seen),
        seconds=round(time.time() - started, 1),
        **{k: int(v) for k, v in sorted(counters.items())},
    )

    assert races_picked == set(RaceId), sorted(
        r.name for r in set(RaceId) - races_picked)
    assert Phase.GHOUL_CONQUER in phases_seen and Phase.GHOUL_REDEPLOY in phases_seen
    assert {'combo', 'decline', 'pass', 'region', 'region_all',
            'sorcerer'} <= set(counters), sorted(counters)
    for key in ('holes_dug', 'lairs_raised', 'skeletons_grown', 'substitutions'):
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
