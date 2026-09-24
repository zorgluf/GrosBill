"""Tests for the Small World engine (`envs/smallw.py`, task T3).

Plain `assert`-based functions, usable either with pytest
(`pytest environments/smallw/tests/test_engine.py`) or through the bundled
runners:

    cd app
    python -m environments.smallw.tests.run_all        # every test module
    python -m environments.smallw.tests.test_engine    # this module only

Three families:

* **fuzz** (`test_fuzz_random_games`): 200 random games, every invariant of
  plan 3.10 checked after every single step;
* **determinism**: same seed + same actions give the same observations and the
  same log, and a `deepcopy` is independent of its original;
* **scenarios**: hand-built states for the first-conquest rule, the conquest
  costs, the final attempt, the decline, the victim redeployment, the abandon,
  the combo prices, the end-of-game ranking, and for the *structural* pieces
  the engine already wires for T4 / T5 (Sorcerer, Dragon, Berserk die, Ghoul
  phases, end-of-turn sub-phases) which are exercised through temporarily
  registered hooks.
"""

from __future__ import annotations

import contextlib
import copy
import time
import traceback

import numpy as np

from ..envs.classes import (
    MARKER_SUPPLY,
    RACES,
    PowerId,
    RaceId,
    RaceInPlay,
    turns_for,
)
from ..envs.hooks import (
    POWER_HOOKS,
    RACE_HOOKS,
    PowerHooks,
    RaceHooks,
    register_power_hooks,
    register_race_hooks,
)
from ..envs.smallw import (
    A_ALLY,
    A_COMBO,
    A_DECLINE,
    A_DRAGON,
    A_PASS,
    A_REGION,
    A_REGION_ALL,
    A_SORCERER,
    N_ACTIONS,
    Phase,
    SmallWorldEnv,
    action_kind,
    ally_action,
    dragon_action,
    region_action,
    region_all_action,
    sorcerer_action,
)

#: Games played by the fuzz test.
FUZZ_GAMES = 200
#: Hard step budget per game (a real game needs ~150 steps).
STEP_BUDGET = 3000

#: Statistics of the last fuzz run, printed by `run_all()`.
FUZZ_STATS: dict = {}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _env(seed: int = 0, n_players: int = 3) -> SmallWorldEnv:
    """A freshly reset environment."""
    env = SmallWorldEnv(n_players, pause_between_turns=False)
    env.reset(seed=seed)
    return env


def _legal(env: SmallWorldEnv) -> list[int]:
    """Legal actions as a plain list of ints."""
    return [int(a) for a in np.flatnonzero(env.action_masks())]


def _force_dice(env: SmallWorldEnv, values: list[int]) -> None:
    """Replace the pre-rolled die sequence (tests poke `Dice` internals)."""
    env.dice._values = list(values) + [0] * 300
    env.dice._pos = 0


def _setup_turn(env: SmallWorldEnv, seat: int, race: RaceId, power: PowerId | None,
                hand: int, held: dict[int, int] | None = None,
                turns_played: int = 1) -> RaceInPlay:
    """Put `seat` in the CONQUER phase with a hand-built active race.

    `held` maps region ids to a token count. Tokens always come from the tray,
    so the conservation invariant keeps holding.
    """
    player = env.players[seat]
    rip = RaceInPlay(race, power, seat)
    rip.tokens_in_hand = env.tray.take(race, hand)
    assert rip.tokens_in_hand == hand
    player.active = rip
    for region_id, count in (held or {}).items():
        region = env.board.by_id(region_id)
        region.lost_tribe = False
        region.owner = seat
        region.race = race
        region.in_decline = False
        region.tokens = env.tray.take(race, count)
        assert region.tokens == count
    player.must_first_conquest = not held
    env._turn_seat = seat
    env.current_player = seat
    env.turn_events = []
    env._acting_this_turn = [rip]
    for _ in range(max(1, turns_played)):
        rip.new_turn()
    env._enter_conquer(rip, ghoul=False)
    return rip


def _occupy(env: SmallWorldEnv, seat: int, race: RaceId, region_id: int,
            tokens: int, in_decline: bool = False,
            power: PowerId | None = None) -> RaceInPlay:
    """Give `seat` a race occupying one region (creating the race if needed)."""
    player = env.players[seat]
    rip = player.race_by(race, in_decline=in_decline)
    if rip is None:
        rip = RaceInPlay(race, power, seat, in_decline=in_decline)
        if in_decline:
            player.declined.append(rip)
        else:
            player.active = rip
    region = env.board.by_id(region_id)
    region.lost_tribe = False
    region.owner = seat
    region.race = race
    region.in_decline = in_decline
    region.tokens = env.tray.take(race, tokens)
    assert region.tokens == tokens
    return rip


@contextlib.contextmanager
def _with_hooks(races: dict | None = None, powers: dict | None = None):
    """Temporarily register race / power hooks, restoring the registries."""
    saved_races, saved_powers = dict(RACE_HOOKS), dict(POWER_HOOKS)
    try:
        for race_id, hooks in (races or {}).items():
            register_race_hooks(race_id, hooks)
        for power_id, hooks in (powers or {}).items():
            register_power_hooks(power_id, hooks)
        yield
    finally:
        RACE_HOOKS.clear()
        RACE_HOOKS.update(saved_races)
        POWER_HOOKS.clear()
        POWER_HOOKS.update(saved_powers)


def _markers_on_board(env: SmallWorldEnv) -> dict[str, int]:
    """Count the markers currently on the map, per kind."""
    regions = env.board.regions
    return {
        'fortress': sum(1 for r in regions if r.fortress),
        'encampment': sum(r.encampments for r in regions),
        'hero': sum(1 for r in regions if r.hero),
        'hole': sum(1 for r in regions if r.hole),
        'dragon': sum(1 for r in regions if r.dragon),
    }


def _check_invariants(env: SmallWorldEnv, ctx: str,
                      vanilla_declined: bool = True) -> None:
    """Every invariant of plan 3.10, checked after one step.

    `vanilla_declined` asserts that a declined race holds exactly one token per
    region; the scenario tests that register Ghoul-like hooks switch it off.
    A race whose registered hooks say `keeps_tokens_in_decline` (the Ghouls,
    T4) is exempt even when the flag is on.
    """
    n = env.n_players

    # -- phase / player ----------------------------------------------------- #
    if env.done:
        assert env.phase == Phase.DONE, f'{ctx}: done but phase {env.phase.name}'
        assert env.winner_player is not None, f'{ctx}: done without a winner'
    else:
        assert env.phase != Phase.DONE, f'{ctx}: phase DONE but not done'
        if env.phase == Phase.TURN_PAUSE:
            # between two turns: nobody to move, only the no-action step
            assert env.current_player == -1, f'{ctx}: pause with a player to move'
            assert not env.action_masks().any(), f'{ctx}: legal action in a pause'
            assert env.current_race is None, f'{ctx}: acting race in a pause'
        else:
            assert 0 <= env.current_player < n, f'{ctx}: current_player {env.current_player}'
            assert env.action_masks().any(), f'{ctx}: all-False mask in {env.phase.name}'
    assert 0 <= env._turn_seat < n, f'{ctx}: turn seat {env._turn_seat}'

    # -- token conservation, per race --------------------------------------- #
    for race_id in RaceId:
        on_board = sum(r.tokens for r in env.board.regions if r.race == race_id)
        holders = [rip for p in env.players for rip in p.all_races()
                   if rip.race == race_id]
        assert len(holders) <= 1, f'{ctx}: {race_id.name} held by {len(holders)} races'
        in_hand = sum(rip.tokens_in_hand for rip in holders)
        tray = env.tray.available(race_id)
        total = RACES[race_id].total_tokens
        assert 0 <= tray <= total, f'{ctx}: tray holds {tray}/{total} {race_id.name}'
        assert on_board + in_hand + tray == total, (
            f'{ctx}: {race_id.name} board {on_board} + hand {in_hand} + tray '
            f'{tray} != {total}')
        assert in_hand >= 0

    # -- markers ------------------------------------------------------------ #
    on_board = _markers_on_board(env)
    for kind, supply in MARKER_SUPPLY.items():
        available = env.tray.available_marker(kind)
        assert 0 <= available <= supply, f'{ctx}: {available} {kind}(s) in the tray'
        assert available + on_board[kind] == supply, (
            f'{ctx}: {on_board[kind]} {kind}(s) on the board + {available} in the '
            f'tray != {supply}')

    # -- regions ------------------------------------------------------------ #
    for region in env.board.regions:
        if region.owner is None:
            assert region.tokens == 0 and region.race is None, (
                f'{ctx}: region {region.id} has no owner but {region.tokens} tokens')
            assert not region.in_decline
        else:
            assert 0 <= region.owner < n, f'{ctx}: region {region.id} owner'
            assert region.race is not None and region.tokens > 0, (
                f'{ctx}: region {region.id} owned but empty')
            assert not region.lost_tribe, (
                f'{ctx}: region {region.id} holds tokens and a Lost Tribe')
            rip = env.players[region.owner].race_by(region.race,
                                                    in_decline=region.in_decline)
            assert rip is not None, (
                f'{ctx}: region {region.id} holds a race nobody owns')
            if (region.in_decline and vanilla_declined
                    and not env._rip_flag('keeps_tokens_in_decline', rip)):
                # vanilla: a declined race keeps exactly one token per region
                # (the Ghouls keep all of theirs, so they are exempt)
                assert region.tokens == 1, (
                    f'{ctx}: declined region {region.id} holds {region.tokens} tokens')
        assert region.encampments >= 0

    # -- players ------------------------------------------------------------ #
    for player in env.players:
        assert player.coins >= 0, f'{ctx}: {player.name} has {player.coins} coins'
        assert len([r for r in player.declined if not r.is_spirit]) <= 1, (
            f'{ctx}: {player.name} has two non-Spirit declined races')
        for rip in player.all_races():
            assert rip.owner == player.seat
            assert rip.in_decline == (rip is not player.active)
            if rip.in_decline:
                assert rip.power is None, f'{ctx}: declined race keeps its power'

    # -- observation -------------------------------------------------------- #
    obs = env.observation
    assert env.observation_space.contains(obs), (
        f'{ctx}: observation outside the space in {env.phase.name}')


def _allowed_terminal_values(env: SmallWorldEnv) -> set[float]:
    """Rank rewards plus every average of a contiguous group of ranks."""
    table = env.rank_rewards()
    allowed = set()
    for i in range(len(table)):
        for j in range(i, len(table)):
            allowed.add(round(sum(table[i:j + 1]) / (j + 1 - i), 9))
    return allowed


# --------------------------------------------------------------------------- #
# Action space and reset
# --------------------------------------------------------------------------- #

def test_action_layout():
    """The action space layout of plan 3.4 and its helpers."""
    assert (A_COMBO, A_DECLINE, A_PASS, A_REGION, A_REGION_ALL, A_SORCERER,
            A_DRAGON, A_ALLY, N_ACTIONS) == (0, 6, 7, 8, 38, 68, 98, 128, 133)
    assert region_action(0) == 8 and region_action(29) == 37
    assert region_all_action(0) == 38 and region_all_action(29) == 67
    assert sorcerer_action(0) == 68 and dragon_action(0) == 98
    assert ally_action(0) == 128 and ally_action(4) == 132
    kinds = {
        0: ('combo', 0), 5: ('combo', 5), 6: ('decline', 0), 7: ('pass', 0),
        8: ('region', 0), 37: ('region', 29), 38: ('region_all', 0),
        67: ('region_all', 29), 68: ('sorcerer', 0), 97: ('sorcerer', 29),
        98: ('dragon', 0), 127: ('dragon', 29), 128: ('ally', 0), 132: ('ally', 4),
    }
    for action, expected in kinds.items():
        assert action_kind(action) == expected, action
    for bad in (-1, N_ACTIONS):
        try:
            action_kind(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'action_kind({bad}) should raise')
    env = _env(0)
    assert env.action_space.n == N_ACTIONS
    assert len(Phase) == 13 and Phase.DONE == 11 and Phase.TURN_PAUSE == 12


def test_reset_initial_state():
    """`reset` deals the board, the players, the column and the first turn."""
    env = _env(7)
    assert env.n_regions == 30 and env.turns_total == turns_for(3)
    assert sum(1 for r in env.board.regions if r.lost_tribe) == 10
    assert [p.coins for p in env.players] == [5, 5, 5]
    assert len(env.combo_column.visible) == 6
    assert env.phase == Phase.PICK_COMBO and env.current_player == 0
    assert env.turn == 1 and env.turns_taken == 0 and not env.done
    assert env.winner_player is None and env.terminal_rewards is None
    assert env.current_race is None
    assert _legal(env) == [A_COMBO + i for i in range(6)]   # 5 coins, 6 combos
    assert env.observation_space.contains(env.observation)
    obs, info = env.reset(seed=7)
    assert info == {'next_step_no_action': False}
    assert len(env.event_log) == 2 and env.event_log[0].startswith('---- NEW GAME')
    # the description helpers work in every phase
    assert env.legal_targets() and all(isinstance(v, str)
                                       for v in env.legal_targets().values())


def test_illegal_actions_raise():
    """An action outside the mask raises, like the other environments."""
    env = _env(1)
    for action in (A_PASS, A_DECLINE, region_action(0), -3, N_ACTIONS + 1):
        try:
            env.step(action)
        except Exception as exc:
            assert 'Illegal action' in str(exc), exc
        else:
            raise AssertionError(f'step({action}) should have raised')
    env.step(A_COMBO)
    while not env.done:
        env.step(_legal(env)[0])
    try:
        env.step(A_PASS)
    except Exception as exc:
        assert 'game is over' in str(exc)
    else:
        raise AssertionError('stepping a finished game should raise')


# --------------------------------------------------------------------------- #
# Conquest rules
# --------------------------------------------------------------------------- #

def test_first_conquest_must_be_border():
    """A fresh race may only enter through a non-water border region."""
    env = _env(2)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.MERCHANT, hand=10)
    targets = {a - A_REGION for a in _legal(env) if A_REGION <= a < A_REGION_ALL}
    expected = {r.index for r in env.board.regions if r.border and not r.is_water}
    assert targets == expected, sorted(targets ^ expected)
    assert env.players[0].must_first_conquest
    # after the first conquest, only the neighbours of region 28 are legal
    env.step(region_action(env.board.by_id(28).index))
    assert not env.players[0].must_first_conquest
    targets = {a - A_REGION for a in _legal(env) if A_REGION <= a < A_REGION_ALL}
    neighbours = {rid - 1 for rid in env.board.by_id(28).adjacent
                  if not env.board.by_id(rid).is_water}
    assert targets == neighbours, sorted(targets ^ neighbours)
    # region 28 is held and a conquest was made: it cannot be abandoned any more
    assert not env.action_masks()[region_action(env.board.by_id(28).index)]
    assert rip.first_conquest_done and rip.conquests_this_turn == [28]


def test_conquest_cost_examples():
    """The cost formula of the rules, on the four examples of the plan."""
    env = _env(3)
    rip = _setup_turn(env, 0, RaceId.RATMEN, None, hand=10, held={9: 1})
    board = env.board
    for region in board.regions:            # start from a clean map
        region.lost_tribe = False

    assert env._conquest_cost(rip, board.by_id(4)) == 2        # empty farmland
    board.by_id(4).lost_tribe = True
    assert env._conquest_cost(rip, board.by_id(4)) == 3        # + a lost tribe
    board.by_id(3).lost_tribe = True                           # region 3 = mountain
    assert env._conquest_cost(rip, board.by_id(3)) == 4        # mountain + lost tribe
    board.by_id(4).lost_tribe = False
    _occupy(env, 1, RaceId.ELVES, 4, 2)
    board.by_id(4).fortress = True
    assert env.tray.take_marker('fortress')
    assert env._conquest_cost(rip, board.by_id(4)) == 5        # 2 enemy + fortress
    _check_invariants(env, 'conquest costs')


def test_water_and_immune_regions_are_never_targets():
    """Seas, the lake and the immune markers are out of reach by default."""
    env = _env(4)
    rip = _setup_turn(env, 0, RaceId.RATMEN, None, hand=12, held={14: 1})
    board = env.board
    board.by_id(9).hole = True                # neighbour of 14, immune
    assert env.tray.take_marker('hole')
    targets = {a - A_REGION for a in _legal(env) if A_REGION <= a < A_REGION_ALL}
    assert board.by_id(15).index not in targets, 'the lake must not be conquerable'
    assert board.by_id(9).index not in targets, 'a hole makes a region immune'
    assert board.by_id(10).index in targets
    assert not env._can_attack(rip, board.by_id(15))
    assert not env._can_attack(rip, board.by_id(9))


def test_diplomat_protection_blocks_conquests():
    """An ally cannot attack the active race of his protector."""
    env = _env(5)
    rip = _setup_turn(env, 0, RaceId.RATMEN, None, hand=12, held={14: 1})
    _occupy(env, 1, RaceId.ELVES, 10, 1)                    # neighbour of 14
    _occupy(env, 2, RaceId.DWARVES, 13, 1, in_decline=True)  # neighbour of 14
    assert env._can_attack(rip, env.board.by_id(10))
    env.players[1].ally = 0                                  # player 2 protects
    assert not env._can_attack(rip, env.board.by_id(10))
    env.players[2].ally = 0                                  # declined tokens: no
    assert env._can_attack(rip, env.board.by_id(13))


def test_final_attempt_success_and_failure():
    """A region short by at most 3 tokens: the die decides, conquests end."""
    for die, success in ((2, True), (0, False)):
        env = _env(6)
        _force_dice(env, [die])
        rip = _setup_turn(env, 0, RaceId.RATMEN, None, hand=1, held={9: 1})
        target = env.board.by_id(10)
        target.lost_tribe = True                  # hill + lost tribe -> cost 3
        assert env._conquest_cost(rip, target) == 3
        options = env._conquest_options(rip)
        assert options[target.index] == (3, True), options
        env.step(region_action(target.index))
        assert rip.die == die
        if success:
            assert target.owner == 0 and target.tokens == 1 and not target.lost_tribe
            assert rip.tokens_in_hand == 0
            assert any('conquers region 10' in line for line in env.event_log)
        else:
            assert target.owner is None and target.lost_tribe
            assert rip.tokens_in_hand == 1
            assert any('fails its final conquest' in line for line in env.event_log)
        # either way the conquests are over: the turn moved on
        assert env.phase != Phase.CONQUER or env._turn_seat != 0
        _check_invariants(env, 'final attempt')


def test_no_conquest_without_a_token_in_hand():
    """With an empty hand a race can only abandon, pass or decline."""
    env = _env(8)
    rip = _setup_turn(env, 0, RaceId.RATMEN, None, hand=0, held={9: 2},
                      turns_played=2)
    legal = _legal(env)
    assert A_PASS in legal and A_DECLINE in legal
    assert legal == sorted([A_DECLINE, A_PASS, region_action(8)])
    assert env._conquest_options(rip) == {}


def test_abandon_frees_the_region_and_its_markers():
    """Abandoning gives the tokens back and returns every marker."""
    env = _env(9)
    rip = _setup_turn(env, 0, RaceId.TROLLS, None, hand=0, held={9: 3, 10: 1})
    region = env.board.by_id(9)
    region.lair = True
    region.fortress = True
    assert env.tray.take_marker('fortress')
    env.step(region_action(region.index))
    assert region.owner is None and region.tokens == 0
    assert not region.lair and not region.fortress
    assert env.tray.available_marker('fortress') == MARKER_SUPPLY['fortress']
    assert rip.tokens_in_hand == 3
    assert env._turn_abandoned and not env._turn_acted
    assert not env.action_masks()[A_DECLINE], 'no decline after an abandon'
    # abandoning everything re-arms the first-conquest rule
    env.step(region_action(env.board.by_id(10).index))
    assert env.players[0].must_first_conquest
    assert rip.tokens_in_hand == 4
    _check_invariants(env, 'abandon')


def test_conquest_of_an_active_race_sends_tokens_back_in_hand():
    """The defender loses one token to the tray and keeps the rest in hand."""
    # the Humans stand in for a vanilla defender: the Elves lose no token (T4)
    env = _env(10)
    rip = _setup_turn(env, 0, RaceId.RATMEN, None, hand=12, held={14: 1})
    victim = _occupy(env, 1, RaceId.HUMANS, 10, 4)
    _occupy(env, 1, RaceId.HUMANS, 11, 1)
    before = env.tray.available(RaceId.HUMANS)
    cost = env._conquest_cost(rip, env.board.by_id(10))
    assert cost == 2 + 4, 'base cost + one per defending token (a hill adds none)'
    env.step(region_action(env.board.by_id(10).index))
    assert env.board.by_id(10).owner == 0 and env.board.by_id(10).tokens == cost
    assert victim.tokens_in_hand == 3 and victim.pending_redeploy
    assert env.tray.available(RaceId.HUMANS) == before + 1
    assert rip.attacked_players == {1}
    _check_invariants(env, 'conquest of an active race')


def test_conquest_of_a_declined_race_removes_it():
    """A declined race loses its token for good, and vanishes when empty."""
    env = _env(11)
    rip = _setup_turn(env, 0, RaceId.RATMEN, None, hand=12, held={14: 1})
    victim = _occupy(env, 1, RaceId.DWARVES, 10, 1, in_decline=True)
    env.players[1].active = None
    before = env.tray.available(RaceId.DWARVES)
    env.step(region_action(env.board.by_id(10).index))
    assert env.tray.available(RaceId.DWARVES) == before + 1
    assert victim.tokens_in_hand == 0 and not victim.pending_redeploy
    assert env.players[1].declined == [], 'the banner should have left the map'
    assert RaceId.DWARVES in env.combo_column.race_stack
    assert rip.attacked_players == set(), 'a declined race is not "attacked"'
    _check_invariants(env, 'conquest of a declined race')


def test_victim_redeploys_at_the_end_of_the_attackers_turn():
    """The victim redeploys his losses, with `current_player` switched to him."""
    # the Humans stand in for a vanilla defender: the Elves lose no token (T4)
    env = _env(12)
    _setup_turn(env, 0, RaceId.RATMEN, None, hand=12, held={14: 1})
    victim = _occupy(env, 2, RaceId.HUMANS, 10, 4)
    _occupy(env, 2, RaceId.HUMANS, 11, 1)
    _occupy(env, 2, RaceId.HUMANS, 12, 1)
    env.step(region_action(env.board.by_id(10).index))       # conquer region 10
    assert victim.tokens_in_hand == 3
    env.step(A_PASS)                                          # end conquests
    # the attacker still has tokens: redeploy them all, then his turn ends
    while env.phase == Phase.REDEPLOY:
        env.step(region_all_action(env.board.by_id(14).index))
    assert env.phase == Phase.VICTIM_REDEPLOY, env.phase.name
    assert env.current_player == 2 and env.current_race is victim
    legal = _legal(env)
    assert set(legal) == {region_action(10), region_action(11),
                          region_all_action(10), region_all_action(11)}
    env.step(region_all_action(env.board.by_id(11).index))
    assert env.board.by_id(11).tokens == 4 and victim.tokens_in_hand == 0
    assert not victim.pending_redeploy
    assert env._turn_seat == 1 and env.current_player == 1   # next seat plays
    _check_invariants(env, 'victim redeploy')


def test_redeployment_moves_tokens_between_regions():
    """Troop redeployment: tokens move freely between the regions held, >= 1
    per region (all but one per region go back in hand, then are placed)."""
    env = _env(40)
    rip = _setup_turn(env, 0, RaceId.RATMEN, None, hand=0,
                      held={9: 4, 14: 1, 10: 2})
    env.step(A_PASS)                                  # no conquest this turn
    assert env.phase == Phase.REDEPLOY
    assert rip.tokens_in_hand == 3 + 1, 'every token but one per region'
    assert [env.board.by_id(rid).tokens for rid in (9, 14, 10)] == [1, 1, 1]
    assert any('lifts 4 token(s)' in line for line in env.turn_events)
    legal = set(_legal(env))
    for rid in (9, 14, 10):
        assert region_action(env.board.by_id(rid).index) in legal
    env.step(region_action(env.board.by_id(9).index))         # 1 stays on 9
    env.step(region_all_action(env.board.by_id(14).index))    # the rest to 14
    assert [env.board.by_id(rid).tokens for rid in (9, 14, 10)] == [2, 4, 1]
    assert rip.tokens_in_hand == 0 and env.phase != Phase.REDEPLOY
    _check_invariants(env, 'free redeployment')


def test_redeployment_with_one_token_per_region_is_skipped():
    """Nothing to lift and nothing in hand: no redeployment decision."""
    env = _env(41)
    rip = _setup_turn(env, 0, RaceId.RATMEN, None, hand=0, held={9: 1, 14: 1})
    env.step(A_PASS)
    assert env.phase != Phase.REDEPLOY and rip.tokens_in_hand == 0
    assert not any('lifts' in line for line in env.turn_events)


def test_victim_redeploys_only_its_losses():
    """A victim puts back what it lost; its other tokens do not move."""
    env = _env(42)
    _setup_turn(env, 0, RaceId.RATMEN, None, hand=12, held={14: 1})
    victim = _occupy(env, 2, RaceId.HUMANS, 10, 4)
    _occupy(env, 2, RaceId.HUMANS, 11, 3)
    _occupy(env, 2, RaceId.HUMANS, 12, 1)
    env.step(region_action(env.board.by_id(10).index))       # conquer region 10
    env.step(A_PASS)
    while env.phase == Phase.REDEPLOY:
        env.step(region_all_action(env.board.by_id(14).index))
    assert env.phase == Phase.VICTIM_REDEPLOY
    assert victim.tokens_in_hand == 3, 'only the 4 - 1 tokens lost'
    assert env.board.by_id(11).tokens == 3, 'region 11 keeps its 3 tokens'
    env.step(region_all_action(env.board.by_id(12).index))
    assert env.board.by_id(11).tokens == 3 and env.board.by_id(12).tokens == 4
    _check_invariants(env, 'victim redeploys its losses only')


def test_victim_with_at_most_one_region_is_auto_resolved():
    """No decision to take: the engine places the tokens (or keeps them)."""
    # the Humans stand in for a vanilla defender: the Elves lose no token (T4)
    env = _env(13)
    _setup_turn(env, 0, RaceId.RATMEN, None, hand=12, held={14: 1})
    victim = _occupy(env, 2, RaceId.HUMANS, 10, 4)
    _occupy(env, 2, RaceId.HUMANS, 11, 1)
    env.step(region_action(env.board.by_id(10).index))        # one region left
    env.step(A_PASS)
    while env.phase == Phase.REDEPLOY:
        env.step(region_all_action(env.board.by_id(14).index))
    assert env.phase != Phase.VICTIM_REDEPLOY
    assert env.board.by_id(11).tokens == 4 and victim.tokens_in_hand == 0
    assert any('only region left' in line for line in env.event_log)

    env = _env(14)
    _setup_turn(env, 0, RaceId.RATMEN, None, hand=12, held={14: 1})
    victim = _occupy(env, 2, RaceId.HUMANS, 10, 4)
    env.step(region_action(env.board.by_id(10).index))        # no region left
    env.step(A_PASS)
    while env.phase == Phase.REDEPLOY:
        env.step(region_all_action(env.board.by_id(14).index))
    assert victim.tokens_in_hand == 3 and not victim.pending_redeploy
    assert env.players[2].must_first_conquest
    _check_invariants(env, 'auto-resolved victim')


# --------------------------------------------------------------------------- #
# Combos, decline, scoring, end of game
# --------------------------------------------------------------------------- #

def test_combo_cost_and_coin_transfer():
    """Picking combo `i` drops one coin on each skipped combo, and takes them."""
    env = _env(15)
    column = env.combo_column
    race, power = column.visible[2].race, column.visible[2].power
    env.step(A_COMBO + 2)
    assert env.players[0].coins == 3
    assert column.visible[0].coins == 1 and column.visible[1].coins == 1
    assert len(column.visible) == 6
    rip = env.players[0].active
    assert rip.race == race and rip.power == power
    assert rip.tokens_in_hand == rip.initial_tokens()
    assert env.tray.available(race) == RACES[race].total_tokens - rip.initial_tokens()
    # finish player 1's turn without conquering, player 2 then takes the coin
    env.step(A_PASS)
    assert env.phase == Phase.PICK_COMBO and env._turn_seat == 1
    env.step(A_COMBO + 0)
    assert env.players[1].coins == 6, 'the coin lying on the combo was collected'
    _check_invariants(env, 'combo prices')


def test_decline_turn_scores_and_keeps_one_token_per_region():
    """Decline: one token per region, the rest to the tray, then scoring."""
    env = _env(16)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.MERCHANT, hand=2,
                      held={9: 3, 10: 2, 11: 1}, turns_played=2)
    player = env.players[0]
    coins = player.coins
    before = env.tray.available(RaceId.RATMEN)
    assert A_DECLINE in _legal(env)
    env.step(A_DECLINE)
    assert player.active is None and len(player.declined) == 1
    declined = player.declined[0]
    assert declined is rip and declined.in_decline and declined.power is None
    assert PowerId.MERCHANT in env.combo_column.power_discard
    assert [r.tokens for r in env.board.regions_of_race(declined)] == [1, 1, 1]
    assert all(r.in_decline for r in env.board.regions_of_race(declined))
    assert declined.tokens_in_hand == 0
    assert env.tray.available(RaceId.RATMEN) == before + 2 + 2 + 1   # hand + extras
    assert player.coins == coins + 3, 'one coin per region, no bonus'
    assert env._turn_seat == 1, 'the turn ends right after a decline'
    _check_invariants(env, 'decline')


def test_declining_again_removes_the_previous_declined_race():
    """Only one declined race per player (Spirit aside)."""
    env = _env(17)
    old = _occupy(env, 0, RaceId.DWARVES, 25, 1, in_decline=True)
    rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.HILL, hand=0,
                      held={9: 1}, turns_played=2)
    before = env.tray.available(RaceId.DWARVES)
    env.step(A_DECLINE)
    assert env.tray.available(RaceId.DWARVES) == before + 1
    assert env.board.by_id(25).owner is None
    assert env.players[0].declined == [rip]
    assert RaceId.DWARVES in env.combo_column.race_stack
    assert not any(d is old for d in env.players[0].declined)
    _check_invariants(env, 'second decline')


def test_scoring_counts_declined_regions_and_water():
    """One coin per region held by any of the player's races."""
    env = _env(18)
    player = env.players[0]
    _setup_turn(env, 0, RaceId.RATMEN, None, hand=0, held={9: 1, 10: 1})
    _occupy(env, 0, RaceId.DWARVES, 25, 1, in_decline=True)
    env.board.by_id(15).owner = 0          # a lake held through Seafaring
    env.board.by_id(15).race = RaceId.RATMEN
    env.board.by_id(15).tokens = env.tray.take(RaceId.RATMEN, 1)
    coins = player.coins
    env._score(player)
    assert player.coins == coins + 4       # 3 active + 1 declined
    assert any('scores 4 coin(s)' in line for line in env.event_log)


def test_end_of_game_ranking_and_tie_break():
    """Most coins wins, ties broken by tokens on the board then by seat."""
    env = _env(19)
    env.players[0].coins, env.players[1].coins, env.players[2].coins = 10, 10, 5
    _occupy(env, 0, RaceId.RATMEN, 25, 2)
    _occupy(env, 1, RaceId.ELVES, 26, 5)
    env._end_game()
    assert env.done and env.phase == Phase.DONE
    assert env.winner_player == 1, 'more tokens on the board breaks the tie'
    assert env.terminal_rewards == [-0.25, 1.0, -0.75]
    assert abs(sum(env.terminal_rewards)) < 1e-9

    env = _env(20)                          # full tie: coins and tokens
    env.players[0].coins, env.players[1].coins, env.players[2].coins = 10, 10, 5
    _occupy(env, 0, RaceId.RATMEN, 25, 3)
    _occupy(env, 1, RaceId.ELVES, 26, 3)
    env._end_game()
    assert env.winner_player == 0, 'the lowest seat wins a complete tie'
    assert env.terminal_rewards == [0.375, 0.375, -0.75]
    assert abs(sum(env.terminal_rewards)) < 1e-9
    assert env.rank_rewards() == (1.0, -0.25, -0.75)


def test_shaping_is_zero_sum():
    """Every scoring event pays the scorer and taxes the others."""
    env = _env(21)
    env._step_rewards = [0.0] * 3
    _setup_turn(env, 0, RaceId.RATMEN, None, hand=0, held={9: 1, 10: 1})
    env._score(env.players[0])
    rewards = env._step_rewards
    assert abs(sum(rewards)) < 1e-12
    assert rewards[0] == 2 / env.SHAPING_SCALE
    assert rewards[1] == rewards[2] == -1 / env.SHAPING_SCALE


# --------------------------------------------------------------------------- #
# Structural pieces wired for T4 / T5
# --------------------------------------------------------------------------- #

def test_sorcerer_substitution_mechanics():
    """`sorcerer_targets` drives the A_SORCERER range; the engine swaps tokens."""
    class Sorcerers(RaceHooks):
        def sorcerer_targets(self, env, rip):
            return {10} if 1 not in rip.sorcerer_used_on else set()

    with _with_hooks(races={RaceId.SORCERERS: Sorcerers()}):
        env = _env(22)
        rip = _setup_turn(env, 0, RaceId.SORCERERS, None, hand=0, held={14: 1})
        victim = _occupy(env, 1, RaceId.ELVES, 10, 1)
        _occupy(env, 1, RaceId.ELVES, 11, 1)
        elves_before = env.tray.available(RaceId.ELVES)
        sorcerers_before = env.tray.available(RaceId.SORCERERS)
        assert sorcerer_action(9) in _legal(env)
        env.step(sorcerer_action(9))
        region = env.board.by_id(10)
        assert region.owner == 0 and region.race == RaceId.SORCERERS
        assert region.tokens == 1
        assert env.tray.available(RaceId.ELVES) == elves_before + 1
        assert env.tray.available(RaceId.SORCERERS) == sorcerers_before - 1
        assert not victim.pending_redeploy, 'the token goes to the tray, not in hand'
        assert rip.sorcerer_used_on == {1} and rip.attacked_players == {1}
        assert rip.conquests_this_turn == [], 'a substitution is not a conquest'
        assert sorcerer_action(9) not in _legal(env), 'once per opponent per turn'
        _check_invariants(env, 'sorcerer')


def test_dragon_conquest_mechanics():
    """`dragon_available` drives the A_DRAGON range; the dragon then moves."""
    class DragonMaster(PowerHooks):
        def dragon_available(self, env, rip):
            return not rip.dragon_used

    with _with_hooks(powers={PowerId.DRAGON_MASTER: DragonMaster()}):
        env = _env(23)
        rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.DRAGON_MASTER, hand=2,
                          held={14: 1})
        target = env.board.by_id(10)
        _occupy(env, 1, RaceId.ELVES, 10, 6)          # far too strong normally
        assert target.index not in env._conquest_options(rip)
        assert dragon_action(target.index) in _legal(env)
        env.step(dragon_action(target.index))
        assert target.owner == 0 and target.tokens == 1 and target.dragon
        assert rip.dragon_used and rip.tokens_in_hand == 1
        assert env.tray.available_marker('dragon') == 0
        assert not env.action_masks()[dragon_action(env.board.by_id(9).index)]
        # a second dragon conquest would move the marker
        rip.dragon_used = False
        env.step(dragon_action(env.board.by_id(9).index))
        assert not target.dragon and env.board.by_id(9).dragon
        assert env.tray.available_marker('dragon') == 0
        _check_invariants(env, 'dragon')


def test_berserk_die_is_rolled_before_each_conquest():
    """`die_before_each_conquest` draws a die at every conquest decision."""
    class Berserk(PowerHooks):
        def die_before_each_conquest(self, rip):
            return True

    with _with_hooks(powers={PowerId.BERSERK: Berserk()}):
        env = _env(24)
        _force_dice(env, [3, 1])
        rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.BERSERK, hand=4,
                          held={14: 1})
        assert rip.die == 3, 'the die is rolled when the conquest phase opens'
        target = env.board.by_id(10)           # hill + lost tribe -> base cost 3
        target.lost_tribe = True
        assert env._conquest_cost(rip, target) == 1, 'the die reduces the cost'
        # with a die in play there is no "final attempt"
        assert all(not final for _, final in env._conquest_options(rip).values())
        env.step(region_action(target.index))
        assert target.tokens == 1 and rip.tokens_in_hand == 3
        assert rip.die == 1, 'a new die is rolled for the next conquest'
        _check_invariants(env, 'berserk')


def test_ghoul_phases_run_before_the_active_race():
    """A declined race that conquers gets GHOUL_CONQUER / GHOUL_REDEPLOY."""
    class Ghouls(RaceHooks):
        def keeps_tokens_in_decline(self, rip):
            return True

        def conquers_in_decline(self, rip):
            return True

        def losses_like_active(self, rip):
            return True

    with _with_hooks(races={RaceId.GHOULS: Ghouls()}):
        env = _env(25)
        ghouls = _occupy(env, 0, RaceId.GHOULS, 25, 3, in_decline=True)
        _occupy(env, 0, RaceId.GHOULS, 26, 2, in_decline=True)
        _occupy(env, 0, RaceId.RATMEN, 9, 2)
        env.players[0].active.power = PowerId.HILL
        env._begin_turn(0)
        assert env.phase == Phase.GHOUL_CONQUER
        assert env.current_race is ghouls
        assert ghouls.tokens_in_hand == 3, 'ready your troops, one token per region'
        assert A_DECLINE not in _legal(env), 'a declined race cannot decline'
        # as if a conquest had left 2 extra tokens on region 26
        ghouls.tokens_in_hand -= 2
        env.board.by_id(26).tokens += 2
        env.step(A_PASS)
        assert env.phase == Phase.GHOUL_REDEPLOY
        assert ghouls.tokens_in_hand == 3, 'the Ghouls redeploy freely too'
        assert env.board.by_id(26).tokens == 1
        env.step(region_all_action(env.board.by_id(25).index))
        assert env.board.by_id(25).tokens == 4
        assert env.phase == Phase.CONQUER, 'the active race plays next'
        assert env.current_race is env.players[0].active
        _check_invariants(env, 'ghoul phases', vanilla_declined=False)


def test_ghoul_decline_keeps_its_tokens_and_redeploys_them():
    """`keeps_tokens_in_decline` turns the decline into a redeployment."""
    class Ghouls(RaceHooks):
        def keeps_tokens_in_decline(self, rip):
            return True

    with _with_hooks(races={RaceId.GHOULS: Ghouls()}):
        env = _env(26)
        rip = _setup_turn(env, 0, RaceId.GHOULS, PowerId.HILL, hand=4,
                          held={9: 2, 10: 1}, turns_played=2)
        env.step(A_DECLINE)
        assert env.phase == Phase.GHOUL_REDEPLOY and rip.in_decline
        assert rip.tokens_in_hand == 4 and env.board.by_id(9).tokens == 2
        env.step(region_all_action(env.board.by_id(9).index))
        assert env.board.by_id(9).tokens == 6 and rip.tokens_in_hand == 0
        assert env.players[0].coins == 5 + 2, 'scoring happens after the redeploy'
        assert env._turn_seat == 1


def test_end_of_turn_sub_phases():
    """The ENCAMPMENTS / FORTRESS / HEROES / ALLY / STOUT_DECLINE chain."""
    class Everything(PowerHooks):
        def encampments_to_place(self, env, rip):
            return 5

        def can_place_fortress(self, env, rip):
            return True

        def heroes_to_place(self, env, rip):
            return 2

        def has_ally_choice(self, env, rip):
            return True

        def can_stout_decline(self, env, rip):
            return True

    with _with_hooks(powers={PowerId.BIVOUACKING: Everything()}):
        env = _env(27)
        rip = _setup_turn(env, 0, RaceId.RATMEN, PowerId.BIVOUACKING, hand=0,
                          held={9: 1, 10: 1}, turns_played=2)
        env.step(A_PASS)                                   # end the conquests
        seen = []
        while env.phase in (Phase.ENCAMPMENTS, Phase.FORTRESS, Phase.HEROES,
                            Phase.ALLY, Phase.STOUT_DECLINE):
            seen.append(env.phase)
            if env.phase == Phase.STOUT_DECLINE:
                assert set(_legal(env)) == {A_DECLINE, A_PASS}
                env.step(A_PASS)
                break
            if env.phase == Phase.ALLY:
                assert set(_legal(env)) == {A_PASS, ally_action(1), ally_action(2)}
                env.step(ally_action(2))
                continue
            env.step([a for a in _legal(env) if a != A_PASS][0])
        assert seen[0] == Phase.ENCAMPMENTS
        assert Phase.FORTRESS in seen and Phase.HEROES in seen
        assert seen[-1] == Phase.STOUT_DECLINE
        assert seen.count(Phase.ENCAMPMENTS) == 5, 'one action per encampment'
        assert seen.count(Phase.HEROES) == 2
        markers = _markers_on_board(env)
        assert markers == {'fortress': 1, 'encampment': 5, 'hero': 2,
                           'hole': 0, 'dragon': 0}
        assert env.players[0].ally == 2 and rip.fortress_used
        assert env.players[0].coins == 5 + 2, 'scoring happens before STOUT_DECLINE'
        _check_invariants(env, 'sub-phases')


def test_encampments_are_taken_back_before_being_re_placed():
    """Encampments already on the board come back to the pool every turn."""
    class Bivouacking(PowerHooks):
        def encampments_to_place(self, env, rip):
            return 5

    with _with_hooks(powers={PowerId.BIVOUACKING: Bivouacking()}):
        env = _env(28)
        _setup_turn(env, 0, RaceId.RATMEN, PowerId.BIVOUACKING, hand=0,
                    held={9: 1}, turns_played=2)
        env.board.by_id(9).encampments = 3
        for _ in range(3):
            assert env.tray.take_marker('encampment')
        env.step(A_PASS)
        assert env.phase == Phase.ENCAMPMENTS and env._encampments_left == 5
        assert env.board.by_id(9).encampments == 0
        for _ in range(5):
            env.step(region_action(8))
        assert env.board.by_id(9).encampments == 5
        assert env.tray.available_marker('encampment') == 0
        _check_invariants(env, 'encampments')


def test_spirit_declined_race_survives_a_second_decline():
    """`is_spirit_power` exempts a declined race from the one-race limit."""
    class Spirit(PowerHooks):
        def is_spirit_power(self, rip):
            return True

    with _with_hooks(powers={PowerId.SPIRIT: Spirit()}):
        env = _env(29)
        first = _setup_turn(env, 0, RaceId.DWARVES, PowerId.SPIRIT, hand=0,
                            held={25: 1}, turns_played=2)
        env.step(A_DECLINE)
        assert first.is_spirit and env.players[0].declined == [first]
        second = _setup_turn(env, 0, RaceId.RATMEN, PowerId.HILL, hand=0,
                             held={9: 1}, turns_played=2)
        env.step(A_DECLINE)
        assert len(env.players[0].declined) == 2
        assert env.board.by_id(25).owner == 0, 'the Spirit race stays on the map'
        assert set(env.players[0].declined) == {first, second}
        _check_invariants(env, 'spirit')


# --------------------------------------------------------------------------- #
# Fuzz
# --------------------------------------------------------------------------- #

def test_fuzz_random_games():
    """`FUZZ_GAMES` random games with every invariant checked at every step."""
    env = SmallWorldEnv(3)
    allowed = _allowed_terminal_values(env)
    steps_per_game = []
    coins = []
    turns = []
    phases_seen = set()
    actions_seen = set()
    started = time.time()

    for seed in range(FUZZ_GAMES):
        obs, info = env.reset(seed=seed)
        assert info['next_step_no_action'] is False
        assert env.observation_space.contains(obs)
        _check_invariants(env, f'seed {seed} reset')
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
            mask = env.action_masks()
            legal = np.flatnonzero(mask)
            assert legal.size, f'seed {seed}: no legal action in {env.phase.name}'
            action = int(rng.choice(legal))
            actions_seen.add(action_kind(action)[0])
            assert env.describe_action(action), env.phase.name
            obs, rewards, terminated, truncated, info = env.step(action)
            steps += 1
            ctx = f'seed {seed} step {steps} ({env.phase.name})'
            assert truncated is False and info == {
                'next_step_no_action': env.phase == Phase.TURN_PAUSE}, ctx
            assert len(rewards) == 3, ctx
            assert abs(sum(rewards)) < 1e-9, f'{ctx}: rewards {rewards} not zero-sum'
            assert terminated == env.done, ctx
            _check_invariants(env, ctx)

        # -- terminal state ------------------------------------------------- #
        assert env.turns_taken == turns_for(3), f'seed {seed}: {env.turns_taken} turns'
        assert env.turn == env.turns_total + 1
        best = max(p.coins for p in env.players)
        assert env.players[env.winner_player].coins == best, f'seed {seed}: winner'
        assert env.terminal_rewards is not None
        assert abs(sum(env.terminal_rewards)) < 1e-9
        for value in env.terminal_rewards:
            assert round(value, 9) in allowed, f'seed {seed}: reward {value}'
        assert not env.action_masks().any(), f'seed {seed}: mask after the end'
        steps_per_game.append(steps)
        coins.append([p.coins for p in env.players])
        turns.append(env.turns_taken)

    FUZZ_STATS.update(
        games=FUZZ_GAMES,
        steps_total=int(np.sum(steps_per_game)),
        steps_mean=round(float(np.mean(steps_per_game)), 1),
        steps_min=int(np.min(steps_per_game)),
        steps_max=int(np.max(steps_per_game)),
        coins_mean=round(float(np.mean(coins)), 1),
        coins_max=int(np.max(coins)),
        turns_mean=round(float(np.mean(turns)), 2),
        phases=sorted(p.name for p in phases_seen),
        action_kinds=sorted(actions_seen),
        seconds=round(time.time() - started, 1),
    )
    # every vanilla decision point must be exercised by the fuzz
    assert {Phase.PICK_COMBO, Phase.CONQUER, Phase.REDEPLOY,
            Phase.VICTIM_REDEPLOY} <= phases_seen, phases_seen
    assert min(turns) == turns_for(3) >= 10


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

def _replay(seed: int, actions: list[int] | None = None):
    """Play a game (recording or replaying `actions`); return the trace."""
    env = SmallWorldEnv(3)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1000)
    played, observations, rewards = [], [], []
    index = 0
    while not env.done:
        if actions is None:
            legal = np.flatnonzero(env.action_masks())
            action = int(rng.choice(legal)) if legal.size else -1   # pause
        else:
            action = actions[index]
            index += 1
        played.append(action)
        obs, reward, *_ = env.step(action)
        observations.append(obs)
        rewards.append(reward)
    return env, played, observations, rewards


def test_determinism_same_seed_same_actions():
    """Same seed + same actions give the same observations, rewards and log."""
    env_a, actions, obs_a, rew_a = _replay(31)
    env_b, replayed, obs_b, rew_b = _replay(31, actions)
    assert replayed == actions
    assert env_a.event_log == env_b.event_log
    assert env_a.winner_player == env_b.winner_player
    assert [p.coins for p in env_a.players] == [p.coins for p in env_b.players]
    assert rew_a == rew_b
    assert len(obs_a) == len(obs_b)
    for step, (first, second) in enumerate(zip(obs_a, obs_b)):
        assert first.keys() == second.keys()
        for key in first:
            assert np.array_equal(first[key], second[key]), f'step {step}, {key}'


def test_deepcopy_does_not_leak():
    """Stepping a deep copy leaves the original untouched (MCTS requirement)."""
    env = _env(32)
    for _ in range(12):
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    clone = copy.deepcopy(env)
    before_log = list(env.event_log)
    before_coins = [p.coins for p in env.players]
    before_obs = env.observation
    before_board = [(r.owner, r.race, r.tokens) for r in env.board.regions]
    rng = np.random.default_rng(5)
    for _ in range(40):
        if clone.done:
            break
        clone.step(int(rng.choice(np.flatnonzero(clone.action_masks()))))
    assert env.event_log == before_log
    assert [p.coins for p in env.players] == before_coins
    assert [(r.owner, r.race, r.tokens) for r in env.board.regions] == before_board
    for key in before_obs:
        assert np.array_equal(env.observation[key], before_obs[key]), key
    assert clone.board is not env.board and clone.players is not env.players
    _check_invariants(env, 'original after a deepcopy')
    _check_invariants(clone, 'clone')


def test_redeterminize_keeps_the_public_state():
    """`redeterminize` only reshuffles the hidden stacks and the unused dice."""
    env = _env(33)
    for _ in range(20):
        if env.done:
            break
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    visible = [(c.race, c.power, c.coins) for c in env.combo_column.visible]
    races = sorted(env.combo_column.race_stack)
    powers = sorted(env.combo_column.power_stack)
    rolled = env.dice._values[:env.dice._pos]
    remaining = sorted(env.dice._values[env.dice._pos:])
    coins = [p.coins for p in env.players]
    obs = env.observation

    env.redeterminize(0)
    assert [(c.race, c.power, c.coins) for c in env.combo_column.visible] == visible
    assert sorted(env.combo_column.race_stack) == races
    assert sorted(env.combo_column.power_stack) == powers
    assert env.dice._values[:env.dice._pos] == rolled
    assert sorted(env.dice._values[env.dice._pos:]) == remaining
    assert [p.coins for p in env.players] == coins
    for key in obs:
        assert np.array_equal(env.observation[key], obs[key]), key


def test_unsupported_player_counts_are_explicit():
    """Only the 3-player board exists so far; the others must say so."""
    for n in (2, 4, 5):
        try:
            SmallWorldEnv(n)
        except NotImplementedError:
            pass
        else:
            raise AssertionError(f'SmallWorldEnv({n}) should raise')


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


def test_turn_pause_between_two_players():
    """TURN_PAUSE: a no-action step after every turn except the last one."""
    env = SmallWorldEnv(3)                        # the pause is on by default
    env.reset(seed=50)
    rng = np.random.default_rng(50)
    pauses, seats_after = 0, []
    info = {'next_step_no_action': False}
    while not env.done:
        if env.phase == Phase.TURN_PAUSE:
            assert info == {'next_step_no_action': True}, 'announced by the step'
            assert env.current_player == -1 and not env.action_masks().any()
            ended = env._turn_seat
            try:
                env.step(0)
            except Exception as error:
                assert 'only the no-action step' in str(error)
            else:
                raise AssertionError('a real action is illegal in a pause')
            _obs, rewards, done, _trunc, info = env.step(-1)
            assert rewards == [0.0] * 3 and not done
            assert env._turn_seat == (ended + 1) % 3, 'the next seat plays'
            assert env.phase != Phase.TURN_PAUSE
            pauses += 1
            seats_after.append(env._turn_seat)
            continue
        legal = np.flatnonzero(env.action_masks())
        _obs, _rewards, _done, _trunc, info = env.step(int(rng.choice(legal)))
    # 10 turns x 3 players, no pause after the very last turn
    assert pauses == env.turns_total * 3 - 1
    assert seats_after[:4] == [1, 2, 0, 1]
    assert env.phase == Phase.DONE


def test_turn_pause_can_be_turned_off():
    """`pause_between_turns=False`: the next turn starts right away."""
    env = SmallWorldEnv(3, pause_between_turns=False)
    env.reset(seed=51)
    rng = np.random.default_rng(51)
    while not env.done:
        assert env.phase != Phase.TURN_PAUSE
        _obs, _r, _d, _t, info = env.step(
            int(rng.choice(np.flatnonzero(env.action_masks()))))
        assert info == {'next_step_no_action': False}
