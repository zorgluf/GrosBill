"""Unit tests for the Small World state classes (`envs/classes.py`, task T2).

Plain `assert`-based functions, usable either with pytest
(`pytest environments/smallw/tests/test_classes.py`) or through the bundled
runners:

    cd app
    python -m environments.smallw.tests.run_all        # every test module
    python -m environments.smallw.tests.test_classes   # this module only
"""

from __future__ import annotations

import copy
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

from ..envs import map3p
from ..envs.classes import (
    DIE_FACES,
    MARKER_KINDS,
    MARKER_SUPPLY,
    MAX_POWER_VALUE,
    N_DRAGONS,
    N_ENCAMPMENTS,
    N_FORTRESSES,
    N_HEROES,
    N_HOLES,
    N_VISIBLE_COMBOS,
    POWERS,
    RACES,
    START_COINS,
    Board,
    Combo,
    ComboColumn,
    Dice,
    PlayerState,
    PowerId,
    RaceId,
    RaceInPlay,
    Region,
    Symbol,
    Terrain,
    Tray,
)

#: `app/environments/smallw/static/`
STATIC_DIR = Path(__file__).resolve().parents[1] / 'static'


def _rng(seed: int = 12345) -> np.random.Generator:
    """A fresh seeded generator (tests must never depend on global state)."""
    return np.random.default_rng(seed)


def _player(seat: int = 0, coins: int = START_COINS) -> PlayerState:
    return PlayerState(seat, coins=coins)


# --------------------------------------------------------------------------- #
# Enums and data tables
# --------------------------------------------------------------------------- #

def test_enum_codes_are_frozen():
    """The enum values are observation codes: their order must not drift."""
    assert [r.name for r in RaceId] == [
        'AMAZONS', 'DWARVES', 'ELVES', 'GHOULS', 'GIANTS', 'HALFLINGS',
        'HUMANS', 'ORCS', 'RATMEN', 'SKELETONS', 'SORCERERS', 'TRITONS',
        'TROLLS', 'WIZARDS',
    ]
    assert [int(r) for r in RaceId] == list(range(14))
    assert [p.name for p in PowerId] == [
        'ALCHEMIST', 'BERSERK', 'BIVOUACKING', 'COMMANDO', 'DIPLOMAT',
        'DRAGON_MASTER', 'FLYING', 'FOREST', 'FORTIFIED', 'HEROIC', 'HILL',
        'MERCHANT', 'MOUNTED', 'PILLAGING', 'SEAFARING', 'SPIRIT', 'STOUT',
        'SWAMP', 'UNDERWORLD', 'WEALTHY',
    ]
    assert [int(p) for p in PowerId] == list(range(20))


def test_race_table():
    """The 14 races with the values of the rules (banner, box, attack bonus)."""
    expected = {
        RaceId.AMAZONS:   ('Amazons',   6, 15, 4),
        RaceId.DWARVES:   ('Dwarves',   3, 8, 0),
        RaceId.ELVES:     ('Elves',     6, 11, 0),
        RaceId.GHOULS:    ('Ghouls',    5, 10, 0),
        RaceId.GIANTS:    ('Giants',    6, 11, 0),
        RaceId.HALFLINGS: ('Halflings', 6, 11, 0),
        RaceId.HUMANS:    ('Humans',    5, 10, 0),
        RaceId.ORCS:      ('Orcs',      5, 10, 0),
        RaceId.RATMEN:    ('Ratmen',    8, 13, 0),
        RaceId.SKELETONS: ('Skeletons', 6, 20, 0),
        RaceId.SORCERERS: ('Sorcerers', 5, 18, 0),
        RaceId.TRITONS:   ('Tritons',   6, 11, 0),
        RaceId.TROLLS:    ('Trolls',    5, 10, 0),
        RaceId.WIZARDS:   ('Wizards',   5, 10, 0),
    }
    assert len(RACES) == 14
    assert set(RACES) == set(RaceId) == set(expected)
    for rid, (name, banner, total, bonus) in expected.items():
        rdef = RACES[rid]
        assert rdef.id == rid, f'{name}: id {rdef.id} != {rid}'
        assert rdef.name == name
        assert rdef.banner_value == banner, f'{name}: banner {rdef.banner_value} != {banner}'
        assert rdef.total_tokens == total, f'{name}: total {rdef.total_tokens} != {total}'
        assert rdef.attack_bonus_tokens == bonus, f'{name}: bonus != {bonus}'
        # the tray can always pay for a freshly picked combo
        assert rdef.total_tokens >= banner + bonus + MAX_POWER_VALUE, (
            f'{name}: only {rdef.total_tokens} tokens in the box for a '
            f'{banner}+{bonus}+{MAX_POWER_VALUE} pick'
        )
    # Amazons are the only race with attack-only tokens
    assert [r for r in RaceId if RACES[r].attack_bonus_tokens] == [RaceId.AMAZONS]


def test_power_table():
    """The 20 powers with the values printed on the badges."""
    expected = {
        PowerId.ALCHEMIST:     ('Alchemist',     4),
        PowerId.BERSERK:       ('Berserk',       4),
        PowerId.BIVOUACKING:   ('Bivouacking',   5),
        PowerId.COMMANDO:      ('Commando',      4),
        PowerId.DIPLOMAT:      ('Diplomat',      5),
        PowerId.DRAGON_MASTER: ('Dragon Master', 5),
        PowerId.FLYING:        ('Flying',        5),
        PowerId.FOREST:        ('Forest',        4),
        PowerId.FORTIFIED:     ('Fortified',     3),
        PowerId.HEROIC:        ('Heroic',        5),
        PowerId.HILL:          ('Hill',          4),
        PowerId.MERCHANT:      ('Merchant',      2),
        PowerId.MOUNTED:       ('Mounted',       5),
        PowerId.PILLAGING:     ('Pillaging',     5),
        PowerId.SEAFARING:     ('Seafaring',     5),
        PowerId.SPIRIT:        ('Spirit',        5),
        PowerId.STOUT:         ('Stout',         4),
        PowerId.SWAMP:         ('Swamp',         4),
        PowerId.UNDERWORLD:    ('Underworld',    5),
        PowerId.WEALTHY:       ('Wealthy',       4),
    }
    assert len(POWERS) == 20
    assert set(POWERS) == set(PowerId) == set(expected)
    for pid, (name, value) in expected.items():
        pdef = POWERS[pid]
        assert pdef.id == pid
        assert pdef.name == name
        assert pdef.value == value, f'{name}: value {pdef.value} != {value}'
    assert MAX_POWER_VALUE == 5
    assert POWERS[PowerId.DRAGON_MASTER].key == 'dragon_master'


def test_keys_are_snake_case():
    for rdef in RACES.values():
        assert rdef.key == rdef.name.lower().replace(' ', '_')
        assert rdef.key.replace('_', '').isalpha() and rdef.key.islower()
    for pdef in POWERS.values():
        assert pdef.key == pdef.name.lower().replace(' ', '_')
        assert pdef.key.replace('_', '').isalpha() and pdef.key.islower()


def test_keys_match_static_assets():
    """Every `key` must name the files committed in `static/` (T1)."""
    assert STATIC_DIR.is_dir(), f'missing static dir {STATIC_DIR}'
    for rdef in RACES.values():
        banner = STATIC_DIR / 'races' / f'{rdef.key}.png'
        token = STATIC_DIR / 'races' / f'{rdef.key}_token.jpg'
        assert banner.is_file(), f'missing banner {banner}'
        assert token.is_file(), f'missing token {token}'
    for pdef in POWERS.values():
        badge = STATIC_DIR / 'powers' / f'{pdef.key}.png'
        assert badge.is_file(), f'missing badge {badge}'
    # no stray / missing file: 14 banners + 14 tokens, 20 badges
    assert len(list((STATIC_DIR / 'races').iterdir())) == 2 * len(RACES)
    assert len(list((STATIC_DIR / 'powers').iterdir())) == len(POWERS)
    # the markers the state classes know about have an image too
    for kind in MARKER_KINDS:
        assert (STATIC_DIR / 'pieces' / f'{kind}.jpg').is_file(), f'missing {kind}.jpg'
    for piece in ('lost_tribe', 'mountain', 'die', 'turn_marker'):
        assert (STATIC_DIR / 'pieces' / f'{piece}.jpg').is_file(), f'missing {piece}.jpg'


def test_constants():
    assert DIE_FACES == (0, 0, 0, 1, 2, 3)
    assert N_VISIBLE_COMBOS == 6
    assert START_COINS == 5
    assert (N_FORTRESSES, N_ENCAMPMENTS, N_HEROES, N_HOLES, N_DRAGONS) == (6, 5, 2, 2, 1)
    assert MARKER_SUPPLY == {
        'fortress': 6, 'encampment': 5, 'hero': 2, 'hole': 2, 'dragon': 1,
    }
    assert set(MARKER_KINDS) == set(MARKER_SUPPLY)
    # re-exported from map3p, never redefined
    assert Terrain is map3p.Terrain and Symbol is map3p.Symbol


# --------------------------------------------------------------------------- #
# Region
# --------------------------------------------------------------------------- #

def test_region_passthroughs():
    board = Board(3)
    farmland = board.by_id(4)       # FARMLAND, no symbol, border
    hill = board.by_id(10)          # HILL, magic + lost tribe, not border
    mountain = board.by_id(13)      # MOUNTAIN, mine + cavern, border
    assert (farmland.id, farmland.index) == (4, 3)
    assert farmland.terrain == Terrain.FARMLAND and farmland.border
    assert not farmland.symbols
    assert hill.terrain == Terrain.HILL and not hill.border
    assert hill.has(Symbol.MAGIC) and hill.has(Symbol.LOST_TRIBE)
    assert mountain.is_mountain and mountain.has(Symbol.CAVERN)
    assert not mountain.is_water and board.by_id(1).is_water
    assert set(farmland.adjacent) == {3, 5, 10, 11}


def test_region_occupation_helpers():
    board = Board(3)
    region = board.by_id(4)
    assert region.is_empty and not region.is_immune
    region.lost_tribe = True
    assert not region.is_empty
    region.lost_tribe = False
    region.owner, region.race, region.tokens = 1, RaceId.ORCS, 3
    assert not region.is_empty
    region.clear_race()
    assert region.is_empty and region.owner is None and region.race is None
    assert region.tokens == 0 and not region.in_decline
    # a mountain alone is empty
    assert board.by_id(13).is_empty
    # immunity markers
    for flag in ('hole', 'hero', 'dragon'):
        setattr(region, flag, True)
        assert region.is_immune
        setattr(region, flag, False)
    assert not region.is_immune
    # clear_race leaves the markers alone (the engine decides their fate)
    region.fortress = region.lair = True
    region.owner, region.race, region.tokens = 0, RaceId.TROLLS, 2
    region.clear_race()
    assert region.fortress and region.lair
    # reset() wipes everything
    region.reset()
    assert not region.fortress and not region.lair and region.is_empty


def test_base_conquest_cost():
    board = Board(3)
    board.setup()
    # empty farmland: the plain 2 tokens
    farmland = board.by_id(4)
    assert farmland.is_empty and farmland.base_conquest_cost() == 2
    # hill with a Lost Tribe: 2 + 1
    hill = board.by_id(10)
    assert hill.lost_tribe and hill.base_conquest_cost() == 3
    # mountain + Lost Tribe: 2 + 1 + 1 (no mountain carries the printed symbol,
    # so the token is placed by hand here)
    mountain = board.by_id(13)
    assert mountain.base_conquest_cost() == 3     # mountain alone
    mountain.lost_tribe = True
    assert mountain.base_conquest_cost() == 4
    # 3 enemy tokens + a fortress on a farmland: 2 + 3 + 1
    farmland.owner, farmland.race, farmland.tokens = 1, RaceId.ELVES, 3
    farmland.fortress = True
    assert farmland.base_conquest_cost() == 6
    assert farmland.defence == 4
    # every defensive element counts once
    farmland.encampments = 2
    farmland.lair = True
    assert farmland.base_conquest_cost() == 9


# --------------------------------------------------------------------------- #
# Board
# --------------------------------------------------------------------------- #

def test_board_setup():
    board = Board(3)
    assert board.n_regions == 30 == len(board.regions)
    for i, region in enumerate(board.regions):
        assert region.index == i and region.id == i + 1
    # before setup: no Lost Tribe anywhere
    assert not any(r.lost_tribe for r in board.regions)
    board.setup()
    lost = [r.id for r in board.regions if r.lost_tribe]
    assert len(lost) == 10, f'expected 10 Lost Tribes, got {lost}'
    assert lost == sorted(r.id for r in board.regions if r.has(Symbol.LOST_TRIBE))
    assert sum(1 for r in board.regions if r.is_mountain) == 7
    assert sum(1 for r in board.regions if r.is_water) == 3
    # nothing else is placed
    assert all(r.owner is None and r.tokens == 0 for r in board.regions)
    assert not any(r.fortress or r.encampments or r.hole or r.hero or r.dragon or r.lair
                   for r in board.regions)
    # setup() is a full reset
    board.by_id(4).owner, board.by_id(4).tokens = 2, 5
    board.by_id(10).lost_tribe = False
    board.by_id(4).dragon = True
    board.setup()
    assert board.by_id(4).is_empty and not board.by_id(4).dragon
    assert board.by_id(10).lost_tribe
    assert len([r for r in board.regions if r.lost_tribe]) == 10


def test_board_queries():
    board = Board(3)
    board.setup()
    orcs = RaceInPlay(RaceId.ORCS, PowerId.COMMANDO, owner=0)
    ghouls = RaceInPlay(RaceId.GHOULS, None, owner=0, in_decline=True)
    for rid, tokens in ((4, 3), (5, 1)):
        region = board.by_id(rid)
        region.owner, region.race, region.tokens = 0, RaceId.ORCS, tokens
        region.lost_tribe = False
    declined = board.by_id(19)
    declined.owner, declined.race, declined.tokens, declined.in_decline = 0, RaceId.GHOULS, 1, True
    enemy = board.by_id(29)
    enemy.owner, enemy.race, enemy.tokens, enemy.lost_tribe = 1, RaceId.DWARVES, 2, False

    assert [r.id for r in board.regions_of(0)] == [4, 5, 19]
    assert [r.id for r in board.regions_of(0, in_decline=False)] == [4, 5]
    assert [r.id for r in board.regions_of(0, in_decline=True)] == [19]
    assert [r.id for r in board.regions_of(1)] == [29]
    assert board.regions_of(2) == []
    assert [r.id for r in board.regions_of_race(orcs)] == [4, 5]
    assert [r.id for r in board.regions_of_race(ghouls)] == [19]
    assert board.count_tokens_on_board(0) == 5
    assert board.count_tokens_on_board(1) == 2
    assert board.count_tokens_on_board(2) == 0
    # geography
    assert [r.id for r in board.neighbours(board.by_id(4))] == [3, 5, 10, 11]
    assert [r.id for r in board.neighbours(1)] == [2, 8, 13]
    assert board.is_coastal(board.by_id(2)) and board.is_coastal(8)
    assert not board.is_coastal(4)
    assert board.is_border(4) and not board.is_border(9)
    assert [r.id for r in board.caverns()] == [5, 13, 16, 20, 24]
    assert len(board.caverns()) == 5


# --------------------------------------------------------------------------- #
# Tray
# --------------------------------------------------------------------------- #

def test_tray_race_tokens():
    tray = Tray()
    for rid, rdef in RACES.items():
        assert tray.available(rid) == rdef.total_tokens
    # a normal pick
    assert tray.take(RaceId.ORCS, 9) == 9
    assert tray.available(RaceId.ORCS) == 1
    # asking for more than the box holds clamps instead of raising
    assert tray.take(RaceId.ORCS, 5) == 1
    assert tray.available(RaceId.ORCS) == 0
    assert tray.take(RaceId.ORCS, 3) == 0
    # Skeletons growth stops at the box limit
    assert tray.take(RaceId.SKELETONS, 20) == 20
    assert tray.take(RaceId.SKELETONS, 1) == 0
    # degenerate counts
    assert tray.take(RaceId.ELVES, 0) == 0
    assert tray.take(RaceId.ELVES, -3) == 0
    assert tray.available(RaceId.ELVES) == RACES[RaceId.ELVES].total_tokens
    # putting back
    tray.put(RaceId.ORCS, 4)
    assert tray.available(RaceId.ORCS) == 4
    tray.put(RaceId.ORCS, 6)
    assert tray.available(RaceId.ORCS) == 10
    try:
        tray.put(RaceId.ORCS, 1)
    except ValueError:
        pass
    else:
        raise AssertionError('Tray.put should refuse to exceed the box')
    try:
        tray.put(RaceId.ORCS, -1)
    except ValueError:
        pass
    else:
        raise AssertionError('Tray.put should refuse a negative count')


def test_tray_markers():
    tray = Tray()
    for kind, supply in MARKER_SUPPLY.items():
        assert tray.available_marker(kind) == supply
        for _ in range(supply):
            assert tray.take_marker(kind) is True
        assert tray.available_marker(kind) == 0
        assert tray.take_marker(kind) is False, f'{kind} supply not enforced'
        tray.put_marker(kind)
        assert tray.available_marker(kind) == 1
        for _ in range(supply - 1):
            tray.put_marker(kind)
        assert tray.available_marker(kind) == supply
        try:
            tray.put_marker(kind)
        except ValueError:
            pass
        else:
            raise AssertionError(f'{kind}: put_marker should refuse to exceed the supply')
    for bad in ('lair', 'mountain', 'coin'):
        try:
            tray.take_marker(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'unknown marker kind {bad!r} should raise')


# --------------------------------------------------------------------------- #
# RaceInPlay / PlayerState
# --------------------------------------------------------------------------- #

def test_initial_tokens():
    assert RaceInPlay(RaceId.AMAZONS, PowerId.MERCHANT, 0).initial_tokens() == 6 + 4 + 2
    assert RaceInPlay(RaceId.DWARVES, PowerId.FORTIFIED, 0).initial_tokens() == 3 + 3
    assert RaceInPlay(RaceId.RATMEN, PowerId.FLYING, 0).initial_tokens() == 8 + 5
    # declined race: no power left, banner value only
    assert RaceInPlay(RaceId.RATMEN, None, 0, in_decline=True).initial_tokens() == 8
    # never more than the box holds
    for rid in RaceId:
        for pid in PowerId:
            rip = RaceInPlay(rid, pid, 0)
            assert rip.initial_tokens() <= RACES[rid].total_tokens
            assert rip.race_def is RACES[rid] and rip.power_def is POWERS[pid]


def test_race_in_play_new_turn():
    rip = RaceInPlay(RaceId.TROLLS, PowerId.BERSERK, owner=1, tokens_in_hand=9)
    assert rip.turns_played == 0 and rip.power_def.name == 'Berserk'
    rip.new_turn()
    assert rip.turns_played == 1
    rip.conquests_this_turn.append(12)
    rip.nonempty_conquests = 1
    rip.dragon_used = rip.fortress_used = rip.first_conquest_done = True
    rip.sorcerer_used_on.add(2)
    rip.attacked_players.add(0)
    rip.die = 3
    rip.holes_placed = 2
    rip.wealthy_paid = True
    rip.extra['x'] = 1
    rip.new_turn()
    assert rip.turns_played == 2
    assert rip.conquests_this_turn == [] and rip.nonempty_conquests == 0
    assert not rip.dragon_used and not rip.fortress_used and not rip.first_conquest_done
    assert rip.sorcerer_used_on == set() and rip.attacked_players == set()
    assert rip.die is None
    # persistent bookkeeping survives
    assert rip.holes_placed == 2 and rip.wealthy_paid and rip.extra == {'x': 1}
    assert rip.tokens_in_hand == 9


def test_player_state_helpers():
    player = _player(seat=2)
    assert player.coins == START_COINS and player.name == 'player 2'
    assert player.all_races() == [] and player.race_by(RaceId.ORCS) is None
    active = RaceInPlay(RaceId.ORCS, PowerId.ALCHEMIST, owner=2)
    ghouls = RaceInPlay(RaceId.GHOULS, None, owner=2, in_decline=True)
    spirit = RaceInPlay(RaceId.ELVES, None, owner=2, in_decline=True, is_spirit=True)
    player.active = active
    player.declined = [ghouls, spirit]
    assert player.all_races() == [active, ghouls, spirit]
    assert player.race_by(RaceId.ORCS) is active
    assert player.race_by(RaceId.ORCS, in_decline=False) is active
    assert player.race_by(RaceId.ORCS, in_decline=True) is None
    assert player.race_by(RaceId.GHOULS, in_decline=True) is ghouls
    assert player.race_by(RaceId.ELVES, in_decline=True) is spirit
    assert player.race_by(RaceId.DWARVES) is None
    assert [r for r in player.declined if r.is_spirit] == [spirit]
    assert player.ally is None and not player.must_first_conquest


# --------------------------------------------------------------------------- #
# ComboColumn
# --------------------------------------------------------------------------- #

def test_combo_column_setup():
    col = ComboColumn(_rng())
    assert len(col.visible) == N_VISIBLE_COMBOS == 6 == len(col)
    assert len(col.race_stack) == 14 - 6
    assert len(col.power_stack) == 20 - 6
    assert col.power_discard == []
    # all distinct, all valid, no coin yet
    assert len({c.race for c in col.visible}) == 6
    assert len({c.power for c in col.visible}) == 6
    assert all(isinstance(c.race, RaceId) and isinstance(c.power, PowerId)
               for c in col.visible)
    assert all(c.coins == 0 for c in col.visible)
    # nothing is revealed twice
    seen = [c.race for c in col.visible] + col.race_stack
    assert sorted(seen) == sorted(RaceId)
    seen_p = [c.power for c in col.visible] + col.power_stack
    assert sorted(seen_p) == sorted(PowerId)
    assert [col.cost(i) for i in range(6)] == [0, 1, 2, 3, 4, 5]


def test_combo_pick_top_is_free():
    col = ComboColumn(_rng())
    player = _player()
    wanted = col.visible[0]
    second = col.visible[1]
    combo = col.pick(0, player)
    assert combo is wanted
    assert player.coins == START_COINS, 'the top combo must be free'
    assert len(col.visible) == 6, 'the column must refill'
    assert col.visible[0] is second, 'the combos must slide up'
    assert all(c.coins == 0 for c in col.visible)


def test_combo_pick_costs_coins_on_skipped_combos():
    col = ComboColumn(_rng())
    player = _player()
    skipped = col.visible[:3]
    combo = col.pick(3, player)
    assert player.coins == START_COINS - 3
    assert [c.coins for c in skipped] == [1, 1, 1]
    assert col.visible[:3] == skipped
    assert [c.coins for c in col.visible] == [1, 1, 1, 0, 0, 0]
    assert combo not in col.visible


def test_combo_coins_are_collected_by_the_next_picker():
    col = ComboColumn(_rng())
    first, second = _player(0), _player(1)
    col.pick(3, first)                      # 1 coin on each of combos 0..2
    assert [c.coins for c in col.visible[:3]] == [1, 1, 1]
    target = col.visible[1]
    combo = col.pick(1, second)             # pays 1 (on combo 0), collects 1
    assert combo is target
    assert combo.coins == 0, 'the coins must move to the player'
    assert second.coins == START_COINS, '1 coin paid, 1 coin collected'
    assert col.visible[0].coins == 2
    # picking that 2-coin combo for free would be too good: it costs nothing and pays 2
    third = _player(2)
    col.pick(0, third)
    assert third.coins == START_COINS + 2


def test_combo_pick_requires_coins():
    col = ComboColumn(_rng())
    poor = _player(coins=2)
    try:
        col.pick(3, poor)
    except ValueError:
        pass
    else:
        raise AssertionError('pick must refuse an unaffordable combo')
    assert poor.coins == 2 and len(col.visible) == 6
    assert all(c.coins == 0 for c in col.visible), 'a refused pick must drop no coin'
    try:
        col.pick(6, _player())
    except IndexError:
        pass
    else:
        raise AssertionError('pick must refuse an index outside the column')


def test_combo_refill_keeps_six_then_shrinks():
    col = ComboColumn(_rng())
    player = _player(coins=100)
    picked = []
    # 8 races left in the stack -> 8 picks with a full column
    for _ in range(8):
        picked.append(col.pick(0, player).race)
        assert len(col.visible) == 6, 'the column must stay full while races remain'
    assert col.race_stack == [], 'the race stack should be exhausted now'
    # from here on the column shrinks: no race left to reveal
    for expected in (5, 4, 3, 2, 1, 0):
        picked.append(col.pick(0, player).race)
        assert len(col.visible) == expected
    assert len(picked) == 14
    assert sorted(picked) == sorted(RaceId), 'every race must have been dealt once'
    assert len(col.power_stack) == 20 - 14
    try:
        col.pick(0, player)
    except IndexError:
        pass
    else:
        raise AssertionError('an empty column must not be pickable')


def test_return_race_goes_to_the_bottom():
    col = ComboColumn(_rng())
    player = _player(coins=100)
    returned = col.pick(0, player).race
    before = list(col.race_stack)
    col.return_race(returned)
    assert col.race_stack == before + [returned], 'the banner must go to the bottom'
    # it is the very last race revealed
    revealed = [col.pick(0, player).race for _ in range(len(col.race_stack) + len(col.visible))]
    assert revealed[-1] == returned
    assert col.visible == []


def test_power_stack_reshuffles_the_discard():
    col = ComboColumn(_rng())
    player = _player(coins=100)
    # empty the power stack by hand, discarding three badges as a decline would
    discarded = col.power_stack[:3]
    col.power_stack = []
    for power in discarded:
        col.discard_power(power)
    assert len(col.power_discard) == 3
    col.pick(0, player)                    # needs one new combo -> one new power
    assert len(col.visible) == 6
    assert col.visible[-1].power in discarded
    assert len(col.power_stack) == 2 and col.power_discard == []
    # both stacks empty -> the slot simply stays empty
    col.power_stack = []
    col.pick(0, player)
    col.pick(0, player)
    assert len(col.visible) == 4, 'no power left: the column cannot refill'
    assert col.race_stack, 'races were still available'


def test_combo_redeterminize():
    col = ComboColumn(_rng(1))
    player = _player(coins=100)
    col.pick(2, player)                                  # coins on combos 0..1
    col.discard_power(col.power_stack.pop(0))
    col.discard_power(col.power_stack.pop(0))
    visible_before = [(c.race, c.power, c.coins) for c in col.visible]
    races_before = list(col.race_stack)
    powers_before = list(col.power_stack)
    discard_before = list(col.power_discard)

    col.redeterminize(_rng(999))

    assert [(c.race, c.power, c.coins) for c in col.visible] == visible_before, \
        'the visible combos are public: redeterminize must not touch them'
    assert Counter(col.race_stack) == Counter(races_before)
    assert Counter(col.power_stack) == Counter(powers_before)
    assert Counter(col.power_discard) == Counter(discard_before)
    assert len(col.race_stack) == len(races_before)
    assert col.power_stack != powers_before or col.race_stack != races_before, \
        'redeterminize must reorder the hidden cards'


# --------------------------------------------------------------------------- #
# Dice
# --------------------------------------------------------------------------- #

def test_dice_faces_distribution():
    n = 6000
    dice = Dice(_rng(7), n=n)
    assert dice.remaining == n
    rolls = [dice.next() for _ in range(n)]
    assert dice.remaining == 0
    counts = Counter(rolls)
    assert set(counts) == {0, 1, 2, 3}, f'unexpected faces: {sorted(counts)}'
    assert all(isinstance(v, int) for v in rolls), 'faces must be plain ints'
    # 0 appears on three of the six faces
    assert 0.46 < counts[0] / n < 0.54, counts
    for face in (1, 2, 3):
        assert 0.13 < counts[face] / n < 0.20, counts
    assert sum(counts.values()) == n


def test_dice_peek_and_extension():
    dice = Dice(_rng(3), n=4)
    assert dice.remaining == 4
    first = dice.peek()
    assert dice.peek() == first and dice.remaining == 4, 'peek must not consume'
    assert dice.next() == first and dice.remaining == 3
    for _ in range(3):
        dice.next()
    assert dice.remaining == 0
    value = dice.next()                     # must extend the sequence
    assert value in DIE_FACES
    assert dice.remaining == 3, 'the sequence must be extended by the initial chunk'


def test_dice_redeterminize():
    dice = Dice(_rng(11), n=50)
    rolled = [dice.next() for _ in range(10)]
    remaining_before = dice.remaining
    hidden_before = list(dice._values[dice._pos:])

    dice.redeterminize(_rng(4242))

    assert dice.remaining == remaining_before == 40
    assert Counter(dice._values[dice._pos:]) == Counter(hidden_before), \
        'redeterminize must keep the multiset of the unused values'
    assert dice._values[:dice._pos] == rolled, 'the rolled values are history'
    assert dice._values[dice._pos:] != hidden_before, 'the unused values must be reordered'
    # and the dice keeps working afterwards
    assert all(dice.next() in DIE_FACES for _ in range(40))
    assert dice.remaining == 0


def test_dice_is_reproducible():
    assert [Dice(_rng(5), n=3).next() for _ in range(1)] == [Dice(_rng(5), n=3).next()]
    a = Dice(_rng(5), n=20)
    b = Dice(_rng(5), n=20)
    assert [a.next() for _ in range(20)] == [b.next() for _ in range(20)]


# --------------------------------------------------------------------------- #
# Deep copy (the MCTS trainer deep-copies the whole env)
# --------------------------------------------------------------------------- #

def _sample_state():
    """A small but complete state: board, player, tray, combo column, dice."""
    rng = _rng(2024)
    board = Board(3)
    board.setup()
    player = _player()
    player.active = RaceInPlay(RaceId.TROLLS, PowerId.FORTIFIED, owner=0, tokens_in_hand=4)
    player.declined = [RaceInPlay(RaceId.GHOULS, None, owner=0, in_decline=True)]
    region = board.by_id(4)
    region.owner, region.race, region.tokens = 0, RaceId.TROLLS, 3
    region.lair = True
    tray = Tray()
    tray.take(RaceId.TROLLS, 7)
    tray.take_marker('fortress')
    return board, player, tray, ComboColumn(rng), Dice(rng, n=20)


def test_deepcopy_independence():
    board, player, tray, col, dice = _sample_state()
    state = (board, player, tray, col, dice)
    clone = copy.deepcopy(state)
    c_board, c_player, c_tray, c_col, c_dice = clone

    # nothing is shared
    assert c_board is not board and c_board.regions[0] is not board.regions[0]
    assert c_player.active is not player.active
    assert c_col.visible[0] is not col.visible[0]

    # mutating the copy leaves the original alone
    c_board.by_id(4).tokens = 9
    c_board.by_id(4).lair = False
    c_board.by_id(9).owner, c_board.by_id(9).race = 1, RaceId.ELVES
    c_player.coins = 42
    c_player.active.tokens_in_hand = 0
    c_player.active.conquests_this_turn.append(4)
    c_player.active.sorcerer_used_on.add(1)
    c_player.active.extra['k'] = 'v'
    c_player.declined.append(RaceInPlay(RaceId.ELVES, None, 0, in_decline=True))
    c_tray.take(RaceId.TROLLS, 3)
    c_tray.take_marker('hero')
    c_col.pick(0, c_player)
    c_col.discard_power(PowerId.STOUT)
    c_dice.next()

    assert board.by_id(4).tokens == 3 and board.by_id(4).lair
    assert board.by_id(9).owner is None and board.by_id(9).race is None
    assert player.coins == START_COINS
    assert player.active.tokens_in_hand == 4
    assert player.active.conquests_this_turn == []
    assert player.active.sorcerer_used_on == set()
    assert player.active.extra == {}
    assert len(player.declined) == 1
    assert tray.available(RaceId.TROLLS) == 3
    assert tray.available_marker('hero') == N_HEROES
    assert len(col.visible) == 6 and col.power_discard == []
    assert dice.remaining == 20
    # ... and the original still works
    col.pick(1, player)
    assert player.coins == START_COINS - 1
    assert board.by_id(4).base_conquest_cost() == 2 + 3 + 1          # tokens + lair
    board.by_id(4).fortress = True
    assert board.by_id(4).base_conquest_cost() == 2 + 3 + 1 + 1      # + fortress


def test_deepcopy_keeps_the_generator_usable():
    _, _, _, col, dice = _sample_state()
    clone_col = copy.deepcopy(col)
    clone_dice = copy.deepcopy(dice)
    # a deep copy gets its own generator with the same state: same future rolls
    assert [dice.next() for _ in range(20)] == [clone_dice.next() for _ in range(20)]
    # ... and both can still extend / refill on their own
    assert dice.next() in DIE_FACES and clone_dice.next() in DIE_FACES
    player_a, player_b = _player(), _player()
    clone_col.power_stack = []
    clone_col.discard_power(PowerId.WEALTHY)
    clone_col.pick(0, player_a)
    col.pick(0, player_b)
    assert len(clone_col.visible) == 6 and len(col.visible) == 6


def test_region_static_is_shared_map_data():
    """A region's static def carries the map data; ids stay stable on a copy."""
    board = Board(3)
    assert board.by_id(7).static is map3p.MAP3P[6]
    clone = copy.deepcopy(board)
    assert [r.id for r in clone.regions] == [r.id for r in board.regions]
    assert clone.by_id(7).terrain == board.by_id(7).terrain
    assert clone.by_id(7).symbols == board.by_id(7).symbols
    assert set(clone.by_id(7).adjacent) == set(board.by_id(7).adjacent)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def run_all() -> bool:
    """Run every `test_*` function of this module; print a summary.

    Returns True if everything passed.
    """
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception:                                   # noqa: BLE001
            failed.append(name)
            print(f'FAIL {name}')
            print(''.join('    ' + line for line in
                          traceback.format_exc().splitlines(keepends=True)))
        else:
            print(f'ok   {name}')
    print(f'\n{__name__}: {len(tests) - len(failed)}/{len(tests)} passed'
          + (f', FAILED: {", ".join(failed)}' if failed else ''))
    return not failed


if __name__ == '__main__':
    import sys
    sys.exit(0 if run_all() else 1)
