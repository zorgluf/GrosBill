"""Static map data of the 3-player Small World board (`smallw`).

Transcribed from `docs/SWPBF/base3p.png` (597x297 px, regions numbered 1..30 as
printed on the board). The shared types (`Terrain`, `Symbol`, `RegionDef`,
`MapDef`) live in `mapdef.py` and are re-exported here; the other player counts
have their own `map<N>p.py` and `maps.py` registers them all.

Run `python -m environments.smallw.envs.map3p` (from `app/`) to self-check the
table.
"""

from __future__ import annotations

from .mapdef import MapDef, RegionDef, Symbol, Terrain, region as _region

__all__ = [
    'Terrain', 'Symbol', 'RegionDef', 'MAP3P', 'MAP_DEF', 'N_REGIONS',
    'BOARD_IMAGE', 'BOARD_SIZE', 'TURN_TRACK', 'ADJACENCY', 'COASTAL_IDS',
    'WATER_EDGE_IDS', 'BORDER_IDS', 'self_check',
]

# Shorthands used by the table only.
_FA, _FO, _HI, _SW, _MO, _WA = (
    Terrain.FARMLAND,
    Terrain.FOREST,
    Terrain.HILL,
    Terrain.SWAMP,
    Terrain.MOUNTAIN,
    Terrain.WATER,
)
_MI, _MA, _CA, _LT = Symbol.MINE, Symbol.MAGIC, Symbol.CAVERN, Symbol.LOST_TRIBE


# --------------------------------------------------------------------------- #
# The 3-player map (docs/SWPBF/base3p.png)
# --------------------------------------------------------------------------- #

#: Terrain counts: 5 farmland, 5 forest, 5 hill, 5 swamp, 7 mountain,
#: 3 water (2 seas + 1 lake). Symbols: 5 mines, 5 magic sources, 5 caverns,
#: 10 lost tribes. 71 undirected adjacency edges.
#:
#: Documented assumptions (visual analysis of the board scan, see the plan):
#: region 21 does NOT touch the board edge (confirmed by the user, it is not a
#: border region); the short contacts 8-9, 5-11, 10-14 and 22-29 are kept; the
#: dashed line north of regions 28/29 is treated as a normal border; regions
#: 27 and 28 share a border (user ruling 2026-09-24).
MAP3P: list[RegionDef] = [
    _region(1,  _WA, (),          True,  (28, 60),   (2, 8, 13)),
    _region(2,  _FO, (_MI,),      True,  (135, 38),  (1, 3, 8, 9)),
    _region(3,  _MO, (),          True,  (218, 30),  (2, 4, 9, 10)),
    _region(4,  _FA, (),          True,  (300, 28),  (3, 5, 10, 11)),
    _region(5,  _SW, (_CA,),      True,  (395, 38),  (4, 6, 11, 12)),
    _region(6,  _FO, (_LT,),      True,  (485, 40),  (5, 7, 12)),
    _region(7,  _SW, (_MI,),      True,  (555, 80),  (6, 12, 17, 18)),
    _region(8,  _FA, (_MA, _LT),  True,  (72, 80),   (1, 2, 9, 13)),
    _region(9,  _SW, (),          False, (185, 80),  (2, 3, 8, 10, 13, 14)),
    _region(10, _HI, (_MA, _LT),  False, (255, 95),  (3, 4, 9, 11, 14, 15)),
    _region(11, _MO, (_MI,),      False, (340, 80),  (4, 5, 10, 12, 15, 16)),
    _region(12, _FA, (_LT,),      False, (455, 80),  (5, 6, 7, 11, 16, 17)),
    _region(13, _MO, (_MI, _CA),  True,  (60, 150),  (1, 8, 9, 14, 19)),
    _region(14, _SW, (_LT,),      False, (205, 150), (9, 10, 13, 15, 19, 20)),
    _region(15, _WA, (),          False, (300, 170), (10, 11, 14, 16, 20, 21, 22)),
    _region(16, _MO, (_CA,),      False, (365, 145), (11, 12, 15, 17, 22, 23)),
    _region(17, _HI, (_MA,),      False, (475, 140), (7, 12, 16, 18, 23, 24)),
    _region(18, _MO, (),          True,  (560, 140), (7, 17, 24, 30)),
    _region(19, _FA, (_MA,),      True,  (95, 215),  (13, 14, 20, 25, 26)),
    _region(20, _FO, (_CA, _LT),  False, (235, 220), (14, 15, 19, 21, 26, 27)),
    _region(21, _HI, (),          False, (335, 250), (15, 20, 22, 27, 28)),
    _region(22, _SW, (_MA, _LT),  False, (380, 215), (15, 16, 21, 23, 28, 29)),
    _region(23, _FA, (_LT,),      False, (450, 195), (16, 17, 22, 24, 29)),
    _region(24, _HI, (_CA, _LT),  True,  (525, 205), (17, 18, 23, 29, 30)),
    _region(25, _HI, (),          True,  (110, 255), (19, 26)),
    _region(26, _MO, (),          True,  (175, 268), (19, 20, 25, 27)),
    _region(27, _MO, (),          True,  (240, 270), (20, 21, 26, 28)),
    _region(28, _FO, (),          True,  (380, 272), (21, 22, 27, 29)),
    _region(29, _FO, (_MI, _LT),  True,  (450, 265), (22, 23, 24, 28, 30)),
    _region(30, _WA, (),          True,  (565, 250), (18, 24, 29)),
]

#: Number of regions of the 3-player map.
N_REGIONS = 30

#: Board image file name inside `static/`, and its pixel size.
BOARD_IMAGE = 'board3p.svg'
BOARD_SIZE = (597, 297)

#: Pixel anchors of the 10 crown icons of the turn track (turn 1 first).
#: Turns 1-7 run down the left edge, turns 8-10 along the bottom edge.
TURN_TRACK: tuple[tuple[int, int], ...] = (
    (22, 130),   # turn 1
    (22, 152),   # turn 2
    (22, 175),   # turn 3
    (22, 197),   # turn 4
    (22, 220),   # turn 5
    (22, 243),   # turn 6
    (22, 266),   # turn 7
    (22, 289),   # turn 8
    (62, 289),   # turn 9
    (100, 289),  # turn 10
)



MAP_DEF = MapDef.build(
    n_players=3,
    regions=MAP3P,
    board_image=BOARD_IMAGE,
    board_size=BOARD_SIZE,
    turn_track=TURN_TRACK,
    turns=10,
    water_edge_ids=frozenset({1, 30}),
)

#: Symmetric adjacency of the 3-player map, keyed by region id.
ADJACENCY: dict[int, frozenset[int]] = dict(MAP_DEF.adjacency)

#: Regions adjacent to a sea or to the lake (Tritons' -1 cost). Water regions
#: themselves are excluded.
COASTAL_IDS: frozenset[int] = MAP_DEF.coastal_ids

#: Seas touching the edge of the board: a region on their shore may host a
#: first conquest (already reflected in the `border` flags).
WATER_EDGE_IDS: frozenset[int] = MAP_DEF.water_edge_ids


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #

#: The 18 border regions of the 3-player map (17 touching the board edge plus
#: regions 8 and 24 whose shore is on an edge sea; region 21 is interior).
BORDER_IDS: frozenset[int] = frozenset(
    {1, 2, 3, 4, 5, 6, 7, 8, 13, 18, 19, 24, 25, 26, 27, 28, 29, 30}
)

_EXPECTED_TERRAIN_COUNTS = {
    Terrain.FARMLAND: 5,
    Terrain.FOREST: 5,
    Terrain.HILL: 5,
    Terrain.SWAMP: 5,
    Terrain.MOUNTAIN: 7,
    Terrain.WATER: 3,
}

_EXPECTED_SYMBOL_COUNTS = {
    Symbol.MINE: 5,
    Symbol.MAGIC: 5,
    Symbol.CAVERN: 5,
    Symbol.LOST_TRIBE: 10,
}

_EXPECTED_N_EDGES = 71


def self_check() -> bool:
    """Validate the 3-player map table (see `MapDef.check`) and its counts."""
    assert len(MAP3P) == N_REGIONS, f'expected {N_REGIONS} regions, got {len(MAP3P)}'
    MAP_DEF.check(
        terrain_counts=_EXPECTED_TERRAIN_COUNTS,
        symbol_counts=_EXPECTED_SYMBOL_COUNTS,
        n_edges=_EXPECTED_N_EDGES,
        border_ids=BORDER_IDS,
        water_ids=frozenset({1, 15, 30}),
    )
    return True


if __name__ == '__main__':
    self_check()
    print('map3p OK')
