"""Static map data of the 4-player Small World board (`smallw4`).

Transcribed from a photo of the printed board (`private/sw4.jpg`, not in the
repository). The board prints no region numbers: regions are numbered here in
reading order (top to bottom, left to right) and drawn with those numbers by
`draw_assets.py`. Coordinates are pixels of the 520x523 board frame
(the photo scaled by 1/4.5).

Run `python -m environments.smallw.envs.map4p` (from `app/`) to self-check
the table.
"""

from __future__ import annotations

from .mapdef import MapDef, RegionDef, Symbol, Terrain, region as _region

_FA, _FO, _HI, _SW, _MO, _WA = (
    Terrain.FARMLAND,
    Terrain.FOREST,
    Terrain.HILL,
    Terrain.SWAMP,
    Terrain.MOUNTAIN,
    Terrain.WATER,
)
_MI, _MA, _CA, _LT = Symbol.MINE, Symbol.MAGIC, Symbol.CAVERN, Symbol.LOST_TRIBE

#: 39 regions: 7 farmland, 7 forest, 7 hill, 7 swamp, 8 mountain,
#: 3 water (2 seas + 1 lake). Symbols: 7 mines, 7 magic sources, 7 caverns,
#: 14 lost tribes. 95 undirected adjacency edges.
MAP4P: list[RegionDef] = [
    _region(1,  _FA, (),             True,  (84, 71),  (2, 7)),
    _region(2,  _WA, (),             True,  (133, 33), (1, 3, 4, 7, 8, 14, 15)),
    _region(3,  _FA, (_MI, _LT),     True,  (204, 93), (2, 4, 5, 8, 9)),
    _region(4,  _FO, (_CA, _LT),     True,  (256, 40), (2, 3, 5)),
    _region(5,  _HI, (_LT,),         True,  (322, 49), (3, 4, 6, 9, 10)),
    _region(6,  _SW, (_MA,),         True,  (433, 67), (5, 10, 12)),
    _region(7,  _SW, (_MA, _LT),     True,  (93, 144), (1, 2, 8, 15)),
    _region(8,  _HI, (_CA, _LT),     True,  (160, 147), (2, 3, 7, 9, 15, 17)),
    _region(9,  _FA, (),             False, (240, 173), (3, 5, 8, 10, 17, 19, 20)),
    _region(10, _MO, (_MI,),         False, (316, 144), (5, 6, 9, 11, 12, 20)),
    _region(11, _FO, (_MA, _LT),     False, (373, 182), (10, 12, 13, 20, 21)),
    _region(12, _FO, (_CA,),         True,  (467, 116), (6, 10, 11, 13)),
    _region(13, _MO, (),             True,  (467, 189), (11, 12, 21, 22)),
    _region(14, _MO, (),             True,  (56, 267), (2, 15, 16, 23)),
    _region(15, _FO, (),             True,  (116, 211), (2, 7, 8, 14, 16, 17)),
    _region(16, _HI, (),             False, (133, 267), (14, 15, 17, 18, 23, 24)),
    _region(17, _SW, (_LT,),         False, (189, 222), (8, 9, 15, 16, 18, 19)),
    _region(18, _MO, (_MI, _CA),     False, (204, 289), (16, 17, 19, 20, 24, 25)),
    _region(19, _HI, (_MA,),         False, (251, 233), (9, 17, 18, 20)),
    _region(20, _WA, (),             False, (307, 256), (9, 10, 11, 18, 19, 21, 25, 26)),
    _region(21, _SW, (_MI,),         False, (378, 278), (11, 13, 20, 22, 26, 27)),
    _region(22, _FA, (),             True,  (478, 280), (13, 21, 27, 29)),
    _region(23, _SW, (_MI, _LT),     True,  (78, 338), (14, 16, 24, 30)),
    _region(24, _FO, (_MA, _LT),     False, (167, 356), (16, 18, 23, 25, 30, 31, 33)),
    _region(25, _MO, (),             False, (256, 367), (18, 20, 24, 26, 33)),
    _region(26, _FA, (_LT,),         False, (344, 344), (20, 21, 25, 27, 33, 35, 36, 37)),
    _region(27, _HI, (_LT,),         False, (433, 322), (21, 22, 26, 28, 29, 37)),
    _region(28, _HI, (_MA,),         True,  (467, 389), (27, 29, 37, 38, 39)),
    _region(29, _SW, (_CA, _LT),     True,  (500, 322), (22, 27, 28)),
    _region(30, _FA, (),             True,  (56, 433), (23, 24, 31)),
    _region(31, _MO, (),             True,  (144, 433), (24, 30, 32, 33)),
    _region(32, _HI, (_MI,),         True,  (178, 489), (31, 33, 34)),
    _region(33, _SW, (_CA, _LT),     False, (244, 422), (24, 25, 26, 31, 32, 34, 35)),
    _region(34, _FA, (),             True,  (278, 500), (32, 33, 35, 36)),
    _region(35, _FO, (_MA, _LT),     False, (300, 444), (26, 33, 34, 36)),
    _region(36, _MO, (_MI,),         True,  (367, 467), (26, 34, 35, 37, 38)),
    _region(37, _MO, (_CA,),         False, (400, 400), (26, 27, 28, 36, 38)),
    _region(38, _FO, (),             True,  (440, 478), (28, 36, 37, 39)),
    _region(39, _WA, (),             True,  (500, 489), (28, 38)),
]

#: Pixel anchors of the 9 crown icons of the turn track (turn 1 first).
TURN_TRACK: tuple[tuple[int, int], ...] = (
    (14, 12),   # turn 1
    (14, 33),   # turn 2
    (14, 51),   # turn 3
    (14, 70),   # turn 4
    (14, 90),   # turn 5
    (14, 110),  # turn 6
    (14, 129),  # turn 7
    (14, 148),  # turn 8
    (14, 168),  # turn 9
)

MAP_DEF = MapDef.build(
    n_players=4,
    regions=MAP4P,
    board_image='board4p.svg',
    board_size=(520, 523),
    turn_track=TURN_TRACK,
    turns=9,
    water_edge_ids=frozenset({2, 39}),
)

# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #

_EXPECTED_TERRAIN_COUNTS = {
    Terrain.FARMLAND: 7,
    Terrain.FOREST: 7,
    Terrain.HILL: 7,
    Terrain.SWAMP: 7,
    Terrain.MOUNTAIN: 8,
    Terrain.WATER: 3,
}

_EXPECTED_SYMBOL_COUNTS = {
    Symbol.MINE: 7,
    Symbol.MAGIC: 7,
    Symbol.CAVERN: 7,
    Symbol.LOST_TRIBE: 14,
}

_EXPECTED_N_EDGES = 95

#: Regions touching the board edge or the shore of a sea touching it.
_EXPECTED_BORDER_IDS = frozenset({
    1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 22, 23, 28, 29,
    30, 31, 32, 34, 36, 38, 39,
})

#: The two seas and the lake.
_EXPECTED_WATER_IDS = frozenset({2, 20, 39})


def self_check() -> bool:
    """Validate the 4-player map table (see `MapDef.check`) and its counts."""
    MAP_DEF.check(
        terrain_counts=_EXPECTED_TERRAIN_COUNTS,
        symbol_counts=_EXPECTED_SYMBOL_COUNTS,
        n_edges=_EXPECTED_N_EDGES,
        border_ids=_EXPECTED_BORDER_IDS,
        water_ids=_EXPECTED_WATER_IDS,
    )
    return True


if __name__ == '__main__':
    self_check()
    print('map4p OK')
