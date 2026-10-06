"""Static map data of the 5-player Small World board (`smallw5`).

Transcribed from a photo of the printed board (`private/sw5.jpg`, not in the
repository). The board prints no region numbers: regions are numbered here in
reading order (top to bottom, left to right) and drawn with those numbers by
`draw_assets.py`. Coordinates are pixels of the 552x548 board frame
(the photo scaled by 1/4.2).

Run `python -m environments.smallw.envs.map5p` (from `app/`) to self-check
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

#: 48 regions: 9 farmland, 9 forest, 9 hill, 9 swamp, 9 mountain,
#: 3 water (2 seas + 1 lake). Symbols: 9 mines, 9 magic sources, 9 caverns,
#: 18 lost tribes. 119 undirected adjacency edges.
MAP5P: list[RegionDef] = [
    _region(1,  _MO, (),             True,  (100, 21), (2, 6, 23)),
    _region(2,  _SW, (_MA,),         True,  (186, 29), (1, 3, 6, 7)),
    _region(3,  _MO, (_CA,),         True,  (257, 33), (2, 4, 7, 8)),
    _region(4,  _FO, (_MA,),         True,  (321, 48), (3, 5, 8, 9)),
    _region(5,  _FA, (_CA, _LT),     True,  (417, 29), (4, 9, 10)),
    _region(6,  _SW, (),             True,  (133, 90), (1, 2, 7, 11, 23)),
    _region(7,  _FA, (_MI, _LT),     False, (214, 100), (2, 3, 6, 8, 11, 12)),
    _region(8,  _HI, (),             False, (281, 100), (3, 4, 7, 9, 12, 13)),
    _region(9,  _MO, (_MI,),         False, (376, 86), (4, 5, 8, 10, 13, 14)),
    _region(10, _FA, (),             True,  (488, 76), (5, 9, 14, 15)),
    _region(11, _HI, (_LT,),         True,  (171, 167), (6, 7, 12, 17, 18, 23)),
    _region(12, _FA, (_CA, _LT),     False, (262, 171), (7, 8, 11, 13, 18, 19, 26)),
    _region(13, _MO, (),             False, (338, 155), (8, 9, 12, 14, 20, 26)),
    _region(14, _SW, (_CA, _LT),     False, (429, 152), (9, 10, 13, 15, 20, 21)),
    _region(15, _HI, (_MI,),         True,  (512, 148), (10, 14, 21)),
    _region(16, _HI, (_LT,),         True,  (62, 190), (17, 23)),
    _region(17, _FO, (_MA,),         True,  (124, 224), (11, 16, 18, 23, 24)),
    _region(18, _SW, (_LT,),         False, (205, 238), (11, 12, 17, 19, 24, 25)),
    _region(19, _HI, (_MA,),         False, (262, 245), (12, 18, 25, 26)),
    _region(20, _FO, (),             False, (393, 202), (13, 14, 21, 26, 27)),
    _region(21, _MO, (_MA,),         True,  (488, 214), (14, 15, 20, 27, 28)),
    _region(22, _FO, (_MI,),         True,  (24, 310), (23, 29, 34)),
    _region(23, _WA, (),             True,  (60, 267), (1, 6, 11, 16, 17, 22, 24, 29)),
    _region(24, _FA, (_LT,),         True,  (143, 298), (17, 18, 23, 25, 29, 30)),
    _region(25, _MO, (_MI, _CA),     False, (226, 298), (18, 19, 24, 26, 30)),
    _region(26, _WA, (),             False, (321, 262), (12, 13, 19, 20, 25, 27, 30, 31, 37, 38)),
    _region(27, _SW, (_MI, _LT),     False, (393, 274), (20, 21, 26, 28, 31, 32)),
    _region(28, _FA, (),             True,  (500, 286), (21, 27, 32, 33)),
    _region(29, _SW, (),             True,  (83, 340), (22, 23, 24, 30, 34, 35)),
    _region(30, _HI, (_LT,),         False, (179, 369), (24, 25, 26, 29, 35, 36, 37)),
    _region(31, _FO, (_LT,),         False, (405, 345), (26, 27, 32, 38, 39)),
    _region(32, _HI, (),             False, (464, 345), (27, 28, 31, 33, 39, 40)),
    _region(33, _SW, (_CA, _LT),     True,  (529, 345), (28, 32, 40)),
    _region(34, _FA, (_CA, _LT),     True,  (29, 405), (22, 29, 35, 41)),
    _region(35, _FO, (_LT,),         False, (95, 417), (29, 30, 34, 36, 41, 42)),
    _region(36, _FA, (),             False, (179, 440), (30, 35, 37, 42, 43)),
    _region(37, _SW, (_CA, _LT),     False, (257, 452), (26, 30, 36, 38, 43, 44, 45)),
    _region(38, _FA, (_MA,),         False, (333, 400), (26, 31, 37, 39, 45, 46)),
    _region(39, _MO, (_MI,),         False, (429, 429), (31, 32, 38, 40, 46, 47)),
    _region(40, _HI, (_MA, _LT),     True,  (512, 417), (32, 33, 39, 47, 48)),
    _region(41, _SW, (_MI, _LT),     True,  (36, 512), (34, 35, 42)),
    _region(42, _FO, (_MA, _LT),     True,  (119, 512), (35, 36, 41, 43)),
    _region(43, _HI, (_MI,),         True,  (202, 512), (36, 37, 42, 44)),
    _region(44, _MO, (_MA,),         True,  (286, 531), (37, 43, 45, 46)),
    _region(45, _FO, (),             False, (321, 471), (37, 38, 44, 46)),
    _region(46, _MO, (),             True,  (393, 512), (38, 39, 44, 45, 47)),
    _region(47, _FO, (_CA,),         True,  (464, 512), (39, 40, 46, 48)),
    _region(48, _WA, (),             True,  (529, 512), (40, 47)),
]

#: Pixel anchors of the 8 crown icons of the turn track (turn 1 first).
TURN_TRACK: tuple[tuple[int, int], ...] = (
    (19, 13),   # turn 1
    (19, 35),   # turn 2
    (19, 54),   # turn 3
    (19, 74),   # turn 4
    (19, 94),   # turn 5
    (19, 115),  # turn 6
    (19, 135),  # turn 7
    (19, 156),  # turn 8
)

MAP_DEF = MapDef.build(
    n_players=5,
    regions=MAP5P,
    board_image='board5p.svg',
    board_size=(552, 548),
    turn_track=TURN_TRACK,
    turns=8,
    water_edge_ids=frozenset({23, 48}),
)

# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #

_EXPECTED_TERRAIN_COUNTS = {
    Terrain.FARMLAND: 9,
    Terrain.FOREST: 9,
    Terrain.HILL: 9,
    Terrain.SWAMP: 9,
    Terrain.MOUNTAIN: 9,
    Terrain.WATER: 3,
}

_EXPECTED_SYMBOL_COUNTS = {
    Symbol.MINE: 9,
    Symbol.MAGIC: 9,
    Symbol.CAVERN: 9,
    Symbol.LOST_TRIBE: 18,
}

_EXPECTED_N_EDGES = 119

#: Regions touching the board edge or the shore of a sea touching it.
_EXPECTED_BORDER_IDS = frozenset({
    1, 2, 3, 4, 5, 6, 10, 11, 15, 16, 17, 21, 22, 23, 24, 28,
    29, 33, 34, 40, 41, 42, 43, 44, 46, 47, 48,
})

#: The two seas and the lake.
_EXPECTED_WATER_IDS = frozenset({23, 26, 48})


def self_check() -> bool:
    """Validate the 5-player map table (see `MapDef.check`) and its counts."""
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
    print('map5p OK')
