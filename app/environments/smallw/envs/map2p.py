"""Static map data of the 2-player Small World board (`smallw2`).

Transcribed from a photo of the printed board (`private/sw2.jpg`, not in the
repository). The board prints no region numbers: regions are numbered here in
reading order (top to bottom, left to right) and drawn with those numbers by
`draw_assets.py`. Coordinates are pixels of the 592x315 board frame
(the photo scaled by 1/5).

Run `python -m environments.smallw.envs.map2p` (from `app/`) to self-check
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

#: 23 regions: 4 farmland, 4 forest, 4 hill, 4 swamp, 4 mountain,
#: 3 water (2 seas + 1 lake). Symbols: 4 mines, 4 magic sources, 4 caverns,
#: 9 lost tribes. 51 undirected adjacency edges.
MAP2P: list[RegionDef] = [
    _region(1,  _WA, (),             True,  (66, 46),  (2, 6)),
    _region(2,  _FA, (_MA,),         True,  (200, 48), (1, 3, 6, 7)),
    _region(3,  _FO, (_MI,),         True,  (280, 46), (2, 4, 7, 8, 12)),
    _region(4,  _SW, (_CA, _LT),     True,  (360, 40), (3, 5, 8, 9)),
    _region(5,  _HI, (),             True,  (500, 34), (4, 9, 10)),
    _region(6,  _MO, (_MI, _CA),     True,  (76, 112), (1, 2, 7, 11)),
    _region(7,  _HI, (_LT,),         False, (210, 112), (2, 3, 6, 11, 12, 15)),
    _region(8,  _MO, (),             False, (360, 100), (3, 4, 9, 12, 13, 16)),
    _region(9,  _FA, (),             False, (430, 90), (4, 5, 8, 10, 13)),
    _region(10, _FO, (_MA, _LT),     True,  (530, 100), (5, 9, 13, 14)),
    _region(11, _FA, (_LT,),         True,  (100, 176), (6, 7, 15, 18, 19)),
    _region(12, _WA, (),             False, (296, 156), (3, 7, 8, 15, 16)),
    _region(13, _HI, (_CA, _LT),     False, (430, 170), (8, 9, 10, 14, 16, 17, 22)),
    _region(14, _MO, (_MI,),         True,  (550, 176), (10, 13, 17, 23)),
    _region(15, _FO, (_LT,),         False, (230, 216), (7, 11, 12, 16, 19, 20)),
    _region(16, _FA, (_MA, _LT),     False, (350, 220), (8, 12, 13, 15, 20, 21, 22)),
    _region(17, _FO, (),             True,  (496, 220), (13, 14, 22, 23)),
    _region(18, _SW, (_MA, _LT),     True,  (90, 250), (11, 19)),
    _region(19, _HI, (_CA,),         True,  (190, 250), (11, 15, 18, 20)),
    _region(20, _SW, (_MI, _LT),     True,  (280, 270), (15, 16, 19, 21)),
    _region(21, _MO, (),             True,  (360, 284), (16, 20, 22)),
    _region(22, _SW, (),             True,  (430, 260), (13, 16, 17, 21, 23)),
    _region(23, _WA, (),             True,  (540, 280), (14, 17, 22)),
]

#: Pixel anchors of the 10 crown icons of the turn track (turn 1 first).
TURN_TRACK: tuple[tuple[int, int], ...] = (
    (20, 142),  # turn 1
    (20, 164),  # turn 2
    (20, 187),  # turn 3
    (20, 210),  # turn 4
    (20, 233),  # turn 5
    (20, 255),  # turn 6
    (20, 278),  # turn 7
    (20, 300),  # turn 8
    (59, 300),  # turn 9
    (96, 300),  # turn 10
)

MAP_DEF = MapDef.build(
    n_players=2,
    regions=MAP2P,
    board_image='board2p.svg',
    board_size=(592, 315),
    turn_track=TURN_TRACK,
    turns=10,
    water_edge_ids=frozenset({1, 23}),
)

# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #

_EXPECTED_TERRAIN_COUNTS = {
    Terrain.FARMLAND: 4,
    Terrain.FOREST: 4,
    Terrain.HILL: 4,
    Terrain.SWAMP: 4,
    Terrain.MOUNTAIN: 4,
    Terrain.WATER: 3,
}

_EXPECTED_SYMBOL_COUNTS = {
    Symbol.MINE: 4,
    Symbol.MAGIC: 4,
    Symbol.CAVERN: 4,
    Symbol.LOST_TRIBE: 9,
}

_EXPECTED_N_EDGES = 51

#: Regions touching the board edge or the shore of a sea touching it.
_EXPECTED_BORDER_IDS = frozenset({
    1, 2, 3, 4, 5, 6, 10, 11, 14, 17, 18, 19, 20, 21, 22, 23,
})

#: The two seas and the lake.
_EXPECTED_WATER_IDS = frozenset({1, 12, 23})


def self_check() -> bool:
    """Validate the 2-player map table (see `MapDef.check`) and its counts."""
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
    print('map2p OK')
