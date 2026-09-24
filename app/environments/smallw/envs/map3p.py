"""Static map data for Small World (`smallw`).

This module holds **only static data**: the terrain / symbols / adjacency of the
printed board, the pixel anchors used by the web renderer, and the turn table.
No game state, no gym dependency — so it can be imported from `classes.py`,
`smallw.py` and `render_web.py` without any cycle.

The `Terrain` and `Symbol` enums live here (and are re-exported by `classes.py`)
because both the state classes and the map table need them.

Currently only the **3-player** map is transcribed (`MAP3P`, from
`docs/SWPBF/base3p.png`, 597x297 px, regions numbered 1..30 as printed on the
board). `map_for(n_players)` raises `NotImplementedError` for the other player
counts so that adding the 2/4/5-player boards later is a pure data change.

Run `python -m environments.smallw.envs.map3p` (from `app/`) to self-check the
table.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #

class Terrain(IntEnum):
    """Terrain type of a region.

    The printed board distinguishes seas from the lake, but the rules treat them
    identically (they cannot be conquered without Seafaring, and they make the
    neighbouring regions *coastal*), so both map to :attr:`WATER`.
    """

    FARMLAND = 0
    FOREST = 1
    HILL = 2
    SWAMP = 3
    MOUNTAIN = 4
    WATER = 5


class Symbol(IntEnum):
    """Static symbol printed in a region.

    `LOST_TRIBE` is the *printed* symbol (it says where a Lost Tribe token is
    placed during setup); whether a Lost Tribe token is still there at a given
    moment is dynamic state, held by `Region.lost_tribe` in `classes.py`.
    """

    MINE = 0
    MAGIC = 1
    CAVERN = 2
    LOST_TRIBE = 3


# --------------------------------------------------------------------------- #
# Region definition
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RegionDef:
    """Immutable description of one printed region.

    Attributes:
        id: number printed on the board, 1..30.
        index: 0-based index into the map list / observation rows (``id - 1``).
        terrain: :class:`Terrain` of the region.
        symbols: frozenset of :class:`Symbol` printed in the region.
        border: True if the region may host a *first conquest* (it touches the
            edge of the board, or its shore is on a sea touching the edge).
        anchor: (x, y) pixel coordinates of a representative point of the region
            on the ``597x297`` board image — used by the renderer to draw the
            tokens and to map a click to a region.
        adjacent: tuple of region **ids** sharing a border with this region.
    """

    id: int
    index: int
    terrain: Terrain
    symbols: frozenset[Symbol]
    border: bool
    anchor: tuple[int, int]
    adjacent: tuple[int, ...]

    # -- derived helpers ---------------------------------------------------- #

    @property
    def is_water(self) -> bool:
        """True for the two seas and the lake (not conquerable without Seafaring)."""
        return self.terrain == Terrain.WATER

    @property
    def is_mountain(self) -> bool:
        """True for mountain regions (they carry an immovable Mountain token)."""
        return self.terrain == Terrain.MOUNTAIN

    def has(self, symbol: Symbol) -> bool:
        """True if `symbol` is printed in this region."""
        return symbol in self.symbols


def _region(
    rid: int,
    terrain: Terrain,
    symbols: tuple[Symbol, ...],
    border: bool,
    anchor: tuple[int, int],
    adjacent: tuple[int, ...],
) -> RegionDef:
    """Small helper keeping the table below readable."""
    return RegionDef(
        id=rid,
        index=rid - 1,
        terrain=terrain,
        symbols=frozenset(symbols),
        border=border,
        anchor=anchor,
        adjacent=adjacent,
    )


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
BOARD_IMAGE = 'board3p.png'
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


# --------------------------------------------------------------------------- #
# Derived tables
# --------------------------------------------------------------------------- #

def _build_adjacency(regions: list[RegionDef]) -> dict[int, frozenset[int]]:
    """Return the adjacency as a dict id -> frozenset(ids), symmetrised."""
    adj: dict[int, set[int]] = {r.id: set(r.adjacent) for r in regions}
    for rid, neighbours in list(adj.items()):
        for other in neighbours:
            adj.setdefault(other, set()).add(rid)
    return {rid: frozenset(neighbours) for rid, neighbours in adj.items()}


def _build_coastal(regions: list[RegionDef],
                   adj: dict[int, frozenset[int]]) -> frozenset[int]:
    """Ids of the (non-water) regions adjacent to a sea or the lake — Tritons."""
    water = {r.id for r in regions if r.is_water}
    return frozenset(
        r.id for r in regions
        if not r.is_water and (adj[r.id] & water)
    )


#: Symmetric adjacency of the 3-player map, keyed by region id.
ADJACENCY: dict[int, frozenset[int]] = _build_adjacency(MAP3P)

#: Regions adjacent to a sea or to the lake (Tritons' -1 cost). Water regions
#: themselves are excluded.
COASTAL_IDS: frozenset[int] = _build_coastal(MAP3P, ADJACENCY)

#: Seas touching the edge of the board: a region on their shore may host a
#: first conquest (already reflected in the `border` flags).
WATER_EDGE_IDS: frozenset[int] = frozenset({1, 30})

#: All the maps, keyed by player count. Only 3 players for now.
MAPS: dict[int, list[RegionDef]] = {3: MAP3P}

#: Number of game turns per player count (the 3-player turn track has 10 spaces).
TURNS_PER_PLAYER_COUNT: dict[int, int] = {2: 10, 3: 10, 4: 9, 5: 8}


def map_for(n_players: int) -> list[RegionDef]:
    """Return the region table for `n_players`.

    Raises:
        NotImplementedError: for any player count whose board has not been
            transcribed yet (everything but 3).
    """
    try:
        return MAPS[n_players]
    except KeyError:
        raise NotImplementedError(
            f"no Small World map transcribed for {n_players} players; "
            f"available: {sorted(MAPS)}. Add the board image to "
            f"environments/smallw/static/ and a MAP<N>P table in map3p.py, "
            f"then register it in MAPS."
        ) from None


def turns_for(n_players: int) -> int:
    """Number of game turns for `n_players` (2p/3p: 10, 4p: 9, 5p: 8)."""
    try:
        return TURNS_PER_PLAYER_COUNT[n_players]
    except KeyError:
        raise NotImplementedError(
            f"unsupported player count {n_players}; "
            f"available: {sorted(TURNS_PER_PLAYER_COUNT)}"
        ) from None


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
    """Validate the 3-player map table.

    Checks the region ids and their order, the adjacency (symmetric, irreflexive,
    71 undirected edges, ids in range), the terrain and symbol counts, that no
    water region carries a symbol, that every anchor lies inside the board and
    that the border flags match :data:`BORDER_IDS`.

    Returns:
        True if everything holds.

    Raises:
        AssertionError: on the first inconsistency found.
    """
    regions = MAP3P

    # -- ids / order -------------------------------------------------------- #
    assert len(regions) == N_REGIONS, f"expected {N_REGIONS} regions, got {len(regions)}"
    for i, r in enumerate(regions):
        assert r.id == i + 1, f"regions must be sorted by id: MAP3P[{i}].id == {r.id}"
        assert r.index == i, f"region {r.id}: index {r.index} != {i}"
    assert {r.id for r in regions} == set(range(1, N_REGIONS + 1))

    # -- adjacency ---------------------------------------------------------- #
    assert set(ADJACENCY) == {r.id for r in regions}, "ADJACENCY keys != region ids"
    edges: set[tuple[int, int]] = set()
    for r in regions:
        neighbours = ADJACENCY[r.id]
        assert r.id not in neighbours, f"region {r.id} is adjacent to itself"
        assert set(r.adjacent) == set(neighbours), (
            f"region {r.id}: RegionDef.adjacent {sorted(r.adjacent)} != "
            f"ADJACENCY {sorted(neighbours)} (adjacency table not symmetric)"
        )
        assert len(r.adjacent) == len(set(r.adjacent)), (
            f"region {r.id}: duplicate entry in adjacent {r.adjacent}"
        )
        for other in neighbours:
            assert 1 <= other <= N_REGIONS, f"region {r.id}: bad neighbour {other}"
            assert r.id in ADJACENCY[other], (
                f"adjacency not symmetric: {r.id}-{other} present, {other}-{r.id} missing"
            )
            edges.add((min(r.id, other), max(r.id, other)))
    assert len(edges) == _EXPECTED_N_EDGES, (
        f"expected {_EXPECTED_N_EDGES} undirected edges, got {len(edges)}"
    )

    # -- terrain / symbols -------------------------------------------------- #
    for terrain, expected in _EXPECTED_TERRAIN_COUNTS.items():
        got = sum(1 for r in regions if r.terrain == terrain)
        assert got == expected, f"{terrain.name}: expected {expected} regions, got {got}"
    for symbol, expected in _EXPECTED_SYMBOL_COUNTS.items():
        got = sum(1 for r in regions if r.has(symbol))
        assert got == expected, f"{symbol.name}: expected {expected} regions, got {got}"
    for r in regions:
        if r.is_water:
            assert not r.symbols, f"water region {r.id} must have no symbol"

    # -- anchors ------------------------------------------------------------ #
    width, height = BOARD_SIZE
    for r in regions:
        x, y = r.anchor
        assert 0 <= x < width and 0 <= y < height, (
            f"region {r.id}: anchor {r.anchor} outside the board {BOARD_SIZE}"
        )
    for i, (x, y) in enumerate(TURN_TRACK):
        assert 0 <= x < width and 0 <= y < height, (
            f"turn track {i + 1}: anchor {(x, y)} outside the board {BOARD_SIZE}"
        )
    assert len(TURN_TRACK) == TURNS_PER_PLAYER_COUNT[3], (
        f"turn track has {len(TURN_TRACK)} spaces, "
        f"expected {TURNS_PER_PLAYER_COUNT[3]}"
    )

    # -- borders ------------------------------------------------------------ #
    got_borders = frozenset(r.id for r in regions if r.border)
    assert got_borders == BORDER_IDS, (
        f"border regions {sorted(got_borders)} != expected {sorted(BORDER_IDS)}"
    )
    assert len(BORDER_IDS) == 18, f"expected 18 border regions, got {len(BORDER_IDS)}"

    # -- derived tables ----------------------------------------------------- #
    water = {r.id for r in regions if r.is_water}
    assert water == {1, 15, 30}, f"water regions {sorted(water)} != [1, 15, 30]"
    assert WATER_EDGE_IDS <= water, "WATER_EDGE_IDS must be water regions"
    for rid in COASTAL_IDS:
        assert rid not in water, f"coastal region {rid} must not be water"
        assert ADJACENCY[rid] & water, f"region {rid} is not adjacent to any water"
    for r in regions:
        if not r.is_water and (ADJACENCY[r.id] & water):
            assert r.id in COASTAL_IDS, f"region {r.id} is coastal but missing from COASTAL_IDS"

    # -- player counts ------------------------------------------------------ #
    assert map_for(3) is MAP3P
    for n in (2, 4, 5):
        try:
            map_for(n)
        except NotImplementedError:
            pass
        else:  # pragma: no cover - guard against a silent map addition
            raise AssertionError(f"map_for({n}) should raise NotImplementedError")

    return True


if __name__ == '__main__':
    self_check()
    print('map3p OK')
