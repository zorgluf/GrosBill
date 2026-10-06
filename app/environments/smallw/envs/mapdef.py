"""Building blocks of the Small World map tables (`smallw`).

This module holds the types shared by the per-player-count map tables
(`map2p.py` ... `map5p.py`): the `Terrain` and `Symbol` enums, the immutable
`RegionDef` of one printed region and `MapDef`, a whole board (regions, turn
count, turn track, board image) plus its derived tables (adjacency, coastal and
border regions). `maps.py` registers one `MapDef` per player count.

Static data only: no game state, no gym dependency — so it can be imported from
`classes.py`, `smallw.py` and `render_web.py` without any cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable, Mapping


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
        id: region number, 1..N (printed on the 3-player board, assigned in
            reading order on the others).
        index: 0-based index into the map list / observation rows (``id - 1``).
        terrain: :class:`Terrain` of the region.
        symbols: frozenset of :class:`Symbol` printed in the region.
        border: True if the region may host a *first conquest* (it touches the
            edge of the board, or its shore is on a sea touching the edge).
        anchor: (x, y) pixel coordinates of a representative point of the region
            on the board image — used by the renderer to draw the tokens and to
            map a click to a region.
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


def region(
    rid: int,
    terrain: Terrain,
    symbols: tuple[Symbol, ...],
    border: bool,
    anchor: tuple[int, int],
    adjacent: tuple[int, ...],
) -> RegionDef:
    """Small helper keeping the map tables readable."""
    return RegionDef(
        id=rid,
        index=rid - 1,
        terrain=terrain,
        symbols=frozenset(symbols),
        border=border,
        anchor=anchor,
        adjacent=adjacent,
    )


# --------------------------------------------------------------------------- #
# Whole board
# --------------------------------------------------------------------------- #

def build_adjacency(regions: Iterable[RegionDef]) -> dict[int, frozenset[int]]:
    """Return the adjacency as a dict id -> frozenset(ids), symmetrised."""
    adj: dict[int, set[int]] = {r.id: set(r.adjacent) for r in regions}
    for rid, neighbours in list(adj.items()):
        for other in neighbours:
            adj.setdefault(other, set()).add(rid)
    return {rid: frozenset(neighbours) for rid, neighbours in adj.items()}


@dataclass(frozen=True, eq=False)
class MapDef:
    """One printed board: its regions and everything derived from them.

    Build it with :meth:`build`. Immutable, and `copy.deepcopy` returns the
    object itself, so deep-copying a game (the MCTS trainer does) never copies
    the map.

    Attributes:
        n_players: player count the board is played with.
        regions: the `RegionDef` table, sorted by id.
        board_image: file name of the board drawing inside `static/`.
        board_size: (width, height) of the board frame, in pixels; anchors and
            the turn track are in this frame.
        turn_track: pixel anchors of the turn track spaces (turn 1 first).
        turns: number of game turns.
        water_edge_ids: seas touching the edge of the board (a region on their
            shore may host a first conquest).
        adjacency: symmetric adjacency, region id -> frozenset of ids.
        coastal_ids: non-water regions adjacent to a sea or the lake (Tritons).
        border_ids: regions that may host a first conquest.
        water_ids: the seas and the lake.
    """

    n_players: int
    regions: tuple[RegionDef, ...]
    board_image: str
    board_size: tuple[int, int]
    turn_track: tuple[tuple[int, int], ...]
    turns: int
    water_edge_ids: frozenset[int]
    adjacency: Mapping[int, frozenset[int]] = field(repr=False)
    coastal_ids: frozenset[int] = field(repr=False)
    border_ids: frozenset[int] = field(repr=False)
    water_ids: frozenset[int] = field(repr=False)

    @classmethod
    def build(cls, *, n_players: int, regions: Iterable[RegionDef], board_image: str,
              board_size: tuple[int, int], turn_track: tuple[tuple[int, int], ...],
              turns: int, water_edge_ids: frozenset[int]) -> 'MapDef':
        """Assemble a board and derive its adjacency / coastal / border tables."""
        regions = tuple(regions)
        adjacency = build_adjacency(regions)
        water = frozenset(r.id for r in regions if r.is_water)
        coastal = frozenset(r.id for r in regions
                            if not r.is_water and adjacency[r.id] & water)
        return cls(
            n_players=n_players,
            regions=regions,
            board_image=board_image,
            board_size=board_size,
            turn_track=tuple(turn_track),
            turns=turns,
            water_edge_ids=frozenset(water_edge_ids),
            adjacency=adjacency,
            coastal_ids=coastal,
            border_ids=frozenset(r.id for r in regions if r.border),
            water_ids=water,
        )

    def __deepcopy__(self, memo) -> 'MapDef':
        return self

    @property
    def n_regions(self) -> int:
        """Number of regions on the board."""
        return len(self.regions)

    def n_edges(self) -> int:
        """Number of undirected adjacency edges."""
        return sum(len(n) for n in self.adjacency.values()) // 2

    def check(self, *, terrain_counts: Mapping[Terrain, int] | None = None,
              symbol_counts: Mapping[Symbol, int] | None = None,
              n_edges: int | None = None, border_ids: frozenset[int] | None = None,
              water_ids: frozenset[int] | None = None) -> None:
        """Validate the table.

        Always checks the region ids and their order, the adjacency (symmetric,
        irreflexive, ids in range, a connected map), that no water region
        carries a symbol, that every anchor and turn-track space lies inside the
        board, that the turn track has one space per turn and that every region
        on the shore of an edge sea is a border region. The keyword arguments
        add the expected counts / sets of a specific board.

        Raises:
            AssertionError: on the first inconsistency found.
        """
        regions, n = self.regions, len(self.regions)
        name = f'{self.n_players}-player map'

        # -- ids / order ---------------------------------------------------- #
        for i, r in enumerate(regions):
            assert r.id == i + 1, f'{name}: regions must be sorted by id: [{i}].id == {r.id}'
            assert r.index == i, f'{name}: region {r.id}: index {r.index} != {i}'

        # -- adjacency ------------------------------------------------------ #
        assert set(self.adjacency) == {r.id for r in regions}, f'{name}: adjacency keys != ids'
        for r in regions:
            neighbours = self.adjacency[r.id]
            assert r.id not in neighbours, f'{name}: region {r.id} is adjacent to itself'
            assert len(r.adjacent) == len(set(r.adjacent)), (
                f'{name}: region {r.id}: duplicate entry in adjacent {r.adjacent}')
            assert set(r.adjacent) == set(neighbours), (
                f'{name}: region {r.id}: adjacent {sorted(r.adjacent)} != '
                f'{sorted(neighbours)} (adjacency table not symmetric)')
            for other in neighbours:
                assert 1 <= other <= n, f'{name}: region {r.id}: bad neighbour {other}'
        seen, todo = {1}, [1]
        while todo:
            for other in self.adjacency[todo.pop()]:
                if other not in seen:
                    seen.add(other)
                    todo.append(other)
        assert len(seen) == n, f'{name}: regions {sorted(set(range(1, n + 1)) - seen)} unreachable'
        if n_edges is not None:
            assert self.n_edges() == n_edges, (
                f'{name}: expected {n_edges} undirected edges, got {self.n_edges()}')

        # -- terrain / symbols ---------------------------------------------- #
        for terrain, expected in (terrain_counts or {}).items():
            got = sum(1 for r in regions if r.terrain == terrain)
            assert got == expected, f'{name}: {terrain.name}: expected {expected} regions, got {got}'
        for symbol, expected in (symbol_counts or {}).items():
            got = sum(1 for r in regions if r.has(symbol))
            assert got == expected, f'{name}: {symbol.name}: expected {expected} regions, got {got}'
        for r in regions:
            if r.is_water:
                assert not r.symbols, f'{name}: water region {r.id} must have no symbol'

        # -- anchors / turn track ------------------------------------------- #
        width, height = self.board_size
        for r in regions:
            x, y = r.anchor
            assert 0 <= x < width and 0 <= y < height, (
                f'{name}: region {r.id}: anchor {r.anchor} outside the board {self.board_size}')
        for i, (x, y) in enumerate(self.turn_track):
            assert 0 <= x < width and 0 <= y < height, (
                f'{name}: turn track {i + 1}: anchor {(x, y)} outside the board')
        assert len(self.turn_track) == self.turns, (
            f'{name}: turn track has {len(self.turn_track)} spaces, expected {self.turns}')

        # -- borders / water ------------------------------------------------ #
        assert self.water_edge_ids <= self.water_ids, f'{name}: edge seas must be water regions'
        for r in regions:
            if not r.is_water and self.adjacency[r.id] & self.water_edge_ids:
                assert r.border, f'{name}: region {r.id} is on the shore of an edge sea'
        for rid in self.coastal_ids:
            assert self.adjacency[rid] & self.water_ids, f'{name}: region {rid} is not coastal'
        if border_ids is not None:
            assert self.border_ids == border_ids, (
                f'{name}: border regions {sorted(self.border_ids)} != expected {sorted(border_ids)}')
        if water_ids is not None:
            assert self.water_ids == water_ids, (
                f'{name}: water regions {sorted(self.water_ids)} != {sorted(water_ids)}')
