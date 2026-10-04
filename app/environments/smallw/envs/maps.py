"""Registry of the Small World boards, one per player count (2 to 5).

The 2/3 and 4/5 player boards are the two sides of the two printed boards; each
side has its own table (`map2p.py` ... `map5p.py`, built on `mapdef.py`).

Run `python -m environments.smallw.envs.maps` (from `app/`) to self-check every
table.
"""

from __future__ import annotations

from . import map2p, map3p, map4p, map5p
from .mapdef import MapDef, RegionDef

#: Every board, keyed by player count.
MAPS: dict[int, MapDef] = {
    2: map2p.MAP_DEF,
    3: map3p.MAP_DEF,
    4: map4p.MAP_DEF,
    5: map5p.MAP_DEF,
}

#: Supported player counts.
PLAYER_COUNTS: tuple[int, ...] = tuple(sorted(MAPS))

#: Number of game turns per player count (the length of each turn track).
TURNS_PER_PLAYER_COUNT: dict[int, int] = {n: m.turns for n, m in MAPS.items()}


def map_def(n_players: int) -> MapDef:
    """The board played with `n_players`.

    Raises:
        ValueError: for a player count Small World is not played with.
    """
    try:
        return MAPS[n_players]
    except KeyError:
        raise ValueError(
            f'Small World is played by {PLAYER_COUNTS[0]} to {PLAYER_COUNTS[-1]} '
            f'players, not {n_players}'
        ) from None


def map_for(n_players: int) -> tuple[RegionDef, ...]:
    """The region table of the board played with `n_players`."""
    return map_def(n_players).regions


def turns_for(n_players: int) -> int:
    """Number of game turns for `n_players` (2p/3p: 10, 4p: 9, 5p: 8)."""
    return map_def(n_players).turns


def map_by_region_count(n_regions: int) -> MapDef:
    """The board with `n_regions` regions (every board has a different count),
    e.g. to recover the map from the size of an observation."""
    for m in MAPS.values():
        if m.n_regions == n_regions:
            return m
    raise ValueError(f'no Small World board has {n_regions} regions')


def self_check() -> bool:
    """Run the self-check of every board."""
    for module in (map2p, map3p, map4p, map5p):
        module.self_check()
    counts = [m.n_regions for m in MAPS.values()]
    assert len(set(counts)) == len(counts), f'two boards share a region count: {counts}'
    return True


if __name__ == '__main__':
    self_check()
    print('maps OK: ' + ', '.join(f'{n}p {m.n_regions} regions' for n, m in MAPS.items()))
