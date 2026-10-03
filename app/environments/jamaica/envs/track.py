"""The race track: a directed graph of spaces around the island.

A ship position is `(node, lap)`. The lap only changes on Port Royal (PR)
edges: entering PR going forward adds 1, leaving PR going backward removes 1.
The start is `(PR, 0)`, a ship behind the start line (after a backward first
move) is on lap -1, and `(PR, 1)` means the ship has finished.

`Track` is immutable and shared by every copy of the env (`__deepcopy__`
returns self); it caches destination sets.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Iterable

from .constants import FLOOR_SCORE, Kind
from . import data

Pos = tuple[int, int]   #: (node, lap)


class Track:
    """Immutable race track built from `data.SPACES` / `data.EDGES` (or test data)."""

    __slots__ = ('n', 'pr', 'kind', 'cost', 'score', 'xy', 'succ', 'pred', 'dist', 'L',
                 'lairs', 'splits', 'merges', '_cache')

    def __init__(self, spaces: Iterable[tuple], edges: Iterable[tuple[int, int]], pr: int = 0):
        spaces = sorted(spaces, key=lambda s: s[0])
        n = len(spaces)
        if [s[0] for s in spaces] != list(range(n)):
            raise ValueError('space ids must be 0..n-1')
        self.n = n
        self.pr = pr
        self.kind = tuple(int(s[1]) for s in spaces)
        self.cost = tuple(int(s[2]) for s in spaces)
        self.score = tuple(s[3] for s in spaces)
        self.xy = tuple((s[4], s[5]) if len(s) > 5 else (0, 0) for s in spaces)
        succ = [[] for _ in range(n)]
        pred = [[] for _ in range(n)]
        for a, b in edges:
            if b in succ[a]:
                raise ValueError(f'duplicate edge {a}->{b}')
            succ[a].append(b)
            pred[b].append(a)
        self.succ = tuple(tuple(x) for x in succ)
        self.pred = tuple(tuple(x) for x in pred)
        self.lairs = tuple(i for i in range(n) if self.kind[i] == Kind.LAIR)
        self.splits = tuple(i for i in range(n) if len(self.succ[i]) > 1)
        self.merges = tuple(i for i in range(n) if len(self.pred[i]) > 1)
        self.dist = self._distances()
        self.L = min(self.dist[s] + 1 for s in self.succ[pr])
        self._cache = {}
        self.validate()

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        spaces = [(i, self.kind[i], self.cost[i], self.score[i], *self.xy[i]) for i in range(self.n)]
        edges = [(a, b) for a in range(self.n) for b in self.succ[a]]
        return (Track, (spaces, edges, self.pr))

    # -- construction ---------------------------------------------------------

    def _distances(self) -> tuple[int, ...]:
        """Fewest forward steps from each node to Port Royal (BFS on `pred`)."""
        dist = [None] * self.n
        dist[self.pr] = 0
        queue = deque([self.pr])
        while queue:
            cur = queue.popleft()
            for p in self.pred[cur]:
                if p != self.pr and dist[p] is None:
                    dist[p] = dist[cur] + 1
                    queue.append(p)
        if any(d is None for d in dist):
            raise ValueError('every space must reach Port Royal')
        return tuple(dist)

    def validate(self) -> None:
        for i in range(self.n):
            if not self.succ[i] or not self.pred[i]:
                raise ValueError(f'space {i} needs a successor and a predecessor')
            if len(self.succ[i]) > 2 or len(self.pred[i]) > 2:
                raise ValueError(f'space {i}: at most 2 successors / predecessors')
        if self.kind[self.pr] != Kind.PORT_ROYAL or self.kind.count(Kind.PORT_ROYAL) != 1:
            raise ValueError('exactly one Port Royal, at `pr`')
        # no forward cycle that avoids Port Royal (DFS colouring)
        state = [0] * self.n
        for root in range(self.n):
            if state[root]:
                continue
            stack = [(root, iter(self.succ[root]))]
            state[root] = 1
            while stack:
                node, it = stack[-1]
                nxt = next(it, None)
                if nxt is None:
                    state[node] = 2
                    stack.pop()
                elif nxt == self.pr:
                    continue
                elif state[nxt] == 1:
                    raise ValueError('forward cycle that avoids Port Royal')
                elif state[nxt] == 0:
                    state[nxt] = 1
                    stack.append((nxt, iter(self.succ[nxt])))
        for i in range(self.n):
            k = self.kind[i]
            if k == Kind.PORT and self.cost[i] <= 0 or k == Kind.SEA and self.cost[i] <= 0:
                raise ValueError(f'space {i}: ports and sea spaces need a cost')

    # -- movement -------------------------------------------------------------

    def step1(self, node: int, lap: int, d: int) -> list[Pos]:
        """Positions one step forward (d > 0) or backward (d < 0)."""
        if d > 0:
            return [(m, lap + (m == self.pr)) for m in self.succ[node]]
        return [(m, lap - (node == self.pr)) for m in self.pred[node]]

    def order_key(self, pos: Pos) -> tuple[int, int]:
        return (self.remaining(*pos), pos[0])

    def destinations(self, pos: Pos, d: int, n: int) -> tuple[Pos, ...]:
        """Every position exactly `n` spaces away through every fork branch.

        Going forward, entering Port Royal after the circuit stops the ship
        there (overshoot). Sorted by (spaces remaining, node id).
        """
        key = (pos, d, n)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        front, stopped = {pos}, set()
        finish = (self.pr, 1)
        for _ in range(n):
            nxt = set()
            for p in front:
                for q in self.step1(p[0], p[1], d):
                    (stopped if q == finish else nxt).add(q)
            front = nxt
        out = tuple(sorted(front | stopped, key=self.order_key))
        self._cache[key] = out
        return out

    def retreat_targets(self, pos: Pos, payable: Callable[[int], bool]) -> tuple[Pos, ...]:
        """First fully payable space on each backward route from `pos`.

        Port Royal on lap 0 is free, so a retreat never crosses the start line
        from lap 0.
        """
        out = set()
        seen = {pos}
        front = [pos]
        while front:
            nxt = []
            for p in front:
                for q in self.step1(p[0], p[1], -1):
                    if q in seen:
                        continue
                    seen.add(q)
                    if payable(q[0]):
                        out.add(q)
                    else:
                        nxt.append(q)
            front = nxt
        return tuple(sorted(out, key=self.order_key))

    # -- race standing --------------------------------------------------------

    def remaining(self, node: int, lap: int) -> int:
        """Spaces left to the finish along the shortest route."""
        if node == self.pr:
            return (1 - lap) * self.L
        return self.dist[node] - lap * self.L

    def pos_score(self, node: int, lap: int) -> int:
        """Printed value of the ship's space (-5 at or before the -5 mark)."""
        if node == self.pr:
            return self.score[self.pr] if lap >= 1 else FLOOR_SCORE
        if lap < 0 or self.score[node] is None:
            return FLOOR_SCORE
        return int(self.score[node])

    def in_floor_zone(self, node: int, lap: int) -> bool:
        return self.pos_score(node, lap) == FLOOR_SCORE and not (node == self.pr and lap >= 1)

    def ring(self, pos: Pos, d: int, k: int) -> tuple[Pos, ...]:
        """Positions exactly k steps away (alias of `destinations`, for features)."""
        return self.destinations(pos, d, k)


_DEFAULT = None


def default_track() -> Track:
    """The real board (built once, shared)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Track(data.SPACES, data.EDGES, data.PR)
    return _DEFAULT
