"""The 8 terrains of Toy Battle, transcribed as graphs.

Every terrain is drawn on the same portrait frame (`W` x `H`, Blue's H.Q. at
the bottom, Red's at the top). A terrain is a planar graph: the nodes are the
bases, the special bases and the H.Q., the edges are the paths, and the
regions are the bounded faces of that graph that hold medals. The coordinates
are ours (read off the boards, then snapped so that the six symmetric terrains
are exactly invariant under a half-turn); the art is not reproduced.

Sources: photos of the physical boards for the medal spaces (`private/toybattle/`,
not committed) — see the README for what was read where.

`build()` checks each transcription: every region must be a bounded face of
the straight-line drawing, and the terrains declared symmetric must really be
(bases, special bases, gates, paths, regions and H.Q.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from .constants import (BLUE, RED, Effect, MAX_BASES, MAX_DIST, MAX_HQ, MAX_REGIONS, N_TERRAINS)

W, H = 100.0, 166.0


@dataclass(frozen=True)
class Base:
    name: str
    x: float
    y: float
    special: bool
    gate: tuple[int, ...] | None        #: Tropical Pool: allowed strengths (Kwak always is)


@dataclass(frozen=True)
class Hq:
    name: str
    x: float
    y: float
    owner: int
    gate: tuple[int, ...] | None
    bases: tuple[int, ...]              #: bases one path away


@dataclass(frozen=True)
class Region:
    bases: tuple[int, ...]
    medals: int
    outline: tuple[tuple[float, float], ...]    #: the face: its nodes (H.Q. included), in order


@dataclass(frozen=True)
class Terrain:
    index: int
    key: str
    name: str
    effect: Effect
    objective: int                      #: medals to win
    bases: tuple[Base, ...]
    hqs: tuple[tuple[Hq, ...], tuple[Hq, ...]]      #: [colour][k]
    adj: tuple[tuple[int, ...], ...]                #: base -> adjacent bases
    edges: tuple[tuple[str, str], ...]              #: every path, by node names (drawing)
    regions: tuple[Region, ...]
    symmetric: bool
    #: views[colour][slot] = base index: the board as that colour sees it. Blue's
    #: view is the identity; on a symmetric terrain Red's is the half-turn, so
    #: both colours have "their side" on the same slots.
    views: tuple[tuple[int, ...], tuple[int, ...]]
    slots: tuple[tuple[int, ...], tuple[int, ...]]          #: inverse of views
    region_views: tuple[tuple[int, ...], tuple[int, ...]]   #: same, for the regions
    #: hq_dist[colour][base] = paths from the base to the nearest H.Q. of that colour
    hq_dist: tuple[tuple[int, ...], tuple[int, ...]]

    @property
    def n_bases(self) -> int:
        return len(self.bases)

    @property
    def n_medals(self) -> int:
        return sum(r.medals for r in self.regions)

    def node_xy(self, name: str) -> tuple[float, float]:
        for b in self.bases:
            if b.name == name:
                return b.x, b.y
        for side in self.hqs:
            for hq in side:
                if hq.name == name:
                    return hq.x, hq.y
        raise KeyError(name)

    def __deepcopy__(self, memo):
        return self                     # immutable and shared

    def __copy__(self):
        return self


# --------------------------------------------------------------------------- #
# the transcription
#
# nodes: "name x y kind [gate]" — kind is base / special / blue / red (an H.Q.)
# edges: "a-b ..." ; regions: (medals, "bases around it")
# --------------------------------------------------------------------------- #

_RAW = [
    dict(
        key='castle', name='Castle Field', effect=Effect.RECALL, objective=7, symmetric=True,
        nodes='''
            R   50    12    red
            a1  16.5  15.5  base
            a2  83.5  15.5  base
            b1  37    37.5  base
            b2  63    37.5  base
            s1  15    51    special
            s2  85    51    special
            m1  20    83    base
            m2  50    83    base
            m3  80    83    base
            s3  15    115   special
            s4  85    115   special
            c1  37    128.5 base
            c2  63    128.5 base
            d1  16.5  150.5 base
            d2  83.5  150.5 base
            B   50    154   blue
        ''',
        edges='''R-a1 R-a2 a1-b1 a2-b2 b1-s1 b2-s2 s1-m1 s2-m3 b1-m1 b1-m2 b2-m2 b2-m3
                 m1-s3 m3-s4 m1-c1 m2-c1 m2-c2 m3-c2 c1-s3 c2-s4 c1-d1 c2-d2 d1-B d2-B''',
        regions=[(3, 'a1 a2 b1 b2 m2'), (1, 's1 b1 m1'), (1, 's2 b2 m3'),
                 (2, 'b1 m1 c1 m2'), (2, 'b2 m3 c2 m2'),
                 (1, 's3 c1 m1'), (1, 's4 c2 m3'), (3, 'd1 d2 c1 c2 m2')],
    ),
    dict(
        key='pool', name='Tropical Pool', effect=Effect.GATE, objective=6, symmetric=True,
        nodes='''
            R1  15    19    red      67
            R2  85    19    red
            t   51    20    base
            l1  16.5  46    base
            c1  51    47    base
            r1  84    45    special  12
            l2  16.5  73    special  345
            r2  82.5  69    base
            c2  50    83    special  67
            l3  17.5  97    base
            r3  83.5  93    special  345
            l4  16    121   special  12
            c3  49    119   base
            r4  83.5  120   base
            b   49    146   base
            B1  15    147   blue
            B2  85    147   blue     67
        ''',
        edges='''R1-c1 t-c1 t-R2 t-r1 l1-c1 c1-r1 l1-l2 l1-c2 c1-c2 r1-r2 r2-c2 l2-l3
                 l3-c2 c2-c3 c2-r4 r2-r3 r3-r4 l3-l4 l4-c3 c3-r4 l4-b c3-b c3-B2 B1-b''',
        regions=[(1, 't c1 r1'), (1, 'l1 c1 c2'), (2, 'c1 r1 r2 c2'), (2, 'l1 l2 l3 c2'),
                 (2, 'c2 r2 r3 r4'), (2, 'l3 l4 c3 c2'), (1, 'c2 c3 r4'), (1, 'l4 c3 b')],
    ),
    dict(
        key='clouds', name='City of Clouds', effect=Effect.DRAW, objective=8, symmetric=True,
        nodes='''
            R   50    12    red
            l1  11.5  25    base
            r1  88.5  25    base
            s1  50    39    special
            l2  11.5  54    base
            r2  88.5  54    base
            s2  50    68.5  special
            l3  11.5  83    base
            r3  88.5  83    base
            s3  50    97.5  special
            l4  11.5  112   base
            r4  88.5  112   base
            s4  50    127   special
            l5  11.5  141   base
            r5  88.5  141   base
            B   50    154   blue
        ''',
        edges='''R-l1 R-r1 l1-s1 r1-s1 l1-l2 r1-r2 l2-s1 r2-s1 l2-s2 r2-s2 l2-l3 r2-r3
                 l3-s2 r3-s2 l3-s3 r3-s3 l3-l4 r3-r4 l4-s3 r4-s3 l4-s4 r4-s4 l4-l5 r4-r5
                 l5-s4 r5-s4 l5-B r5-B''',
        regions=[(1, 'l1 s1 r1'), (1, 'l1 l2 s1'), (1, 'r1 r2 s1'), (2, 's1 l2 s2 r2'),
                 (1, 'l2 l3 s2'), (1, 'r2 r3 s2'), (2, 's2 l3 s3 r3'),
                 (1, 'l3 l4 s3'), (1, 'r3 r4 s3'), (2, 's3 l4 s4 r4'),
                 (1, 'l4 l5 s4'), (1, 'r4 r5 s4'), (1, 'l5 s4 r5')],
    ),
    dict(
        key='jungle', name='Volcanic Jungle', effect=Effect.ERUPT, objective=7, symmetric=True,
        nodes='''
            a   16    15    base
            b   51.5  15    base
            R   88    15    red
            v1  12    49    special
            c   51.5  49    base
            d   84    49    base
            e   14    83    base
            f   50    83    base
            g   86    83    base
            h   16    117   base
            i   48.5  117   base
            v2  88    117   special
            B   12    151   blue
            j   48.5  151   base
            k   84    151   base
        ''',
        edges='''a-b b-R a-v1 v1-e v1-c b-c c-f b-d R-d d-g e-f f-g e-h f-i g-v2 v2-k
                 v2-i h-B h-j i-j B-j j-k''',
        regions=[(2, 'a b c v1'), (2, 'v1 c f e'), (3, 'b c f g d'),
                 (3, 'h j i f e'), (2, 'i f g v2'), (2, 'j i v2 k')],
    ),
    dict(
        key='cemetery', name='Cursed Cemetery', effect=Effect.EXHUME, objective=7, symmetric=True,
        nodes='''
            a   31    16    base
            R   83    12    red
            p1  14    38    special
            b   53    38    base
            p2  86    38    special
            c   33    59.5  base
            d   68    59.5  base
            e   13    83    base
            f   50    83    base
            g   87    83    base
            h   32    106.5 base
            i   67    106.5 base
            p3  14    128   special
            j   47    128   base
            p4  86    128   special
            B   17    154   blue
            k   69    150   base
        ''',
        edges='''a-R a-p1 a-c a-b b-R b-p2 b-d p1-c p2-d c-d c-e c-f d-f d-g
                 k-B k-p4 k-i k-j j-B j-p3 j-h p4-i p3-h i-h i-g i-f h-f h-e''',
        regions=[(1, 'a p1 c'), (2, 'a b d c'), (1, 'b p2 d'), (1, 'c d f'),
                 (2, 'c e h f'), (2, 'd g i f'), (1, 'h i f'),
                 (1, 'j p3 h'), (2, 'k j h i'), (1, 'k p4 i')],
    ),
    dict(
        key='caribbean', name='Caribbean Sea', effect=Effect.NONE, objective=5, symmetric=False,
        nodes='''
            R   50.5  17    red
            a   16.5  19.5  base
            b   77    31.5  base
            c   16.5  50    base
            d   49.5  50    base
            e   86.5  66    base
            f   16.5  84    base
            g   50    84    base
            h   86.5  103.5 base
            i   16.5  114.5 base
            j   41.5  114.5 base
            k   66    123.5 base
            l   49.5  151   base
            B1  16.5  150.5 blue
            B2  87.5  150.5 blue
        ''',
        edges='''a-R a-c R-d d-b c-d c-f d-g b-e b-k f-g f-j g-k e-h h-k h-B2 i-j i-B1
                 j-l k-l B1-l''',
        regions=[(1, 'a c d'), (2, 'c d g f'), (2, 'd b k g'), (2, 'b e h k'),
                 (3, 'f g k l j'), (1, 'i j l')],
    ),
    dict(
        key='station', name='Station Metal-X', effect=Effect.SHIELD, objective=7, symmetric=True,
        nodes='''
            a   11    11    base
            R   89    11    red
            b   56    26    base
            c   11    48    base
            d   90    48    base
            e   50    58.5  base
            s1  29    83    special
            s2  50    83    special
            s3  71    83    special
            f   50    107.5 base
            g   10    118   base
            h   89    118   base
            i   44    140   base
            B   11    155   blue
            j   89    155   base
        ''',
        edges='''a-R a-c a-e b-R b-e b-d c-e d-e c-s1 e-s1 e-s2 e-s3 d-s3 s1-s2 s2-s3
                 s1-g s1-f s2-f s3-f s3-h f-g f-h f-i f-j g-i h-j i-B j-B''',
        regions=[(1, 'a b e'), (1, 'a c e'), (1, 'b d e'), (1, 'c e s1'), (1, 'd e s3'),
                 (1, 'e s1 s2'), (1, 'e s2 s3'), (1, 'f s1 s2'), (1, 'f s2 s3'),
                 (1, 'g f s1'), (1, 'h f s3'), (1, 'g f i'), (1, 'h f j'), (1, 'i f j')],
    ),
    dict(
        key='battlefield', name='Battlefield', effect=Effect.SNIPER, objective=8, symmetric=True,
        nodes='''
            a1  21    16.5  base
            R   50    16.5  red
            a2  79    16.5  base
            b1  21    44    base
            st  50    43    special
            b2  79    44    base
            sl  10    83    special
            m1  37    83    base
            m2  63    83    base
            sr  90    83    special
            c1  21    122   base
            sb  50    123   special
            c2  79    122   base
            d1  21    149.5 base
            B   50    149.5 blue
            d2  79    149.5 base
        ''',
        edges='''a1-R R-a2 a1-b1 a2-b2 b1-st st-b2 b1-sl b1-m1 st-m1 st-m2 b2-m2 b2-sr
                 sl-c1 m1-c1 m1-sb m2-sb m2-c2 sr-c2 c1-sb sb-c2 c1-d1 c2-d2 d1-B B-d2''',
        regions=[(3, 'a1 a2 b1 st b2'), (1, 'b1 st m1'), (1, 'st b2 m2'),
                 (2, 'b1 sl c1 m1'), (2, 'st m1 sb m2'), (2, 'b2 m2 c2 sr'),
                 (1, 'm1 c1 sb'), (1, 'm2 sb c2'), (3, 'c1 sb c2 d1 d2')],
    ),
]


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

def faces(xy: dict[str, tuple[float, float]], edges) -> list[list[str]]:
    """Bounded faces of the straight-line drawing of a connected planar graph,
    each as the cycle of its nodes."""
    nbrs: dict[str, list[str]] = {n: [] for n in xy}
    for a, b in edges:
        nbrs[a].append(b)
        nbrs[b].append(a)
    for n, lst in nbrs.items():
        x, y = xy[n]
        lst.sort(key=lambda m: math.atan2(xy[m][1] - y, xy[m][0] - x))
    seen = set()
    found = []
    for a, b in [(a, b) for a, b in edges] + [(b, a) for a, b in edges]:
        if (a, b) in seen:
            continue
        cycle = []
        u, v = a, b
        while (u, v) not in seen:
            seen.add((u, v))
            cycle.append(u)
            ring = nbrs[v]
            w = ring[(ring.index(u) - 1) % len(ring)]     # next edge, turning the same way
            u, v = v, w
        area = sum(xy[p][0] * xy[q][1] - xy[q][0] * xy[p][1]
                   for p, q in zip(cycle, cycle[1:] + cycle[:1])) / 2
        found.append((area, cycle))
    # every bounded face is walked the same way round, the outer one the other way
    positive = [f for f in found if f[0] > 0]
    negative = [f for f in found if f[0] < 0]
    inner = positive if len(positive) >= len(negative) else negative
    return [cycle for _, cycle in inner]


def _centroid(points) -> tuple[float, float]:
    area = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    return cx / (3 * area), cy / (3 * area)


def _segment_distance(x: float, y: float, a, b) -> float:
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(x - ax - t * dx, y - ay - t * dy)


def _inside(x: float, y: float, points) -> bool:
    hit = False
    for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            hit = not hit
    return hit


MEDAL_STEP = 3.75           #: board units between two medals of a region
NODE_CLEARANCE, PATH_CLEARANCE = 9.5, 3.5


@lru_cache(maxsize=None)
def medal_spot(region: Region) -> tuple[float, float]:
    """Where the row of medals of a region is drawn: inside its face, as clear
    of its bases and paths as possible (the centroid often sits under a tile).
    Only the drawings need it: it is searched on demand, not at import."""
    points, n = list(region.outline), region.medals
    half = (n - 1) * MEDAL_STEP / 2
    cx, cy = _centroid(points)
    sides = list(zip(points, points[1:] + points[:1]))
    best, best_score = (cx, cy), -1.0
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    y = math.ceil(min(ys))
    while y < max(ys):
        x = math.ceil(min(xs))
        while x < max(xs):
            row = [(x - half, y), (x, y), (x + half, y)]
            if all(_inside(px, py, points) for px, py in row):
                nodes = min(math.hypot(px - qx, py - qy) for px, py in row for qx, qy in points)
                paths = min(_segment_distance(px, py, a, b) for px, py in row for a, b in sides)
                score = min(nodes / NODE_CLEARANCE, paths / PATH_CLEARANCE, 1.6) \
                    - 0.004 * math.hypot(x - cx, y - cy)
                if score > best_score:
                    best, best_score = (float(x), float(y)), score
            x += 1.0
        y += 1.0
    return best


def _turn(x: float, y: float) -> tuple[float, float]:
    """The half-turn of the frame."""
    return W - x, H - y


def _build(index: int, raw: dict) -> Terrain:
    key = raw['key']
    xy: dict[str, tuple[float, float]] = {}
    bases: list[Base] = []
    hq_rows: list[tuple[str, float, float, int, tuple | None]] = []
    for line in raw['nodes'].strip().splitlines():
        parts = line.split()
        name, x, y, kind = parts[0], float(parts[1]), float(parts[2]), parts[3]
        gate = tuple(int(ch) for ch in parts[4]) if len(parts) > 4 else None
        assert name not in xy, f'{key}: duplicate node {name}'
        xy[name] = (x, y)
        if kind in ('base', 'special'):
            bases.append(Base(name, x, y, kind == 'special', gate))
        else:
            hq_rows.append((name, x, y, {'blue': BLUE, 'red': RED}[kind], gate))
    index_of = {b.name: i for i, b in enumerate(bases)}

    edges = tuple(tuple(e.split('-')) for e in raw['edges'].split())
    assert len(set(frozenset(e) for e in edges)) == len(edges), f'{key}: duplicate path'
    for a, b in edges:
        assert a in xy and b in xy and a != b, f'{key}: bad path {a}-{b}'
    adj: list[list[int]] = [[] for _ in bases]
    hq_adj: dict[str, list[int]] = {row[0]: [] for row in hq_rows}
    for a, b in edges:
        if a in index_of and b in index_of:
            adj[index_of[a]].append(index_of[b])
            adj[index_of[b]].append(index_of[a])
        else:
            hq, base = (a, b) if a in hq_adj else (b, a)
            assert base in index_of, f'{key}: path between two H.Q.'
            hq_adj[hq].append(index_of[base])

    # regions = bounded faces holding medals
    face_sets = {}
    for cycle in faces(xy, edges):
        members = frozenset(index_of[n] for n in cycle if n in index_of)
        face_sets[members] = [xy[n] for n in cycle]
    regions = []
    for medals, names in raw['regions']:
        members = frozenset(index_of[n] for n in names.split())
        assert members in face_sets, f'{key}: region {names!r} is not a face of the board'
        regions.append(Region(tuple(sorted(members)), medals, tuple(face_sets[members])))

    hqs = [[], []]
    for name, x, y, owner, gate in hq_rows:
        hqs[owner].append(Hq(name, x, y, owner, gate, tuple(sorted(hq_adj[name]))))

    n = len(bases)
    identity = tuple(range(n))
    region_identity = tuple(range(len(regions)))
    red_view, red_regions = identity, region_identity
    if raw['symmetric']:
        where = {(b.x, b.y): i for i, b in enumerate(bases)}
        turn = [where[_turn(b.x, b.y)] for b in bases]            # KeyError = not symmetric
        for i, b in enumerate(bases):
            t = bases[turn[i]]
            assert (b.special, b.gate) == (t.special, t.gate), f'{key}: {b.name} / {t.name} differ'
            assert sorted(turn[j] for j in adj[i]) == sorted(adj[turn[i]]), f'{key}: paths of {b.name}'
        by_members = {frozenset(r.bases): k for k, r in enumerate(regions)}
        red_regions = tuple(by_members[frozenset(turn[b] for b in r.bases)] for r in regions)
        for k, r in enumerate(regions):
            assert regions[red_regions[k]].medals == r.medals, f'{key}: medals of region {k}'
        # Red's k-th H.Q. is the half-turn of Blue's k-th
        red_by_xy = {(h.x, h.y): h for h in hqs[RED]}
        hqs[RED] = [red_by_xy[_turn(h.x, h.y)] for h in hqs[BLUE]]
        for blue, red in zip(hqs[BLUE], hqs[RED]):
            assert blue.gate == red.gate and sorted(turn[b] for b in blue.bases) == list(red.bases), \
                f'{key}: H.Q. {blue.name} / {red.name} differ'
        red_view = tuple(turn)

    hq_dist = []
    for side in hqs:
        dist = {b: 1 for hq in side for b in hq.bases}
        todo = list(dist)
        while todo:
            b = todo.pop(0)
            for nb in adj[b]:
                if nb not in dist:
                    dist[nb] = dist[b] + 1
                    todo.append(nb)
        hq_dist.append(tuple(dist[b] for b in range(n)))
        assert max(hq_dist[-1]) < MAX_DIST, key

    assert n <= MAX_BASES and len(regions) <= MAX_REGIONS, key
    assert all(1 <= len(side) <= MAX_HQ for side in hqs), key
    views = (identity, red_view)
    slots = tuple(tuple(view.index(b) for b in range(n)) for view in views)
    return Terrain(index=index, key=key, name=raw['name'], effect=raw['effect'],
                   objective=raw['objective'], bases=tuple(bases),
                   hqs=(tuple(hqs[BLUE]), tuple(hqs[RED])),
                   adj=tuple(tuple(sorted(a)) for a in adj), edges=edges,
                   regions=tuple(regions), symmetric=raw['symmetric'],
                   views=views, slots=slots, region_views=(region_identity, red_regions),
                   hq_dist=tuple(hq_dist))


def build() -> dict[str, Terrain]:
    return {raw['key']: _build(i, raw) for i, raw in enumerate(_RAW)}


TERRAINS: dict[str, Terrain] = build()
TERRAIN_KEYS: tuple[str, ...] = tuple(TERRAINS)
assert len(TERRAINS) == N_TERRAINS
