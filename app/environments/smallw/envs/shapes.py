"""Region outlines of the Small World boards (`smallw`).

Every board is a planar map in the pixel frame of its `MapDef`: the points
where three regions meet (or two regions and the board edge, `JUNCTIONS`), the
borders between two regions (`BORDERS`, each running from one junction to
another through a few points, smoothed with Catmull-Rom when drawn) and the
board edge (`FRAME`). Each border is shared by the two regions it separates, so
the outlines tile the board exactly. Two regions may share several borders
(a region lying on the coast between them, an inlet).

`draw_assets.py` draws `board<N>p.svg` from these outlines, and the web
renderer uses them to map a click to a region (`BoardShape.region_at`). The
data lives in `shape<N>p.py`: hand-drawn for 3 players, traced from photos of
the printed boards for 2, 4 and 5 players. `BoardShape.check` verifies a shape
against its map table (adjacency, border regions, anchors inside their region,
the regions cover the board).

Static data and pure geometry: no gym, no NiceGUI.
"""

from __future__ import annotations

import importlib
from typing import Mapping, Sequence, Union

from .mapdef import MapDef

Point = tuple[float, float]
Cubic = tuple[Point, Point, Point, Point]
#: ``(region id, region id, start junction, [points], end junction)``
Border = tuple[int, int, str, list[Point], str]
#: ``(point or junction name, region the edge runs along after it)``
FrameEntry = tuple[Union[Point, str], int]


def _catmull_rom(points: Sequence[Point]) -> list[Cubic]:
    """Cubic Bézier segments of the Catmull-Rom spline through `points`."""
    if len(points) == 2:
        p0, p1 = points
        return [(p0, p0, p1, p1)]
    out = []
    padded = [points[0]] + list(points) + [points[-1]]
    for i in range(1, len(padded) - 2):
        a, b, c, d = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        c1 = (b[0] + (c[0] - a[0]) / 6, b[1] + (c[1] - a[1]) / 6)
        c2 = (c[0] - (d[0] - b[0]) / 6, c[1] - (d[1] - b[1]) / 6)
        out.append((b, c1, c2, c))
    return out


def _reverse(cubics: list[Cubic]) -> list[Cubic]:
    return [(p1, c2, c1, p0) for (p0, c1, c2, p1) in reversed(cubics)]


def _close(a: Point, b: Point) -> bool:
    return abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) < 1e-6


def sample(cubics: list[Cubic], steps: int = 8) -> list[Point]:
    """Points along a chain of cubics (geometry checks, hit testing)."""
    out = []
    for p0, c1, c2, p1 in cubics:
        for k in range(steps):
            t = k / steps
            u = 1 - t
            out.append((
                u ** 3 * p0[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0]
                + t ** 3 * p1[0],
                u ** 3 * p0[1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1]
                + t ** 3 * p1[1]))
    return out


def polygon_area(points: list[Point]) -> float:
    return 0.5 * abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1)
                         in zip(points, points[1:] + points[:1])))


def inside(point: Point, polygon: list[Point]) -> bool:
    """Even-odd point-in-polygon test."""
    x, y = point
    result = False
    for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y0 > y) != (y1 > y):
            if x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
                result = not result
    return result


class BoardShape:
    """The outlines of the regions of one board.

    Args:
        map_def: the board (region table, frame size).
        junctions: ``{name: (x, y)}``.
        borders: a list of `Border` tuples, or the ``{(id, id): (start,
            [points], end)}`` form of the hand-drawn 3-player table.
        frame: the board edge clockwise from the top left corner, as
            ``(point or junction name, region)`` entries — the region the edge
            runs along *after* that point. Corners keep the region.
    """

    def __init__(self, map_def: MapDef, junctions: Mapping[str, Point],
                 borders: Union[Sequence[Border], Mapping[tuple[int, int], tuple]],
                 frame: Sequence[FrameEntry]):
        self.map = map_def
        self.junctions = dict(junctions)
        if isinstance(borders, Mapping):
            borders = [(a, b, *value) for (a, b), value in borders.items()]
        self.borders: list[Border] = [tuple(b) for b in borders]
        self.frame: list[tuple[Point, int]] = [
            (self.junctions[p] if isinstance(p, str) else (float(p[0]), float(p[1])), r)
            for p, r in frame]
        self._outlines: dict[int, list[Cubic]] = {}
        self._polygons: dict[int, list[Point]] = {}

    # -- outlines ----------------------------------------------------------- #

    def border_cubics(self, border: Border) -> list[Cubic]:
        """The smoothed curve of one border, from its first to its last junction."""
        _, _, start, middle, end = border
        return _catmull_rom([self.junctions[start], *middle, self.junctions[end]])

    def frame_pieces(self) -> dict[int, list[list[Cubic]]]:
        """Straight board-edge pieces of every region touching the edge."""
        pieces: dict[int, list[list[Cubic]]] = {}
        frame = self.frame
        count = len(frame)
        i = 0
        while i < count:
            region = frame[i][1]
            points = [frame[i][0]]
            j = i + 1
            while True:                         # corners keep the same region
                point, nxt = frame[j % count]
                points.append(point)
                if nxt != region or j % count == 0:
                    break
                j += 1
            pieces.setdefault(region, []).append(
                [(a, a, b, b) for a, b in zip(points, points[1:])])
            i = j
        return pieces

    def region_outline(self, region_id: int) -> list[Cubic]:
        """Closed outline of one region, as a chain of cubic segments."""
        if region_id in self._outlines:
            return self._outlines[region_id]
        pieces = [self.border_cubics(b) for b in self.borders if region_id in b[:2]]
        pieces += self.frame_pieces().get(region_id, [])
        chain = pieces.pop(0)
        while pieces:
            end = chain[-1][3]
            for index, piece in enumerate(pieces):
                if _close(piece[0][0], end):
                    chain += piece
                    break
                if _close(piece[-1][3], end):
                    chain += _reverse(piece)
                    break
            else:
                raise ValueError(f'{self.map.n_players}p region {region_id}: '
                                 f'outline is not closed at {end}')
            pieces.pop(index)
        if not _close(chain[0][0], chain[-1][3]):
            raise ValueError(f'{self.map.n_players}p region {region_id}: '
                             'outline is not closed')
        self._outlines[region_id] = chain
        return chain

    def region_polygon(self, region_id: int) -> list[Point]:
        """The outline of a region sampled as a polygon."""
        if region_id not in self._polygons:
            self._polygons[region_id] = sample(self.region_outline(region_id))
        return self._polygons[region_id]

    def region_at(self, x: float, y: float) -> int | None:
        """Id of the region drawn under `(x, y)`, None outside every region."""
        for region in self.map.regions:
            if inside((x, y), self.region_polygon(region.id)):
                return region.id
        return None

    # -- check -------------------------------------------------------------- #

    def check(self) -> list[str]:
        """Problems of the planar map (empty list = consistent with the map table)."""
        problems = []
        name = f'{self.map.n_players}p'
        drawn = {tuple(sorted(b[:2])) for b in self.borders}
        truth = {tuple(sorted((a, b))) for a, ns in self.map.adjacency.items() for b in ns}
        for pair in sorted(truth - drawn):
            problems.append(f'{name}: border {pair} missing from the drawing')
        for pair in sorted(drawn - truth):
            problems.append(f'{name}: border {pair} drawn but not adjacent in the map')
        for b in self.borders:
            for junction in (b[2], b[4]):
                if junction not in self.junctions:
                    problems.append(f'{name}: border {b[:2]}: unknown junction {junction}')
        if problems:
            return problems
        on_edge = set(self.frame_pieces())
        for region in self.map.regions:
            if region.id in on_edge and not region.border:
                problems.append(f'{name}: region {region.id} touches the edge but is '
                                'not a border region')
        width, height = self.map.board_size
        total = 0.0
        for region in self.map.regions:
            try:
                polygon = self.region_polygon(region.id)
            except ValueError as error:
                problems.append(str(error))
                continue
            total += polygon_area(polygon)
            if not inside(region.anchor, polygon):
                problems.append(f'{name}: anchor of region {region.id} is outside it')
        if abs(total - width * height) > 0.002 * width * height:
            problems.append(f'{name}: regions cover {total:.0f} px², '
                            f'board is {width * height}')
        return problems


def shape_for(n_players: int) -> BoardShape:
    """The region outlines of the board played with `n_players`."""
    module = importlib.import_module(f'{__package__}.shape{n_players}p')
    return module.SHAPE
