"""Draw every image of the `smallw` web UI as plain SVG into `static/`.

Run from `app/`::

    python3 -m environments.smallw.draw_assets           # (re)write static/
    python3 -m environments.smallw.draw_assets --check   # verify, write nothing

The original Small World artwork is copyrighted and is not shipped: everything
served at `/smallw_static` is drawn here from scratch, on purpose as simple as
possible (flat colours, no texture, no scenery) while keeping what matters to
play — the race and power names, the token values, the bonus pictograms, and a
recognisable figure per race.

* `board<N>p.svg` — the map of each player count (2 to 5). The region shapes
  are the planar maps of `envs/shapes.py` (hand-drawn for 3 players, traced
  from photos of the printed boards for the others) laid out on the pixel
  frame of `map<N>p.py`, so the region anchors and the turn track stay valid.
  Each border is drawn once and shared by the two regions it separates;
  `--check` verifies the adjacency against the map table, that every anchor
  lies inside its region and that the regions tile the board.
* `races/<key>.svg` (banner, 429x230) and `races/<key>_token.svg` (88x88).
* `powers/<key>.svg` (badge, 115x115).
* `pieces/<name>.svg` — coins, markers and the turn marker.

`--check` also fails if a committed file differs from what the script draws.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from .envs.classes import POWERS, RACES, PowerId, RaceId
from .envs.mapdef import Symbol, Terrain
from .envs.maps import PLAYER_COUNTS
from .envs.shapes import BoardShape, Cubic, Point, shape_for

STATIC_DIR = Path(__file__).resolve().parent / 'static'

FONT = "font-family=\"'DejaVu Sans', Verdana, Arial, sans-serif\""


# --------------------------------------------------------------------------- #
# Small SVG helpers
# --------------------------------------------------------------------------- #

_CLIP_IDS = [0]


def new_clip_id() -> str:
    """A clipPath id unique inside the document being built."""
    _CLIP_IDS[0] += 1
    return f'c{_CLIP_IDS[0]}'


def svg_doc(width: float, height: float, body: str, title: str = '') -> str:
    """A standalone SVG document (explicit width/height = natural size).

    Also restarts the clipPath ids, so every file numbers them from 1.
    """
    _CLIP_IDS[0] = 0
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}" '
            f'height="{height:g}" viewBox="0 0 {width:g} {height:g}">')
    tag = f'<title>{escape(title)}</title>' if title else ''
    return f'{head}{tag}\n{body}\n</svg>\n'


def fmt(value: float) -> str:
    """Compact number for path data."""
    text = f'{value:.1f}'
    return text[:-2] if text.endswith('.0') else text


def text(x: float, y: float, value: str, size: float, fill: str = '#222',
         weight: str = 'bold', anchor: str = 'middle', halo: str = '',
         halo_width: float = 0.0, extra: str = '') -> str:
    """Centred text, optionally with an outline halo painted under the fill."""
    stroke = ''
    if halo:
        stroke = (f' stroke="{halo}" stroke-width="{halo_width:g}" '
                  'stroke-linejoin="round" paint-order="stroke"')
    return (f'<text x="{fmt(x)}" y="{fmt(y)}" {FONT} font-size="{size:g}" '
            f'font-weight="{weight}" text-anchor="{anchor}" '
            f'dominant-baseline="central" fill="{fill}"{stroke}{extra}>'
            f'{escape(value)}</text>')


def group(body: str, x: float = 0, y: float = 0, scale: float = 1.0) -> str:
    """`body` translated by (x, y) and scaled."""
    return (f'<g transform="translate({fmt(x)} {fmt(y)}) '
            f'scale({scale:g})">{body}</g>')


# --------------------------------------------------------------------------- #
# Board geometry (the outlines themselves are in `envs/shapes.py`)
# --------------------------------------------------------------------------- #

def cubics_path(cubics: list[Cubic], closed: bool = True) -> str:
    """SVG path data of a chain of cubics (straight ones as `L`)."""
    parts = [f'M{fmt(cubics[0][0][0])} {fmt(cubics[0][0][1])}']
    for p0, c1, c2, p1 in cubics:
        if c1 == p0 and c2 == p1:
            parts.append(f'L{fmt(p1[0])} {fmt(p1[1])}')
        else:
            parts.append(f'C{fmt(c1[0])} {fmt(c1[1])} {fmt(c2[0])} '
                         f'{fmt(c2[1])} {fmt(p1[0])} {fmt(p1[1])}')
    return ' '.join(parts) + (' Z' if closed else '')


# --------------------------------------------------------------------------- #
# Board drawing
# --------------------------------------------------------------------------- #

#: Flat terrain colours: (fill, darker shade for the terrain glyph).
TERRAIN_COLOURS: dict[Terrain, tuple[str, str]] = {
    Terrain.FARMLAND: ('#f2d36b', '#c9a43a'),
    Terrain.FOREST: ('#4f9a58', '#2f6e3a'),
    Terrain.HILL: ('#b3db7a', '#80ad4c'),
    Terrain.SWAMP: ('#a99470', '#7c6847'),
    Terrain.MOUNTAIN: ('#bfc3cc', '#8a909c'),
    Terrain.WATER: ('#7cbde8', '#4f95c9'),
}
BORDER_COLOUR = '#fbf7ec'
FRAME_COLOUR = '#5a4a3a'


def terrain_glyph(terrain: Terrain, colour: str) -> str:
    """Small terrain pictogram centred on (0, 0), about 14 px wide."""
    s = f'stroke="{colour}" stroke-width="1.6" stroke-linecap="round" ' \
        'stroke-linejoin="round"'
    if terrain == Terrain.FARMLAND:            # an ear of wheat
        grains = ''.join(
            f'<ellipse cx="{side * 2.2:g}" cy="{y:g}" rx="1.6" ry="2.6" '
            f'transform="rotate({side * 30:g} {side * 2.2:g} {y:g})" '
            f'fill="{colour}"/>'
            for y in (-4, -0.5, 3) for side in (-1, 1))
        return (f'<path d="M0 -7 V7" {s} fill="none"/>'
                f'<ellipse cx="0" cy="-7" rx="1.4" ry="2.4" fill="{colour}"/>'
                + grains)
    if terrain == Terrain.FOREST:              # a fir tree
        return (f'<path d="M0 -8 L5 -1 H2.5 L6.5 5 H-6.5 L-2.5 -1 H-5 Z" '
                f'fill="{colour}"/><path d="M0 5 V8" {s}/>')
    if terrain == Terrain.HILL:                # two rounded hills
        return (f'<path d="M-8 5 Q-4 -5 1 5 Z M-2 5 Q3 -2 8 5 Z" '
                f'fill="{colour}"/>')
    if terrain == Terrain.SWAMP:               # reeds over water
        return (f'<path d="M-4 4 V-4 M0 4 V-7 M4 4 V-3" {s} fill="none"/>'
                f'<ellipse cx="-4" cy="-5" rx="1.3" ry="2.4" fill="{colour}"/>'
                f'<ellipse cx="0" cy="-8" rx="1.3" ry="2.4" fill="{colour}"/>'
                f'<ellipse cx="4" cy="-4" rx="1.3" ry="2.4" fill="{colour}"/>'
                f'<path d="M-8 6 Q-6 4 -4 6 T0 6 T4 6 T8 6" {s} fill="none"/>')
    if terrain == Terrain.MOUNTAIN:            # two peaks
        return (f'<path d="M-9 6 L-3 -6 L1 1 L4 -3 L9 6 Z" fill="{colour}"/>'
                '<path d="M-3 -6 L-5 -2 L-3 -3 L-1.5 -2.5 Z" fill="#ffffff"/>')
    return (f'<path d="M-8 -2 Q-6 -4 -4 -2 T0 -2 T4 -2 T8 -2 '         # waves
            f'M-8 3 Q-6 1 -4 3 T0 3 T4 3 T8 3" {s} fill="none"/>')


def symbol_icon(symbol: Symbol) -> str:
    """Printed region symbol centred on (0, 0), radius 7."""
    if symbol == Symbol.MINE:                  # crossed pick and hammer
        return ('<circle r="7" fill="#c0392b" stroke="#ffffff" stroke-width="1"/>'
                '<path d="M-3.5 3.5 L3 -3 M3.5 3.5 L-3 -3" stroke="#ffffff" '
                'stroke-width="1.5" stroke-linecap="round"/>'
                '<path d="M0.5 -5 Q4.5 -4.5 5 -0.5" stroke="#ffffff" '
                'stroke-width="1.6" fill="none" stroke-linecap="round"/>'
                '<rect x="-5.2" y="-5.2" width="3.4" height="2.2" '
                'transform="rotate(45 -3.5 -4.1)" fill="#ffffff"/>')
    if symbol == Symbol.MAGIC:                 # magic crystals
        return ('<circle r="7" fill="#2f5fbf" stroke="#ffffff" stroke-width="1"/>'
                '<path d="M0 -5 L2 0 L0 5 L-2 0 Z M-3.6 -2 L-2.4 1.2 L-3.6 4.4 '
                'L-4.8 1.2 Z M3.6 -2 L4.8 1.2 L3.6 4.4 L2.4 1.2 Z" '
                'fill="#dff3ff"/>')
    if symbol == Symbol.CAVERN:                # a cave mouth
        return ('<circle r="7" fill="#4a4a4a" stroke="#ffffff" stroke-width="1"/>'
                '<path d="M-5 4 Q-5 -5 0 -5 Q5 -5 5 4 Z" fill="#9a9a9a"/>'
                '<path d="M-2.8 4 Q-2.8 -2 0 -2 Q2.8 -2 2.8 4 Z" fill="#111111"/>')
    # Lost Tribe setup spot: a small square with a figure
    return ('<rect x="-5" y="-5" width="10" height="10" rx="1.5" '
            'fill="#ece7dc" stroke="#6f6a60" stroke-width="0.9"/>'
            '<circle cy="-2" r="1.6" fill="#6f6a60"/>'
            '<path d="M-2.6 3.6 Q0 -1.2 2.6 3.6 Z" fill="#6f6a60"/>')


def _place(polygon: list[Point], avoid: list[tuple[Point, float]],
           radius: float, near: Point | None = None) -> Point | None:
    """Free spot of `radius` inside `polygon`, away from the `avoid` discs.

    Scans a 2 px grid and keeps the point with the largest clearance (capped,
    so that among comfortable spots the one closest to `near` wins).
    """
    poly = np.array(polygon)
    xs = np.arange(poly[:, 0].min() + 2, poly[:, 0].max() - 1, 2)
    ys = np.arange(poly[:, 1].min() + 2, poly[:, 1].max() - 1, 2)
    gx, gy = (a.ravel() for a in np.meshgrid(xs, ys))
    # even-odd point-in-polygon, vectorised over the grid
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    crosses = (y0[None] > gy[:, None]) != (y1[None] > gy[:, None])
    with np.errstate(divide='ignore', invalid='ignore'):
        xcut = x0[None] + (gy[:, None] - y0[None]) * (x1 - x0)[None] / (y1 - y0)[None]
    keep = (crosses & (gx[:, None] < xcut)).sum(axis=1) % 2 == 1
    gx, gy = gx[keep], gy[keep]
    if gx.size == 0:
        return None
    # distance to the outline
    dx, dy = (x1 - x0)[None], (y1 - y0)[None]
    length = np.maximum(dx * dx + dy * dy, 1e-9)
    t = np.clip(((gx[:, None] - x0[None]) * dx + (gy[:, None] - y0[None]) * dy)
                / length, 0, 1)
    dist = np.hypot(gx[:, None] - x0[None] - t * dx,
                    gy[:, None] - y0[None] - t * dy).min(axis=1)
    clear = dist - radius - 1.5
    for (cx, cy), size in avoid:
        clear = np.minimum(clear, np.hypot(gx - cx, gy - cy) - size - radius)
    score = np.minimum(clear, 4.0)
    if near is not None:
        score = score - 0.01 * np.hypot(gx - near[0], gy - near[1])
    best = int(np.argmax(score))
    return float(gx[best]), float(gy[best])


def _token_zone(anchor: Point) -> list[tuple[Point, float]]:
    """Discs covered by what the renderer draws at an anchor (race token,
    count badge, marker row, the dashed target ring of radius 19 and the
    conquest cost above its right side), kept free of printed symbols."""
    cx, cy = anchor
    return [((cx, cy), 21.0), ((cx + 17, cy - 16), 7.0)]


def board_layout(shape: BoardShape) -> dict[int, dict[str, Point]]:
    """Where the number, the symbols and the terrain glyph of every region go."""
    layout = {}
    for region in shape.map.regions:
        polygon = shape.region_polygon(region.id)
        avoid = _token_zone(region.anchor)
        # keep off the turn track (a far disc never binds, see `_place`)
        avoid += [(p, 15.0) for p in shape.map.turn_track]
        spots = {}
        spot = _place(polygon, avoid, 6.0)
        spots['number'] = spot
        avoid.append((spot, 6.0))
        for symbol in sorted(region.symbols):
            spot = _place(polygon, avoid, 7.0, near=region.anchor)
            spots[symbol.name] = spot
            avoid.append((spot, 7.0))
        # mountains carry the renderer's Mountain marker: no second glyph
        if region.terrain != Terrain.MOUNTAIN:
            spot = _place(polygon, avoid, 8.0)
            if spot is not None:
                spots['terrain'] = spot
        layout[region.id] = spots
    return layout


def draw_board(shape: BoardShape) -> str:
    """The whole board of one player count."""
    W, H = shape.map.board_size
    parts = [f'<rect width="{W}" height="{H}" fill="{BORDER_COLOUR}"/>']
    for region in shape.map.regions:
        fill, _ = TERRAIN_COLOURS[region.terrain]
        parts.append(f'<path id="r{region.id}" '
                     f'd="{cubics_path(shape.region_outline(region.id))}" '
                     f'fill="{fill}"/>')
    # the borders, drawn once each over the fills
    border_paths = ' '.join(cubics_path(shape.border_cubics(b), closed=False)
                            for b in shape.borders)
    parts.append(f'<path d="{border_paths}" fill="none" stroke="{BORDER_COLOUR}" '
                 'stroke-width="2.2" stroke-linejoin="round" '
                 'stroke-linecap="round"/>')
    layout = board_layout(shape)
    for region in shape.map.regions:
        spots = layout[region.id]
        _, dark = TERRAIN_COLOURS[region.terrain]
        if 'terrain' in spots:
            x, y = spots['terrain']
            parts.append(f'<g transform="translate({fmt(x)} {fmt(y)})" '
                         f'opacity="0.85">{terrain_glyph(region.terrain, dark)}</g>')
        for symbol in sorted(region.symbols):
            x, y = spots[symbol.name]
            parts.append(f'<g transform="translate({fmt(x)} {fmt(y)})">'
                         f'{symbol_icon(symbol)}</g>')
        x, y = spots['number']
        parts.append(text(x, y, str(region.id), 10, fill='#2b2b2b',
                          halo='#ffffff', halo_width=2.4))
    # turn track
    for turn, (x, y) in enumerate(shape.map.turn_track, start=1):
        parts.append(f'<rect x="{x - 15}" y="{y - 8}" width="30" height="16" '
                     f'rx="6" fill="#e9e4f0" fill-opacity="0.92" '
                     f'stroke="#6d6585" stroke-width="1"/>')
        parts.append(text(x, y + 0.5, str(turn), 10, fill='#4d4566'))
    parts.append(f'<rect x="0.75" y="0.75" width="{W - 1.5}" height="{H - 1.5}" '
                 f'fill="none" stroke="{FRAME_COLOUR}" stroke-width="1.5"/>')
    return svg_doc(W, H, '\n'.join(parts),
                   f'Small World — {shape.map.n_players} players')


# --------------------------------------------------------------------------- #
# Shared pictograms
# --------------------------------------------------------------------------- #

ORANGE = '#e8762c'
INK = '#2b2b2b'


def octagon(r: float) -> str:
    """Points of a regular octagon of circumradius `r` centred on (0, 0)."""
    return ' '.join(f'{fmt(r * math.cos(math.pi / 8 + k * math.pi / 4))},'
                    f'{fmt(r * math.sin(math.pi / 8 + k * math.pi / 4))}'
                    for k in range(8))


def star(cx: float, cy: float, r: float, fill: str) -> str:
    points = []
    for k in range(10):
        angle = -math.pi / 2 + k * math.pi / 5
        rad = r if k % 2 == 0 else r * 0.45
        points.append(f'{fmt(cx + rad * math.cos(angle))},'
                      f'{fmt(cy + rad * math.sin(angle))}')
    return f'<polygon points="{" ".join(points)}" fill="{fill}"/>'


def coin_icon(label: str, r: float = 14) -> str:
    """Gold victory-coin octagon with a label ('+1'), centred on (0, 0)."""
    return (f'<polygon points="{octagon(r)}" fill="#e9c24f" stroke="#a57f14" '
            f'stroke-width="{r * 0.14:g}" stroke-linejoin="round"/>'
            + text(0, 0.5, label, r * 1.05, fill='#5a4300'))


def minus_tile(label: str = '-1', r: float = 13) -> str:
    """'Cost' tile: a light square with '-1'."""
    return (f'<rect x="{-r:g}" y="{-r:g}" width="{2 * r:g}" height="{2 * r:g}" '
            f'rx="{r * 0.25:g}" fill="#ffffff" stroke="#5f6b7a" '
            f'stroke-width="{r * 0.13:g}"/>'
            + text(0, 0.5, label, r * 1.1, fill='#1f2d3d'))


def shield_icon(label: str = '+1', r: float = 13) -> str:
    """Defence shield with a label."""
    return (f'<path d="M{-r:g} {-r:g} H{r:g} V{r * 0.1:g} Q{r:g} {r * 0.8:g} 0 '
            f'{r * 1.2:g} Q{-r:g} {r * 0.8:g} {-r:g} {r * 0.1:g} Z" '
            f'fill="#3d7fd1" stroke="#ffffff" stroke-width="{r * 0.12:g}"/>'
            + text(0, -r * 0.05, label, r * 1.0, fill='#ffffff'))


def no_entry(r: float = 12) -> str:
    """Red 'immune' disc."""
    return (f'<circle r="{r:g}" fill="#d32f2f" stroke="#ffffff" '
            f'stroke-width="{r * 0.15:g}"/><rect x="{-r * 0.6:g}" '
            f'y="{-r * 0.18:g}" width="{r * 1.2:g}" height="{r * 0.36:g}" '
            'fill="#ffffff"/>')


def pillar_icon(r: float = 12) -> str:
    """The 'decline' symbol: a grey column."""
    return (f'<path d="M{-r * 0.7:g} {-r:g} H{r * 0.7:g} V{-r * 0.75:g} '
            f'H{r * 0.45:g} V{r * 0.75:g} H{r * 0.7:g} V{r:g} H{-r * 0.7:g} '
            f'V{r * 0.75:g} H{-r * 0.45:g} V{-r * 0.75:g} H{-r * 0.7:g} Z" '
            f'fill="#d9d9d9" stroke="#6b6b6b" stroke-width="{r * 0.12:g}" '
            'stroke-linejoin="round"/>'
            f'<path d="M{-r * 0.15:g} {-r * 0.6:g} V{r * 0.6:g} '
            f'M{r * 0.15:g} {-r * 0.6:g} V{r * 0.6:g}" stroke="#9a9a9a" '
            f'stroke-width="{r * 0.08:g}"/>')


def region_tile(content: str = '', r: float = 16, fill: str = '#f2d36b',
                crossed: bool = False) -> str:
    """A generic region: a rounded square of land, optionally crossed out."""
    out = (f'<rect x="{-r:g}" y="{-r:g}" width="{2 * r:g}" height="{2 * r:g}" '
           f'rx="{r * 0.3:g}" fill="{fill}" stroke="{FRAME_COLOUR}" '
           f'stroke-width="{r * 0.1:g}"/>' + content)
    if crossed:
        out += (f'<path d="M{-r * 0.7:g} {-r * 0.7:g} L{r * 0.7:g} {r * 0.7:g} '
                f'M{r * 0.7:g} {-r * 0.7:g} L{-r * 0.7:g} {r * 0.7:g}" '
                f'stroke="#d32f2f" stroke-width="{r * 0.22:g}" '
                'stroke-linecap="round"/>')
    return out


def arrow(x0: float, y0: float, x1: float, y1: float, colour: str = '#43a047',
          width: float = 4) -> str:
    """Straight arrow from (x0, y0) to (x1, y1)."""
    angle = math.atan2(y1 - y0, x1 - x0)
    head = width * 2.4
    bx, by = x1 - head * math.cos(angle), y1 - head * math.sin(angle)
    left = (bx + head * 0.6 * math.sin(angle), by - head * 0.6 * math.cos(angle))
    right = (bx - head * 0.6 * math.sin(angle), by + head * 0.6 * math.cos(angle))
    return (f'<path d="M{fmt(x0)} {fmt(y0)} L{fmt(bx)} {fmt(by)}" '
            f'stroke="{colour}" stroke-width="{width:g}" stroke-linecap="round"/>'
            f'<polygon points="{fmt(x1)},{fmt(y1)} {fmt(left[0])},{fmt(left[1])} '
            f'{fmt(right[0])},{fmt(right[1])}" fill="{colour}"/>')


def at(x: float, y: float, body: str, scale: float = 1.0) -> str:
    return group(body, x, y, scale)


# --------------------------------------------------------------------------- #
# Race figures (an 88x88 bust, shoulders running past the bottom edge)
# --------------------------------------------------------------------------- #

def _shoulders(fill: str, wide: bool = False) -> str:
    if wide:
        return f'<path d="M0 104 Q2 64 44 62 Q86 64 88 104 Z" fill="{fill}"/>'
    return f'<path d="M8 104 Q10 68 44 66 Q78 68 80 104 Z" fill="{fill}"/>'


def _eyes(y: float, dx: float = 5.5, r: float = 2.1, fill: str = INK,
          cx: float = 44) -> str:
    return (f'<circle cx="{cx - dx:g}" cy="{y:g}" r="{r:g}" fill="{fill}"/>'
            f'<circle cx="{cx + dx:g}" cy="{y:g}" r="{r:g}" fill="{fill}"/>')


def _smile(y: float, w: float = 4.5, colour: str = '#7a3b2a', depth: float = 3) -> str:
    return (f'<path d="M{44 - w:g} {y:g} Q44 {y + depth:g} {44 + w:g} {y:g}" '
            f'stroke="{colour}" stroke-width="1.8" fill="none" '
            'stroke-linecap="round"/>')


RACE_FIGURES: dict[RaceId, str] = {
    RaceId.AMAZONS: (
        '<path d="M23 46 Q20 16 44 16 Q68 16 65 46 Q69 62 72 76 L58 70 H30 '
        'L16 76 Q19 62 23 46 Z" fill="#3a2316"/>'
        + _shoulders('#d79a66')
        + '<path d="M26 104 L29 77 Q44 85 59 77 L62 104 Z" fill="#2f7d32"/>'
        '<ellipse cx="44" cy="42" rx="14" ry="16" fill="#d79a66"/>'
        '<path d="M30 33 Q44 25 58 33" stroke="#c62828" stroke-width="4" '
        'fill="none"/>'
        + _eyes(42)
        + '<path d="M34 47 L40 46 M48 46 L54 47" stroke="#c62828" '
        'stroke-width="2" stroke-linecap="round"/>'
        + _smile(51)
        + '<path d="M73 8 Q91 46 73 84" stroke="#6d4c2f" stroke-width="3.5" '
        'fill="none" stroke-linecap="round"/>'
        '<path d="M73 8 V84" stroke="#fafafa" stroke-width="1"/>'),
    RaceId.DWARVES: (
        _shoulders('#7d8590')
        + '<path d="M71 92 L76 26" stroke="#7a4f2a" stroke-width="3.5" '
        'stroke-linecap="round"/>'
        '<path d="M63 32 Q75 20 87 30" stroke="#9aa1ab" stroke-width="4.5" '
        'fill="none" stroke-linecap="round"/>'
        '<ellipse cx="44" cy="44" rx="15" ry="15" fill="#eab08a"/>'
        '<path d="M27 40 Q27 16 44 16 Q61 16 61 40 Z" fill="#8f96a3"/>'
        '<rect x="25" y="37" width="38" height="5" rx="2" fill="#6b7280"/>'
        '<rect x="42" y="37" width="4" height="9" fill="#6b7280"/>'
        + _eyes(46, dx=7)
        + '<path d="M27 48 Q26 84 44 88 Q62 84 61 48 Q55 57 44 56 Q33 57 27 48 Z" '
        'fill="#d35400"/>'
        '<ellipse cx="44" cy="51" rx="4" ry="4.5" fill="#d98c6c"/>'
        '<path d="M34 57 Q44 51 54 57 Q44 60 34 57 Z" fill="#a84300"/>'),
    RaceId.ELVES: (
        '<path d="M27 42 Q24 16 44 16 Q64 16 61 42 L64 72 H24 Z" fill="#f2d16b"/>'
        + _shoulders('#3e9b4f')
        + '<path d="M32 42 L11 27 L32 51 Z M56 42 L77 27 L56 51 Z" '
        'fill="#f7dcc8"/>'
        '<ellipse cx="44" cy="43" rx="13" ry="16" fill="#f7dcc8"/>'
        '<path d="M31 37 Q35 22 44 22 Q53 22 57 37 Q50 30 44 31 Q38 30 31 37 Z" '
        'fill="#f2d16b"/>'
        '<ellipse cx="38.5" cy="43" rx="2.6" ry="1.8" fill="#2e7d32"/>'
        '<ellipse cx="49.5" cy="43" rx="2.6" ry="1.8" fill="#2e7d32"/>'
        + _smile(51, w=3.5, colour='#b5546a')
        + ''.join(f'<circle cx="{fmt(70 + 4 * math.cos(k * 1.2566))}" '
                  f'cy="{fmt(70 + 4 * math.sin(k * 1.2566))}" r="3.2" '
                  'fill="#ffffff" stroke="#d9a0b5" stroke-width="0.6"/>'
                  for k in range(5))
        + '<circle cx="70" cy="70" r="2.2" fill="#f2c12e"/>'),
    RaceId.GHOULS: (
        '<path d="M14 104 L18 42 Q20 10 44 10 Q68 10 70 42 L74 104 Z" '
        'fill="#5b4636"/>'
        '<path d="M26 104 L30 72 Q44 78 58 72 L62 104 Z" fill="#7a6a58"/>'
        '<path d="M31 32 Q44 26 57 32 Q60 50 52 64 Q44 70 36 64 Q28 50 31 32 Z" '
        'fill="#b8c7a3"/>'
        '<ellipse cx="38" cy="42" rx="4.5" ry="4" fill="#2b2b2b"/>'
        '<ellipse cx="50" cy="42" rx="4.5" ry="4" fill="#2b2b2b"/>'
        + _eyes(42, dx=6, r=1.3, fill='#e8e14a')
        + '<path d="M44 46 L42 51 H46 Z" fill="#6f7c5e"/>'
        '<ellipse cx="44" cy="58" rx="5" ry="4.5" fill="#2b2b2b"/>'
        '<path d="M40.5 55 V57.5 M43 54.3 V57.5 M45.5 54.3 V57.5 M48 55 V57.5" '
        'stroke="#f0f0e0" stroke-width="1.3"/>'),
    RaceId.GIANTS: (
        '<path d="M56 14 Q66 3 78 9 Q89 18 83 30 Q72 37 60 30 Q51 22 56 14 Z" '
        'fill="#9e9e9e"/>'
        + _shoulders('#8d6e63', wide=True)
        + '<path d="M66 64 Q80 46 72 30" stroke="#e3a483" stroke-width="9" '
        'fill="none" stroke-linecap="round"/>'
        '<circle cx="22" cy="46" r="4.5" fill="#e3a483"/>'
        '<circle cx="66" cy="46" r="4.5" fill="#e3a483"/>'
        '<path d="M22 46 Q22 18 44 18 Q66 18 66 46 Q66 66 44 66 Q22 66 22 46 Z" '
        'fill="#e3a483"/>'
        '<path d="M31 38 L41 40.5 M47 40.5 L57 38" stroke="#5d4037" '
        'stroke-width="3.2" stroke-linecap="round"/>'
        + _eyes(44.5, dx=7.5, r=1.8)
        + '<path d="M44 41 Q50 51 44 53 Q38 51 44 41 Z" fill="#cf8c6c"/>'
        '<path d="M37 59 H51" stroke="#6d3b2a" stroke-width="2.2" '
        'stroke-linecap="round"/>'),
    RaceId.HALFLINGS: (
        _shoulders('#2e7d32')
        + '<circle cx="44" cy="74" r="3.5" fill="#e0b13a"/>'
        '<circle cx="28" cy="48" r="3.8" fill="#f2c49e"/>'
        '<circle cx="60" cy="48" r="3.8" fill="#f2c49e"/>'
        '<circle cx="44" cy="46" r="16" fill="#f2c49e"/>'
        + ''.join(f'<circle cx="{x}" cy="{y}" r="{r}" fill="#6d4423"/>'
                  for x, y, r in ((30, 33, 7), (38, 27, 7), (47, 26, 7),
                                  (56, 31, 7), (60, 40, 4.5), (28, 41, 4.5)))
        + _eyes(46)
        + '<circle cx="33" cy="53" r="3" fill="#f29b8f" opacity="0.7"/>'
        '<circle cx="55" cy="53" r="3" fill="#f29b8f" opacity="0.7"/>'
        + _smile(55, w=6, depth=6)),
    RaceId.HUMANS: (
        '<path d="M74 92 V22" stroke="#7a4f2a" stroke-width="3" '
        'stroke-linecap="round"/>'
        '<path d="M66 9 V22 H82 V9 M74 9 V22" stroke="#9aa1ab" '
        'stroke-width="2.6" fill="none" stroke-linecap="round" '
        'stroke-linejoin="round"/>'
        + _shoulders('#f4efe4')
        + '<path d="M10 104 Q12 74 30 69 L40 104 Z M78 104 Q76 74 58 69 L48 104 Z" '
        'fill="#8d5a3b"/>'
        '<ellipse cx="44" cy="45" rx="14" ry="16" fill="#f0bf98"/>'
        '<path d="M32 32 Q31 14 44 14 Q57 14 56 32 Z" fill="#e6b832"/>'
        '<rect x="32" y="26" width="24" height="4" fill="#8d5a3b"/>'
        '<ellipse cx="44" cy="32" rx="25" ry="5.5" fill="#d9a521"/>'
        + _eyes(43)
        + '<ellipse cx="44" cy="48" rx="2.5" ry="3" fill="#dc9f7c"/>'
        '<path d="M35 54 Q40 49 44 52 Q48 49 53 54 Q48 56 44 54 Q40 56 35 54 Z" '
        'fill="#5d3a22"/>'),
    RaceId.ORCS: (
        _shoulders('#6d4c41')
        + '<path d="M18 80 L24 68 L30 78 M58 78 L64 68 L70 80" fill="#9e9e9e"/>'
        '<path d="M30 40 L9 33 L30 51 Z M58 40 L79 33 L58 51 Z" fill="#6f9a3a"/>'
        '<path d="M28 38 Q28 18 44 18 Q60 18 60 38 V50 Q60 66 44 66 Q28 66 28 50 Z" '
        'fill="#7aa843"/>'
        '<path d="M33 37 L42 41 M55 37 L46 41" stroke="#2f4a14" '
        'stroke-width="2.8" stroke-linecap="round"/>'
        + _eyes(44, dx=6, r=2.2, fill='#d32f2f')
        + '<path d="M41 49 Q44 52 47 49" stroke="#2f4a14" stroke-width="1.6" '
        'fill="none" stroke-linecap="round"/>'
        '<path d="M35 57 Q44 61 53 57" stroke="#2f4a14" stroke-width="2" '
        'fill="none" stroke-linecap="round"/>'
        '<path d="M37 58.5 L36 50 L40.5 57.5 Z M51 58.5 L52 50 L47.5 57.5 Z" '
        'fill="#fffde7"/>'),
    RaceId.RATMEN: (
        _shoulders('#6b5a4a')
        + '<circle cx="27" cy="26" r="10" fill="#8a7563"/>'
        '<circle cx="61" cy="26" r="10" fill="#8a7563"/>'
        '<circle cx="27" cy="26" r="6" fill="#e8a0a8"/>'
        '<circle cx="61" cy="26" r="6" fill="#e8a0a8"/>'
        '<path d="M26 38 Q26 20 44 20 Q62 20 62 38 Q62 52 52 62 Q48 72 44 72 '
        'Q40 72 36 62 Q26 52 26 38 Z" fill="#8a7563"/>'
        '<ellipse cx="44" cy="60" rx="8" ry="8" fill="#a8927f"/>'
        + _eyes(40, dx=7, r=2.6, fill='#1a1a1a')
        + '<circle cx="38" cy="39" r="0.9" fill="#ffffff"/>'
        '<circle cx="52" cy="39" r="0.9" fill="#ffffff"/>'
        '<path d="M38 62 L24 58 M38 64 L24 66 M50 62 L64 58 M50 64 L64 66" '
        'stroke="#3a3a3a" stroke-width="0.9"/>'
        '<circle cx="44" cy="66" r="2.8" fill="#e57f8f"/>'
        '<rect x="41.6" y="69" width="2.3" height="4.5" fill="#ffffff"/>'
        '<rect x="44.1" y="69" width="2.3" height="4.5" fill="#ffffff"/>'),
    RaceId.SKELETONS: (
        _shoulders('#3a3348')
        + '<path d="M44 66 V104" stroke="#f3f0e6" stroke-width="3"/>'
        '<path d="M32 77 Q44 72 56 77 M30 84 Q44 79 58 84 M29 91 Q44 86 59 91" '
        'stroke="#f3f0e6" stroke-width="2.6" fill="none" stroke-linecap="round"/>'
        '<rect x="40" y="58" width="8" height="10" fill="#f3f0e6"/>'
        '<path d="M26 40 Q26 16 44 16 Q62 16 62 40 Q62 50 56 54 V61 H32 V54 '
        'Q26 50 26 40 Z" fill="#f3f0e6"/>'
        '<ellipse cx="37" cy="40" rx="5" ry="5.5" fill="#2b2236"/>'
        '<ellipse cx="51" cy="40" rx="5" ry="5.5" fill="#2b2236"/>'
        '<path d="M44 46 L41 52 H47 Z" fill="#2b2236"/>'
        '<path d="M34 56 H54 M37.5 53.5 V61 M42 53.5 V61 M46 53.5 V61 '
        'M50.5 53.5 V61" stroke="#2b2236" stroke-width="1.2"/>'),
    RaceId.SORCERERS: (
        _shoulders('#262230')
        + '<path d="M18 66 L14 28 L44 50 L74 28 L70 66 Z" fill="#c62828"/>'
        '<ellipse cx="44" cy="44" rx="13" ry="16" fill="#efe3d6"/>'
        '<path d="M31 41 Q30 20 44 20 Q58 20 57 41 Q55 30 49 30 L44 37 L39 30 '
        'Q33 30 31 41 Z" fill="#1c1c1c"/>'
        '<path d="M34 39 L42 42 M54 39 L46 42" stroke="#1c1c1c" '
        'stroke-width="1.8" stroke-linecap="round"/>'
        + _eyes(44.5, r=1.8)
        + '<path d="M40 55 Q44 53 48 55" stroke="#7a3b2a" stroke-width="1.5" '
        'fill="none"/>'
        '<path d="M41 57 L44 66 L47 57 Z" fill="#1c1c1c"/>'
        '<circle cx="33" cy="82" r="4.5" fill="#efe3d6"/>'
        '<circle cx="55" cy="82" r="4.5" fill="#efe3d6"/>'
        '<circle cx="44" cy="80" r="9" fill="#9be7ff" stroke="#e1f7ff" '
        'stroke-width="1.5"/>'
        '<circle cx="41" cy="77" r="2.5" fill="#ffffff"/>'),
    RaceId.TRITONS: (
        '<path d="M75 92 V20" stroke="#c9a227" stroke-width="3" '
        'stroke-linecap="round"/>'
        '<path d="M67 10 V22 Q75 28 83 22 V10 M75 8 V24" stroke="#c9a227" '
        'stroke-width="2.6" fill="none" stroke-linecap="round" '
        'stroke-linejoin="round"/>'
        + _shoulders('#3f9b8e')
        + '<path d="M30 30 L28 12 L36 22 L40 8 L46 20 L52 8 L54 22 L62 12 '
        'L58 30 Z" fill="#e2566e"/>'
        '<path d="M27 42 L15 35 L18 51 Z M61 42 L73 35 L70 51 Z" fill="#e2566e"/>'
        '<path d="M26 44 Q26 22 44 22 Q62 22 62 44 Q62 64 44 66 Q26 64 26 44 Z" '
        'fill="#4fb3a5"/>'
        '<circle cx="37" cy="40" r="5" fill="#ffffff"/>'
        '<circle cx="51" cy="40" r="5" fill="#ffffff"/>'
        + _eyes(40.5, dx=7, r=2.4)
        + '<path d="M34 54 Q44 60 54 54" stroke="#1d5a52" stroke-width="2.2" '
        'fill="none" stroke-linecap="round"/>'
        '<path d="M30 49 Q32 51 30 53 M58 49 Q56 51 58 53" stroke="#1d5a52" '
        'stroke-width="1.3" fill="none"/>'),
    RaceId.TROLLS: (
        '<path d="M12 92 L27 26" stroke="#6d4c2f" stroke-width="5" '
        'stroke-linecap="round"/>'
        '<ellipse cx="26" cy="22" rx="8" ry="13" transform="rotate(14 26 22)" '
        'fill="#7a5532"/>'
        + _shoulders('#7f9878')
        + '<path d="M30 40 L17 33 L28 50 Z M58 40 L71 33 L60 50 Z" fill="#8fa888"/>'
        '<path d="M28 42 Q28 20 44 20 Q60 20 60 42 Q60 62 44 64 Q28 62 28 42 Z" '
        'fill="#8fa888"/>'
        '<path d="M31 37 Q38 32 44 38 Q50 32 57 37" stroke="#4f6149" '
        'stroke-width="3" fill="none" stroke-linecap="round"/>'
        + _eyes(41.5, dx=6, r=1.9)
        + '<path d="M44 39 Q53 52 47 61 Q40 61 42 50 Z" fill="#748e6c"/>'
        '<path d="M36 59 Q44 62 52 59" stroke="#3d4d38" stroke-width="2" '
        'fill="none" stroke-linecap="round"/>'
        '<path d="M38 59.5 L39 56 L40.5 60 Z" fill="#fffde7"/>'),
    RaceId.WIZARDS: (
        _shoulders('#5c6bc0')
        + '<ellipse cx="44" cy="43" rx="12" ry="12" fill="#f1c9a5"/>'
        '<path d="M24 32 L50 2 L64 32 Z" fill="#3949ab"/>'
        + star(46, 18, 4, '#ffd54f') + star(53, 26, 2.6, '#ffd54f')
        + '<ellipse cx="44" cy="32" rx="25" ry="5" fill="#303f9f"/>'
        '<path d="M33 37 Q38 34 42 37 M46 37 Q50 34 55 37" stroke="#ffffff" '
        'stroke-width="2.6" fill="none" stroke-linecap="round"/>'
        + _eyes(41, r=1.8)
        + '<path d="M30 44 Q30 90 44 94 Q58 90 58 44 Q52 53 44 51 Q36 53 30 44 Z" '
        'fill="#f5f5f5"/>'
        '<path d="M36 52 Q44 47 52 52 Q44 55 36 52 Z" fill="#e0e0e0"/>'
        '<ellipse cx="44" cy="47" rx="2.6" ry="3.2" fill="#e0a882"/>'),
}

#: Per race: (light background, dark accent).
RACE_COLOURS: dict[RaceId, tuple[str, str]] = {
    RaceId.AMAZONS: ('#a7d67f', '#3f7a2c'),
    RaceId.DWARVES: ('#f2d27a', '#9c7414'),
    RaceId.ELVES: ('#f6cddc', '#b8506f'),
    RaceId.GHOULS: ('#b5c2b9', '#56675c'),
    RaceId.GIANTS: ('#c3e3f4', '#4f87a8'),
    RaceId.HALFLINGS: ('#c6e3ad', '#5e9445'),
    RaceId.HUMANS: ('#f8d3a0', '#b0742a'),
    RaceId.ORCS: ('#f0a596', '#a8423a'),
    RaceId.RATMEN: ('#d3cfca', '#6f6862'),
    RaceId.SKELETONS: ('#d2c5ec', '#6e58a3'),
    RaceId.SORCERERS: ('#aab4ea', '#3f4fa8'),
    RaceId.TRITONS: ('#a3ddd6', '#2f8c82'),
    RaceId.TROLLS: ('#e2dbcd', '#8a7d63'),
    RaceId.WIZARDS: ('#bcc3ee', '#4a55a8'),
}


def mini_token(race: RaceId, size: float = 22) -> str:
    """A race token drawn at `size`, centred on (0, 0)."""
    return at(-size / 2, -size / 2, _token_body(race), scale=size / 88)


def _token_body(race: RaceId) -> str:
    """The 88x88 token tile: background, figure (clipped), outline."""
    light, dark = RACE_COLOURS[race]
    clip_id = new_clip_id()
    figure = (f'<clipPath id="{clip_id}"><rect x="3" y="3" width="82" '
              'height="82" rx="12"/></clipPath>'
              f'<g clip-path="url(#{clip_id})">{RACE_FIGURES[race]}</g>')
    return (f'<rect x="3" y="3" width="82" height="82" rx="12" fill="{light}"/>'
            f'{figure}<rect x="3" y="3" width="82" height="82" rx="12" '
            f'fill="none" stroke="{dark}" stroke-width="4"/>')


def draw_race_token(race: RaceId) -> str:
    return svg_doc(88, 88, _token_body(race),
                   f'{RACES[race].name} token')


# --------------------------------------------------------------------------- #
# Race banners (429x230): name, figure, banner value, ability pictogram
# --------------------------------------------------------------------------- #

def _ability_amazons() -> tuple[str, list[str]]:
    tokens = ''.join(at(x, y, mini_token(RaceId.AMAZONS, 30))
                     for x, y in ((-48, -18), (-16, -18), (-48, 14), (-16, 14)))
    return (tokens + at(38, -2, '<circle r="30" fill="#e8762c" stroke="#ffffff" '
                        'stroke-width="3"/>' + text(0, 1, '+4', 30, fill='#fff'))
            , ['+4 tokens, only', 'when conquering'])


def _ability_coin_per(icon: str, caption: list[str]) -> tuple[str, list[str]]:
    return (at(-34, 0, icon) + arrow(-4, 0, 14, 0, colour='#8a7a5a', width=3)
            + at(42, 0, coin_icon('+1', 22)), caption)


def race_ability(race: RaceId) -> tuple[str, list[str]]:
    """Pictogram (centred on (0, 0), ~180x90) and caption lines of a race."""
    big_symbol = lambda symbol: at(0, 0, symbol_icon(symbol), 3.4)
    tile = lambda terrain: region_tile(at(0, 0, terrain_glyph(
        terrain, TERRAIN_COLOURS[terrain][1]), 1.8),
        r=24, fill=TERRAIN_COLOURS[terrain][0])
    if race == RaceId.AMAZONS:
        return _ability_amazons()
    if race == RaceId.DWARVES:
        return _ability_coin_per(big_symbol(Symbol.MINE),
                                 ['per Mine region,', 'even in decline'])
    if race == RaceId.HUMANS:
        return _ability_coin_per(tile(Terrain.FARMLAND),
                                 ['per Farmland region'])
    if race == RaceId.WIZARDS:
        return _ability_coin_per(big_symbol(Symbol.MAGIC),
                                 ['per Magic Source region'])
    if race == RaceId.ORCS:
        return _ability_coin_per(region_tile(at(0, 0, mini_token(RaceId.RATMEN, 22)),
                                             r=24, crossed=True),
                                 ['per non-empty region', 'conquered this turn'])
    if race == RaceId.GIANTS:
        return (at(-34, 0, tile(Terrain.MOUNTAIN)) + at(30, 0, minus_tile('-1', 22)),
                ['to conquer next to', 'your Mountains'])
    if race == RaceId.TRITONS:
        return (at(-34, 0, tile(Terrain.WATER)) + at(30, 0, minus_tile('-1', 22)),
                ['to conquer a', 'coastal region'])
    if race == RaceId.ELVES:
        return (at(-48, 10, region_tile(at(0, 0, mini_token(RaceId.ELVES, 32)), r=26))
                + arrow(-18, 4, 20, -14, colour='#43a047', width=5)
                + at(48, -16, mini_token(RaceId.ELVES, 40)),
                ['attacked: lose no token,', 'all come back to hand'])
    if race == RaceId.GHOULS:
        return (at(-40, 0, pillar_icon(26))
                + ''.join(at(x, y, mini_token(RaceId.GHOULS, 28))
                          for x, y in ((6, -16), (36, -16), (21, 14), (51, 14))),
                ['decline with all tokens,', 'still conquer in decline'])
    if race == RaceId.HALFLINGS:
        return (at(-44, 0, hole_shape(), 1.0) + text(-8, 22, '×2', 18, fill=INK)
                + at(40, 0, no_entry(22)),
                ['enter anywhere; 2 holes', 'make regions immune'])
    if race == RaceId.RATMEN:
        return (''.join(at(-48 + 32 * (k % 4), -16 + 32 * (k // 4),
                           mini_token(RaceId.RATMEN, 28)) for k in range(8)),
                ['no power:', 'strength in numbers'])
    if race == RaceId.SKELETONS:
        return (''.join(at(x, 0, region_tile(at(0, 0, mini_token(RaceId.ORCS, 18)),
                                             r=17, crossed=True))
                        for x in (-72, -36))
                + arrow(-14, 0, 4, 0, colour='#8a7a5a', width=3)
                + text(22, 1, '+1', 24, fill=INK)
                + at(60, 0, mini_token(RaceId.SKELETONS, 32)),
                ['+1 token for every 2', 'non-empty regions taken'])
    if race == RaceId.SORCERERS:
        return (at(-46, 0, region_tile(at(0, 0, mini_token(RaceId.ORCS, 26)), r=22))
                + arrow(-18, 0, 18, 0, colour='#8e44ad', width=5)
                + at(46, 0, region_tile(at(0, 0, mini_token(RaceId.SORCERERS, 26)),
                                        r=22)),
                ['replace a lone enemy', 'token next to you'])
    if race == RaceId.TROLLS:
        return (at(-34, 0, lair_shape(52)) + at(32, -2, shield_icon('+1', 22)),
                ["a Lair in each region", '(kept in decline)'])
    raise KeyError(race)                                    # pragma: no cover


def draw_race_banner(race: RaceId) -> str:
    """The race banner: name, value, ability and figure."""
    rdef = RACES[race]
    light, dark = RACE_COLOURS[race]
    width, height = 429, 230
    pictogram, caption = race_ability(race)
    parts = [
        f'<rect x="2" y="2" width="{width - 4}" height="{height - 4}" rx="26" '
        f'fill="#fbf6ea" stroke="{dark}" stroke-width="4"/>',
        f'<path d="M2 28 Q2 2 28 2 H262 L246 56 H2 Z" fill="{dark}"/>',
        text(124, 30, rdef.name.upper(), 30, fill='#ffffff',
             extra=' letter-spacing="1"'),
        at(126, 110, pictogram, 1.25),
    ]
    for line_no, line in enumerate(caption):
        parts.append(text(126, 182 + 22 * line_no, line, 17, fill=INK,
                          weight='normal'))
    cx, cy, r = 330, 113, 93
    parts += [
        f'<clipPath id="portrait"><circle cx="{cx}" cy="{cy}" r="{r}"/></clipPath>',
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{light}"/>',
        f'<g clip-path="url(#portrait)">'
        f'{at(cx - 88, cy - 82, RACE_FIGURES[race], 2.0)}</g>',
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{dark}" '
        'stroke-width="5"/>',
        f'<circle cx="388" cy="188" r="32" fill="{ORANGE}" stroke="#ffffff" '
        'stroke-width="4"/>',
        text(388, 190, str(rdef.banner_value), 40, fill='#ffffff'),
    ]
    return svg_doc(width, height, '\n'.join(parts), f'{rdef.name} banner')


# --------------------------------------------------------------------------- #
# Power badges (115x115): name, pictogram, token value, bonus
# --------------------------------------------------------------------------- #

def dove() -> str:
    return ('<path d="M-20 4 Q-6 -2 2 -10 Q10 -18 18 -14 L24 -14 L19 -10 '
            'Q16 4 2 10 Q-10 14 -20 4 Z" fill="#ffffff" stroke="#7a8a99" '
            'stroke-width="1.5" stroke-linejoin="round"/>'
            '<path d="M-6 0 Q-2 -22 12 -26 Q8 -10 4 0 Z" fill="#eef2f6" '
            'stroke="#7a8a99" stroke-width="1.5" stroke-linejoin="round"/>'
            '<circle cx="16" cy="-12" r="1.4" fill="#2b2b2b"/>'
            '<path d="M22 -12 Q28 -6 26 0" stroke="#43a047" stroke-width="2" '
            'fill="none"/><ellipse cx="27" cy="-4" rx="2" ry="3.5" '
            'fill="#43a047"/>')


def dragon_head() -> str:
    return ('<path d="M-22 14 Q-24 -8 -8 -14 L-4 -26 L2 -14 Q14 -14 22 -6 '
            'L26 4 Q20 10 8 8 L-2 10 Q-6 18 -22 14 Z" fill="#d84332" '
            'stroke="#8e1f15" stroke-width="2" stroke-linejoin="round"/>'
            '<path d="M-14 -10 L-18 -24 L-8 -14 Z" fill="#8e1f15"/>'
            '<circle cx="4" cy="-6" r="2.6" fill="#ffd54f"/>'
            '<path d="M10 4 L24 4" stroke="#8e1f15" stroke-width="1.6"/>'
            '<path d="M14 4 L15 8 L17 4 M19 4 L20 7 L22 4" fill="#ffffff"/>')


def wings() -> str:
    feather = ('<path d="M0 0 Q-10 -24 -30 -20 Q-24 -14 -28 -10 Q-20 -8 -24 -2 '
               'Q-14 0 -18 6 Q-8 6 0 0 Z" fill="#ffffff" stroke="#7a8a99" '
               'stroke-width="1.6" stroke-linejoin="round"/>')
    return (f'<g transform="translate(-3 4)">{feather}</g>'
            f'<g transform="translate(3 4) scale(-1 1)">{feather}</g>')


def helmet() -> str:
    return ('<path d="M-14 12 V-2 Q-14 -18 0 -18 Q14 -18 14 -2 V12 H6 V2 H-6 '
            'V12 Z" fill="#e2b33c" stroke="#8a6410" stroke-width="2" '
            'stroke-linejoin="round"/>'
            '<path d="M-6 -4 H6" stroke="#5a4108" stroke-width="3"/>'
            '<path d="M-14 -6 Q-26 -14 -22 -24 Q-18 -14 -12 -12 Z '
            'M14 -6 Q26 -14 22 -24 Q18 -14 12 -12 Z" fill="#ffffff" '
            'stroke="#8a6410" stroke-width="1.2"/>')


def flask() -> str:
    return ('<path d="M-6 -20 H6 V-8 L15 10 Q17 18 10 18 H-10 Q-17 18 -15 10 '
            'L-6 -8 Z" fill="#e3f2fb" stroke="#4a6572" stroke-width="2" '
            'stroke-linejoin="round"/>'
            '<path d="M-11 4 H11 L15 10 Q17 18 10 18 H-10 Q-17 18 -15 10 Z" '
            'fill="#43a047"/><rect x="-8" y="-24" width="16" height="5" rx="2" '
            'fill="#8d6e63"/><circle cx="-3" cy="-2" r="2" fill="#a5d6a7"/>'
            '<circle cx="3" cy="-8" r="1.5" fill="#a5d6a7"/>')


def die_face() -> str:
    pips = ''.join(f'<circle cx="{x}" cy="{y}" r="2.6" fill="#2b2b2b"/>'
                   for x, y in ((-7, -7), (7, -7), (0, 0), (-7, 7), (7, 7)))
    return (f'<g transform="rotate(-12)"><rect x="-15" y="-15" width="30" '
            f'height="30" rx="6" fill="#ffffff" stroke="#2b2b2b" '
            f'stroke-width="2"/>{pips}</g>')


def tent() -> str:
    return ('<path d="M-20 14 L0 -18 L20 14 Z" fill="#e0a64a" stroke="#7a5320" '
            'stroke-width="2" stroke-linejoin="round"/>'
            '<path d="M-6 14 L0 2 L6 14 Z" fill="#7a5320"/>'
            '<path d="M0 -18 V-26 L8 -23 L0 -20" fill="#d32f2f" stroke="#7a5320" '
            'stroke-width="1"/>')


def dagger() -> str:
    return ('<g transform="rotate(35)"><path d="M0 -26 L5 -6 H-5 Z" '
            'fill="#cfd8dc" stroke="#546e7a" stroke-width="1.5" '
            'stroke-linejoin="round"/><rect x="-11" y="-7" width="22" '
            'height="4" rx="2" fill="#8d6e63"/><rect x="-2.5" y="-3" width="5" '
            'height="14" rx="1.5" fill="#6d4c41"/><circle cy="13" r="3" '
            'fill="#8d6e63"/></g>')


def fortress_shape() -> str:
    return ('<path d="M-18 16 V-12 H-12 V-6 H-6 V-12 H0 V-6 H6 V-12 H12 V-6 '
            'H18 V-12 H24 V16 Z" transform="translate(-3 0)" fill="#e7d3a8" '
            'stroke="#7a6232" stroke-width="2" stroke-linejoin="round"/>'
            '<path d="M-3 16 V6 Q0 0 3 6 V16 Z" fill="#7a6232"/>')


def horse_head() -> str:
    return ('<path d="M-10 20 L-8 0 Q-10 -16 2 -22 L6 -28 L8 -20 Q20 -14 22 0 '
            'Q24 6 18 8 Q12 8 8 2 L6 20 Z" fill="#a1683a" stroke="#5d3a1a" '
            'stroke-width="2" stroke-linejoin="round"/>'
            '<path d="M-8 0 Q-12 -14 0 -22 Q-6 -10 -4 2 Z" fill="#3e2716"/>'
            '<circle cx="8" cy="-10" r="2" fill="#1b1b1b"/>')


def boat() -> str:
    return ('<path d="M-22 6 H22 L14 16 H-14 Z" fill="#8d5a3b" stroke="#5d3a1a" '
            'stroke-width="2" stroke-linejoin="round"/>'
            '<path d="M0 6 V-24" stroke="#5d3a1a" stroke-width="2"/>'
            '<path d="M2 -22 L18 2 H2 Z" fill="#ffffff" stroke="#7a8a99" '
            'stroke-width="1.5" stroke-linejoin="round"/>'
            '<path d="M-2 -18 L-16 2 H-2 Z" fill="#f5f5f5" stroke="#7a8a99" '
            'stroke-width="1.5" stroke-linejoin="round"/>'
            '<path d="M-26 20 Q-20 16 -14 20 T-2 20 T10 20 T22 20" '
            'stroke="#4f95c9" stroke-width="2.5" fill="none"/>')


def ghost() -> str:
    return ('<path d="M-16 20 V-6 Q-16 -24 0 -24 Q16 -24 16 -6 V20 L10 14 L5 20 '
            'L0 14 L-5 20 L-10 14 Z" fill="#e0f4f6" stroke="#5aa6b0" '
            'stroke-width="2" stroke-linejoin="round"/>'
            '<ellipse cx="-6" cy="-8" rx="2.6" ry="3.6" fill="#2b3d40"/>'
            '<ellipse cx="6" cy="-8" rx="2.6" ry="3.6" fill="#2b3d40"/>'
            '<ellipse cx="0" cy="3" rx="3" ry="4" fill="#2b3d40"/>')


def arm() -> str:
    return ('<path d="M-22 18 V2 Q-22 -6 -14 -6 H-4 L0 -18 Q4 -26 12 -22 '
            'Q18 -18 14 -10 Q22 -6 20 6 Q16 18 0 18 Z" fill="#e3a483" '
            'stroke="#8d5a3b" stroke-width="2" stroke-linejoin="round"/>'
            '<path d="M4 -4 Q10 0 14 -6" stroke="#8d5a3b" stroke-width="1.6" '
            'fill="none"/>')


def coin_stack() -> str:
    """Two piles of gold coins seen from the side."""
    out = ''
    for x, count in ((-11, 5), (11, 3)):
        for k in range(count):
            y = 16 - 7 * k
            out += (f'<ellipse cx="{x}" cy="{y + 2.5:g}" rx="11" ry="4" '
                    'fill="#a57f14"/>'
                    f'<ellipse cx="{x}" cy="{y:g}" rx="11" ry="4" '
                    'fill="#e9c24f" stroke="#a57f14" stroke-width="1.2"/>')
    return out


def hole_shape() -> str:
    return ('<path d="M-26 14 Q-24 -16 0 -16 Q24 -16 26 14 Z" fill="#7cb342" '
            'stroke="#33691e" stroke-width="2" stroke-linejoin="round"/>'
            '<ellipse cx="0" cy="6" rx="12" ry="9" fill="#3e2a1a"/>')


def lair_shape(size: float = 40) -> str:
    s = size / 40
    return (f'<g transform="scale({s:g})"><path d="M-20 16 L-16 -8 L-6 -18 '
            'L8 -16 L18 -6 L20 16 Z" fill="#9e9e9e" stroke="#555555" '
            'stroke-width="2" stroke-linejoin="round"/>'
            '<path d="M-8 16 V4 Q0 -6 8 4 V16 Z" fill="#2b2b2b"/></g>')


def mountain_shape() -> str:
    return ('<path d="M-30 18 L-10 -18 L2 0 L12 -12 L30 18 Z" fill="#9aa3b2" '
            'stroke="#5f6878" stroke-width="2" stroke-linejoin="round"/>'
            '<path d="M-10 -18 L-16 -7 L-11 -9 L-7 -6 L-4 -9 Z M12 -12 L7 -5 '
            'L11 -6 L15 -4 L17 -4 Z" fill="#ffffff"/>')


def _terrain_badge(terrain: Terrain) -> str:
    fill, dark = TERRAIN_COLOURS[terrain]
    return region_tile(at(0, 0, terrain_glyph(terrain, dark), 2.0), r=22,
                       fill=fill)


#: Per power: (pictogram, bonus items [(x, y, svg)], small caption).
def power_art(power: PowerId) -> tuple[str, list[tuple[float, float, str]], str]:
    coin = lambda label='+1': coin_icon(label, 13)
    if power == PowerId.ALCHEMIST:
        return flask(), [(90, 92, coin('+2'))], '/turn'
    if power == PowerId.BERSERK:
        return die_face(), [(90, 92, minus_tile('-?', 12))], ''
    if power == PowerId.BIVOUACKING:
        return tent(), [(90, 90, shield_icon('+1', 12))], '×5'
    if power == PowerId.COMMANDO:
        return dagger(), [(90, 92, minus_tile('-1', 12))], ''
    if power == PowerId.DIPLOMAT:
        return dove(), [], 'ally'
    if power == PowerId.DRAGON_MASTER:
        return dragon_head(), [(90, 92, no_entry(12))], ''
    if power == PowerId.FLYING:
        return wings(), [], 'anywhere'
    if power == PowerId.FOREST:
        return _terrain_badge(Terrain.FOREST), [(90, 92, coin())], ''
    if power == PowerId.FORTIFIED:
        return (fortress_shape(), [(64, 92, shield_icon('+1', 11)),
                                   (92, 92, coin())], '')
    if power == PowerId.HEROIC:
        return helmet(), [(90, 92, no_entry(12))], '×2'
    if power == PowerId.HILL:
        return _terrain_badge(Terrain.HILL), [(90, 92, coin())], ''
    if power == PowerId.MERCHANT:
        return (region_tile(text(0, 1, '?', 26, fill=FRAME_COLOUR), r=20,
                            fill='#efe3c8'), [(90, 92, coin())], '')
    if power == PowerId.MOUNTED:
        terrains = [(x, 99, region_tile(at(0, 0, terrain_glyph(t, TERRAIN_COLOURS[t][1]),
                                          0.6), r=7, fill=TERRAIN_COLOURS[t][0]))
                    for x, t in ((50, Terrain.HILL), (66, Terrain.FARMLAND))]
        return horse_head(), [(90, 92, minus_tile('-1', 12))] + terrains, ''
    if power == PowerId.PILLAGING:
        return (region_tile(at(0, 0, mini_token(RaceId.RATMEN, 24)), r=20,
                            crossed=True), [(90, 92, coin())], '')
    if power == PowerId.SEAFARING:
        return boat(), [], 'water'
    if power == PowerId.SPIRIT:
        return ghost(), [(80, 92, pillar_icon(10)), (98, 92, pillar_icon(10))], ''
    if power == PowerId.STOUT:
        return arm(), [(90, 92, pillar_icon(12))], ''
    if power == PowerId.SWAMP:
        return _terrain_badge(Terrain.SWAMP), [(90, 92, coin())], ''
    if power == PowerId.UNDERWORLD:
        return (at(0, 0, symbol_icon(Symbol.CAVERN), 3.0),
                [(90, 92, minus_tile('-1', 12))], '')
    if power == PowerId.WEALTHY:
        return coin_stack(), [(90, 92, coin('+7'))], 'once'
    raise KeyError(power)                                   # pragma: no cover


def draw_power_badge(power: PowerId) -> str:
    pdef = POWERS[power]
    size = 115
    pictogram, bonus, caption = power_art(power)
    name = pdef.name.upper()
    font = 13 if len(name) <= 11 else 11
    parts = [
        f'<rect x="2" y="2" width="{size - 4}" height="{size - 4}" rx="12" '
        'fill="#fbf6ea" stroke="#8a5a2b" stroke-width="3"/>',
        '<path d="M2 14 Q2 2 14 2 H101 Q113 2 113 14 V24 H2 Z" fill="#8a5a2b"/>',
        text(57.5, 13.5, name, font, fill='#ffffff'),
        at(57.5, 54, pictogram),
        f'<circle cx="23" cy="92" r="17" fill="{ORANGE}" stroke="#ffffff" '
        'stroke-width="2.5"/>',
        text(23, 93, str(pdef.value), 22, fill='#ffffff'),
    ]
    parts += [at(x, y, body) for x, y, body in bonus]
    if caption:
        # between the value disc and the bonus (or centred if no bonus)
        x = 58 if bonus else 74
        parts.append(text(x, 99, caption, 10, fill=INK, weight='normal'))
    return svg_doc(size, size, '\n'.join(parts), f'{pdef.name} badge')


# --------------------------------------------------------------------------- #
# Pieces: coins, markers, turn marker
# --------------------------------------------------------------------------- #

COIN_COLOURS = {1: ('#d08a5a', '#8a4a22'), 3: ('#d3d7dd', '#7d838c'),
                5: ('#ecc955', '#a57f14'), 10: ('#f2b445', '#9c6b0c')}


def draw_coin(value: int) -> str:
    fill, dark = COIN_COLOURS[value]
    body = (f'<polygon points="{octagon(26)}" fill="{fill}" stroke="{dark}" '
            'stroke-width="3" stroke-linejoin="round"/>'
            f'<polygon points="{octagon(20)}" fill="none" stroke="{dark}" '
            'stroke-width="1.2" opacity="0.6"/>'
            + text(0, 1, str(value), 24 if value < 10 else 20, fill=dark))
    return svg_doc(56, 56, at(28, 28, body), f'{value} coin')


def draw_piece(name: str) -> tuple[float, float, str]:
    """(width, height, body) of a marker piece."""
    if name == 'dragon':
        return 78, 78, at(39, 41, dragon_head(), 1.45)
    if name == 'encampment':
        return 100, 100, at(50, 50, '<circle r="44" fill="#9ccc65" '
                            'stroke="#558b2f" stroke-width="4"/>' + tent(), 1.0)
    if name == 'fortress':
        return 94, 94, at(47, 48, fortress_shape(), 1.6)
    if name == 'hero':
        return 78, 78, at(39, 44, helmet(), 1.45)
    if name == 'hole':
        return 78, 78, at(39, 42, hole_shape(), 1.35)
    if name == 'lost_tribe':
        return 88, 88, ('<rect x="3" y="3" width="82" height="82" rx="12" '
                        'fill="#c9c5bd" stroke="#6f6a60" stroke-width="4"/>'
                        '<circle cx="44" cy="30" r="11" fill="#6f6a60"/>'
                        '<path d="M22 80 Q24 44 44 44 Q64 44 66 80 Z" '
                        'fill="#6f6a60"/>'
                        '<path d="M70 82 V16 M70 10 L66 20 H74 Z" '
                        'stroke="#6f6a60" stroke-width="3" fill="#6f6a60"/>')
    if name == 'mountain':
        return 100, 100, at(50, 54, mountain_shape(), 1.5)
    if name == 'turn_marker':
        return 134, 72, ('<path d="M14 62 L8 14 L38 38 L67 6 L96 38 L126 14 '
                         'L120 62 Z" fill="#f2c12e" stroke="#9c7414" '
                         'stroke-width="4" stroke-linejoin="round"/>'
                         '<circle cx="40" cy="50" r="6" fill="#d32f2f"/>'
                         '<circle cx="67" cy="50" r="7" fill="#2e7d32"/>'
                         '<circle cx="94" cy="50" r="6" fill="#d32f2f"/>')
    raise KeyError(name)


PIECES = ('dragon', 'encampment', 'fortress', 'hero', 'hole', 'lost_tribe',
          'mountain', 'turn_marker')
COINS = (1, 3, 5, 10)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def all_files() -> dict[str, str]:
    """``{path relative to static/: SVG text}`` of every asset."""
    files = {}
    for n in PLAYER_COUNTS:
        shape = shape_for(n)
        files[shape.map.board_image] = draw_board(shape)
    for race in RaceId:
        key = RACES[race].key
        files[f'races/{key}.svg'] = draw_race_banner(race)
        files[f'races/{key}_token.svg'] = draw_race_token(race)
    for power in PowerId:
        files[f'powers/{POWERS[power].key}.svg'] = draw_power_badge(power)
    for value in COINS:
        files[f'pieces/coin_{value}.svg'] = draw_coin(value)
    for name in PIECES:
        width, height, body = draw_piece(name)
        files[f'pieces/{name}.svg'] = svg_doc(width, height, body,
                                              name.replace('_', ' '))
    files['MANIFEST.md'] = manifest(files)
    return files


def manifest(files: dict[str, str]) -> str:
    lines = [
        '# `smallw` static assets', '',
        'Drawn by `environments/smallw/draw_assets.py` (run from `app/`:',
        '`python3 -m environments.smallw.draw_assets`). Do not edit by hand:',
        'change the script and re-run it. Everything here is original, simple',
        'flat artwork made for this project (no artwork of the published game).',
        '',
        "No Troll's Lair image: `render_web.py` draws it inline. Declined races",
        'are rendered with a CSS `filter: grayscale(1)`.', '',
        '| file | size (px) |', '|------|-----------|',
    ]
    for path in sorted(files):
        head = files[path].split('\n', 1)[0]
        width = head.split('width="')[1].split('"')[0]
        height = head.split('height="')[1].split('"')[0]
        lines.append(f'| `{path}` | {width}x{height} |')
    lines += ['', f'{len(files)} image files.', '']
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--check', action='store_true',
                        help='verify the map and static/, write nothing')
    args = parser.parse_args(argv)

    problems = [problem for n in PLAYER_COUNTS for problem in shape_for(n).check()]
    files = all_files()
    expected = {STATIC_DIR / path for path in files}
    present = {p for p in STATIC_DIR.rglob('*') if p.is_file()}
    if args.check:
        for path, content in files.items():
            target = STATIC_DIR / path
            if not target.is_file():
                problems.append(f'missing {path}')
            elif target.read_text(encoding='utf-8') != content:
                problems.append(f'out of date: {path}')
        problems += [f'stray file {p.relative_to(STATIC_DIR)}'
                     for p in sorted(present - expected)]
        for problem in problems:
            print(problem)
        print('OK' if not problems else f'{len(problems)} problem(s)')
        return 1 if problems else 0

    if problems:
        for problem in problems:
            print(problem)
        return 1
    for path in sorted(present - expected):
        path.unlink()
        print(f'removed {path.relative_to(STATIC_DIR)}')
    for path, content in files.items():
        target = STATIC_DIR / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
    print(f'wrote {len(files)} files to {STATIC_DIR}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
