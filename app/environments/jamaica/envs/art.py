"""SVG drawings for the Jamaica web UI (our own design, no game artwork).

Pure string builders, no nicegui import: `render_web.py` assembles them and
`tests/test_render.py` parses them.
"""

from __future__ import annotations

import math
from html import escape

from . import data
from .constants import Kind, Res, STAR, Sym, card_syms

#: board drawing size (pixels) and the scale from the reference photo
BOARD_W = 600
BOARD_H = round(BOARD_W * data.BOARD_SIZE[1] / data.BOARD_SIZE[0])
SCALE = BOARD_W / data.BOARD_SIZE[0]
SPACE_R = 13

SEAT_COLORS = ('#d32f2f', '#1565c0', '#2e7d32', '#f9a825', '#7b1fa2', '#37474f')
SEAT_NAMES = ('red', 'blue', 'green', 'yellow', 'purple', 'black')

SEA_TOP, SEA_BOTTOM = '#0f5e9c', '#2389da'
SAND, SAND_EDGE, FOREST = '#e9d8a6', '#c2a86b', '#7a9a4a'
GOLD, GOLD_EDGE = '#ffca28', '#a67c00'
FOOD, FOOD_EDGE = '#8bc34a', '#4a7a1d'
POWDER, POWDER_EDGE = '#424242', '#111111'


def xy(point) -> tuple[float, float]:
    """Reference-photo pixels -> board drawing pixels."""
    return point[0] * SCALE, point[1] * SCALE


def _f(v: float) -> str:
    return f'{v:.1f}'


def smooth_path(points, closed: bool = True) -> str:
    """Catmull-Rom spline through `points` as an SVG path."""
    pts = [xy(p) for p in points]
    n = len(pts)
    if n < 3:
        return ''
    d = [f'M{_f(pts[0][0])},{_f(pts[0][1])}']
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        p0 = pts[(i - 1) % n] if closed or i > 0 else pts[i]
        p1, p2 = pts[i], pts[(i + 1) % n]
        p3 = pts[(i + 2) % n] if closed or i + 2 < n else p2
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d.append(f'C{_f(c1[0])},{_f(c1[1])} {_f(c2[0])},{_f(c2[1])} {_f(p2[0])},{_f(p2[1])}')
    if closed:
        d.append('Z')
    return ' '.join(d)


# --------------------------------------------------------------------------- #
# small glyphs
# --------------------------------------------------------------------------- #

def diamonds(cx: float, cy: float, k: int, size: float = 3.2, fill: str = '#ffffff') -> str:
    """k white squares (sea cost), laid out like on a die."""
    offsets = {1: [(0, 0)], 2: [(-1, 0), (1, 0)], 3: [(-1, -0.8), (1, -0.8), (0, 0.9)],
               4: [(0, -1.1), (-1.1, 0), (1.1, 0), (0, 1.1)]}.get(k, [(0, 0)])
    out = []
    for ox, oy in offsets:
        x, y = cx + ox * size * 1.3, cy + oy * size * 1.3
        out.append(f'<rect x="{_f(x - size / 2)}" y="{_f(y - size / 2)}" width="{_f(size)}" '
                   f'height="{_f(size)}" fill="{fill}" transform="rotate(45 {_f(x)} {_f(y)})"/>')
    return ''.join(out)


def skull(cx: float, cy: float, r: float, fill: str = '#eceff1', ink: str = '#263238') -> str:
    """A small stylised skull."""
    return (f'<g><circle cx="{_f(cx)}" cy="{_f(cy - r * 0.15)}" r="{_f(r * 0.62)}" fill="{fill}"/>'
            f'<rect x="{_f(cx - r * 0.35)}" y="{_f(cy + r * 0.2)}" width="{_f(r * 0.7)}" '
            f'height="{_f(r * 0.38)}" rx="{_f(r * 0.1)}" fill="{fill}"/>'
            f'<circle cx="{_f(cx - r * 0.24)}" cy="{_f(cy - r * 0.18)}" r="{_f(r * 0.17)}" fill="{ink}"/>'
            f'<circle cx="{_f(cx + r * 0.24)}" cy="{_f(cy - r * 0.18)}" r="{_f(r * 0.17)}" fill="{ink}"/>'
            f'<path d="M{_f(cx)},{_f(cy + r * 0.02)} l{_f(-r * 0.08)},{_f(r * 0.14)} '
            f'h{_f(r * 0.16)} z" fill="{ink}"/></g>')


def coin(cx: float, cy: float, r: float, label: str = '') -> str:
    text = (f'<text x="{_f(cx)}" y="{_f(cy + r * 0.38)}" text-anchor="middle" font-size="{_f(r * 1.1)}" '
            f'font-weight="bold" font-family="sans-serif" fill="#5d4037">{escape(label)}</text>'
            if label else '')
    return (f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" fill="{GOLD}" stroke="{GOLD_EDGE}" '
            f'stroke-width="{_f(max(1, r * 0.15))}"/>{text}')


def food_icon(cx: float, cy: float, r: float) -> str:
    return (f'<rect x="{_f(cx - r)}" y="{_f(cy - r)}" width="{_f(2 * r)}" height="{_f(2 * r)}" '
            f'rx="{_f(r * 0.25)}" fill="{FOOD}" stroke="{FOOD_EDGE}" stroke-width="1"/>'
            f'<circle cx="{_f(cx - r * 0.2)}" cy="{_f(cy + r * 0.1)}" r="{_f(r * 0.5)}" fill="#e53935"/>'
            f'<path d="M{_f(cx - r * 0.2)},{_f(cy - r * 0.35)} q{_f(r * 0.3)},{_f(-r * 0.45)} '
            f'{_f(r * 0.6)},{_f(-r * 0.2)}" stroke="#33691e" stroke-width="{_f(r * 0.18)}" fill="none"/>'
            f'<circle cx="{_f(cx + r * 0.45)}" cy="{_f(cy + r * 0.35)}" r="{_f(r * 0.32)}" fill="#ffb300"/>')


def powder_icon(cx: float, cy: float, r: float) -> str:
    pts = ' '.join(f'{_f(cx + r * math.cos(math.pi / 8 + k * math.pi / 4))},'
                   f'{_f(cy + r * math.sin(math.pi / 8 + k * math.pi / 4))}' for k in range(8))
    return (f'<polygon points="{pts}" fill="{POWDER}" stroke="{POWDER_EDGE}" stroke-width="1"/>'
            f'<rect x="{_f(cx - r * 0.42)}" y="{_f(cy - r * 0.45)}" width="{_f(r * 0.84)}" '
            f'height="{_f(r * 0.9)}" rx="{_f(r * 0.2)}" fill="#8d6e63"/>'
            f'<line x1="{_f(cx - r * 0.42)}" y1="{_f(cy)}" x2="{_f(cx + r * 0.42)}" y2="{_f(cy)}" '
            f'stroke="#3e2723" stroke-width="{_f(r * 0.1)}"/>')


def arrow_icon(cx: float, cy: float, r: float, forward: bool) -> str:
    color, edge = ('#43a047', '#1b5e20') if forward else ('#e53935', '#8e0000')
    s = 1 if forward else -1
    pts = [(-0.8, -0.3), (0.1, -0.3), (0.1, -0.75), (0.85, 0), (0.1, 0.75), (0.1, 0.3), (-0.8, 0.3)]
    p = ' '.join(f'{_f(cx + s * x * r)},{_f(cy + y * r)}' for x, y in pts)
    return f'<polygon points="{p}" fill="{color}" stroke="{edge}" stroke-width="1"/>'


def symbol_icon(sym: int, cx: float, cy: float, r: float) -> str:
    if sym == Sym.GOLD:
        return coin(cx, cy, r)
    if sym == Sym.FOOD:
        return food_icon(cx, cy, r * 0.9)
    if sym == Sym.POWDER:
        return powder_icon(cx, cy, r)
    return arrow_icon(cx, cy, r, sym == Sym.FWD)


def res_icon(res: int, cx: float, cy: float, r: float) -> str:
    return {Res.GOLD: coin, Res.FOOD: food_icon, Res.POWDER: powder_icon}[res](cx, cy, r) \
        if res != Res.EMPTY else ''


def ship(cx: float, cy: float, color: str, size: float = 14.0, title: str = '') -> str:
    """A small pirate ship in a seat colour."""
    s = size
    t = f'<title>{escape(title)}</title>' if title else ''
    return (f'<g>{t}<path d="M{_f(cx - s * 0.75)},{_f(cy + s * 0.1)} L{_f(cx + s * 0.75)},'
            f'{_f(cy + s * 0.1)} L{_f(cx + s * 0.45)},{_f(cy + s * 0.5)} L{_f(cx - s * 0.5)},'
            f'{_f(cy + s * 0.5)} Z" fill="#5d4037" stroke="#1b0000" stroke-width="1"/>'
            f'<line x1="{_f(cx)}" y1="{_f(cy + s * 0.1)}" x2="{_f(cx)}" y2="{_f(cy - s * 0.8)}" '
            f'stroke="#3e2723" stroke-width="1.4"/>'
            f'<path d="M{_f(cx + 1)},{_f(cy - s * 0.75)} L{_f(cx + s * 0.62)},{_f(cy - s * 0.05)} '
            f'L{_f(cx + 1)},{_f(cy - s * 0.05)} Z" fill="{color}" stroke="#ffffff" stroke-width="1"/>'
            f'<path d="M{_f(cx - 1)},{_f(cy - s * 0.6)} L{_f(cx - s * 0.5)},{_f(cy - s * 0.05)} '
            f'L{_f(cx - 1)},{_f(cy - s * 0.05)} Z" fill="{color}" stroke="#ffffff" stroke-width="1" '
            f'opacity="0.85"/></g>')


def chest(cx: float, cy: float, s: float = 7.0) -> str:
    """Treasure token still waiting in a lair."""
    return (f'<g><title>treasure</title><rect x="{_f(cx - s)}" y="{_f(cy - s * 0.55)}" width="{_f(2 * s)}" '
            f'height="{_f(s * 1.2)}" rx="{_f(s * 0.2)}" fill="#8d4e19" stroke="#3e2000" stroke-width="1"/>'
            f'<rect x="{_f(cx - s)}" y="{_f(cy - s * 0.55)}" width="{_f(2 * s)}" height="{_f(s * 0.4)}" '
            f'fill="#b26a2a" stroke="#3e2000" stroke-width="1"/>'
            f'<rect x="{_f(cx - s * 0.18)}" y="{_f(cy - s * 0.3)}" width="{_f(s * 0.36)}" '
            f'height="{_f(s * 0.5)}" fill="{GOLD}"/></g>')


def star(cx: float, cy: float, r: float = 9.0, fill: str = '#fdff00') -> str:
    pts = []
    for k in range(10):
        rr = r if k % 2 == 0 else r * 0.45
        a = -math.pi / 2 + k * math.pi / 5
        pts.append(f'{_f(cx + rr * math.cos(a))},{_f(cy + rr * math.sin(a))}')
    return f'<polygon points="{" ".join(pts)}" fill="{fill}" stroke="#605a00" stroke-width="1.2"/>'


# --------------------------------------------------------------------------- #
# the board
# --------------------------------------------------------------------------- #

def space_glyph(kind: int, cost: int, cx: float, cy: float, r: float = SPACE_R) -> str:
    if kind == Kind.SEA:
        return (f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" fill="#1f78b4" stroke="#e3f2fd" '
                f'stroke-width="1.6"/>' + diamonds(cx, cy, cost))
    if kind == Kind.PORT:
        return (f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" fill="#fff8e1" stroke="{GOLD_EDGE}" '
                f'stroke-width="1.6"/>' + coin(cx, cy, r * 0.72, str(cost)))
    if kind == Kind.LAIR:
        return (f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" fill="#6d4c41" stroke="#d7ccc8" '
                f'stroke-width="1.6"/>' + skull(cx, cy, r * 0.95))
    # Port Royal
    return (f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r * 1.35)}" fill="#b71c1c" stroke="{GOLD}" '
            f'stroke-width="2.2"/>' + star(cx, cy, r * 0.9, fill=GOLD))


def board_svg(track) -> str:
    """Static board: sea, land, track links, spaces, printed scores, the -5 line."""
    w, h = BOARD_W, BOARD_H
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
             '<defs><linearGradient id="jm-sea" x1="0" y1="0" x2="0" y2="1">'
             f'<stop offset="0" stop-color="{SEA_TOP}"/><stop offset="1" stop-color="{SEA_BOTTOM}"/>'
             '</linearGradient>'
             '<pattern id="jm-waves" width="40" height="18" patternUnits="userSpaceOnUse">'
             '<path d="M0,9 q10,-6 20,0 t20,0" fill="none" stroke="#ffffff" stroke-opacity="0.08" '
             'stroke-width="1.5"/></pattern></defs>',
             f'<rect width="{w}" height="{h}" fill="url(#jm-sea)"/>',
             f'<rect width="{w}" height="{h}" fill="url(#jm-waves)"/>']
    for outline in (data.MAIN_ISLAND,) + data.OUTER_ISLANDS:
        parts.append(f'<path d="{smooth_path(outline)}" fill="{SAND}" stroke="{SAND_EDGE}" '
                     f'stroke-width="2"/>')
    # a forested heart for the main island
    cx = sum(p[0] for p in data.MAIN_ISLAND) / len(data.MAIN_ISLAND)
    cy = sum(p[1] for p in data.MAIN_ISLAND) / len(data.MAIN_ISLAND)
    inner = [(cx + (x - cx) * 0.72, cy + (y - cy) * 0.8) for x, y in data.MAIN_ISLAND]
    parts.append(f'<path d="{smooth_path(inner)}" fill="{FOREST}" opacity="0.55"/>')
    parts.append(f'<text x="{_f(xy((cx, cy))[0])}" y="{_f(xy((cx, cy))[1])}" text-anchor="middle" '
                 f'font-family="Georgia, serif" font-size="30" font-style="italic" fill="#3e2723" '
                 f'opacity="0.55" transform="rotate(-90 {_f(xy((cx, cy))[0])} {_f(xy((cx, cy))[1])})">'
                 f'Jamaica</text>')
    for node in track.lairs:                   # a rocky islet under every lair
        px, py = xy(track.xy[node])
        parts.append(f'<circle cx="{_f(px)}" cy="{_f(py)}" r="{_f(SPACE_R * 1.9)}" fill="#8d8378" '
                     f'stroke="#5d5348" stroke-width="1.5" opacity="0.85"/>')
    # track links
    for a in range(track.n):
        for b in track.succ[a]:
            (x1, y1), (x2, y2) = xy(track.xy[a]), xy(track.xy[b])
            parts.append(f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" '
                         f'stroke="#ffffff" stroke-opacity="0.7" stroke-width="2" stroke-dasharray="5 4"/>')
    (fx1, fy1), (fx2, fy2) = xy(data.FLOOR_LINE[0]), xy(data.FLOOR_LINE[1])
    parts.append(f'<line x1="{_f(fx1)}" y1="{_f(fy1)}" x2="{_f(fx2)}" y2="{_f(fy2)}" stroke="#ff1744" '
                 f'stroke-width="2.5"/>')
    parts.append(f'<text x="{_f(fx2 - 4)}" y="{_f(fy2 - 5)}" text-anchor="end" font-size="11" '
                 f'font-family="sans-serif" font-weight="bold" fill="#ff1744">-5</text>')
    for node in range(track.n):
        px, py = xy(track.xy[node])
        parts.append(f'<g id="jm-space-{node}"><title>space {node}</title>'
                     + space_glyph(track.kind[node], track.cost[node], px, py) + '</g>')
        score = track.score[node]
        if score is not None:
            parts.append(f'<text x="{_f(px + SPACE_R + 9)}" y="{_f(py + 4)}" text-anchor="middle" '
                         f'font-size="11" font-family="sans-serif" font-weight="bold" fill="#ffffff" '
                         f'stroke="#0d47a1" stroke-width="0.5">{score}</text>')
    parts.append('</svg>')
    return ''.join(parts)


# --------------------------------------------------------------------------- #
# cards, dice, holds
# --------------------------------------------------------------------------- #

def card_svg(code: int, dice=None, width: int = 104, height: int = 58, dim: bool = False) -> str:
    """An action card: sun + morning symbol, moon + evening symbol."""
    m, e = card_syms(code)
    op = ' opacity="0.45"' if dim else ''
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 104 58"{op}>',
            '<rect x="1" y="1" width="102" height="56" rx="8" fill="#fdf6e3" stroke="#8d6e63" stroke-width="2"/>',
            '<line x1="52" y1="6" x2="52" y2="52" stroke="#d7ccc8" stroke-width="1.5"/>',
            '<circle cx="12" cy="12" r="5" fill="#ffb300"/>',
            '<path d="M92,7 a5,5 0 1,0 5,8 a4,4 0 1,1 -5,-8" fill="#5c6bc0"/>',
            symbol_icon(m, 28, 32, 13), symbol_icon(e, 78, 32, 13)]
    if dice is not None:
        body.append(f'<text x="28" y="54" text-anchor="middle" font-size="10" font-family="sans-serif" '
                    f'fill="#5d4037">{dice[0]}</text>')
        body.append(f'<text x="78" y="54" text-anchor="middle" font-size="10" font-family="sans-serif" '
                    f'fill="#5d4037">{dice[1]}</text>')
    body.append('</svg>')
    return ''.join(body)


def card_back_svg(width: int = 52, height: int = 29) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 104 58">'
            '<rect x="1" y="1" width="102" height="56" rx="8" fill="#6d4c41" stroke="#3e2723" stroke-width="2"/>'
            + skull(52, 30, 20, fill='#d7ccc8', ink='#3e2723') + '</svg>')


_PIPS = {1: [(0, 0)], 2: [(-1, -1), (1, 1)], 3: [(-1, -1), (0, 0), (1, 1)],
         4: [(-1, -1), (1, -1), (-1, 1), (1, 1)], 5: [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)],
         6: [(-1, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (1, 1)]}


def die_svg(value: int, size: int = 34, label: str = '') -> str:
    s = size
    pips = ''.join(f'<circle cx="{_f(s / 2 + x * s * 0.25)}" cy="{_f(s / 2 + y * s * 0.25)}" '
                   f'r="{_f(s * 0.08)}" fill="#212121"/>' for x, y in _PIPS.get(value, []))
    t = f'<title>{escape(label)}</title>' if label else ''
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 {s} {s}">{t}'
            f'<rect x="1" y="1" width="{s - 2}" height="{s - 2}" rx="{_f(s * 0.18)}" fill="#fffde7" '
            f'stroke="#6d4c41" stroke-width="1.5"/>{pips}</svg>')


def combat_face_svg(face: int, size: int = 34) -> str:
    s = size
    inner = star(s / 2, s / 2, s * 0.36, fill='#ffffff') if face == STAR else (
        f'<text x="{_f(s / 2)}" y="{_f(s * 0.68)}" text-anchor="middle" font-size="{_f(s * 0.5)}" '
        f'font-family="Georgia, serif" font-weight="bold" fill="#212121">{face}</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 {s} {s}">'
            f'<rect x="1" y="1" width="{s - 2}" height="{s - 2}" rx="{_f(s * 0.18)}" fill="#efebe9" '
            f'stroke="#4e342e" stroke-width="1.5"/>{inner}</svg>')


def hold_svg(res: int, count: int, sixth: bool = False, width: int = 44, height: int = 44,
             highlight: bool = False) -> str:
    """One hold of a ship, with its tokens."""
    edge = '#ff6f00' if highlight else ('#4e342e' if not sixth else '#1565c0')
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 44 44">',
             f'<rect x="1.5" y="1.5" width="41" height="41" rx="5" fill="#a1887f" stroke="{edge}" '
             f'stroke-width="{3 if highlight else 2}"/>',
             '<line x1="4" y1="15" x2="40" y2="15" stroke="#8d6e63"/>',
             '<line x1="4" y1="29" x2="40" y2="29" stroke="#8d6e63"/>']
    if res != Res.EMPTY:
        parts.append(res_icon(res, 16, 22, 9))
        parts.append(f'<text x="34" y="28" text-anchor="middle" font-size="15" font-weight="bold" '
                     f'font-family="sans-serif" fill="#ffffff" stroke="#3e2723" stroke-width="0.6">{count}</text>')
    if sixth:
        parts.append('<text x="38" y="11" text-anchor="end" font-size="8" font-family="sans-serif" '
                     'fill="#e3f2fd">6th</text>')
    parts.append('</svg>')
    return ''.join(parts)
