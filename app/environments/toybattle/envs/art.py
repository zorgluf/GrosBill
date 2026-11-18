"""Our own drawings of Toy Battle: flat SVG, sized for a phone screen.

Nothing here reproduces the published art. A board is its graph, drawn the
same way for the 8 terrains (rounded squares for the bases, a double outline
and a glyph for the special bases, a flag for the H.Q., thick lines for the
paths, gold dots for the medals) on a background tinted per terrain. Troops
are coloured tiles showing their strength.

Board units are those of `terrains.py` (100 x 166); `PX` pixels per unit.
`flip=True` draws the half-turn of the board (Red's point of view), so that
the viewer's H.Q. is always at the bottom.
"""

from __future__ import annotations

import math

from .constants import BLUE, Effect, RED, TROOP_NAMES, Troop
from .terrains import H, W, Terrain

PX = 4
BOARD_W, BOARD_H = int(W * PX), int(H * PX)
BASE = 15.0                 #: side of a base, in board units
TILE = 13.0                 #: side of a troop tile on the board
HQ_R = 9.5

PLAYER_FILL = ('#2f6fde', '#dc4a3d')
PLAYER_DARK = ('#17408f', '#8f241b')
GOLD = '#ffc928'

#: background, background accent, path, base fill, base outline, special accent, glyph
THEMES = {
    'castle':      ('#9ccc65', '#8bc34a', '#f4e7b0', '#dfe4e6', '#78909c', '#8d5a2b', '#5d3a1a'),
    'pool':        ('#63c8f2', '#4db6e6', '#ffffff', '#455a64', '#263238', '#ff9800', '#ffb74d'),
    'clouds':      ('#f9cfe0', '#f4b6d0', '#ffffff', '#fde047', '#c79a00', '#8e5bd6', '#5e35b1'),
    'jungle':      ('#3f8f47', '#357a3c', '#e2c79a', '#b9654a', '#6d3324', '#ff7a1a', '#ffe0b2'),
    'cemetery':    ('#4a6f60', '#3f6153', '#b9ab98', '#a9b9c2', '#546e7a', '#ff8f1f', '#bf360c'),
    'caribbean':   ('#3b9be8', '#2f89d6', '#b47a3c', '#8bc34a', '#f1dfa5', '#ffffff', '#33691e'),
    'station':     ('#4a5560', '#3e4852', '#aab3ea', '#eef1f3', '#90a4ae', '#4dd8ff', '#00838f'),
    'battlefield': ('#ecd08a', '#e3c373', '#9a7355', '#c9b8ab', '#7b6457', '#43a047', '#1b5e20'),
}

#: glyph of the special bases (plain text, in every font)
EFFECT_GLYPH = {Effect.RECALL: '↩', Effect.DRAW: '+1', Effect.ERUPT: '⇄', Effect.EXHUME: '†',
                Effect.SHIELD: '∅', Effect.SNIPER: '⌖'}
#: reminder of the troop effect, in a corner of its tile
TROOP_GLYPH = ('', '+2', '×2', '✕', '↷', '−1', '+1', '')


def pt(x: float, y: float, flip: bool = False) -> tuple[float, float]:
    """Board units -> pixels."""
    if flip:
        x, y = W - x, H - y
    return x * PX, y * PX


def _f(v: float) -> str:
    return f'{v:.1f}'.rstrip('0').rstrip('.')


def gate_text(gate) -> str:
    return '/'.join(str(v) for v in gate)


def _star(cx: float, cy: float, r: float, fill: str, stroke: str = 'none', sw: float = 0) -> str:
    pts = []
    for k in range(10):
        rad = r if k % 2 == 0 else r * 0.45
        a = -math.pi / 2 + k * math.pi / 5
        pts.append(f'{_f(cx + rad * math.cos(a))},{_f(cy + rad * math.sin(a))}')
    return f'<polygon points="{" ".join(pts)}" fill="{fill}" stroke="{stroke}" stroke-width="{_f(sw)}"/>'


def _text(x: float, y: float, text: str, size: float, fill: str = '#ffffff', weight: str = '700',
          extra: str = '') -> str:
    return (f'<text x="{_f(x)}" y="{_f(y)}" font-size="{_f(size)}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="middle" dominant-baseline="central" '
            f'font-family="Arial, Helvetica, sans-serif" {extra}>{text}</text>')


# --------------------------------------------------------------------------- #
# the board (static part)
# --------------------------------------------------------------------------- #

def board_svg(T: Terrain, flip: bool = False) -> str:
    """The empty terrain: background, paths, bases, H.Q. (no medal, no troop)."""
    bg, accent, path, base_fill, base_line, special, glyph_fill = THEMES[T.key]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{BOARD_W}" height="{BOARD_H}" '
           f'viewBox="0 0 {BOARD_W} {BOARD_H}">',
           f'<rect width="{BOARD_W}" height="{BOARD_H}" rx="14" fill="{bg}"/>',
           f'<rect x="6" y="6" width="{BOARD_W - 12}" height="{BOARD_H - 12}" rx="10" fill="none" '
           f'stroke="{accent}" stroke-width="6"/>']
    for a, b in T.edges:
        (x0, y0), (x1, y1) = pt(*T.node_xy(a), flip), pt(*T.node_xy(b), flip)
        out.append(f'<line x1="{_f(x0)}" y1="{_f(y0)}" x2="{_f(x1)}" y2="{_f(y1)}" stroke="#00000033" '
                   f'stroke-width="{_f(4.2 * PX)}" stroke-linecap="round"/>')
    for a, b in T.edges:
        (x0, y0), (x1, y1) = pt(*T.node_xy(a), flip), pt(*T.node_xy(b), flip)
        out.append(f'<line x1="{_f(x0)}" y1="{_f(y0)}" x2="{_f(x1)}" y2="{_f(y1)}" stroke="{path}" '
                   f'stroke-width="{_f(3 * PX)}" stroke-linecap="round"/>')
    half = BASE * PX / 2
    for i, base in enumerate(T.bases):
        x, y = pt(base.x, base.y, flip)
        out.append(f'<g id="tb-base-{i}">')
        if base.special:
            out.append(f'<rect x="{_f(x - half - 3)}" y="{_f(y - half - 3)}" width="{_f(2 * half + 6)}" '
                       f'height="{_f(2 * half + 6)}" rx="13" fill="{special}" stroke="#00000055" stroke-width="1.5"/>')
        out.append(f'<rect x="{_f(x - half)}" y="{_f(y - half)}" width="{_f(2 * half)}" height="{_f(2 * half)}" '
                   f'rx="10" fill="{base_fill}" stroke="{base_line}" stroke-width="2.5"/>')
        if base.special:
            glyph = gate_text(base.gate) if base.gate else EFFECT_GLYPH.get(T.effect, '')
            size = 16 if len(glyph) > 3 else 24
            out.append(_text(x, y, glyph, size, glyph_fill))
        out.append('</g>')
    for owner in (BLUE, RED):
        for k, hq in enumerate(T.hqs[owner]):
            x, y = pt(hq.x, hq.y, flip)
            r = HQ_R * PX
            out.append(f'<g id="tb-hq-{owner}-{k}">'
                       f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(r)}" fill="{PLAYER_FILL[owner]}" '
                       f'stroke="{PLAYER_DARK[owner]}" stroke-width="4"/>'
                       f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(r - 7)}" fill="none" stroke="#ffffff" '
                       f'stroke-width="2" stroke-dasharray="5 4"/>')
            out.append(_text(x, y - (5 if hq.gate else 0), 'HQ', 17))
            if hq.gate:
                out.append(_text(x, y + 11, gate_text(hq.gate), 13, '#ffe082'))
            out.append('</g>')
    out.append('</svg>')
    return ''.join(out)


# --------------------------------------------------------------------------- #
# troops
# --------------------------------------------------------------------------- #

def _tile(x: float, y: float, side: float, owner: int, troop: int, opacity: float = 1.0) -> str:
    """A troop tile centred on (x, y), `side` pixels wide."""
    h = side / 2
    out = [f'<g opacity="{_f(opacity)}">'
           f'<rect x="{_f(x - h)}" y="{_f(y - h)}" width="{_f(side)}" height="{_f(side)}" rx="{_f(side * 0.18)}" '
           f'fill="{PLAYER_FILL[owner]}" stroke="#ffffff" stroke-width="{_f(max(side * 0.05, 1.5))}"/>']
    if troop == Troop.KWAK:
        out.append(_star(x, y - side * 0.04, side * 0.3, GOLD, PLAYER_DARK[owner], 1))
    else:
        out.append(_text(x - side * 0.08, y - side * 0.02, str(int(troop)), side * 0.62))
    glyph = TROOP_GLYPH[troop]
    if glyph:
        out.append(_text(x + side * 0.27, y + side * 0.3, glyph, side * 0.26, '#ffe082'))
    out.append('</g>')
    return ''.join(out)


def troop_svg(owner: int, troop: int, size: int = 44, dim: bool = False) -> str:
    """A troop tile as a standalone SVG (the rack)."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
            f'{_tile(size / 2, size / 2, size - 3, owner, troop, 0.35 if dim else 1.0)}</svg>')


def back_svg(owner: int, size: int = 26) -> str:
    """A face-down tile (the opponent's rack)."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
            f'<rect x="1.5" y="1.5" width="{size - 3}" height="{size - 3}" rx="{_f(size * 0.18)}" '
            f'fill="{PLAYER_DARK[owner]}" stroke="#ffffff" stroke-width="1.5"/>'
            f'{_star(size / 2, size / 2, size * 0.22, PLAYER_FILL[owner])}</svg>')


def medal_svg(size: int = 18) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
            f'<circle cx="{size / 2}" cy="{size / 2}" r="{size / 2 - 1}" fill="#37474f"/>'
            f'{_star(size / 2, size / 2, size * 0.36, GOLD)}</svg>')


def stack_svg(x: float, y: float, stack) -> str:
    """The tiles of a base: the visible one, the edges of up to two covered ones."""
    side = TILE * PX
    out = []
    below = stack[:-1][-2:]
    for depth, (owner, _) in enumerate(below):
        off = (len(below) - depth) * 4.0
        out.append(f'<rect x="{_f(x - side / 2 + off)}" y="{_f(y - side / 2 + off)}" width="{_f(side)}" '
                   f'height="{_f(side)}" rx="{_f(side * 0.18)}" fill="{PLAYER_DARK[owner]}" '
                   f'stroke="#ffffff" stroke-width="1.5"/>')
    owner, troop = stack[-1]
    out.append(_tile(x, y, side, owner, troop))
    if len(stack) > 1:
        bx, by = x + side / 2 - 2, y - side / 2 + 2
        out.append(f'<circle cx="{_f(bx)}" cy="{_f(by)}" r="8.5" fill="#263238" stroke="#ffffff" stroke-width="1.2"/>')
        out.append(_text(bx, by, str(len(stack)), 11))
    return ''.join(out)


def medals_svg(x: float, y: float, n: int) -> str:
    """`n` medals side by side around (x, y)."""
    out = []
    step = 15.0
    for k in range(n):
        cx = x + (k - (n - 1) / 2) * step
        out.append(f'<circle cx="{_f(cx)}" cy="{_f(y)}" r="6.8" fill="#37474f" stroke="#ffffff" stroke-width="1"/>')
        out.append(_star(cx, y, 4.8, GOLD))
    return ''.join(out)


def ring_svg(x: float, y: float, color: str, dashed: bool = False, grow: float = 0.0, pulse: bool = False) -> str:
    side = (BASE + 3 + grow) * PX
    dash = ' stroke-dasharray="9 6"' if dashed else ''
    cls = ' class="tb-pulse"' if pulse else ''
    return (f'<rect{cls} x="{_f(x - side / 2)}" y="{_f(y - side / 2)}" width="{_f(side)}" height="{_f(side)}" '
            f'rx="14" fill="none" stroke="{color}" stroke-width="4"{dash}/>')


def troop_title(troop: int) -> str:
    strength = 'joker' if troop == Troop.KWAK else f'strength {int(troop)}'
    return f'{TROOP_NAMES[troop]} ({strength})'
