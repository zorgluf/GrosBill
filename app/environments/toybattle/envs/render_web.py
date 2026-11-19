"""NiceGUI renderer for Toy Battle (`toybattle`), laid out for a phone held upright.

Same contract as the other games: `ToyBattleEnv.nicegui_page()` builds a
:class:`RenderWeb` and calls :meth:`RenderWeb.init_web`, then every
`ToyBattleEnv.render()` calls :meth:`RenderWeb.render_web`.

One narrow column (it is centred on a wide screen)::

    terrain, whose turn, what to do
    opponent: medals, face-down rack, reserve, discard
    the board (the viewer's H.Q. at the bottom; tap a base or an H.Q.)
    me: medals, reserve, discard, my rack (tap a troop), Draw / Skip
    the last events, the full log and a rules reminder (folded)

To place a troop: tap it on the rack (the bases it may go to light up), then
tap the base. The other decisions (Jumbo, the special bases) light their
targets up directly.

The board is our own drawing (`art.board_svg`, a data URI) under an SVG overlay
(medals, troops, highlights). Painting is synchronous, section by section, as
in the other recent renderers (see `environments/smallw/envs/render_web.py`).
Everything that decides *what* to show is a pure module-level function, tested
headlessly in `tests/test_render.py`.
"""

from __future__ import annotations

import base64
import math
from functools import lru_cache

from nicegui import ui

from . import art
from .constants import (A_DRAW, A_PLACE, A_SKIP, A_TARGET, A_TYPE, BLUE, COLOR_NAMES, EFFECT_TEXT,
                        MAX_BASES, N_SLOTS, N_TYPES, Phase, RED, TROOP_EFFECTS, TROOP_NAMES,
                        hq_slot, place_action)
from .terrains import TERRAINS, medal_spot

PROMPTS = {
    Phase.MAIN: 'Tap a troop then a base, or draw.',
    Phase.EXTRA: "Cap'n: place an extra troop, or skip.",
    Phase.JUMBO: 'Jumbo: tap an adjacent enemy troop to discard it, or skip.',
    Phase.RECALL: 'Special base: tap one of your troops to take it back, or skip.',
    Phase.EXHUME: 'Special base: take a troop back from your discard, or skip.',
    Phase.ERUPT_SRC: 'Eruption: tap an adjacent enemy troop to move it, or skip.',
    Phase.ERUPT_DST: 'Eruption: tap the base it moves to.',
}
LEGAL, MARK, SUGGEST, ORIGIN = '#00e676', '#ffeb3b', '#ff9100', '#ffffff'
CLICK_RADIUS = 10.5         #: board units around a node that count as a tap on it
CSS = """
@keyframes tb-pulse { 50% { opacity: 0.3; } }
.tb-pulse { animation: tb-pulse 1.3s ease-in-out infinite; }
@media (prefers-reduced-motion: reduce) { .tb-pulse { animation: none; } }
.tb-tile { cursor: pointer; border-radius: 9px; padding: 1px; line-height: 0; }
.tb-tile.tb-selected { outline: 3px solid #00c853; outline-offset: 1px; }
.tb-tile.tb-suggested { outline: 3px dashed #ff9100; outline-offset: 1px; }
"""


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #

def viewer(env) -> int:
    """Seat whose rack the page shows (-1 = both)."""
    pov = env.pov_player
    return pov if pov in (BLUE, RED) else -1


def bottom_seat(env) -> int:
    """Seat drawn at the bottom of the board (Blue when watching)."""
    return max(viewer(env), BLUE)


def is_human_decision(env, callback=None) -> bool:
    """True when the page may send an action for the seat it displays."""
    if callback is None or env.done or env.current_player < 0:
        return False
    return viewer(env) in (-1, env.current_player)


def nodes(env, flip: bool) -> list[tuple[tuple, float, float]]:
    """Every node of the terrain: (('base', b) or ('hq', owner, k), x, y) in pixels."""
    T = env.terrain
    out = [(('base', b), *art.pt(base.x, base.y, flip)) for b, base in enumerate(T.bases)]
    for owner in (BLUE, RED):
        out += [(('hq', owner, k), *art.pt(hq.x, hq.y, flip)) for k, hq in enumerate(T.hqs[owner])]
    return out


def node_at(env, x: float, y: float, flip: bool) -> tuple | None:
    """The node tapped at pixel (x, y) of the board image."""
    best, best_d = None, CLICK_RADIUS * art.PX
    for node, nx, ny in nodes(env, flip):
        d = math.hypot(nx - x, ny - y)
        if d <= best_d:
            best, best_d = node, d
    return best


def node_slot(env, node: tuple) -> int | None:
    """Action slot of `node` for the player to move (None for their own H.Q.)."""
    p = env.current_player
    if node[0] == 'base':
        return env.terrain.slots[p][node[1]]
    return hq_slot(node[2]) if node[1] != p else None


def selectable_troops(env) -> list[int]:
    """Troop types the player to move may place now."""
    if env.done or env.phase not in (Phase.MAIN, Phase.EXTRA):
        return []
    return sorted({t for t, _ in env.placements(env.current_player)})


def action_for_click(env, node: tuple | None, troop: int | None) -> int | None:
    """The legal action a tap on `node` means (with `troop` selected on the rack)."""
    if node is None or env.done:
        return None
    slot = node_slot(env, node)
    if slot is None:
        return None
    if env.phase in (Phase.MAIN, Phase.EXTRA):
        action = None if troop is None else place_action(troop, slot)
    elif env.phase == Phase.EXHUME or node[0] != 'base':
        action = None
    else:
        action = A_TARGET + slot
    return action if action is not None and env.action_masks()[action] else None


def legal_nodes(env, troop: int | None) -> list[tuple]:
    """Nodes a tap would act on."""
    flip = False
    return [node for node, _, _ in nodes(env, flip) if action_for_click(env, node, troop) is not None]


def suggested_parts(env, action: int | None) -> tuple[int | None, tuple | None]:
    """(troop, node) a suggested action points at."""
    if action is None or action in (A_DRAW, A_SKIP):
        return None, None
    T, p = env.terrain, env.current_player
    if action >= A_TYPE:
        return action - A_TYPE, None
    troop, slot = (None, action - A_TARGET) if action >= A_TARGET else divmod(action - A_PLACE, N_SLOTS)
    if slot >= MAX_BASES:
        return troop, ('hq', 1 - p, slot - MAX_BASES)
    return troop, (('base', T.views[p][slot]) if slot < T.n_bases else None)


def stack_text(env, b: int) -> str:
    T = env.terrain
    base = T.bases[b]
    kind = 'Special base' if base.special else 'Base'
    stack = env.stacks[b]
    if not stack:
        return f'{kind} {base.name}: empty'
    tiles = ', '.join(f'{COLOR_NAMES[o]} {TROOP_NAMES[t]}' for o, t in reversed(stack))
    return f'{kind} {base.name}, top first: {tiles}'


def status_text(env) -> str:
    T = env.terrain
    if env.done:
        if env.winner_player is None:
            return f'Draw ({env.end_reason}).'
        return f'{env.player_names[env.winner_player]} ({COLOR_NAMES[env.winner_player]}) wins: {env.end_reason}.'
    p = env.current_player
    return f'{T.name} · turn {env.turns_taken + 1} · {env.player_names[p]} ({COLOR_NAMES[p]}) to play'


def buttons(env) -> list[dict]:
    """The decisions that are not a tap on the board: draw, skip, exhume."""
    out = []
    for a in env.legal_actions():
        if a == A_DRAW:
            n = min(2, 8 - env.rack_size(env.current_player), len(env.reserve[env.current_player]))
            out.append({'action': a, 'label': f'Draw {n}'})
        elif a == A_SKIP:
            out.append({'action': a, 'label': 'Skip'})
        elif a >= A_TYPE:
            out.append({'action': a, 'label': TROOP_NAMES[a - A_TYPE], 'troop': a - A_TYPE})
    return out


def overlay_svg(env, selected: int | None = None, suggested: int | None = None, playable: bool = False) -> str:
    """Medals, troops and highlights, drawn over `art.board_svg`."""
    T = env.terrain
    flip = bottom_seat(env) == RED
    out = []
    for r, region in enumerate(T.regions):
        x, y = art.pt(*medal_spot(region), flip)
        taker = env.region_owner[r]
        if taker < 0:
            out.append(art.medals_svg(x, y, region.medals))
        else:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{art.PLAYER_FILL[taker]}" '
                       f'stroke="#ffffff" stroke-width="1.5" opacity="0.85"/>')
    xy = {node: (x, y) for node, x, y in nodes(env, flip)}
    for b in set(env.marks):
        out.append(art.ring_svg(*xy[('base', b)], MARK))
    if env.pending_base is not None and not env.done:
        out.append(art.ring_svg(*xy[('base', env.pending_base)], ORIGIN, grow=1.5))
    for b, stack in enumerate(env.stacks):
        if stack:
            out.append(art.stack_svg(*xy[('base', b)], stack))
    if env.captured is not None:
        owner, k, troop = env.captured
        out.append(art.stack_svg(*xy[('hq', owner, k)], [(1 - owner, troop)]))
    if playable:
        for node in legal_nodes(env, selected):
            out.append(art.ring_svg(*xy[node], LEGAL, dashed=True, pulse=True))
        _, node = suggested_parts(env, suggested)
        if node is not None and node in xy:
            out.append(art.ring_svg(*xy[node], SUGGEST, grow=2.5))
    return ''.join(out)


@lru_cache(maxsize=32)
def _board_uri(key: str, flip: bool) -> str:
    svg = art.board_svg(TERRAINS[key], flip)
    return 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode()


def _svg(content: str):
    return ui.html(content, sanitize=False)


# --------------------------------------------------------------------------- #
# the page
# --------------------------------------------------------------------------- #

SECTIONS = ('status', 'opponent', 'board', 'me', 'log')
LAST_LINES = 4


class RenderWeb:

    def __init__(self):
        self._slots: dict[str, object] = {}
        self._env = None
        self._callback = None
        self._suggested: int | None = None
        self.selected: int | None = None        #: troop picked on the rack

    # -- actions --------------------------------------------------------- #

    def _playable(self) -> bool:
        return self._env is not None and is_human_decision(self._env, self._callback)

    def _send(self, action: int) -> None:
        env, callback = self._env, self._callback
        if env is None or callback is None or not is_human_decision(env, callback):
            ui.notify('It is not your turn.', type='info')
            return
        if not env.action_masks()[action]:
            ui.notify(f'Not possible: {env.describe_action(action)}', type='warning')
            return
        self.selected = None
        callback(action)

    def _select(self, troop: int) -> None:
        if not self._playable() or troop not in selectable_troops(self._env):
            ui.notify(f'{art.troop_title(troop)}: {TROOP_EFFECTS[troop]}', type='info')
            return
        self.selected = None if self.selected == troop else troop
        self._paint(('board', 'me'))

    def _on_board(self, event) -> None:
        env = self._env
        x, y = getattr(event, 'image_x', None), getattr(event, 'image_y', None)
        if env is None or x is None or y is None:
            return
        node = node_at(env, x, y, bottom_seat(env) == RED)
        if node is None:
            return
        info = stack_text(env, node[1]) if node[0] == 'base' else f'{COLOR_NAMES[node[1]]} H.Q.'
        if not self._playable():
            ui.notify(info, type='info')
            return
        action = action_for_click(env, node, self.selected)
        if action is None:
            if env.phase in (Phase.MAIN, Phase.EXTRA) and self.selected is None:
                info = f'Tap a troop of your rack first. {info}'
            ui.notify(info, type='info')
            return
        self._send(action)

    # -- painting -------------------------------------------------------- #

    def _paint(self, names=None) -> None:
        if self._env is None:
            return
        for name in names or SECTIONS:
            slot = self._slots.get(name)
            if slot is None:
                continue
            slot.clear()
            with slot:
                getattr(self, f'_build_{name}')()

    def _build_status(self):
        env = self._env
        ui.label(status_text(env)).classes('text-sm font-bold')
        if self._playable():
            ui.label(PROMPTS[env.phase]).classes('text-sm text-deep-orange-9')
        elif not env.done:
            ui.label(f'Waiting for {env.player_names[env.current_player]}…').classes('text-sm text-italic')

    def _player_line(self, seat: int):
        """Name, medals, reserve and discard of a seat, on one line."""
        env = self._env
        T = env.terrain
        with ui.row().classes('w-full items-center gap-2 no-wrap'):
            ui.element('div').style(f'width: 12px; height: 12px; border-radius: 3px; flex: none; '
                                    f'background: {art.PLAYER_FILL[seat]};')
            ui.label(env.player_names[seat]).classes('text-sm font-bold ellipsis')
            _svg(art.medal_svg(18))
            ui.label(f'{env.medals[seat]}/{T.objective}').classes('text-sm font-bold')
            ui.label(f'reserve {len(env.reserve[seat])}').classes('text-xs text-grey-8')
            ui.space()
            discard = [t for t in range(N_TYPES) for _ in range(env.discard[seat][t])]
            if discard:
                ui.label('discard').classes('text-xs text-grey-8')
                with ui.row().classes('gap-0 no-wrap'):
                    for t in discard:
                        _svg(art.troop_svg(seat, t, 22, dim=True))

    def _rack(self, seat: int, open_: bool):
        env = self._env
        playable = self._playable() and env.current_player == seat
        choices = selectable_troops(env) if playable else []
        hint, _ = suggested_parts(env, self._suggested) if playable else (None, None)
        with ui.row().classes('w-full gap-1 items-center').style('min-height: 30px;'):
            if not open_:
                for _ in range(env.rack_size(seat)):
                    _svg(art.back_svg(seat, 26))
                if not env.rack_size(seat):
                    ui.label('empty rack').classes('text-xs text-grey-7')
                return
            if not env.rack_size(seat):
                ui.label('empty rack').classes('text-xs text-grey-7')
            for t in range(N_TYPES):
                for i in range(env.racks[seat][t]):
                    pinned = i < env.blocked[seat][t]
                    usable = t in choices and not pinned
                    classes = 'tb-tile'
                    if usable and self.selected == t:
                        classes += ' tb-selected'
                    elif usable and hint == t and env.phase in (Phase.MAIN, Phase.EXTRA):
                        classes += ' tb-suggested'
                    tile = ui.element('div').classes(classes).on('click', lambda _, t=t: self._select(t))
                    with tile:
                        _svg(art.troop_svg(seat, t, 42, dim=playable and not usable))
                        tip = f'{art.troop_title(t)}: {TROOP_EFFECTS[t]}' + (' (pinned down this turn)' if pinned else '')
                        ui.tooltip(tip)

    def _build_opponent(self):
        env = self._env
        seat = 1 - bottom_seat(env)
        self._player_line(seat)
        self._rack(seat, open_=viewer(env) == -1)

    def _build_board(self):
        env = self._env
        flip = bottom_seat(env) == RED
        playable = self._playable()
        if playable and env.phase in (Phase.MAIN, Phase.EXTRA):
            choices = selectable_troops(env)
            if self.selected not in choices:
                self.selected = choices[0] if len(choices) == 1 else None
        else:
            self.selected = None
        ui.interactive_image(_board_uri(env.terrain_key, flip),
                             content=overlay_svg(env, self.selected, self._suggested, playable),
                             on_mouse=self._on_board, events=['click'], cross=False, sanitize=False
                             ).style(f'width: min(100%, calc(66vh * {art.BOARD_W} / {art.BOARD_H})); '
                                     f'max-width: {art.BOARD_W + 80}px;')

    def _build_me(self):
        env = self._env
        seat = bottom_seat(env)
        self._player_line(seat)
        self._rack(seat, open_=True)
        if not self._playable():
            return
        with ui.row().classes('w-full gap-2 items-center'):
            for b in buttons(env):
                star = '★ ' if b['action'] == self._suggested else ''
                if 'troop' in b:
                    with ui.element('div').classes('tb-tile').on('click', lambda _, a=b['action']: self._send(a)):
                        _svg(art.troop_svg(env.current_player, b['troop'], 42))
                        ui.tooltip(f"{star}Take {b['label']} back")
                else:
                    color = 'primary' if b['action'] == A_DRAW else 'grey-8'
                    ui.button(star + b['label'], on_click=lambda _, a=b['action']: self._send(a)) \
                        .props(f'no-caps unelevated color={color}')

    def _build_log(self):
        env = self._env
        T = env.terrain
        lines = [text for _, text in env.log_lines(viewer(env))]
        with ui.column().classes('w-full gap-0'):
            for text in lines[-LAST_LINES:]:
                ui.label(text).classes('text-xs text-grey-9').style('line-height: 1.25;')
        with ui.expansion('Game log').classes('w-full').props('dense'):
            for text in lines[-300:]:
                ui.label(text).classes('text-xs').style('line-height: 1.25;')
        with ui.expansion('Rules reminder').classes('w-full').props('dense'):
            ui.label(f'{T.name}: {T.objective} medals to win, or place a troop on the enemy H.Q. '
                     f'{EFFECT_TEXT[T.effect]}').classes('text-xs')
            ui.label('On your turn: draw 2 troops (8 on the rack at most), or place 1 troop on a base '
                     'connected to your H.Q. through bases you occupy: an empty base, one of yours, or an '
                     'enemy troop of strictly lower strength. Surround a region to take its medals.').classes('text-xs')
            for t in range(N_TYPES):
                with ui.row().classes('items-center gap-2 no-wrap'):
                    _svg(art.troop_svg(bottom_seat(env), t, 24))
                    ui.label(f'{art.troop_title(t)}: {TROOP_EFFECTS[t]}').classes('text-xs')

    # -- contract -------------------------------------------------------- #

    def init_web(self, env, callback=None):
        ui.add_css(CSS)
        self._env, self._callback, self._suggested = env, callback, None
        slots = {}
        with ui.column().classes('w-full items-center q-pa-xs'):
            with ui.column().classes('w-full gap-1 items-stretch').style('max-width: 520px;'):
                slots['status'] = ui.column().classes('w-full gap-0')
                slots['opponent'] = ui.column().classes('w-full gap-1')
                slots['board'] = ui.column().classes('w-full gap-0 items-center')
                slots['me'] = ui.column().classes('w-full gap-1')
                slots['log'] = ui.column().classes('w-full gap-0')
        self._slots = slots
        self._paint()

    def render_web(self, env, callback=None, suggested_action: int | None = None, **kwargs):
        self._env, self._callback, self._suggested = env, callback, suggested_action
        self._paint()
