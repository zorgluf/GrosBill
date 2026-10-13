"""NiceGUI renderer for Jamaica (`jamaica`).

Same contract as the other games: `JamaicaEnv.nicegui_page()` builds a
:class:`RenderWeb` and calls :meth:`RenderWeb.init_web`, then every
`JamaicaEnv.render()` calls :meth:`RenderWeb.render_web` (extra keyword
arguments from `play.py` are ignored).

Layout::

    +---------------------------+------------------------------------+
    | status (round, Captain, dice, whose decision, my instruction)  |
    +---------------------------+------------------------------------+
    | board (interactive image) | controls (buttons of the decision) |
    |                           | my hand, my holds, my treasures    |
    |                           | battle panel                       |
    |                           | players (one card per seat,        |
    |                           |   clockwise from me)               |
    +---------------------------+------------------------------------+
    | game log                                                       |
    +----------------------------------------------------------------+

New log lines also pop as short toasts at the top of the page (as in smallw).

The board is our own drawing (`art.board_svg`, served as a data URI, so no
static route and no reverse-proxy prefix issue) with an SVG overlay for the
ships, the treasure chests, the legal destinations and the suggestion star.

Ship moves (`env.move_log`) logged since the last repaint are animated in the
overlay: each ship sails space by space along its route, `move_duration`
seconds per move (one after another when a repaint shows several). They are
CSS keyframes, which restart whenever the overlay is rebuilt, so a repaint
without new moves shows the ships still. A move that starts a battle ends with
cannon smoke flashing on that space, and the battle panel flashes in step the
first time it shows that battle (`battle_flash` seconds, both).

Painting is synchronous (clear and re-fill one container per section) for the
reason explained in `environments/smallw/envs/render_web.py`: since NiceGUI 3
`ui.refreshable(...).refresh()` only schedules the rebuild.

Every function that decides *what* to show is a pure module-level function
(no nicegui call), tested headlessly in `tests/test_render.py`.
"""

from __future__ import annotations

import base64
import math
from functools import lru_cache

from nicegui import ui

from . import art, rules, sim
from .constants import (A_CAPTAIN, A_CARD, A_DEST, A_POWDER, A_SABRE, N_CODES, POWER_NAMES,
                        Phase, Power, RES_NAMES, Res, STAR, SYM_NAMES, card_syms)
from .rules import SIXTH_SLOT, hold_at

# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #

PHASE_PROMPTS = {
    Phase.CAPTAIN: 'you are the Captain: place the dice, then pick your card',
    Phase.CARD: 'pick your card for this round',
    Phase.MOVE_DEST: 'choose your route (click a highlighted space)',
    Phase.RETREAT_DEST: 'short of supplies: choose where to fall back',
    Phase.LOAD_HOLD: 'no empty hold: choose the hold to throw overboard',
    Phase.PAY_HOLD: 'choose the hold to pay from',
    Phase.TARGET: 'choose the ship to attack',
    Phase.ATTACK_POWDER: 'battle! how much gunpowder do you spend?',
    Phase.DEFENSE_POWDER: 'you are attacked! how much gunpowder do you spend?',
    Phase.SABRE: "Saran's Sabre: keep this roll or reroll?",
    Phase.REWARD: 'you won the battle: choose your loot',
}


def is_human_decision(env, callback=None, pov_player=None) -> bool:
    """True when the page may send an action for the seat it displays."""
    if callback is None or env.done:
        return False
    pov = env.pov_player if pov_player is None else pov_player
    if env.current_player < 0:
        return False
    return pov == -1 or env.current_player == pov


def viewer(env) -> int:
    """Seat whose private information the page shows (-1 = everything)."""
    pov = env.pov_player
    return -1 if pov is None else pov


def seat_order(env) -> list[int]:
    """Seats clockwise, starting with the viewer (seat 0 in god mode)."""
    start = max(viewer(env), 0)
    return [(start + k) % env.n_players for k in range(env.n_players)]


def board_xy(env, node: int) -> tuple[float, float]:
    return art.xy(env.track.xy[node])


def nearest_space(env, x: float, y: float, max_dist: float = art.SPACE_R * 2.2) -> int | None:
    best, best_d = None, max_dist
    for node in range(env.track.n):
        px, py = board_xy(env, node)
        d = math.hypot(px - x, py - y)
        if d <= best_d:
            best, best_d = node, d
    return best


def action_for_board_click(env, node: int | None) -> int | None:
    """Destination decisions only: the action that sends the ship to `node`."""
    d = env.pending
    if node is None or d is None or d.kind not in (Phase.MOVE_DEST, Phase.RETREAT_DEST):
        return None
    for k, (n, _) in enumerate(d.data):
        if n == node:
            return A_DEST + k
    return None


#: Where a lone ship sits relative to the centre of its space.
SHIP_OFFSET = (6, -14)


def ship_slots(env) -> dict[int, tuple[float, float]]:
    """Drawing position of every ship (fanned out when several share a space)."""
    by_node: dict[int, list[int]] = {}
    for s in env.ships:
        by_node.setdefault(s.node, []).append(s.seat)
    out = {}
    for node, seats in by_node.items():
        cx, cy = board_xy(env, node)
        k = len(seats)
        for i, seat in enumerate(seats):
            if k == 1:
                out[seat] = (cx + SHIP_OFFSET[0], cy + SHIP_OFFSET[1])
            else:
                a = -math.pi / 2 + 2 * math.pi * i / k
                out[seat] = (cx + 17 * math.cos(a), cy + 17 * math.sin(a) - 4)
    return out


def card_preview(env, seat: int, code: int, dice) -> str:
    """One-line summary of what a card would do (simulated, no randomness)."""
    if dice is None:
        return ''
    o = sim.play(env, seat, code, dice)
    bits = []
    if o.progress:
        bits.append(f'{o.progress:+d} spaces')
    for label, v in (('gold', o.d_gold), ('food', o.d_food), ('powder', o.d_powder)):
        if v:
            bits.append(f'{v:+d} {label}')
    if o.shortages:
        bits.append('shortage!')
    if o.combats:
        bits.append('battle!')
    if o.lairs:
        bits.append('treasure!')
    if o.finished:
        bits.append('Port Royal!')
    return ', '.join(bits) or 'no change'


def decision_buttons(env) -> list[dict]:
    """Every legal action of the pending decision as {label, action, detail}."""
    d = env.pending
    if d is None or env.done or d.kind in (Phase.TURN_PAUSE, Phase.DONE):
        return []
    out = []
    seat = d.seat
    for a in env.legal_actions():
        detail = ''
        if d.kind == Phase.CAPTAIN:
            order, code = divmod(a - A_CAPTAIN, N_CODES)
            hi, lo = max(env.raw_dice), min(env.raw_dice)
            dice = (hi, lo) if order == 0 else (lo, hi)
            detail = card_preview(env, seat, code, dice)
        elif d.kind == Phase.CARD:
            detail = card_preview(env, seat, a - A_CARD, env.dice)
        elif d.kind in (Phase.ATTACK_POWDER, Phase.DEFENSE_POWDER):
            detail = powder_odds(env, a - A_POWDER)
        out.append({'label': env.describe_action(a), 'action': a, 'detail': detail})
    return out


def powder_odds(env, k: int) -> str:
    b = env.battle
    att_seat = b.attacker
    me = env.ships[env.pending.seat]
    opp = env.ships[b.defender if env.pending.seat == att_seat else att_seat]
    if env.pending.kind == Phase.ATTACK_POWDER:
        w, t = sim.p_battle(k, rules.beth(me), 0, rules.beth(opp), env.faces)
        return f'win {w:.0%} if they add nothing'
    w, t = sim.p_defend(env.strength(b, 0), k, rules.beth(me), env.faces)
    return f'win {w:.0%}, tie {t:.0%}'


def visible_treasures(env, pov: int, seat: int) -> list[str]:
    """Treasure cards of `seat` as `pov` sees them."""
    s = env.ships[seat]
    out = [POWER_NAMES[p] for p in Power if s.has(p)]
    for tid in s.hidden:
        if pov == -1 or env.done or env.known[tid] >> pov & 1:
            out.append(f'{env.treasure_value(tid):+d}')
        elif env.public_cursed[tid]:
            out.append('cursed ?')
        else:
            out.append('?')
    return out


def status_text(env) -> str:
    if env.done:
        w = env.winner_player
        return f'Game over after {env.round} rounds: {env.player_names[w]} wins'
    dice = (f'morning {env.dice[0]}, evening {env.dice[1]}' if env.dice is not None
            else f'rolled {env.raw_dice[0]} and {env.raw_dice[1]}')
    who = ('next ship' if env.current_player < 0
           else f'{env.player_names[env.current_player]} to decide')
    return f'Round {env.round} · Captain {env.player_names[env.captain]} · dice {dice} · {who}'


def active_seat(env) -> int:
    """The seat to mark as playing: the one deciding now, or, while the UI
    waits for "next", the ship whose turn just played (-1 once the game is over)."""
    if env.done:
        return -1
    if env.current_player >= 0:
        return env.current_player
    return env.resolving if env.resolving >= 0 else env._last_seat


def standings(env) -> list[dict]:
    rows = []
    for seat, s in enumerate(env.ships):
        rows.append({
            'seat': seat,
            'name': env.player_names[seat],
            'position': env.track.pos_score(s.node, s.lap),
            'gold': rules.total(s, Res.GOLD),
            'treasures': sum(env.treasure_value(t) for t in s.hidden),
            'score': env.final_score(seat),
            'remaining': env.track.remaining(s.node, s.lap),
        })
    rows.sort(key=lambda r: (-r['score'], r['remaining'], r['seat']))
    return rows


#: Seconds of animation per ship move, and how many moves one repaint may
#: animate (the older ones are skipped: the ships just start further along).
MOVE_ANIMATION_S = 2.0
MOVE_ANIMATION_MAX = 6
#: Seconds the battle effects flash (smoke on the board, the battle panel)
#: once a ship arrives where it starts a battle, and the length of one flash.
BATTLE_FLASH_S = 2.0
FLASH_PERIOD_S = 0.4
SMOKE_CSS = ('@keyframes jm-smoke { 0% { opacity: 0; transform: scale(0.6); } '
             '40% { opacity: 1; transform: scale(1.05); } 100% { opacity: 0; transform: scale(1.25); } }')


def new_moves(log: list, seen: tuple[int, int] | None) -> tuple[list, tuple[int, int]]:
    """The entries of `log` (`env.move_log`) after the `seen` marker
    (``(id(log), count)``, as for `new_log_lines`) and the new marker."""
    start = seen[1] if seen is not None and seen[0] == id(log) else 0
    return list(log[min(start, len(log)):]), (id(log), len(log))


def move_route(env, move) -> list[int]:
    """Spaces a move goes through, start and end included."""
    return [node for node, _ in env.track.path(move.src, move.dst, move.d, move.steps)]


def move_keyframes(env, moves, duration: float = MOVE_ANIMATION_S
                   ) -> dict[int, list[tuple[float, float, float]]]:
    """`(time, dx, dy)` keyframes of every moved ship, as offsets from where
    it is drawn now; move `i` runs from `i * duration` to `(i + 1) * duration`,
    one equal share per space crossed, and a ship waits in place between
    its moves."""
    slots = ship_slots(env)
    last = {m.seat: i for i, m in enumerate(moves)}
    frames: dict[int, list[tuple[float, float, float]]] = {}
    for i, m in enumerate(moves):
        fx, fy = slots[m.seat]
        route = move_route(env, m)
        out = frames.setdefault(m.seat, [])
        for j, node in enumerate(route):
            if i == last[m.seat] and j == len(route) - 1:
                dx, dy = 0.0, 0.0
            else:
                cx, cy = board_xy(env, node)
                dx, dy = cx + SHIP_OFFSET[0] - fx, cy + SHIP_OFFSET[1] - fy
            if not out:
                out.append((0.0, dx, dy))
            t = (i + j / max(len(route) - 1, 1)) * duration
            if out[-1][0] == t:     # one keyframe per time: browsers mishandle duplicates
                out.pop()
            out.append((t, dx, dy))
    total = len(moves) * duration
    for out in frames.values():
        if out[-1][0] < total:
            out.append((total, out[-1][1], out[-1][2]))
    return frames


def move_animation_css(frames: dict[int, list[tuple[float, float, float]]], tag: str) -> str:
    """One `@keyframes jm-mv-<tag>-<seat>` rule per animated ship."""
    rules_ = []
    for seat, out in frames.items():
        total = out[-1][0] or 1.0
        steps = ' '.join(f'{100 * t / total:.3f}% {{ transform: translate({dx:.1f}px, {dy:.1f}px); }}'
                         for t, dx, dy in out)
        rules_.append(f'@keyframes jm-mv-{tag}-{seat} {{ {steps} }}')
    rules_.append('@media (prefers-reduced-motion: reduce) { .jm-moving, .jm-smoke '
                  '{ animation: none !important; } }')
    return '\n'.join(rules_)


def flash_animation(name: str, delay: float, duration: float = BATTLE_FLASH_S) -> str:
    """CSS `animation` value: `name` repeated for `duration` s after `delay` s,
    showing its first keyframe before and its last one after."""
    n = max(1, round(duration / FLASH_PERIOD_S))
    return f'{name} {duration / n:.3f}s ease-in-out {delay:.3f}s {n} both'


def battle_arrivals(moves, duration: float = MOVE_ANIMATION_S) -> list[tuple[int, float]]:
    """`(space, seconds)` of every animated move that starts a battle: where
    and when (in the move animation) the ship arrives."""
    return [(m.dst[0], (i + 1) * duration) for i, m in enumerate(moves) if m.battle]


def battle_flash_delay(moves, battle, duration: float = MOVE_ANIMATION_S) -> float:
    """When the battle panel should start flashing: the arrival of the move
    that started `battle` if it is animated, else at once."""
    if battle is None:
        return 0.0
    for i in range(len(moves) - 1, -1, -1):
        m = moves[i]
        if m.battle and m.seat == battle.attacker and m.dst[0] == battle.node:
            return (i + 1) * duration
    return 0.0


def board_overlay_svg(env, suggested_action: int | None = None, callback=None,
                      moves=(), duration: float = MOVE_ANIMATION_S, tag: str = '0',
                      flash: float = BATTLE_FLASH_S) -> str:
    """Ships, treasure chests, legal destinations and the suggestion star;
    `moves` (entries of `env.move_log`) are animated, `duration` s each, and
    those starting a battle end with `flash` s of smoke."""
    parts = []
    for node, token in env.lair_token.items():
        if token:
            x, y = board_xy(env, node)
            parts.append(art.chest(x - 12, y + 11, 5.5))
    d = env.pending
    if d is not None and d.kind in (Phase.MOVE_DEST, Phase.RETREAT_DEST) \
            and is_human_decision(env, callback):
        for n, _ in d.data:
            x, y = board_xy(env, n)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{art.SPACE_R + 5}" fill="none" '
                         f'stroke="#ffeb3b" stroke-width="3.5" stroke-dasharray="6 3"/>')
    active = active_seat(env)
    frames = move_keyframes(env, moves, duration) if moves and duration > 0 else {}
    smokes = battle_arrivals(moves, max(duration, 0.0)) if flash > 0 else []
    if frames or smokes:
        parts.append(f'<style>{move_animation_css(frames, tag)}\n{SMOKE_CSS}</style>')
    for seat, (x, y) in ship_slots(env).items():
        if seat in frames:
            parts.append(f'<g class="jm-moving" style="animation: jm-mv-{tag}-{seat} '
                         f'{frames[seat][-1][0]:.3f}s linear both">')
        if seat == active:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="13" fill="#ffffff" fill-opacity="0.55" '
                         f'stroke="{art.SEAT_COLORS[seat]}" stroke-width="3"/>')
        parts.append(art.ship(x, y, art.SEAT_COLORS[seat], 15, title=env.player_names[seat]))
        if seat in frames:
            parts.append('</g>')
    for node, delay in smokes:
        x, y = board_xy(env, node)
        parts.append(f'<g class="jm-smoke" style="opacity: 0; transform-box: fill-box; '
                     f'transform-origin: center; animation: {flash_animation("jm-smoke", delay, flash)}">'
                     f'{art.smoke(x, y - 4, 15)}</g>')
    if suggested_action is not None and d is not None and d.kind in (Phase.MOVE_DEST, Phase.RETREAT_DEST):
        k = suggested_action - A_DEST
        if 0 <= k < len(d.data):
            x, y = board_xy(env, d.data[k][0])
            parts.append(art.star(x + art.SPACE_R, y - art.SPACE_R, 8))
    return ''.join(parts)


@lru_cache(maxsize=4)
def _board_uri(track) -> str:
    svg = art.board_svg(track)
    return 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode()


def _svg(content: str):
    return ui.html(content, sanitize=False)


# --------------------------------------------------------------------------- #
# the page
# --------------------------------------------------------------------------- #

#: Toasts: how long each log line stays on screen, and how many lines one
#: repaint may pop (the bots can log dozens of lines between two human moves;
#: the older ones are summed up in one toast, the full text stays in the log).
TOAST_TIMEOUT_MS = 4000
TOAST_MAX_LINES = 8
#: Compact toasts, so a burst covers less of the board.
TOAST_CSS = """
.q-notification.jamaica-toast {
    min-height: 0; padding: 2px 12px; margin-top: 3px;
    font-size: 12px; opacity: 0.92;
}
.q-notification.jamaica-toast .q-notification__message { padding: 2px 0; }
"""
#: The battle panel's flash (see `flash_animation`).
BATTLE_CSS = """
@keyframes jm-battle-flash {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255, 111, 0, 0); background-color: #ffffff; }
    50% { box-shadow: 0 0 0 4px #ff6f00; background-color: #ffe0b2; }
}
@media (prefers-reduced-motion: reduce) { .jm-battle-flash { animation: none !important; } }
"""


def new_log_lines(log: list[tuple[int, str, int]], pov: int, seen: tuple[int, int] | None
                  ) -> tuple[list[str], tuple[int, int]]:
    """The lines of `log` visible to `pov` and not toasted yet, and the new
    `seen` marker.

    `seen` is ``(id(log), count)`` from the previous call: a reset of the game
    replaces the list, so a new id means everything is new.
    """
    start = seen[1] if seen is not None and seen[0] == id(log) else 0
    start = min(start, len(log))
    lines = [t for _, t, m in log[start:] if t.strip() and (pov < 0 or m >> pov & 1)]
    return lines, (id(log), len(log))


def toast_messages(lines: list[str], limit: int = TOAST_MAX_LINES) -> list[str]:
    """What to pop for `lines`: the last `limit`, the older ones summed up."""
    if len(lines) <= limit:
        return list(lines)
    skipped = len(lines) - (limit - 1)
    return [f'… {skipped} earlier event(s), see the game log'] + lines[-(limit - 1):]


SECTIONS = ('status', 'board', 'controls', 'side', 'players', 'log')


class RenderWeb:
    """The `jamaica` page; `move_duration` = seconds of animation per ship move,
    `battle_flash` = seconds the battle effects flash (0 disables either)."""

    def __init__(self, move_duration: float = MOVE_ANIMATION_S, battle_flash: float = BATTLE_FLASH_S):
        self.move_duration = move_duration
        self.battle_flash = battle_flash
        self._slots: dict[str, object] = {}
        self._env = None
        self._callback = None
        self._suggested: int | None = None
        self.die_order = 0          #: Captain's choice before clicking a card
        #: `(id(event_log), length)` of the log lines already toasted
        self._toasted: tuple[int, int] | None = None
        #: `(id(move_log), length)` of the moves already animated, the moves
        #: to animate at the next board build and a counter for unique names
        self._moved: tuple[int, int] | None = None
        self._moves: list = []
        self._repaints = 0
        #: the battle whose panel already flashed, and when the next flash starts
        self._flashed = None
        self._flash_delay = 0.0

    # -- actions --------------------------------------------------------- #

    def _playable(self) -> bool:
        return self._env is not None and is_human_decision(self._env, self._callback)

    def _send(self, action: int) -> None:
        env, callback = self._env, self._callback
        if env is None or callback is None or not is_human_decision(env, callback):
            ui.notify('watching: it is not your turn', type='info')
            return
        if not env.action_masks()[action]:
            ui.notify(f'illegal move: {env.describe_action(action)}', type='warning')
            return
        callback(action)

    def _next(self) -> None:
        if self._callback is not None and self._env is not None and self._env.current_player == -1:
            self._callback(None)

    def _on_board(self, event) -> None:
        env = self._env
        if env is None:
            return
        x, y = getattr(event, 'image_x', None), getattr(event, 'image_y', None)
        if x is None or y is None:
            return
        node = nearest_space(env, x, y)
        if node is not None and not self._playable():
            ui.notify(env.space_label(node), type='info')
            return
        action = action_for_board_click(env, node)
        if action is None:
            if node is not None:
                ui.notify(env.space_label(node), type='info')
            return
        self._send(action)

    def _set_order(self, value) -> None:
        self.die_order = int(value or 0)
        self._paint(('controls',))

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
        with ui.row().classes('items-center gap-3'):
            ui.label('Jamaica').classes('text-h6')
            ui.label(status_text(env)).classes('text-body1')
            if env.dice is not None:
                _svg(art.die_svg(env.dice[0], 28, 'morning'))
                ui.label('☀').classes('text-lg')
                _svg(art.die_svg(env.dice[1], 28, 'evening'))
                ui.label('☾').classes('text-lg')
            if self._playable():
                prompt = PHASE_PROMPTS.get(Phase(self._env.pending.kind), '')
                ui.label(prompt).classes('text-body1 text-weight-bold text-deep-orange-9')

    def _build_board(self):
        env = self._env
        moves, self._moves = self._moves, []    # animate them once
        ui.interactive_image(_board_uri(env.track),
                             content=board_overlay_svg(env, self._suggested, self._callback, moves,
                                                       self.move_duration, str(self._repaints),
                                                       self.battle_flash),
                             on_mouse=self._on_board, events=['click'], cross=False,
                             sanitize=False).style(f'width: {art.BOARD_W}px; max-width: 100%;')

    def _build_controls(self):
        env = self._env
        if env.done:
            self._build_standings()
            return
        if env.current_player == -1:
            if self._callback is not None:
                ui.button('Next', on_click=self._next).props('color=primary')
            return
        if not self._playable():
            ui.label(f'waiting for {env.player_names[env.current_player]}…').classes('text-italic')
            return
        d = env.pending
        buttons = decision_buttons(env)
        if d.kind == Phase.CAPTAIN:
            hi, lo = max(env.raw_dice), min(env.raw_dice)
            options = {0: f'morning {hi} / evening {lo}'}
            if hi != lo:
                options[1] = f'morning {lo} / evening {hi}'
            else:
                self.die_order = 0
            ui.toggle(options, value=self.die_order, on_change=lambda e: self._set_order(e.value))
            buttons = [b for b in buttons if (b['action'] - A_CAPTAIN) // N_CODES == self.die_order]
        if d.kind in (Phase.CAPTAIN, Phase.CARD):
            dice = env.dice
            if d.kind == Phase.CAPTAIN:
                dice = (hi, lo) if self.die_order == 0 else (lo, hi)
            with ui.row().classes('gap-2'):
                for b in buttons:
                    code = (b['action'] - A_CAPTAIN) % N_CODES if d.kind == Phase.CAPTAIN else b['action'] - A_CARD
                    with ui.column().classes('items-center gap-0 cursor-pointer').on(
                            'click', lambda _, a=b['action']: self._send(a)):
                        _svg(art.card_svg(code, dice))
                        ui.label(b['detail']).classes('text-caption').style('max-width: 110px')
                        if b['action'] == self._suggested:
                            ui.label('★ suggested').classes('text-caption text-amber-9')
            return
        with ui.row().classes('gap-2 flex-wrap'):
            for b in buttons:
                text = b['label'] + (f" ({b['detail']})" if b['detail'] else '')
                if b['action'] == self._suggested:
                    text = '★ ' + text
                ui.button(text, on_click=lambda _, a=b['action']: self._send(a)).props('no-caps dense')

    def _build_standings(self):
        env = self._env
        ui.label(status_text(env)).classes('text-h6')
        with ui.column().classes('gap-0'):
            for rank, r in enumerate(standings(env)):
                ui.label(f"#{rank + 1} {r['name']}: {r['score']} points "
                         f"(position {r['position']}, gold {r['gold']}, treasures {r['treasures']:+d})")

    def _build_side(self):
        env = self._env
        pov = viewer(env)
        seat = pov if pov >= 0 else (env.current_player if env.current_player >= 0 else 0)
        s = env.ships[seat]
        ui.label(f'{env.player_names[seat]} ({art.SEAT_NAMES[seat]})').classes('text-subtitle1')
        with ui.row().classes('gap-1 items-end'):
            for slot in rules.slots(s):
                h = hold_at(s, slot)
                _svg(art.hold_svg(h[0], h[1], sixth=slot == SIXTH_SLOT))
        treasures = visible_treasures(env, pov, seat)
        if treasures:
            ui.label('treasures: ' + ', '.join(treasures)).classes('text-body2')
        if pov >= 0 or env.done:
            with ui.row().classes('gap-1 items-center'):
                ui.label('hand:').classes('text-body2')
                for code in sorted(s.hand):
                    _svg(art.card_svg(code, None, 70, 39))
                if s.chosen >= 0:
                    ui.label('played:').classes('text-body2')
                    _svg(art.card_svg(s.chosen, None, 70, 39))
        self._build_battle()

    def _build_battle(self):
        env = self._env
        b = env.battle or env.last_battle
        if b is None or b.defender < 0:
            return
        live = env.battle is not None
        card = ui.card().classes('q-pa-sm')
        if b is not self._flashed and self.battle_flash > 0:    # flash once per battle
            self._flashed = b
            card.classes('jm-battle-flash').style(
                'animation: ' + flash_animation('jm-battle-flash', self._flash_delay, self.battle_flash))
        with card:
            ui.label(('Battle: ' if live else 'Last battle: ')
                     + f'{env.player_names[b.attacker]} attacks {env.player_names[b.defender]}'
                     ).classes('text-subtitle2')
            with ui.row().classes('items-center gap-2'):
                for seat, k, face, rolled in ((b.attacker, b.att_k, b.att_face, b.att_rolled),
                                              (b.defender, b.def_k, b.def_face, b.def_rolled)):
                    ui.label(f'{env.player_names[seat]}: {k} powder'
                             + (' +2 Lady Beth' if env.ships[seat].has(Power.BETH) else ''))
                    if rolled:
                        _svg(art.combat_face_svg(face, 30))
            if not live:
                result = ('tie: nothing happens' if b.winner < 0
                          else f'{env.player_names[b.winner]} wins, takes {b.loot or "…"}')
                ui.label(result).classes('text-body2')

    def _build_players(self):
        env = self._env
        pov = viewer(env)
        active = active_seat(env)
        with ui.row().classes('w-full gap-2 items-stretch flex-wrap'):
            for seat in seat_order(env):
                s = env.ships[seat]
                color = art.SEAT_COLORS[seat]
                style = f'border-top: 5px solid {color}; min-width: 190px'
                if seat == active:
                    style += f'; outline: 3px solid {color}; outline-offset: 1px'
                with ui.card().classes('q-pa-sm').style(style):
                    title = env.player_names[seat]
                    if seat == env.captain:
                        title += '  ⚓ Captain'
                    with ui.row().classes('w-full items-center justify-between no-wrap gap-1'):
                        ui.label(title).classes('text-subtitle2')
                        if seat == active:
                            ui.label('▶ playing' if env.current_player >= 0 else '▶ just played'
                                     ).classes('text-caption text-weight-bold').style(f'color: {color}')
                    ui.label(env.space_label(s.node, s.lap) if not s.finished else 'in Port Royal (finished)'
                             ).classes('text-caption')
                    with ui.row().classes('gap-1'):
                        for slot in rules.slots(s):
                            h = hold_at(s, slot)
                            _svg(art.hold_svg(h[0], h[1], sixth=slot == SIXTH_SLOT, width=30, height=30))
                    card = ''
                    if s.chosen >= 0 and (s.revealed or seat == pov or pov == -1):
                        m, e = card_syms(s.chosen)
                        card = f'card: {SYM_NAMES[m]} / {SYM_NAMES[e]}'
                    elif s.chosen >= 0:
                        card = 'card chosen (hidden)'
                    if card:
                        ui.label(card).classes('text-caption')
                    tr = visible_treasures(env, pov, seat)
                    if tr:
                        ui.label('treasures: ' + ', '.join(tr)).classes('text-caption')
                    ui.label(f'hand {len(s.hand)} · deck {len(s.deck)} · discard {len(s.discard)} · '
                             f'score ≈ {env.public_score(seat):.0f}').classes('text-caption')

    def _build_log(self):
        env = self._env
        ui.label('Game log').classes('text-sm font-bold')
        with ui.scroll_area().classes('w-full border rounded').style('height: 170px;') as area:
            for rnd, text in env.log_lines(viewer(env))[-300:]:
                ui.label(text).classes('text-xs').style('white-space: pre-wrap; line-height: 1.15;')
        area.scroll_to(percent=1.0)

    def _toast_new_lines(self) -> None:
        """Pop the log lines logged since the last repaint as short toasts,
        stacked at the top of the page (Quasar stacks same-position ones)."""
        env = self._env
        lines, self._toasted = new_log_lines(env.event_log, viewer(env), self._toasted)
        messages = toast_messages(lines)
        if not messages:
            return
        # `ui.notify` needs a live slot: the clicked button may have just been
        # deleted by the repaint, the section containers are only cleared.
        # Quasar puts the newest toast on top: send them in reverse so a burst
        # reads top-down in order.
        with self._slots['log']:
            for message in reversed(messages):
                ui.notify(message, position='top', timeout=TOAST_TIMEOUT_MS,
                          group=False, classes='jamaica-toast')

    # -- contract -------------------------------------------------------- #

    def init_web(self, env, callback=None):
        ui.add_css(TOAST_CSS)
        ui.add_css(BATTLE_CSS)
        self._env, self._callback, self._suggested = env, callback, None
        slots = {}
        with ui.column().classes('w-full gap-2 q-pa-sm'):
            slots['status'] = ui.column().classes('w-full gap-1')
            with ui.row().classes('w-full no-wrap items-start gap-3'):
                slots['board'] = ui.column().classes('gap-0').style(f'flex: 0 0 auto; width: {art.BOARD_W}px;')
                with ui.column().classes('gap-2').style('flex: 1 1 auto; min-width: 0;'):
                    slots['controls'] = ui.column().classes('w-full gap-1')
                    slots['side'] = ui.column().classes('w-full gap-1')
                    slots['players'] = ui.column().classes('w-full gap-1')
            slots['log'] = ui.column().classes('w-full gap-1')
        self._slots = slots
        _, self._moved = new_moves(env.move_log, None)   # opening the page replays nothing
        self._flashed = env.battle or env.last_battle
        self._paint()
        self._toast_new_lines()

    def render_web(self, env, callback=None, suggested_action: int | None = None, **kwargs):
        self._env = env
        self._callback = callback
        self._suggested = suggested_action
        moves, self._moved = new_moves(env.move_log, self._moved)
        self._moves = moves[-MOVE_ANIMATION_MAX:]
        self._flash_delay = battle_flash_delay(self._moves, env.battle or env.last_battle,
                                               max(self.move_duration, 0.0))
        self._repaints += 1
        self._paint()
        self._toast_new_lines()
