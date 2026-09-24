"""NiceGUI renderer for Small World (`smallw`) — task T7, plan 3.9.

Same contract as the other games: `SmallWorldEnv.nicegui_page()` builds a
:class:`RenderWeb` and calls :meth:`RenderWeb.init_web`, then every
`SmallWorldEnv.render()` calls :meth:`RenderWeb.render_web` (which also
accepts — and ignores — the extra keyword arguments `play.py` passes on some
branches: `pov_player`, `suggest`, `moves`, ...).

Layout
------
::

    +--------------------------------------------+-------------+
    | status chips (turn / phase / player / die) |  combo      |
    | controls (mode toggle, buttons, hint)      |  column     |
    | board: interactive_image + SVG overlay     |  (6 cards)  |
    +--------------------------------------------+-------------+
    | players panel (one card per seat)                        |
    | event log (scrollable, newest at the bottom)             |
    +----------------------------------------------------------+

The board is `static/board3p.png` (597x297 px) with an SVG overlay drawn in
the **image pixel coordinates**: the race token of every occupied region with
its token count, greyscaled while the race is in decline, the markers (Lost
Tribe, Mountain, Troll's Lair — drawn by hand, no artwork exists —, fortress,
encampments, hole, hero, dragon), the turn marker on the turn track, a
coloured ring around every legal target and a star on the suggested action.

Click mapping (`action_for_click`)
----------------------------------
A click gives image coordinates, the nearest region anchor gives the region,
and the phase plus the current *mode* give the action:

===================================  =========================================
phase                                action sent by a click
===================================  =========================================
CONQUER / GHOUL_CONQUER, *Normal*    `A_REGION + i` (conquer, or abandon one of
                                     the acting race's own regions)
CONQUER / GHOUL_CONQUER, *Sorcerer*  `A_SORCERER + i`
CONQUER / GHOUL_CONQUER, *Dragon*    `A_DRAGON + i`
REDEPLOY / GHOUL_REDEPLOY /          `A_REGION + i` (*one token*) or
VICTIM_REDEPLOY                      `A_REGION_ALL + i` (*all remaining*)
ENCAMPMENTS / FORTRESS / HEROES      `A_REGION + i` (place the marker)
PICK_COMBO / ALLY / STOUT_DECLINE    nothing: those use the cards / buttons
===================================  =========================================

Illegal or meaningless clicks are answered with a `ui.notify` explaining why,
and every click is ignored unless it really is the human's decision
(`callback is not None` and `current_player == pov_player`, or god mode
`pov_player == -1`).

Everything that computes *what* to draw is a module-level pure function (no
`nicegui` call, no UI state), so `tests/test_render.py` can check the overlay,
the click mapping, the coin rule and the labels headlessly.

Note on painting: the page is *not* built out of `@ui.refreshable` sections
like the other two games. Since NiceGUI 3, `refresh()` returns an
`AwaitableResponse` that schedules the rebuild as a *background task*
(`nicegui/awaitable_response.py`), so a refresh issued while `play.py` is
still building the page — it walks the game up to the human's first decision
inside the `@ui.page` function — is applied only after that function returns;
the server-rendered HTML can therefore show the pre-refresh tree (what `curl`
sees). A real browser still ends up with the refreshed one, because it
receives the element tree over the websocket afterwards: checked on
`/stotten` with NiceGUI 3.13.0, where the first human decision is drawn *and*
clickable (T8). Painting synchronously does not depend on that ordering at
all: `init_web` creates one container per section and
:meth:`RenderWeb._paint` clears and re-fills them, which works both during the
page build and from a later event handler.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

from nicegui import app, ui

from .classes import POWERS, RACES
from .map3p import BOARD_IMAGE, BOARD_SIZE, TURN_TRACK
from .rules_text import POWER_RULES, RACE_RULES

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .smallw import SmallWorldEnv


# --------------------------------------------------------------------------- #
# Static files
# --------------------------------------------------------------------------- #

#: URL prefix the board, the banners, the badges and the pieces are served at.
STATIC_URL = '/smallw_static'

#: `app/environments/smallw/static` — the directory served at `STATIC_URL`.
STATIC_DIR = Path(__file__).resolve().parent.parent / 'static'

#: Module-level guard: the route must be registered exactly once per process.
_STATIC_REGISTERED = False


def register_static_files() -> None:
    """Serve `static/` at :data:`STATIC_URL` (idempotent)."""
    global _STATIC_REGISTERED
    if _STATIC_REGISTERED:
        return
    app.add_static_files(STATIC_URL, str(STATIC_DIR))
    _STATIC_REGISTERED = True


def race_banner_url(race) -> str:
    """URL of the race banner image (≈430x230)."""
    return f'{STATIC_URL}/races/{RACES[race].key}.png'


def race_token_url(race) -> str:
    """URL of the 88x88 race token image."""
    return f'{STATIC_URL}/races/{RACES[race].key}_token.jpg'


def power_badge_url(power) -> str:
    """URL of the 115x115 special power badge image."""
    return f'{STATIC_URL}/powers/{POWERS[power].key}.png'


def piece_url(name: str) -> str:
    """URL of one of the `pieces/` images (`coin_1`, `fortress`, `die`, ...)."""
    return f'{STATIC_URL}/pieces/{name}.jpg'


def board_url() -> str:
    """URL of the board image."""
    return f'{STATIC_URL}/{BOARD_IMAGE}'


# --------------------------------------------------------------------------- #
# Lazy access to the engine module
# --------------------------------------------------------------------------- #

_SW = None


def _sw():
    """The `smallw` module.

    Imported lazily because `smallw.py` imports *this* module at the top (the
    action constants and `Phase` are defined further down that file, so a
    module-level `from .smallw import ...` would be a circular import).
    """
    global _SW
    if _SW is None:
        from . import smallw as module
        _SW = module
    return _SW


def _phase_name(phase) -> str:
    """Name of a `Phase` member (works with a plain int or a string too)."""
    return getattr(phase, 'name', str(phase))


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #

#: Phase → plain words for the status bar.
PHASE_LABELS: dict[str, str] = {
    'PICK_COMBO': 'Pick a race + power combo',
    'CONQUER': 'Conquests',
    'GHOUL_CONQUER': 'Conquests of the declined Ghouls',
    'REDEPLOY': 'Redeployment',
    'GHOUL_REDEPLOY': 'Redeployment of the declined Ghouls',
    'ENCAMPMENTS': 'Place the encampments (Bivouacking)',
    'FORTRESS': 'Build a fortress (Fortified)',
    'HEROES': 'Send the heroes (Heroic)',
    'ALLY': 'Choose an ally (Diplomat)',
    'STOUT_DECLINE': 'Decline right now? (Stout)',
    'VICTIM_REDEPLOY': 'Redeploy the tokens lost this turn',
    'DONE': 'Game over',
    'TURN_PAUSE': 'End of turn',
}

#: Phase → label of the `A_PASS` button.
PASS_LABELS: dict[str, str] = {
    'CONQUER': 'End conquests',
    'GHOUL_CONQUER': 'End conquests',
    'FORTRESS': 'No fortress',
    'ALLY': 'No ally',
    'STOUT_DECLINE': 'Stay active',
}

#: Phase → one-line instruction for the human.
PHASE_HINTS: dict[str, str] = {
    'PICK_COMBO': 'Pick one of the combos on the right (the top one is free, '
                  'each one below costs one more coin).',
    'CONQUER': 'Click a region to conquer it (red ring, cost shown) or one of '
               'your own regions to abandon it (orange, before your first '
               'conquest only).',
    'GHOUL_CONQUER': 'Your declined Ghouls conquer first: click a region '
                     '(red) or abandon one of theirs (orange).',
    'REDEPLOY': 'Every token but one per region is back in hand: click '
                'your regions to spread them as you like (blue rings).',
    'GHOUL_REDEPLOY': 'Every Ghoul token but one per region is back in hand: '
                      'spread them over their regions (blue rings).',
    'ENCAMPMENTS': 'Click your regions to place the encampments (they may be '
                   'stacked).',
    'FORTRESS': 'Click one of your regions to build the fortress, or pass.',
    'HEROES': 'Click one of your regions to send a hero there.',
    'ALLY': 'Choose the player you ally with, or pass (he may not attack you '
            'until your next turn).',
    'STOUT_DECLINE': 'Stout: go in decline right now (and still score this '
                     'turn), or stay active.',
    'VICTIM_REDEPLOY': 'You lost tokens: click your regions to put them back '
                       'on the board.',
    'DONE': 'The game is over.',
    'TURN_PAUSE': 'Look at the move just played, then click "Next player".',
}

MODE_NORMAL = 'normal'
MODE_SORCERER = 'sorcerer'
MODE_DRAGON = 'dragon'
MODE_ONE = 'one'
MODE_ALL = 'all'

#: Mode → label of the toggle button.
MODE_LABELS: dict[str, str] = {
    MODE_NORMAL: 'Normal',
    MODE_SORCERER: 'Sorcerer',
    MODE_DRAGON: 'Dragon',
    MODE_ONE: 'one token',
    MODE_ALL: 'all remaining',
}

TARGET_CONQUER = 'conquer'
TARGET_ABANDON = 'abandon'
TARGET_REDEPLOY = 'redeploy'
TARGET_MARKER = 'marker'
TARGET_SORCERER = 'sorcerer'
TARGET_DRAGON = 'dragon'

#: Target kind → ring colour on the board.
TARGET_COLORS: dict[str, str] = {
    TARGET_CONQUER: '#ff1744',    # red
    TARGET_ABANDON: '#ff9100',    # orange
    TARGET_REDEPLOY: '#2979ff',   # blue
    TARGET_MARKER: '#aa00ff',     # purple
    TARGET_SORCERER: '#00e5ff',   # cyan
    TARGET_DRAGON: '#f50057',     # pink
}


def phase_label(phase) -> str:
    """Plain-words name of a phase (`'Conquests'`, `'Redeployment'`, ...)."""
    return PHASE_LABELS.get(_phase_name(phase), _phase_name(phase))


def pass_label(phase) -> str:
    """Label of the `A_PASS` button in `phase`."""
    return PASS_LABELS.get(_phase_name(phase), 'Pass')


def phase_hint(phase) -> str:
    """One-line instruction telling the human what to click in `phase`."""
    return PHASE_HINTS.get(_phase_name(phase), '')


# --------------------------------------------------------------------------- #
# Small state queries (pure, engine read-only)
# --------------------------------------------------------------------------- #

def _holds(rip, region) -> bool:
    """True if `region` is held by exactly that race of that player.

    Same test as `SmallWorldEnv._holds`, duplicated here so the renderer never
    reaches into the engine for something this small.
    """
    if rip is None:
        return False
    return (region.owner == rip.owner and region.race == rip.race
            and region.in_decline == rip.in_decline)


def is_human_decision(env, callback=None, pov_player=None) -> bool:
    """True when the page may send an action for the seat it displays.

    `callback` is None while the page is being built and on the final render,
    and the AI turns are rendered for information only. God mode
    (`pov_player == -1`) drives whatever seat is to move.
    """
    if callback is None or env.done:
        return False
    pov = env.pov_player if pov_player is None else pov_player
    if pov == -1:
        return True
    return env.current_player == pov


def shows_coin_value(seat: int, pov_player: int) -> bool:
    """True if the *value* of `seat`'s coins may be displayed (plan T3b).

    Only the point of view sees its own value; god mode (`-1`) sees them all.
    Everybody else only shows how many coin *tokens* he holds.
    """
    return pov_player == -1 or seat == pov_player


def coin_text(env, seat: int, pov_player: int) -> str:
    """Coin line of one seat: the value for the pov, the token count else."""
    player = env.players[seat]
    if shows_coin_value(seat, pov_player):
        return f'{player.coins} coins in {player.coin_count} token(s)'
    return f'{player.coin_count} coin token(s)'


def combo_tokens(race, power) -> int:
    """Tokens a fresh `race` + `power` combo brings (banner + badge + bonus)."""
    race_def = RACES[race]
    total = race_def.banner_value + race_def.attack_bonus_tokens
    if power is not None:
        total += POWERS[power].value
    return total


def available_modes(env) -> list[str]:
    """Click modes offered in the current phase (empty = no toggle).

    CONQUER / GHOUL_CONQUER: *Normal*, plus *Sorcerer* / *Dragon* only while
    such an action really is legal. The redeployment phases always offer
    *one token* / *all remaining*.
    """
    sw = _sw()
    if env.done:
        return []
    phase = env.phase
    if phase in (sw.Phase.CONQUER, sw.Phase.GHOUL_CONQUER):
        masks = env.action_masks()
        modes = [MODE_NORMAL]
        if any(masks[sw.A_SORCERER:sw.A_DRAGON]):
            modes.append(MODE_SORCERER)
        if any(masks[sw.A_DRAGON:sw.A_ALLY]):
            modes.append(MODE_DRAGON)
        return modes
    if phase in (sw.Phase.REDEPLOY, sw.Phase.GHOUL_REDEPLOY,
                 sw.Phase.VICTIM_REDEPLOY):
        return [MODE_ONE, MODE_ALL]
    return []


def region_targets(env, mode: str = MODE_NORMAL) -> dict[int, str]:
    """``{region index: target kind}`` for what a board click can do now.

    The kinds are the `TARGET_*` constants and give the ring colour; only the
    targets reachable **in the current mode** are returned, so the board always
    shows exactly what a click would do.
    """
    sw = _sw()
    if env.done or env.board is None:
        return {}
    masks = env.action_masks()
    phase = env.phase
    rip = env.current_race
    targets: dict[int, str] = {}

    if phase in (sw.Phase.CONQUER, sw.Phase.GHOUL_CONQUER):
        if mode == MODE_SORCERER:
            base, kind = sw.A_SORCERER, TARGET_SORCERER
        elif mode == MODE_DRAGON:
            base, kind = sw.A_DRAGON, TARGET_DRAGON
        else:
            base, kind = sw.A_REGION, None
        for index in range(env.n_regions):
            if not masks[base + index]:
                continue
            if kind is not None:
                targets[index] = kind
            else:
                targets[index] = (TARGET_ABANDON
                                  if _holds(rip, env.board.by_index(index))
                                  else TARGET_CONQUER)
    elif phase in (sw.Phase.REDEPLOY, sw.Phase.GHOUL_REDEPLOY,
                   sw.Phase.VICTIM_REDEPLOY):
        for index in range(env.n_regions):
            if masks[sw.A_REGION + index]:
                targets[index] = TARGET_REDEPLOY
    elif phase in (sw.Phase.ENCAMPMENTS, sw.Phase.FORTRESS, sw.Phase.HEROES):
        for index in range(env.n_regions):
            if masks[sw.A_REGION + index]:
                targets[index] = TARGET_MARKER
    return targets


def conquest_costs(env) -> dict[int, tuple[int, bool]]:
    """``{region index: (cost, final attempt)}`` of the acting race, or ``{}``.

    Reads the engine's own cost table (`_conquest_options`) so the numbers
    drawn on the board are exactly the ones `step` will use.
    """
    sw = _sw()
    rip = env.current_race
    if rip is None or env.phase not in (sw.Phase.CONQUER, sw.Phase.GHOUL_CONQUER):
        return {}
    try:
        return dict(env._conquest_options(rip))
    except Exception:                                       # noqa: BLE001
        return {}


def target_reachable(env, region) -> bool:
    """True if the acting race may target `region`, its cost aside.

    Asks the engine itself (`_can_attack`), so the refusal messages never
    disagree with the masks.
    """
    rip = env.current_race
    if rip is None:
        return False
    try:
        return bool(env._can_attack(rip, region))
    except Exception:                                       # noqa: BLE001
        return False


def conquest_cost(env, region) -> int | None:
    """Tokens the acting race needs to conquer `region` (None if unknown)."""
    rip = env.current_race
    if rip is None:
        return None
    try:
        return int(env._conquest_cost(rip, region))
    except Exception:                                       # noqa: BLE001
        return None


def nearest_region_index(env, x: float, y: float) -> int:
    """0-based index of the region whose anchor is closest to `(x, y)`.

    `(x, y)` are the image pixel coordinates `ui.interactive_image` yields.
    """
    best_index, best_dist = 0, None
    for region in env.board.regions:
        ax, ay = region.static.anchor
        dist = (ax - x) ** 2 + (ay - y) ** 2
        if best_dist is None or dist < best_dist:
            best_index, best_dist = region.index, dist
    return best_index


def action_for_click(env, index: int, mode: str = MODE_NORMAL
                     ) -> tuple[int | None, str]:
    """Action a click on region `index` sends, or `(None, why not)`.

    Returns `(action, description)` when the click maps to a legal action and
    `(None, message)` otherwise — the message is what the page notifies.
    """
    sw = _sw()
    if env.done:
        return None, 'the game is over'
    phase = env.phase
    if phase == sw.Phase.PICK_COMBO:
        return None, 'pick one of the combos in the column on the right'
    if phase == sw.Phase.ALLY:
        return None, 'choose your ally with the buttons above the board'
    if phase == sw.Phase.STOUT_DECLINE:
        return None, 'use the "Go in decline" / "Stay active" buttons'

    if phase in (sw.Phase.CONQUER, sw.Phase.GHOUL_CONQUER):
        if mode == MODE_SORCERER:
            action = sw.A_SORCERER + index
        elif mode == MODE_DRAGON:
            action = sw.A_DRAGON + index
        else:
            action = sw.A_REGION + index
    elif phase in (sw.Phase.REDEPLOY, sw.Phase.GHOUL_REDEPLOY,
                   sw.Phase.VICTIM_REDEPLOY):
        action = (sw.A_REGION_ALL + index if mode == MODE_ALL
                  else sw.A_REGION + index)
    elif phase in (sw.Phase.ENCAMPMENTS, sw.Phase.FORTRESS, sw.Phase.HEROES):
        action = sw.A_REGION + index
    else:                                                   # pragma: no cover
        return None, f'nothing to click in {phase_label(phase)}'

    if env.action_masks()[action]:
        return action, env.describe_action(action)
    return None, illegal_reason(env, index, mode)


def illegal_reason(env, index: int, mode: str = MODE_NORMAL) -> str:
    """Plain-words explanation of why region `index` cannot be clicked."""
    sw = _sw()
    region = env.board.by_index(index)
    rip = env.current_race
    phase = env.phase
    label = f'region {region.id}'

    if phase in (sw.Phase.CONQUER, sw.Phase.GHOUL_CONQUER):
        if mode == MODE_SORCERER:
            return (f'no Sorcerer substitution possible on {label} (one lone '
                    'enemy token, one substitution per opponent and per turn)')
        if mode == MODE_DRAGON:
            return f'the dragon cannot be sent to {label}'
        if _holds(rip, region):
            return (f'{label} is yours: you may only abandon a region before '
                    'your first conquest of the turn')
        if region.is_immune:
            return f'{label} is protected (hole in the ground / hero / dragon)'
        if region.is_water:
            return f'{label} is water: only a Seafaring race may enter it'
        if rip is not None and rip.tokens_in_hand <= 0:
            return 'no token left in hand'
        if (region.owner is not None and not region.in_decline
                and rip is not None
                and env.players[region.owner].ally == rip.owner):
            protector = env.players[region.owner].name
            return (f'you are {protector}\'s ally: you may not attack his '
                    'active race this turn')
        if not target_reachable(env, region):
            return (f'{label} is out of reach: you must be adjacent to it '
                    '(or make your first conquest on a border region)')
        cost = conquest_cost(env, region)
        hand = 0 if rip is None else rip.tokens_in_hand
        if cost is not None:
            return (f'{label} costs {cost} token(s) and you only have {hand} '
                    '(a final attempt needs the cost to be at most 3 more)')
        return f'{label} is not a legal target'                 # pragma: no cover

    if phase in (sw.Phase.REDEPLOY, sw.Phase.GHOUL_REDEPLOY,
                 sw.Phase.VICTIM_REDEPLOY):
        if not _holds(rip, region):
            return f'{label} is not yours: redeploy on your own regions only'
        return 'no token left to redeploy'
    if phase == sw.Phase.ENCAMPMENTS:
        return f'{label} is not one of your regions'
    if phase == sw.Phase.FORTRESS:
        if region.fortress:
            return f'{label} already has a fortress'
        return f'{label} is not one of your regions'
    if phase == sw.Phase.HEROES:
        if region.hero:
            return f'{label} already has a hero'
        return f'{label} is not one of your regions'
    return f'{label} is not a legal target'                        # unreachable


def button_actions(env) -> dict[str, dict]:
    """State of the DECLINE / PASS / ALLY buttons.

    ``{'decline': {...}, 'pass': {...}, 'ally': {seat: {...}}}`` with a
    `label`, `action` and `legal` entry each — everything the page needs to
    build them enabled or disabled.
    """
    sw = _sw()
    masks = env.action_masks()
    allies = {}
    for seat in range(env.n_players):
        action = sw.A_ALLY + seat
        allies[seat] = {
            'label': f'Ally with {env.players[seat].name}',
            'action': action,
            'legal': bool(masks[action]),
        }
    return {
        'decline': {'label': 'Go in decline', 'action': sw.A_DECLINE,
                    'legal': bool(masks[sw.A_DECLINE])},
        'pass': {'label': pass_label(env.phase), 'action': sw.A_PASS,
                 'legal': bool(masks[sw.A_PASS])},
        'ally': allies,
    }


def race_title(race) -> str:
    """Tooltip heading of a race: name and banner value (Amazons' bonus)."""
    race_def = RACES[race]
    value = f'{race_def.banner_value}'
    if race_def.attack_bonus_tokens:
        value += f' +{race_def.attack_bonus_tokens}'
    return f'{race_def.name} ({value} tokens)'


def power_title(power) -> str:
    """Tooltip heading of a power: name and badge value."""
    power_def = POWERS[power]
    return f'{power_def.name} (+{power_def.value} tokens)'


def combo_entries(env, suggested_action: int | None = None) -> list[dict]:
    """One dict per visible combo, top of the column first."""
    sw = _sw()
    masks = env.action_masks()
    entries = []
    for index, combo in enumerate(env.combo_column.visible):
        action = sw.A_COMBO + index
        entries.append({
            'index': index,
            'action': action,
            'race': combo.race,
            'power': combo.power,
            'race_name': RACES[combo.race].name,
            'power_name': POWERS[combo.power].name,
            'race_title': race_title(combo.race),
            'power_title': power_title(combo.power),
            'race_rules': RACE_RULES[combo.race],
            'power_rules': POWER_RULES[combo.power],
            'banner': race_banner_url(combo.race),
            'badge': power_badge_url(combo.power),
            'cost': env.combo_column.cost(index),
            'coins': combo.coins,
            'tokens': combo_tokens(combo.race, combo.power),
            'legal': bool(masks[action]),
            'suggested': suggested_action is not None and suggested_action == action,
        })
    return entries


def _race_entry(env, rip, in_decline: bool) -> dict:
    """Display data of one race banner of a player."""
    return {
        'race': rip.race,
        'race_name': RACES[rip.race].name,
        'banner': race_banner_url(rip.race),
        'power': rip.power,
        'power_name': None if rip.power is None else POWERS[rip.power].name,
        'badge': None if rip.power is None else power_badge_url(rip.power),
        'race_title': race_title(rip.race),
        'power_title': None if rip.power is None else power_title(rip.power),
        'race_rules': RACE_RULES[rip.race],
        'power_rules': None if rip.power is None else POWER_RULES[rip.power],
        'in_decline': in_decline,
        'hand': rip.tokens_in_hand,
        'tray': env.tray.available(rip.race),
        'regions': len(env.board.regions_of_race(rip)),
        'tokens_on_board': sum(r.tokens for r in env.board.regions_of_race(rip)),
        'is_spirit': rip.is_spirit,
        'acting': rip is env.current_race,
    }


def player_entries(env, pov_player: int | None = None) -> list[dict]:
    """One dict per seat: names, races, coins (hidden per plan T3b), ally."""
    pov = env.pov_player if pov_player is None else pov_player
    entries = []
    for seat, player in enumerate(env.players):
        active = (None if player.active is None
                  else _race_entry(env, player.active, False))
        declined = [_race_entry(env, rip, True) for rip in player.declined]
        entries.append({
            'seat': seat,
            'name': player.name,
            'current': seat == env.current_player,
            'is_pov': seat == pov,
            'shows_value': shows_coin_value(seat, pov),
            'coins': player.coins,
            'coin_count': player.coin_count,
            'coin_text': coin_text(env, seat, pov),
            'active': active,
            'declined': declined,
            'ally': (None if player.ally is None
                     else env.players[player.ally].name),
            'must_first_conquest': bool(player.must_first_conquest),
        })
    return entries


_DIE_RE = re.compile(r'die(?:\s+rolls|:)\s*(\d)')


def last_die_in_log(env) -> int | None:
    """Last reinforcement die result mentioned in the event log, else None."""
    for line in reversed(env.event_log):
        found = _DIE_RE.search(line)
        if found:
            return int(found.group(1))
    return None


def tokens_to_place(env) -> int | None:
    """How many tokens / markers the acting player still has to place.

    None outside a placement phase.
    """
    sw = _sw()
    rip = env.current_race
    phase = env.phase
    if phase in (sw.Phase.REDEPLOY, sw.Phase.GHOUL_REDEPLOY,
                 sw.Phase.VICTIM_REDEPLOY):
        if rip is None:
            return None
        try:
            return int(env._redeployable(rip))
        except Exception:                                   # noqa: BLE001
            return int(rip.tokens_in_hand)
    if phase == sw.Phase.ENCAMPMENTS:
        return int(getattr(env, '_encampments_left', 0))
    if phase == sw.Phase.HEROES:
        return int(getattr(env, '_heroes_left', 0))
    if phase == sw.Phase.FORTRESS:
        return 1
    return None


def status_entries(env, pov_player: int | None = None) -> list[tuple[str, str]]:
    """``[(label, value)]`` chips of the status bar."""
    sw = _sw()
    pov = env.pov_player if pov_player is None else pov_player
    turn = min(env.turn, env.turns_total)
    rows: list[tuple[str, str]] = [
        ('Turn', f'{turn}/{env.turns_total}'),
        ('Phase', phase_label(env.phase)),
    ]
    if env.done:
        winner = ('undetermined' if env.winner_player is None
                  else env.players[env.winner_player].name)
        rows.append(('Winner', winner))
        return rows
    if env.players and env.phase == sw.Phase.TURN_PAUSE:
        rows.append(('Just played', env.players[env._turn_seat].name))
        return rows
    if not env.players or env.current_player < 0:       # before the first deal
        return rows
    player = env.players[env.current_player]
    who = player.name + (' (you)' if env.current_player == pov else '')
    rows.append(('To move', who))
    rip = env.current_race
    if rip is not None:
        power = ('in decline' if rip.power is None
                 else POWERS[rip.power].name)
        rows.append(('Race', f'{RACES[rip.race].name} / {power}'))
        rows.append(('In hand', f'{rip.tokens_in_hand} token(s)'))
        if rip.die is not None:
            rows.append(('Die', str(rip.die)))
        if env.phase in (sw.Phase.CONQUER, sw.Phase.GHOUL_CONQUER):
            rows.append(('Conquests', str(len(rip.conquests_this_turn))))
    left = tokens_to_place(env)
    if left is not None:
        rows.append(('To place', str(left)))
    if rip is None or rip.die is None:
        die = last_die_in_log(env)
        if die is not None:
            rows.append(('Last die', str(die)))
    return rows


#: Toasts: how long each log line stays on screen, and how many lines one
#: repaint may pop (the bots can log dozens of lines between two human moves;
#: the older ones are summed up in one toast, the full text stays in the log).
TOAST_TIMEOUT_MS = 2000
TOAST_MAX_LINES = 8
#: Compact toasts, so a burst covers less of the board.
TOAST_CSS = """
.q-notification.smallw-toast {
    min-height: 0; padding: 2px 12px; margin-top: 3px;
    font-size: 12px; opacity: 0.92;
}
.q-notification.smallw-toast .q-notification__message { padding: 2px 0; }
"""


def new_log_lines(log: list[str], seen: tuple[int, int] | None
                  ) -> tuple[list[str], tuple[int, int]]:
    """The lines of `log` not toasted yet, and the new `seen` marker.

    `seen` is ``(id(log), count)`` from the previous call: a reset of the game
    replaces the list, so a new id means everything is new.
    """
    start = seen[1] if seen is not None and seen[0] == id(log) else 0
    start = min(start, len(log))
    lines = [line for line in log[start:] if line.strip()]
    return lines, (id(log), len(log))


def toast_messages(lines: list[str], limit: int = TOAST_MAX_LINES) -> list[str]:
    """What to pop for `lines`: the last `limit`, the older ones summed up."""
    if len(lines) <= limit:
        return list(lines)
    skipped = len(lines) - (limit - 1)
    return ([f'… {skipped} earlier event(s), see the game log'] +
            lines[-(limit - 1):])


def standings(env) -> list[dict]:
    """Final ranking: coins first, tokens on board as tie-break, then seat."""
    rows = []
    for seat, player in enumerate(env.players):
        rows.append({
            'seat': seat,
            'name': player.name,
            'coins': player.coins,
            'coin_count': player.coin_count,
            'tokens': env.board.count_tokens_on_board(seat),
        })
    rows.sort(key=lambda row: (-row['coins'], -row['tokens'], row['seat']))
    for rank, row in enumerate(rows):
        row['rank'] = rank + 1
    return rows


# --------------------------------------------------------------------------- #
# SVG overlay
# --------------------------------------------------------------------------- #

#: Combo column of the page: the power badge (square) then the race banner,
#: side by side at the same height (the banners are 429x230 px).
COMBO_BADGE_PX = 76
COMBO_BANNER_PX = round(COMBO_BADGE_PX * 429 / 230)
COMBO_COLUMN_PX = COMBO_BADGE_PX + COMBO_BANNER_PX + 20

#: Size of a race token drawn on the board, in board pixels.
TOKEN_SIZE = 20
#: Size of a marker glyph (fortress, hero, ...), in board pixels.
MARKER_SIZE = 11
#: Radius of the "legal target" ring.
RING_RADIUS = 19


def _svg_image(url: str, x: float, y: float, size: float, title: str = '',
               grey: bool = False, opacity: float = 1.0) -> str:
    """One `<image>` of the overlay, optionally greyscaled (decline)."""
    style = ' style="filter:grayscale(1)"' if grey else ''
    tooltip = f'<title>{title}</title>' if title else ''
    return (f'<image href="{url}" x="{x:.1f}" y="{y:.1f}" '
            f'width="{size:.1f}" height="{size:.1f}" opacity="{opacity:g}"'
            f'{style} preserveAspectRatio="xMidYMid meet">{tooltip}</image>')


def _svg_badge(cx: float, cy: float, text: str, radius: float = 6.0,
               fill: str = '#111111') -> str:
    """Small dark disc with a number in it (token / encampment count)."""
    return (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" '
            f'fill="{fill}" fill-opacity="0.88" stroke="#ffffff" '
            f'stroke-width="0.8"/>'
            f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" '
            f'dominant-baseline="central" font-family="sans-serif" '
            f'font-size="{radius * 1.5:.1f}" font-weight="bold" '
            f'fill="#ffffff">{text}</text>')


def _svg_text(x: float, y: float, text: str, size: float = 10.0,
              fill: str = '#ffffff') -> str:
    """Outlined text, legible over any part of the board."""
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" '
            f'dominant-baseline="central" font-family="sans-serif" '
            f'font-size="{size:.1f}" font-weight="bold" fill="{fill}" '
            f'stroke="#000000" stroke-width="2.2" paint-order="stroke">'
            f'{text}</text>')


def _svg_star(cx: float, cy: float, radius: float = 8.0) -> str:
    """The suggested-action star, same colours as `_element_star`."""
    points = []
    for step in range(10):
        angle = -math.pi / 2 + step * math.pi / 5
        rad = radius if step % 2 == 0 else radius * 0.45
        points.append(f'{cx + rad * math.cos(angle):.1f},'
                      f'{cy + rad * math.sin(angle):.1f}')
    return (f'<polygon points="{" ".join(points)}" fill="#fdff00" '
            f'stroke="#605a00" stroke-width="1"/>')


def _svg_lair(x: float, y: float, size: float) -> str:
    """Troll's Lair glyph — drawn by hand, the component set has no image."""
    s = size
    return (
        '<g><title>Troll\'s Lair</title>'
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{s:.1f}" height="{s:.1f}" '
        f'rx="{s * 0.28:.1f}" fill="#7a6a58" stroke="#241c12" '
        f'stroke-width="0.9"/>'
        f'<path d="M {x + s * 0.24:.1f} {y + s * 0.86:.1f} '
        f'L {x + s * 0.24:.1f} {y + s * 0.52:.1f} '
        f'A {s * 0.26:.1f} {s * 0.30:.1f} 0 0 1 '
        f'{x + s * 0.76:.1f} {y + s * 0.52:.1f} '
        f'L {x + s * 0.76:.1f} {y + s * 0.86:.1f} Z" fill="#1a1410"/>'
        '</g>'
    )


def region_markers(region) -> list[tuple[str, str, int]]:
    """``[(kind, tooltip, count)]`` of the markers to draw on `region`.

    The Mountain is static (every mountain region carries one from the setup),
    the others are dynamic state.
    """
    markers: list[tuple[str, str, int]] = []
    if region.lost_tribe:
        markers.append(('lost_tribe', 'Lost Tribe', 1))
    if region.is_mountain:
        markers.append(('mountain', 'Mountain', 1))
    if region.lair:
        markers.append(('lair', "Troll's Lair", 1))
    if region.fortress:
        markers.append(('fortress', 'Fortress', 1))
    if region.encampments:
        markers.append(('encampment', 'Encampments', region.encampments))
    if region.hole:
        markers.append(('hole', 'Hole in the ground', 1))
    if region.hero:
        markers.append(('hero', 'Hero', 1))
    if region.dragon:
        markers.append(('dragon', 'Dragon', 1))
    return markers


def _svg_markers(cx: float, cy: float, region) -> str:
    """The marker row of one region, centred under its anchor.

    The Lost Tribe is left out: it is drawn at race-token size in the token
    slot by :func:`board_overlay_svg`.
    """
    markers = [m for m in region_markers(region) if m[0] != 'lost_tribe']
    if not markers:
        return ''
    gap = 1.0
    width = len(markers) * MARKER_SIZE + (len(markers) - 1) * gap
    x = cx - width / 2
    y = cy + TOKEN_SIZE * 0.35
    out = []
    for kind, tooltip, count in markers:
        if kind == 'lair':
            out.append(_svg_lair(x, y, MARKER_SIZE))
        else:
            out.append(_svg_image(piece_url(kind), x, y, MARKER_SIZE,
                                  title=tooltip))
        if count > 1:
            out.append(_svg_badge(x + MARKER_SIZE, y + MARKER_SIZE,
                                  str(count), radius=4.2))
        x += MARKER_SIZE + gap
    return ''.join(out)


def board_overlay_svg(env, pov_player: int | None = None,
                      suggested_action: int | None = None,
                      mode: str = MODE_NORMAL) -> str:
    """The whole SVG overlay of the board, in image pixel coordinates.

    Draws, in this order: the turn marker, the rings of the legal targets with
    the conquest costs, then for every region its race token (greyscaled in
    decline) with the token count and the marker row, and finally the star of
    the suggested action.
    """
    sw = _sw()
    if env.board is None:                                   # pragma: no cover
        return ''
    width, height = BOARD_SIZE
    parts: list[str] = [f'<!-- smallw overlay {width}x{height} -->']

    # turn marker on the turn track
    turn_index = max(0, min(env.turn, env.turns_total) - 1)
    if turn_index < len(TURN_TRACK):
        tx, ty = TURN_TRACK[turn_index]
        parts.append(f'<image href="{piece_url("turn_marker")}" '
                     f'x="{tx - 12:.1f}" y="{ty - 7:.1f}" width="24" '
                     f'height="13" opacity="0.95">'
                     f'<title>Turn {min(env.turn, env.turns_total)}</title>'
                     f'</image>')

    targets = region_targets(env, mode)
    costs = conquest_costs(env)
    suggested_index = None
    if suggested_action is not None:
        try:
            kind, arg = sw.action_kind(int(suggested_action))
        except ValueError:                                  # pragma: no cover
            kind, arg = '', 0
        if kind in (sw.KIND_REGION, sw.KIND_REGION_ALL, sw.KIND_SORCERER,
                    sw.KIND_DRAGON):
            suggested_index = arg

    for region in env.board.regions:
        cx, cy = region.static.anchor
        group: list[str] = []

        # legal-target ring (+ conquest cost)
        kind = targets.get(region.index)
        if kind is not None:
            colour = TARGET_COLORS.get(kind, '#ffffff')
            group.append(
                f'<circle cx="{cx}" cy="{cy}" r="{RING_RADIUS}" fill="none" '
                f'stroke="{colour}" stroke-width="2.4" stroke-opacity="0.95" '
                f'stroke-dasharray="5 3"><title>{kind}</title></circle>')
            if kind == TARGET_CONQUER and region.index in costs:
                cost, final = costs[region.index]
                group.append(_svg_text(cx + RING_RADIUS - 2,
                                       cy - RING_RADIUS + 3,
                                       f'{cost}{"!" if final else ""}'))

        # race token + count
        if (region.race is not None and region.owner is not None
                and region.tokens > 0):
            size = TOKEN_SIZE
            group.append(_svg_image(
                race_token_url(region.race), cx - size / 2,
                cy - size * 0.75, size,
                title=f'{RACES[region.race].name}'
                      f'{" in decline" if region.in_decline else ""} — '
                      f'{env.players[region.owner].name}',
                grey=region.in_decline))
            group.append(_svg_badge(cx + size * 0.42, cy - size * 0.72,
                                    str(region.tokens)))
        elif region.lost_tribe:
            # a Lost Tribe occupies the region like a race: same slot, same size
            size = TOKEN_SIZE
            group.append(_svg_image(piece_url('lost_tribe'), cx - size / 2,
                                    cy - size * 0.75, size,
                                    title='Lost Tribe'))

        group.append(_svg_markers(cx, cy, region))

        if suggested_index == region.index:
            group.append(_svg_star(cx - RING_RADIUS + 2, cy - RING_RADIUS + 2))

        body = ''.join(bit for bit in group if bit)
        if body:
            parts.append(f'<g id="sw-region-{region.id}">{body}</g>')
    return '\n'.join(parts)


# --------------------------------------------------------------------------- #
# NiceGUI page
# --------------------------------------------------------------------------- #

#: The sections of the page, painted in this order.
SECTIONS: tuple[str, ...] = ('status', 'controls', 'board', 'combos',
                             'players', 'log')


def _rules_tooltip(title: str, text: str) -> None:
    """Hover reminder of a race or power rule, attached to the parent element."""
    with ui.tooltip().classes('text-body2').props('max-width="280px"'):
        ui.label(title).classes('font-bold')
        ui.label(text)


def _element_star():
    """The suggestion star, same helper as the other games."""
    content = '''<svg viewBox="0 0 300 275" xmlns="http://www.w3.org/2000/svg" version="1.1">
  <polygon fill="#fdff00" stroke="#605a00" stroke-width="15" points="150,25  179,111 269,111 197,165                     223,251  150,200 77,251  103,165                     31,111 121,111"/>
</svg>'''
    return ui.html(content, sanitize=False)


class RenderWeb:
    """The `smallw` page: board, combo column, players panel and event log.

    `init_web` creates one container per section and paints them; `render_web`
    repaints them. The painting is **synchronous** on purpose: since NiceGUI 3
    `ui.refreshable(...).refresh()` only schedules the rebuild as a background
    task, so a refresh issued from inside the page builder (`play.py` walks up
    to the human's first decision before the page is serialised) is applied
    after that function returns, and the served HTML may still show the
    pre-refresh state. Clearing a container and re-filling it works at both
    moments, so `_paint` does exactly that.

    Every handler reads the *current* `env` / `callback` (`self._env`,
    `self._callback`) instead of what its build-time closure captured, so a
    click can never send an action computed from a stale page either.
    """

    def __init__(self):
        #: current click mode (`MODE_*`), kept across repaints
        self.mode: str = MODE_NORMAL
        #: one container element per section name
        self._slots: dict[str, object] = {}
        #: last render arguments (what the handlers and the painters read)
        self._env = None
        self._callback = None
        self._suggested: int | None = None
        #: `(id(event_log), length)` of the log lines already toasted
        self._toasted: tuple[int, int] | None = None

    # -- state ---------------------------------------------------------- #

    def current_mode(self, env) -> str:
        """The click mode, forced back into what the phase offers."""
        modes = available_modes(env)
        if not modes:
            return MODE_NORMAL
        if self.mode not in modes:
            self.mode = modes[0]
        return self.mode

    def _playable(self) -> bool:
        """True while the page may send an action for the seat it shows."""
        return (self._env is not None
                and is_human_decision(self._env, self._callback))

    def _send(self, action: int) -> None:
        """Send `action` to `play.py`'s callback, refusing anything illegal."""
        env, callback = self._env, self._callback
        if env is None or callback is None:
            ui.notify('the game is not waiting for you', type='info')
            return
        if not is_human_decision(env, callback):
            ui.notify('watching: it is not your turn', type='info')
            return
        if not env.action_masks()[action]:
            ui.notify(f'illegal move: {env.describe_action(action)}',
                      type='warning')
            return
        callback(action)

    def _on_mode(self, value: str) -> None:
        """Mode toggle: remember it and redraw the highlights."""
        self.mode = value or MODE_NORMAL
        self._paint(('controls', 'board'))

    def _on_board_mouse(self, event) -> None:
        """Board click → nearest region → action of the phase and mode."""
        env = self._env
        if env is None or env.board is None:                # pragma: no cover
            return
        if not self._playable():
            ui.notify('watching: it is not your turn', type='info')
            return
        x, y = getattr(event, 'image_x', None), getattr(event, 'image_y', None)
        if x is None or y is None:                          # pragma: no cover
            return
        index = nearest_region_index(env, x, y)
        action, message = action_for_click(env, index, self.current_mode(env))
        if action is None:
            ui.notify(message, type='warning')
            return
        self._send(action)

    # -- painting ------------------------------------------------------- #

    def _paint(self, names: tuple[str, ...] | None = None) -> None:
        """Rebuild the given sections (all of them by default)."""
        if self._env is None:                               # pragma: no cover
            return
        for name in (names or SECTIONS):
            slot = self._slots.get(name)
            if slot is None:                                # pragma: no cover
                continue
            slot.clear()
            with slot:
                getattr(self, f'_build_{name}')()

    def _build_status(self):
        """Turn / phase / player / die chips, and the final standings."""
        env = self._env
        with ui.row().classes('items-center gap-2 w-full'):
            ui.label('Small World').classes('text-lg font-bold')
            if env.board is None:                           # pragma: no cover
                ui.label('waiting for the first deal')
                return
            for label, value in status_entries(env):
                with ui.element('div').classes(
                        'rounded px-2 py-0.5 bg-blue-1 border'):
                    ui.label(f'{label}: {value}').classes('text-xs')
        if env.done:
            self._build_final_scores()

    def _build_final_scores(self):
        """End of the game: a small score box pinned at the top left."""
        env = self._env
        rows = standings(env)
        box = ui.card().tight().classes('q-pa-sm shadow-6').style(
            'position: fixed; top: 8px; left: 8px; z-index: 2000; '
            'min-width: 210px; border: 2px solid #f2c037; '
            'background: #fffdf3;')
        with box:
            with ui.row().classes('items-center gap-1 no-wrap'):
                ui.icon('emoji_events', size='20px').classes('text-amber-8')
                ui.label('Final score').classes('text-subtitle2 font-bold')
            for row in rows:
                winner = row['rank'] == 1
                with ui.row().classes(
                        'items-center justify-between no-wrap w-full gap-3 '
                        'q-px-xs rounded'
                        + (' bg-amber-2' if winner else '')):
                    ui.label(f'{row["rank"]}. {row["name"]}').classes(
                        'text-body2' + (' font-bold' if winner else ''))
                    with ui.row().classes('items-center gap-1 no-wrap'):
                        ui.label(str(row['coins'])).classes(
                            'text-h6 font-bold' if winner else 'text-subtitle1')
                        ui.image(piece_url('coin_1')).style('width: 18px;')
            coins = [row['coins'] for row in rows]
            if len(set(coins)) < len(coins):
                ui.label('tie on coins: more tokens on the board wins').classes(
                    'text-caption text-grey-7')

    def _build_controls(self):
        """Mode toggle, DECLINE / PASS / ALLY buttons and the phase hint."""
        env = self._env
        if env.board is None:                               # pragma: no cover
            return
        playable = self._playable()
        suggested = self._suggested
        buttons = button_actions(env)
        modes = available_modes(env)
        with ui.row().classes('items-center gap-2 w-full'):
            if len(modes) > 1:
                ui.label('Click mode:').classes('text-sm')
                toggle = ui.toggle({mode: MODE_LABELS[mode] for mode in modes},
                                   value=self.current_mode(env),
                                   on_change=lambda e: self._on_mode(e.value))
                toggle.props('dense no-caps')
                toggle.set_enabled(playable)
            paused = env.phase == _sw().Phase.TURN_PAUSE
            specs = [] if paused else [buttons['decline'], buttons['pass']]
            if _phase_name(env.phase) == 'ALLY':
                specs += [spec for seat, spec in buttons['ally'].items()
                          if seat != env.current_player]
            for spec in specs:
                self._action_button(spec, playable, suggested)
            if paused:
                self._next_player_button()
        hint = phase_hint(env.phase)
        if hint:
            prefix = '' if playable or paused else 'Watching — '
            ui.label(prefix + hint).classes('text-xs text-grey-8')

    def _next_player_button(self):
        """TURN_PAUSE: go on with the next turn (a no-action step)."""
        callback = self._callback
        # `play.py` treats `callback(None)` as "Next step": it steps with -1
        button = ui.button('Next player', icon='skip_next',
                           on_click=lambda: callback(None))
        button.props('no-caps color=primary')
        button.set_enabled(callback is not None)

    def _action_button(self, spec: dict, playable: bool,
                       suggested: int | None):
        """One DECLINE / PASS / ALLY button, starred when suggested."""
        button = ui.button(spec['label'],
                           on_click=lambda a=spec['action']: self._send(a))
        button.props('dense no-caps')
        button.set_enabled(playable and spec['legal'])
        if suggested is not None and suggested == spec['action']:
            with button:
                _element_star().style('width: 14px; position: absolute; '
                                      'top: -4px; right: -4px;')

    def _build_board(self):
        """The board image with its SVG overlay."""
        env = self._env
        if env.board is None:                               # pragma: no cover
            return
        content = board_overlay_svg(env, suggested_action=self._suggested,
                                    mode=self.current_mode(env))
        image = ui.interactive_image(
            board_url(), content=content, on_mouse=self._on_board_mouse,
            events=['click'], cross=False, sanitize=False,
        )
        image.style('width: 68vw; max-width: 100%; min-width: 320px;')
        image.classes('border rounded')

    def _build_combos(self):
        """The visible combo column, clickable in PICK_COMBO."""
        env = self._env
        if env.board is None:                               # pragma: no cover
            return
        ui.label('Combos').classes('text-sm font-bold')
        for entry in combo_entries(env, self._suggested):
            card = ui.card().tight().classes('w-full cursor-pointer')
            # the handler checks the mask itself, so the card stays usable even
            # if the page was painted before `play.py` handed over the callback
            card.on('click', lambda a=entry['action']: self._send(a))
            if not entry['legal']:
                card.style('opacity: 0.55;')
            with card:
                with ui.row().classes('no-wrap items-center gap-1 q-pa-xs') \
                        .style('position: relative;'):
                    with ui.image(entry['badge']).style(
                            f'width: {COMBO_BADGE_PX}px; flex: 0 0 auto;'):
                        _rules_tooltip(entry['power_title'],
                                       entry['power_rules'])
                    with ui.image(entry['banner']).style(
                            f'width: {COMBO_BANNER_PX}px; flex: 0 0 auto;'):
                        _rules_tooltip(entry['race_title'],
                                       entry['race_rules'])
                    if entry['suggested']:
                        _element_star().style(
                            'position: absolute; top: 0; left: 0; '
                            'width: 22px;')
                ui.label(f'{entry["power_name"]} {entry["race_name"]}'
                         ).classes('text-xs font-bold q-px-xs')
                with ui.row().classes('items-center justify-between w-full '
                                      'q-px-xs no-wrap'):
                    ui.label(f'cost {entry["cost"]}').classes(
                        'text-xs font-bold'
                        + ('' if entry['legal'] else ' text-grey-5'))
                    ui.label(f'{entry["tokens"]} tokens').classes('text-xs')
                    if entry['coins']:
                        with ui.row().classes('items-center gap-0 no-wrap'):
                            ui.image(piece_url('coin_1')).style('width: 14px;')
                            ui.label(f'x{entry["coins"]}').classes('text-xs')
        if not env.combo_column.visible:                    # pragma: no cover
            ui.label('no combo left').classes('text-xs')

    def _build_players(self):
        """One card per seat: races, tokens, ally and coins (plan T3b rule)."""
        env = self._env
        if env.board is None:                               # pragma: no cover
            return
        for entry in player_entries(env):
            card = ui.card().tight().classes('q-pa-sm')
            card.style('min-width: 220px;')
            if entry['current']:
                card.style('border: 2px solid #1976d2;')
            with card:
                with ui.row().classes('items-center gap-1 no-wrap'):
                    name = entry['name'] + (' (you)' if entry['is_pov'] else '')
                    ui.label(name).classes(
                        'text-sm font-bold'
                        + (' text-blue-9' if entry['current'] else ''))
                    if entry['current']:
                        ui.icon('play_arrow', size='16px')
                self._build_race_row(entry['active'])
                for declined in entry['declined']:
                    self._build_race_row(declined)
                if entry['active'] is None and not entry['declined']:
                    ui.label('no race yet').classes('text-xs text-grey-7')
                with ui.row().classes('items-center gap-1 no-wrap'):
                    ui.image(piece_url('coin_1')).style('width: 16px;')
                    ui.label(entry['coin_text']).classes('text-xs')
                    if not entry['shows_value']:
                        ui.icon('visibility_off', size='14px').tooltip(
                            'the value of the coins is hidden, only the '
                            'number of tokens is public')
                if entry['ally']:
                    ui.label(f'ally: {entry["ally"]}').classes(
                        'text-xs text-green-9')
                if entry['must_first_conquest']:
                    ui.label('must start again from a border region').classes(
                        'text-xs text-orange-9')

    def _build_race_row(self, race: dict | None):
        """One banner + badge line inside a player card."""
        if race is None:
            return
        with ui.row().classes('items-center gap-1 no-wrap'):
            with ui.element('div').style('position: relative; width: 84px;'):
                with ui.image(race['banner']).style('width: 84px;') as banner:
                    _rules_tooltip(race['race_title'], race['race_rules'])
                if race['in_decline']:
                    banner.style('filter: grayscale(1); opacity: 0.85;')
                if race['badge']:
                    with ui.image(race['badge']).style(
                            'position: absolute; right: -2px; bottom: -2px; '
                            'width: 30px;'):
                        _rules_tooltip(race['power_title'],
                                       race['power_rules'])
            with ui.column().classes('gap-0'):
                title = race['race_name']
                if race['power_name']:
                    title += f' / {race["power_name"]}'
                if race['in_decline']:
                    title += ' (decline)'
                if race['is_spirit']:
                    title += ' [spirit]'
                ui.label(title).classes(
                    'text-xs' + (' text-grey-7' if race['in_decline']
                                 else ' font-bold'))
                ui.label(f'hand {race["hand"]} · board '
                         f'{race["tokens_on_board"]} in {race["regions"]} '
                         f'region(s) · tray {race["tray"]}'
                         ).classes('text-xs text-grey-8')

    def _toast_new_lines(self) -> None:
        """Pop the log lines logged since the last repaint as short toasts,
        stacked at the top of the page (Quasar stacks same-position ones)."""
        lines, self._toasted = new_log_lines(self._env.event_log, self._toasted)
        messages = toast_messages(lines)
        if not messages:
            return
        # `ui.notify` finds the page through the current slot, which may be the
        # clicked button that the repaint just deleted: use a section container
        # instead (they are cleared, never deleted). Quasar starts the timeout
        # once its slide-in is over, so each toast is fully visible for
        # TOAST_TIMEOUT_MS, then fades out. Quasar puts the newest toast on
        # top: sending them in reverse makes a burst read top-down in order.
        with self._slots['log']:
            for message in reversed(messages):
                ui.notify(message, position='top', timeout=TOAST_TIMEOUT_MS,
                          group=False, classes='smallw-toast')

    def _build_log(self):
        """Scrollable event log, newest at the bottom, auto-scrolled."""
        env = self._env
        ui.label('Game log').classes('text-sm font-bold')
        with ui.scroll_area().classes('w-full border rounded').style(
                'height: 150px;') as area:
            for line in env.event_log[-400:]:
                ui.label(line).classes('text-xs').style(
                    'white-space: pre-wrap; line-height: 1.15;')
        area.scroll_to(percent=1.0)

    # -- contract ------------------------------------------------------- #

    def init_web(self, env: 'SmallWorldEnv', callback=None):
        """Build the page once (called by `SmallWorldEnv.nicegui_page`)."""
        register_static_files()
        ui.add_css(TOAST_CSS)
        self._env, self._callback, self._suggested = env, callback, None
        slots = {}
        with ui.column().classes('w-full gap-2 q-pa-sm'):
            slots['status'] = ui.column().classes('w-full gap-1')
            with ui.row().classes('w-full no-wrap items-start gap-3'):
                with ui.column().classes('gap-2').style(
                        'flex: 1 1 auto; min-width: 0;'):
                    slots['controls'] = ui.column().classes('w-full gap-1')
                    slots['board'] = ui.column().classes('w-full gap-0')
                slots['combos'] = ui.column().classes('gap-1').style(
                    f'flex: 0 0 auto; width: {COMBO_COLUMN_PX}px;')
            slots['players'] = ui.row().classes('w-full items-stretch gap-2')
            slots['log'] = ui.column().classes('w-full gap-1')
        self._slots = slots
        self._paint()
        self._toast_new_lines()

    def render_web(self, env: 'SmallWorldEnv', callback=None,
                   suggested_action: int | None = None, **kwargs):
        """Repaint the page (the extra `play.py` kwargs are ignored)."""
        self._env = env
        self._callback = callback
        self._suggested = suggested_action
        self._paint()
        self._toast_new_lines()
