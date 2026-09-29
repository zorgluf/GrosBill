"""Tests for the NiceGUI renderer of Small World (`envs/render_web.py`, T7).

Plain `assert`-based functions, usable either with pytest or through the
bundled runners:

    cd app
    python -m environments.smallw.tests.run_all        # every test module
    python -m environments.smallw.tests.test_render    # this module only

**No browser and no NiceGUI element is ever built here**: everything the page
displays is computed by the pure module-level helpers of `render_web.py`
(`board_overlay_svg`, `action_for_click`, `region_targets`, `coin_text`,
`phase_label`, ...), and those are what this module checks — the SVG overlay
of hand-built states, the click → region → action mapping of every phase and
mode against the real `action_masks()`, the coin-hiding rule of plan T3b, the
phase labels and a random-game sweep calling every helper in every phase.

The scenario helpers (`_env`, `_setup_turn`, `_occupy`, `_legal`,
`_force_dice`) are the ones of `test_engine.py`, as `test_races.py` and
`test_powers.py` do.
"""

from __future__ import annotations

import traceback

import numpy as np

from ..envs import render_web as rw
from ..envs.classes import N_ENCAMPMENTS, POWERS, RACES, PowerId, RaceId
from ..envs.map3p import BOARD_IMAGE, MAP3P, TURN_TRACK
from ..envs.smallw import (
    A_ALLY,
    A_COMBO,
    A_DECLINE,
    A_DRAGON,
    A_PASS,
    A_REGION,
    A_REGION_ALL,
    A_SORCERER,
    Phase,
    SmallWorldEnv,
    region_action,
    region_all_action,
)
from .test_engine import _env, _force_dice, _legal, _occupy, _setup_turn

#: Random games played by the sweep test.
SWEEP_GAMES = 6


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _index(region_id: int) -> int:
    """0-based index of the region printed `region_id`."""
    return region_id - 1


def _group(svg: str, region_id: int) -> str:
    """The overlay group of one region (one line of the generated SVG)."""
    needle = f'id="sw-region-{region_id}"'
    for line in svg.splitlines():
        if needle in line:
            return line
    return ''


def _anchor(region_id: int) -> tuple[int, int]:
    """Pixel anchor of the region printed `region_id`."""
    return MAP3P[region_id - 1].anchor


def _finish_redeploy(env: SmallWorldEnv) -> None:
    """Dump the acting race's hand on its first region.

    Stops *before* a VICTIM_REDEPLOY, so a test can inspect that phase.
    """
    while env.phase in (Phase.REDEPLOY, Phase.GHOUL_REDEPLOY):
        legal = [a for a in _legal(env) if A_REGION_ALL <= a < A_SORCERER]
        env.step(legal[0])


# --------------------------------------------------------------------------- #
# Static assets
# --------------------------------------------------------------------------- #

def test_asset_urls_point_at_existing_files():
    """Every URL the renderer builds resolves to a committed static file."""
    assert rw.STATIC_DIR.is_dir(), rw.STATIC_DIR
    assert rw.board_url() == f'/smallw_static/{BOARD_IMAGE}'

    urls = [rw.board_url(), rw.piece_url('coin_1'), rw.piece_url('turn_marker'),
            rw.piece_url('die')]
    for race_id in RaceId:
        urls += [rw.race_banner_url(race_id), rw.race_token_url(race_id)]
    for power_id in PowerId:
        urls.append(rw.power_badge_url(power_id))
    # every marker the overlay can draw, except the hand-drawn Troll's Lair
    for kind in ('lost_tribe', 'mountain', 'fortress', 'encampment', 'hole',
                 'hero', 'dragon'):
        urls.append(rw.piece_url(kind))

    for url in urls:
        assert url.startswith(rw.STATIC_URL + '/'), url
        path = rw.STATIC_DIR / url[len(rw.STATIC_URL) + 1:]
        assert path.is_file(), f'{url} -> {path} is missing'


def test_asset_keys_match_the_definitions():
    """The URLs are built from the `key` of the race / power definitions."""
    assert rw.race_banner_url(RaceId.TROLLS).endswith('/races/trolls.png')
    assert rw.race_token_url(RaceId.TROLLS).endswith('/races/trolls_token.jpg')
    assert rw.power_badge_url(PowerId.DRAGON_MASTER).endswith(
        '/powers/dragon_master.png')
    for race_id in RaceId:
        assert RACES[race_id].key in rw.race_banner_url(race_id)
    for power_id in PowerId:
        assert POWERS[power_id].key in rw.power_badge_url(power_id)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #

def test_phase_labels_cover_every_phase():
    """Every phase has a distinct plain-words label and an instruction."""
    labels = {phase: rw.phase_label(phase) for phase in Phase}
    assert len(set(labels.values())) == len(Phase), labels
    for phase, label in labels.items():
        assert label and label != phase.name, phase
        assert rw.phase_hint(phase), phase
    assert rw.phase_label(Phase.CONQUER) == 'Conquests'
    assert rw.phase_label(Phase.PICK_COMBO).startswith('Pick a race')
    assert 'Ghouls' in rw.phase_label(Phase.GHOUL_CONQUER)
    assert rw.phase_label(Phase.DONE) == 'Game over'


def test_pass_labels_per_phase():
    """The `A_PASS` button is labelled for each phase that offers it."""
    assert rw.pass_label(Phase.CONQUER) == 'End conquests'
    assert rw.pass_label(Phase.GHOUL_CONQUER) == 'End conquests'
    assert rw.pass_label(Phase.FORTRESS) == 'No fortress'
    assert rw.pass_label(Phase.ALLY) == 'No ally'
    assert rw.pass_label(Phase.STOUT_DECLINE) == 'Stay active'
    assert rw.pass_label(Phase.REDEPLOY) == 'Pass'


def test_mode_labels():
    """Every mode has a label, and the toggles read as the plan asks."""
    assert rw.MODE_LABELS[rw.MODE_NORMAL] == 'Normal'
    assert rw.MODE_LABELS[rw.MODE_SORCERER] == 'Sorcerer'
    assert rw.MODE_LABELS[rw.MODE_DRAGON] == 'Dragon'
    assert rw.MODE_LABELS[rw.MODE_ONE] == 'one token'
    assert rw.MODE_LABELS[rw.MODE_ALL] == 'all remaining'


# --------------------------------------------------------------------------- #
# Click geometry
# --------------------------------------------------------------------------- #

def test_nearest_region_index_on_every_anchor():
    """A click on an anchor (or 2 px away) selects that very region."""
    env = _env(700)
    for region in env.board.regions:
        ax, ay = region.static.anchor
        assert rw.nearest_region_index(env, ax, ay) == region.index
        assert rw.nearest_region_index(env, ax + 2, ay - 2) == region.index


def test_nearest_region_index_outside_the_board():
    """Coordinates off the board still resolve to the closest anchor."""
    env = _env(701)
    assert rw.nearest_region_index(env, 0, 0) == _index(1)        # anchor 28,60
    assert rw.nearest_region_index(env, 596, 0) == _index(7)      # anchor 555,80
    assert rw.nearest_region_index(env, 596, 296) == _index(30)   # anchor 565,250
    # halfway between two anchors: one of the two, never a third one
    (x1, y1), (x2, y2) = _anchor(9), _anchor(10)
    mid = rw.nearest_region_index(env, (x1 + x2) / 2, (y1 + y2) / 2)
    assert mid in (_index(9), _index(10))


# --------------------------------------------------------------------------- #
# SVG overlay
# --------------------------------------------------------------------------- #

def test_overlay_draws_tokens_counts_and_greyscale():
    """Every held region shows its race token, its count, grey in decline."""
    env = _env(710)
    _setup_turn(env, 0, RaceId.RATMEN, None, hand=2, held={9: 3})
    _occupy(env, 1, RaceId.HUMANS, 10, 1, in_decline=True)
    svg = rw.board_overlay_svg(env)

    active = _group(svg, 9)
    assert rw.race_token_url(RaceId.RATMEN) in active
    assert '>3</text>' in active, 'the token count badge'
    assert 'grayscale' not in active, 'an active race keeps its colours'
    assert 'Ratmen' in active and env.players[0].name in active, 'tooltip'

    declined = _group(svg, 10)
    assert rw.race_token_url(RaceId.HUMANS) in declined
    assert 'grayscale' in declined, 'a declined race is greyscaled'
    assert 'in decline' in declined
    # an empty region draws no token at all
    assert rw.race_token_url(RaceId.RATMEN) not in _group(svg, 4)


def test_overlay_draws_every_marker():
    """Lost Tribe, Mountain, Lair, fortress, encampments, hole, hero, dragon."""
    env = _env(711)
    _setup_turn(env, 0, RaceId.TROLLS, None, hand=1, held={9: 1})
    region = env.board.by_id(9)
    region.lair = True
    region.fortress = True
    region.encampments = 3
    region.hole = True
    region.hero = True
    region.dragon = True
    lost = env.board.by_id(6)
    lost.lost_tribe = True
    svg = rw.board_overlay_svg(env)

    group = _group(svg, 9)
    for kind in ('fortress', 'encampment', 'hole', 'hero', 'dragon'):
        assert rw.piece_url(kind) in group, kind
    assert "Troll's Lair" in group and 'rx=' in group, 'the Lair is hand-drawn'
    assert '>3</text>' in group, 'the encampment count'
    assert rw.piece_url('lost_tribe') in _group(svg, 6)
    size = f'width="{rw.TOKEN_SIZE:.1f}"'
    assert size in _group(svg, 6), 'the Lost Tribe is drawn at race-token size'
    # the mountains carry their static token even while empty
    assert rw.piece_url('mountain') in _group(svg, 3)
    assert rw.piece_url('mountain') not in _group(svg, 6), 'region 6 is a forest'


def test_region_markers_listing():
    """`region_markers` lists exactly what sits on the region."""
    env = _env(712)
    region = env.board.by_id(9)
    assert rw.region_markers(region) == []
    region.lost_tribe = True
    region.encampments = 2
    kinds = [kind for kind, _tooltip, _count in rw.region_markers(region)]
    assert kinds == ['lost_tribe', 'encampment']
    assert rw.region_markers(region)[1][2] == 2
    assert [k for k, _t, _c in rw.region_markers(env.board.by_id(3))] == ['mountain']


def test_overlay_turn_marker_follows_the_turn_track():
    """The turn marker sits on `TURN_TRACK[turn - 1]`, clamped at the end."""
    env = _env(713)
    x, y = TURN_TRACK[0]
    svg = rw.board_overlay_svg(env)
    assert rw.piece_url('turn_marker') in svg
    assert f'x="{x - 12:.1f}" y="{y - 7:.1f}"' in svg
    assert '<title>Turn 1</title>' in svg

    env.turn = 5
    x, y = TURN_TRACK[4]
    assert f'x="{x - 12:.1f}" y="{y - 7:.1f}"' in rw.board_overlay_svg(env)

    env.turn = env.turns_total + 1                 # what `_end_game` leaves
    env.done = True
    x, y = TURN_TRACK[env.turns_total - 1]
    svg = rw.board_overlay_svg(env)
    assert f'x="{x - 12:.1f}" y="{y - 7:.1f}"' in svg
    assert f'<title>Turn {env.turns_total}</title>' in svg


def test_overlay_highlights_legal_targets_with_distinct_colours():
    """Abandon / conquer / redeploy / marker rings use different colours."""
    env = _env(714)
    _setup_turn(env, 0, RaceId.RATMEN, None, hand=6, held={9: 1})
    svg = rw.board_overlay_svg(env)
    assert rw.TARGET_COLORS[rw.TARGET_ABANDON] in _group(svg, 9), 'own region'
    assert rw.TARGET_COLORS[rw.TARGET_CONQUER] in _group(svg, 10), 'neighbour'
    cost, final = rw.conquest_costs(env)[_index(10)]
    assert not final and f'>{cost}</text>' in _group(svg, 10), 'the cost is drawn'
    assert rw.TARGET_COLORS[rw.TARGET_CONQUER] not in _group(svg, 29), 'too far'

    env.step(A_PASS)                                        # -> REDEPLOY
    assert env.phase == Phase.REDEPLOY
    svg = rw.board_overlay_svg(env, mode=rw.MODE_ALL)
    assert rw.TARGET_COLORS[rw.TARGET_REDEPLOY] in _group(svg, 9)
    assert rw.TARGET_COLORS[rw.TARGET_CONQUER] not in _group(svg, 10)

    env = _env(715)
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.BIVOUACKING, hand=0,
                held={9: 1, 10: 1}, turns_played=2)
    env.step(A_PASS)
    assert env.phase == Phase.ENCAMPMENTS
    svg = rw.board_overlay_svg(env)
    assert rw.TARGET_COLORS[rw.TARGET_MARKER] in _group(svg, 9)
    assert rw.TARGET_COLORS[rw.TARGET_MARKER] in _group(svg, 10)
    assert rw.TARGET_COLORS[rw.TARGET_MARKER] not in _group(svg, 14)


def test_overlay_star_marks_the_suggested_region_only():
    """The star lands on the suggested region, and never on a combo pick."""
    env = _env(716)
    _setup_turn(env, 0, RaceId.RATMEN, None, hand=6, held={9: 1})
    svg = rw.board_overlay_svg(env, suggested_action=region_action(_index(10)))
    assert '<polygon' in _group(svg, 10), 'the suggestion star'
    assert '<polygon' not in _group(svg, 9)
    # a combo suggestion is drawn on the card, not on the board
    svg = rw.board_overlay_svg(env, suggested_action=A_COMBO + 2)
    assert '<polygon' not in svg
    # so are DECLINE / PASS / ALLY
    for action in (A_DECLINE, A_PASS, A_ALLY + 1):
        assert '<polygon' not in rw.board_overlay_svg(env, suggested_action=action)


def test_overlay_of_a_fresh_game_has_no_race_token():
    """Right after `reset()` only the Lost Tribes and the mountains show."""
    env = _env(717)
    svg = rw.board_overlay_svg(env)
    assert '_token.jpg' not in svg, 'no race is in play yet'
    assert rw.piece_url('lost_tribe') in svg
    assert rw.piece_url('mountain') in svg
    assert svg.count('<g id="sw-region-') > 0
    # and it survives being asked for a game that is over
    env = _env(718)
    while not env.done:
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert rw.board_overlay_svg(env)
    assert rw.status_entries(env)[-1][0] == 'Winner'


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #

def test_available_modes_in_the_conquest_phases():
    """*Sorcerer* / *Dragon* only appear when such an action really is legal."""
    env = _env(720)
    _setup_turn(env, 0, RaceId.RATMEN, None, hand=6, held={9: 1})
    assert rw.available_modes(env) == [rw.MODE_NORMAL]

    env = _env(721)
    _setup_turn(env, 0, RaceId.SORCERERS, None, hand=4, held={14: 1})
    _occupy(env, 1, RaceId.HUMANS, 10, 1)
    assert rw.available_modes(env) == [rw.MODE_NORMAL, rw.MODE_SORCERER]

    env = _env(722)
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.DRAGON_MASTER, hand=4,
                held={14: 1})
    assert rw.available_modes(env) == [rw.MODE_NORMAL, rw.MODE_DRAGON]


def test_available_modes_in_the_other_phases():
    """The redeployments offer one/all; the card and button phases offer none."""
    env = _env(723)
    _setup_turn(env, 0, RaceId.RATMEN, None, hand=4, held={9: 1})
    env.step(A_PASS)
    assert env.phase == Phase.REDEPLOY
    assert rw.available_modes(env) == [rw.MODE_ONE, rw.MODE_ALL]

    assert rw.available_modes(_env(724)) == [], 'PICK_COMBO'

    env = _env(725)
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.DIPLOMAT, hand=0, held={9: 1},
                turns_played=2)
    env.step(A_PASS)
    assert env.phase == Phase.ALLY and rw.available_modes(env) == []


# --------------------------------------------------------------------------- #
# Click -> action
# --------------------------------------------------------------------------- #

def test_click_is_refused_when_the_cards_or_buttons_are_the_only_way():
    """PICK_COMBO / ALLY / STOUT_DECLINE ignore the board, with a reason."""
    env = _env(730)
    action, message = rw.action_for_click(env, _index(9))
    assert action is None and 'combo' in message

    env = _env(731)
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.DIPLOMAT, hand=0, held={9: 1},
                turns_played=2)
    env.step(A_PASS)
    assert env.phase == Phase.ALLY
    action, message = rw.action_for_click(env, _index(9))
    assert action is None and 'ally' in message

    env = _env(732)
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.STOUT, hand=0, held={9: 1},
                turns_played=2)
    env.step(A_PASS)
    assert env.phase == Phase.STOUT_DECLINE
    action, message = rw.action_for_click(env, _index(9))
    assert action is None and 'decline' in message


def test_click_conquer_and_abandon():
    """In *Normal* mode a click conquers, or abandons one of your regions."""
    env = _env(733)
    _setup_turn(env, 0, RaceId.RATMEN, None, hand=6, held={9: 1})
    action, message = rw.action_for_click(env, _index(10))
    assert action == region_action(_index(10))
    assert 'conquer' in message and 'region 10' in message
    assert rw.region_targets(env)[_index(10)] == rw.TARGET_CONQUER

    action, message = rw.action_for_click(env, _index(9))
    assert action == region_action(_index(9))
    assert 'abandon' in message
    assert rw.region_targets(env)[_index(9)] == rw.TARGET_ABANDON

    env.step(region_action(_index(10)))                     # first conquest
    action, message = rw.action_for_click(env, _index(9))
    assert action is None and 'abandon' in message, message
    assert _index(9) not in rw.region_targets(env)


def test_click_conquer_illegal_reasons():
    """The refusal message says *why* the region cannot be taken."""
    env = _env(734)
    _setup_turn(env, 0, RaceId.RATMEN, None, hand=6, held={14: 1})
    env.board.by_id(10).hero = True
    action, message = rw.action_for_click(env, _index(10))
    assert action is None and 'protected' in message

    action, message = rw.action_for_click(env, _index(15))     # the lake
    assert action is None and 'water' in message and 'Seafaring' in message

    action, message = rw.action_for_click(env, _index(29))     # far away
    assert action is None and 'out of reach' in message

    _occupy(env, 1, RaceId.HUMANS, 20, 6)                      # 8 tokens needed
    action, message = rw.action_for_click(env, _index(20))
    assert action == region_action(_index(20)), 'a final attempt is legal'
    assert 'final attempt' in message
    env.board.by_id(20).tokens += 2                            # 10 needed now
    action, message = rw.action_for_click(env, _index(20))
    assert action is None and 'out of reach' not in message
    assert 'costs 10 token(s) and you only have 6' in message, message

    env.players[1].ally = 0                                    # Diplomat pact
    _occupy(env, 1, RaceId.HUMANS, 13, 1)
    action, message = rw.action_for_click(env, _index(13))
    assert action is None and 'ally' in message, message

    env.current_race.tokens_in_hand = 0
    action, message = rw.action_for_click(env, _index(9))
    assert action is None and 'no token left in hand' in message


def test_click_sorcerer_and_dragon_modes():
    """The mode decides which action range a click lands in."""
    env = _env(735)
    _setup_turn(env, 0, RaceId.SORCERERS, None, hand=4, held={14: 1})
    _occupy(env, 1, RaceId.HUMANS, 10, 1)
    assert rw.action_for_click(env, _index(10), rw.MODE_SORCERER)[0] == \
        A_SORCERER + _index(10)
    assert rw.region_targets(env, rw.MODE_SORCERER) == \
        {_index(10): rw.TARGET_SORCERER}
    # the same click in Normal mode is an ordinary conquest
    assert rw.action_for_click(env, _index(10), rw.MODE_NORMAL)[0] == \
        region_action(_index(10))
    action, message = rw.action_for_click(env, _index(4), rw.MODE_SORCERER)
    assert action is None and 'Sorcerer' in message

    env = _env(736)
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.DRAGON_MASTER, hand=4,
                held={14: 1})
    _occupy(env, 1, RaceId.HUMANS, 10, 8)                   # 10 tokens needed
    assert _index(10) not in rw.region_targets(env, rw.MODE_NORMAL), \
        'far beyond a final attempt'
    assert rw.action_for_click(env, _index(10), rw.MODE_DRAGON)[0] == \
        A_DRAGON + _index(10)
    assert rw.region_targets(env, rw.MODE_DRAGON)[_index(10)] == rw.TARGET_DRAGON
    action, message = rw.action_for_click(env, _index(15), rw.MODE_DRAGON)
    assert action is None and 'dragon' in message


def test_click_redeploy_one_and_all():
    """*one token* sends `A_REGION`, *all remaining* sends `A_REGION_ALL`."""
    env = _env(737)
    _setup_turn(env, 0, RaceId.RATMEN, None, hand=5, held={9: 1})
    env.step(A_PASS)
    assert env.phase == Phase.REDEPLOY
    assert rw.action_for_click(env, _index(9), rw.MODE_ONE)[0] == \
        region_action(_index(9))
    assert rw.action_for_click(env, _index(9), rw.MODE_ALL)[0] == \
        region_all_action(_index(9))
    assert rw.region_targets(env, rw.MODE_ONE) == {_index(9): rw.TARGET_REDEPLOY}
    assert rw.region_targets(env, rw.MODE_ALL) == {_index(9): rw.TARGET_REDEPLOY}
    action, message = rw.action_for_click(env, _index(10), rw.MODE_ALL)
    assert action is None and 'not yours' in message
    assert rw.tokens_to_place(env) == 5


def test_click_victim_and_ghoul_redeploys():
    """The victim's and the Ghouls' redeployments behave like a normal one."""
    env = _env(738)
    _setup_turn(env, 0, RaceId.RATMEN, None, hand=8, held={14: 1})
    _occupy(env, 1, RaceId.HUMANS, 10, 2)
    _occupy(env, 1, RaceId.HUMANS, 4, 2)
    _occupy(env, 1, RaceId.HUMANS, 5, 2)                    # 2 regions are left
    env.step(region_action(_index(10)))                     # player 1 loses one
    env.step(A_PASS)
    _finish_redeploy(env)
    assert env.phase == Phase.VICTIM_REDEPLOY and env.current_player == 1
    assert rw.action_for_click(env, _index(4), rw.MODE_ONE)[0] == \
        region_action(_index(4))
    assert rw.region_targets(env, rw.MODE_ALL)[_index(4)] == rw.TARGET_REDEPLOY
    assert rw.phase_label(env.phase).startswith('Redeploy the tokens lost')

    env = _env(739)
    ghouls = _occupy(env, 0, RaceId.GHOULS, 9, 3, in_decline=True)
    ghouls.tokens_in_hand = 0
    env._begin_turn(0)
    assert env.phase == Phase.GHOUL_CONQUER
    assert rw.available_modes(env) == [rw.MODE_NORMAL]
    assert rw.region_targets(env)[_index(9)] == rw.TARGET_ABANDON
    env.step(A_PASS)
    assert env.phase == Phase.GHOUL_REDEPLOY
    assert rw.available_modes(env) == [rw.MODE_ONE, rw.MODE_ALL]


def test_click_marker_phases():
    """ENCAMPMENTS / FORTRESS / HEROES all place their marker with `A_REGION`."""
    env = _env(740)
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.BIVOUACKING, hand=0,
                held={9: 1, 10: 1}, turns_played=2)
    env.step(A_PASS)
    assert env.phase == Phase.ENCAMPMENTS
    assert rw.tokens_to_place(env) == N_ENCAMPMENTS
    action, message = rw.action_for_click(env, _index(9))
    assert action == region_action(_index(9)) and 'encampment' in message
    assert rw.action_for_click(env, _index(14))[0] is None

    env = _env(741)
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.FORTIFIED, hand=0,
                held={9: 1, 10: 1}, turns_played=2)
    env.step(A_PASS)
    assert env.phase == Phase.FORTRESS
    action, message = rw.action_for_click(env, _index(10))
    assert action == region_action(_index(10)) and 'fortress' in message
    assert rw.button_actions(env)['pass']['label'] == 'No fortress'
    env.board.by_id(9).fortress = True
    action, message = rw.action_for_click(env, _index(9))
    assert action is None and 'already has a fortress' in message

    env = _env(742)
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.HEROIC, hand=0,
                held={9: 1, 10: 1}, turns_played=2)
    env.step(A_PASS)
    assert env.phase == Phase.HEROES
    action, message = rw.action_for_click(env, _index(9))
    assert action == region_action(_index(9)) and 'hero' in message
    env.step(action)
    action, message = rw.action_for_click(env, _index(9))
    assert action is None and 'already has a hero' in message


def test_region_targets_always_match_the_masks():
    """`region_targets` is exactly the set of region clicks `step` accepts."""
    env = _env(743)
    _setup_turn(env, 0, RaceId.SORCERERS, PowerId.DRAGON_MASTER, hand=5,
                held={14: 1})
    _occupy(env, 1, RaceId.HUMANS, 10, 1)
    for mode, base in ((rw.MODE_NORMAL, A_REGION),
                       (rw.MODE_SORCERER, A_SORCERER),
                       (rw.MODE_DRAGON, A_DRAGON)):
        masks = env.action_masks()
        expected = {i for i in range(env.n_regions) if masks[base + i]}
        assert set(rw.region_targets(env, mode)) == expected, mode
        for index in range(env.n_regions):
            action, _message = rw.action_for_click(env, index, mode)
            assert (action is not None) == (index in expected), (mode, index)


def test_game_over_refuses_every_click():
    """After `done` nothing is clickable and the standings are available."""
    env = _env(744)
    while not env.done:
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    action, message = rw.action_for_click(env, _index(9))
    assert action is None and 'over' in message
    assert rw.region_targets(env) == {}
    assert rw.available_modes(env) == []
    rows = rw.standings(env)
    assert [row['rank'] for row in rows] == [1, 2, 3]
    assert rows[0]['seat'] == env.winner_player
    assert rows[0]['coins'] == max(p.coins for p in env.players)


def test_standings_tie_breaks_on_tokens_then_seat():
    """Same coins → more tokens on board first, then the lowest seat."""
    env = _env(745)
    _occupy(env, 1, RaceId.HUMANS, 10, 3)
    for player in env.players:
        player.coins = 20
    rows = rw.standings(env)
    assert [row['seat'] for row in rows] == [1, 0, 2]
    assert rows[0]['tokens'] == 3


# --------------------------------------------------------------------------- #
# Coins (plan T3b), players panel, combos, buttons, status
# --------------------------------------------------------------------------- #

def test_coin_display_rule():
    """The value is shown to its owner only; the others show token counts."""
    env = _env(750)
    env.players[0].coins = 13          # -> 10 + 3, two tokens
    env.players[1].coins = 7           # -> 5 + 1 + 1, three tokens
    assert env.players[0].coin_count == 2 and env.players[1].coin_count == 3

    assert rw.shows_coin_value(0, 0) and not rw.shows_coin_value(1, 0)
    assert rw.shows_coin_value(0, -1) and rw.shows_coin_value(1, -1)

    assert rw.coin_text(env, 0, 0) == '13 coins in 2 token(s)'
    assert rw.coin_text(env, 1, 0) == '3 coin token(s)'
    assert '7' not in rw.coin_text(env, 1, 0), 'the value must stay hidden'
    assert rw.coin_text(env, 1, -1) == '7 coins in 3 token(s)'


def test_player_entries_hide_the_opponents_values():
    """The panel data follows the same rule and describes every banner."""
    env = _env(751)
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.BERSERK, hand=4, held={9: 2})
    _occupy(env, 1, RaceId.GHOULS, 10, 1, in_decline=True)
    env.players[1].coins = 11
    env.players[2].ally = 0

    entries = rw.player_entries(env, pov_player=0)
    assert [entry['seat'] for entry in entries] == [0, 1, 2]
    assert entries[0]['current'] and entries[0]['is_pov']
    assert entries[0]['shows_value'] and '5 coins' in entries[0]['coin_text']
    assert not entries[1]['shows_value']
    assert env.players[1].coin_count == 2, '11 = 10 + 1'
    assert entries[1]['coin_text'] == '2 coin token(s)'
    assert '11' not in entries[1]['coin_text']
    assert entries[2]['ally'] == env.players[0].name

    active = entries[0]['active']
    assert active['race_name'] == 'Ratmen' and active['power_name'] == 'Berserk'
    assert active['hand'] == 4 and active['regions'] == 1
    assert active['tokens_on_board'] == 2 and active['acting']
    assert active['banner'].endswith('ratmen.png')
    assert active['badge'].endswith('berserk.png')
    assert not active['in_decline']

    declined = entries[1]['declined'][0]
    assert declined['in_decline'] and declined['power_name'] is None
    assert declined['badge'] is None and declined['race_name'] == 'Ghouls'
    assert entries[1]['active'] is None

    # god mode shows everything
    everything = rw.player_entries(env, pov_player=-1)
    assert all(entry['shows_value'] for entry in everything)
    assert '11 coins' in everything[1]['coin_text']


def test_player_entries_support_three_declined_races():
    """Two Spirits give three declined banners; all of them are listed."""
    env = _env(752)
    _occupy(env, 0, RaceId.DWARVES, 9, 1, in_decline=True)
    _occupy(env, 0, RaceId.HUMANS, 10, 1, in_decline=True)
    _occupy(env, 0, RaceId.ORCS, 4, 1, in_decline=True)
    env.players[0].declined[1].is_spirit = True
    env.players[0].declined[2].is_spirit = True
    entry = rw.player_entries(env, pov_player=0)[0]
    assert len(entry['declined']) == 3
    assert [race['race_name'] for race in entry['declined']] == [
        'Dwarves', 'Humans', 'Orcs']
    assert [race['is_spirit'] for race in entry['declined']] == [False, True, True]


def test_combo_entries():
    """Six clickable cards, cost = row, coins on them, suggestion star."""
    env = _env(753)
    entries = rw.combo_entries(env, suggested_action=A_COMBO + 2)
    assert len(entries) == 6
    assert [entry['cost'] for entry in entries] == [0, 1, 2, 3, 4, 5]
    assert [entry['action'] for entry in entries] == [A_COMBO + i for i in range(6)]
    assert all(entry['legal'] for entry in entries), '5 coins pays every row'
    assert [entry['suggested'] for entry in entries] == [
        False, False, True, False, False, False]
    for entry in entries:
        assert entry['banner'].endswith('.png') and entry['badge'].endswith('.png')
        assert entry['race_name'] == RACES[entry['race']].name
        assert entry['power_name'] == POWERS[entry['power']].name
        assert entry['tokens'] == rw.combo_tokens(entry['race'], entry['power'])
        assert entry['coins'] == 0

    env.players[0].coins = 2                       # only the first three rows
    assert [entry['legal'] for entry in rw.combo_entries(env)] == [
        True, True, True, False, False, False]

    env.combo_column.visible[3].coins = 2
    assert rw.combo_entries(env)[3]['coins'] == 2


def test_every_race_and_power_has_a_rules_reminder():
    """The hover tooltips cover the 14 races and the 20 powers."""
    from ..envs.rules_text import POWER_RULES, RACE_RULES
    assert set(RACE_RULES) == set(RaceId)
    assert set(POWER_RULES) == set(PowerId)
    assert all(text.strip() for text in RACE_RULES.values())
    assert all(text.strip() for text in POWER_RULES.values())
    assert rw.race_title(RaceId.AMAZONS) == 'Amazons (6 +4 tokens)'
    assert rw.power_title(PowerId.MERCHANT) == 'Merchant (+2 tokens)'


def test_combo_and_race_entries_carry_the_rules():
    """Combo cards and player race rows expose the tooltip texts."""
    from ..envs.rules_text import POWER_RULES, RACE_RULES
    env = _env(754)
    for entry in rw.combo_entries(env):
        assert entry['race_rules'] == RACE_RULES[entry['race']]
        assert entry['power_rules'] == POWER_RULES[entry['power']]
        assert entry['race_title'].startswith(entry['race_name'])
        assert entry['power_title'].startswith(entry['power_name'])
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.MERCHANT, hand=5)
    active = rw.player_entries(env, pov_player=0)[0]['active']
    assert active['race_rules'] == RACE_RULES[RaceId.RATMEN]
    assert active['power_rules'] == POWER_RULES[PowerId.MERCHANT]


def test_combo_tokens_counts_the_banner_the_badge_and_the_bonus():
    """Amazons + Berserk = 6 banner + 4 attack-only + 4 badge."""
    assert rw.combo_tokens(RaceId.AMAZONS, PowerId.BERSERK) == 6 + 4 + 4
    assert rw.combo_tokens(RaceId.DWARVES, PowerId.MERCHANT) == 3 + 2
    assert rw.combo_tokens(RaceId.DWARVES, None) == 3


def test_button_actions_follow_the_masks():
    """DECLINE / PASS / ALLY are only offered when the masks allow them."""
    env = _env(754)
    buttons = rw.button_actions(env)                        # PICK_COMBO
    assert buttons['decline']['label'] == 'Go in decline'
    assert not buttons['decline']['legal'] and not buttons['pass']['legal']
    assert not any(spec['legal'] for spec in buttons['ally'].values())

    env = _env(755)
    _setup_turn(env, 0, RaceId.RATMEN, None, hand=6, held={9: 1})
    buttons = rw.button_actions(env)
    assert buttons['decline']['legal'], 'nothing done yet this turn'
    assert buttons['pass']['legal'] and buttons['pass']['label'] == 'End conquests'
    env.step(region_action(_index(10)))
    assert not rw.button_actions(env)['decline']['legal'], 'already conquered'

    env = _env(756)
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.DIPLOMAT, hand=0, held={9: 1},
                turns_played=2)
    env.step(A_PASS)
    buttons = rw.button_actions(env)
    assert env.phase == Phase.ALLY and buttons['pass']['label'] == 'No ally'
    assert not buttons['ally'][0]['legal'], 'cannot ally with yourself'
    assert buttons['ally'][1]['legal'] and buttons['ally'][2]['legal']
    assert env.players[1].name in buttons['ally'][1]['label']
    assert buttons['ally'][1]['action'] == A_ALLY + 1

    env = _env(757)
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.STOUT, hand=0, held={9: 1},
                turns_played=2)
    env.step(A_PASS)
    buttons = rw.button_actions(env)
    assert buttons['decline']['legal'] and buttons['pass']['legal']
    assert buttons['pass']['label'] == 'Stay active'


def test_status_entries_and_the_dice():
    """The status bar carries the turn, the phase, the die and the hand."""
    env = _env(758)
    rows = dict(rw.status_entries(env, pov_player=0))
    assert rows['Turn'] == '1/10'
    assert rows['Phase'] == rw.phase_label(Phase.PICK_COMBO)
    assert rows['To move'].endswith('(you)')
    assert 'Die' not in rows and 'Race' not in rows

    env = _env(759)
    _force_dice(env, [2, 0, 0])
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.BERSERK, hand=3, held={9: 1})
    assert env.current_race.die == 2
    rows = dict(rw.status_entries(env, pov_player=1))
    assert rows['Die'] == '2'
    assert rows['Race'] == 'Ratmen / Berserk'
    assert rows['In hand'] == '3 token(s)'
    assert rows['Conquests'] == '0'
    assert not rows['To move'].endswith('(you)')
    assert rw.last_die_in_log(env) == 2


def test_last_die_in_log_reads_a_final_attempt():
    """The failed final attempt leaves its die value in the log."""
    env = _env(760)
    _force_dice(env, [0, 0, 0])
    _setup_turn(env, 0, RaceId.RATMEN, None, hand=1, held={9: 1})
    assert rw.last_die_in_log(env) is None
    env.step(region_action(_index(10)))                     # final attempt
    assert any('die rolls' in line for line in env.event_log)
    assert rw.last_die_in_log(env) == 0
    assert dict(rw.status_entries(env))['Die'] == '0', 'the race keeps its die'
    env._begin_turn(1)                                      # a race with no die
    assert env.current_race is None or env.current_race.die is None
    assert dict(rw.status_entries(env))['Last die'] == '0'


def test_tokens_to_place_per_phase():
    """`tokens_to_place` follows the redeployment and the marker phases."""
    env = _env(761)
    assert rw.tokens_to_place(env) is None, 'PICK_COMBO'
    _setup_turn(env, 0, RaceId.RATMEN, None, hand=4, held={9: 1})
    assert rw.tokens_to_place(env) is None, 'CONQUER'
    env.step(A_PASS)
    assert env.phase == Phase.REDEPLOY and rw.tokens_to_place(env) == 4
    env.step(region_action(_index(9)))
    assert rw.tokens_to_place(env) == 3

    env = _env(762)
    _setup_turn(env, 0, RaceId.AMAZONS, None, hand=10, held={9: 1})
    env.step(A_PASS)
    assert rw.tokens_to_place(env) == 10 - 4, 'the 4 Amazon attack tokens stay'

    env = _env(763)
    _setup_turn(env, 0, RaceId.RATMEN, PowerId.HEROIC, hand=0,
                held={9: 1, 10: 1}, turns_played=2)
    env.step(A_PASS)
    assert env.phase == Phase.HEROES and rw.tokens_to_place(env) == 2
    env.step(region_action(_index(9)))
    assert rw.tokens_to_place(env) == 1


def test_is_human_decision():
    """Clicks are only accepted for the seat the page plays, and never idle."""
    env = _env(764)
    _setup_turn(env, 0, RaceId.RATMEN, None, hand=4, held={9: 1})
    env.pov_player = 0
    assert not rw.is_human_decision(env, None), 'no callback = page building'
    assert rw.is_human_decision(env, lambda a: None)
    env.pov_player = 1
    assert not rw.is_human_decision(env, lambda a: None), 'an AI is playing'
    env.pov_player = -1
    assert rw.is_human_decision(env, lambda a: None), 'god mode drives all seats'
    assert not rw.is_human_decision(env, lambda a: None, pov_player=2)
    env.done = True
    assert not rw.is_human_decision(env, lambda a: None)


# --------------------------------------------------------------------------- #
# Whole-game sweep
# --------------------------------------------------------------------------- #

def test_every_helper_survives_random_games():
    """Random games, every helper called in every phase, nothing raises.

    Also re-checks the `region_targets` / `action_for_click` agreement, the
    coin rule and the anchor → region mapping at every single step.
    """
    phases_seen: set[str] = set()
    modes_seen: set[str] = set()
    kinds_seen: set[str] = set()
    for seed in range(SWEEP_GAMES):
        env = SmallWorldEnv(3)
        env.reset(seed=1000 + seed)
        rng = np.random.default_rng(seed)
        env.pov_player = int(rng.integers(0, env.n_players))
        steps = 0
        while not env.done and steps < 400:
            phases_seen.add(env.phase.name)
            if env.phase == Phase.TURN_PAUSE:
                # nobody to move: the page shows who just played, no target
                status = dict(rw.status_entries(env))
                assert status['Just played'] == env.players[env._turn_seat].name
                assert 'To move' not in status
                assert rw.region_targets(env) == {}
                assert not rw.is_human_decision(env, callback=print, pov_player=0)
                rw.board_overlay_svg(env)
                rw.player_entries(env)
                rw.combo_entries(env)
                rw.button_actions(env)
                rw.tokens_to_place(env)
                env.step(-1)
                steps += 1
                continue
            legal = np.flatnonzero(env.action_masks())
            assert len(legal) > 0, env.phase
            suggested = int(rng.choice(legal))
            modes = rw.available_modes(env) or [rw.MODE_NORMAL]
            for mode in modes:
                modes_seen.add(mode)
                svg = rw.board_overlay_svg(env, suggested_action=suggested,
                                           mode=mode)
                assert svg.startswith('<!-- smallw overlay')
                targets = rw.region_targets(env, mode)
                kinds_seen.update(targets.values())
                for index in range(env.n_regions):
                    action, message = rw.action_for_click(env, index, mode)
                    assert message, (env.phase, index, mode)
                    assert (action is not None) == (index in targets), (
                        f'{env.phase.name} region {index + 1} mode {mode}')
                    if action is not None:
                        assert env.action_masks()[action]
            rw.status_entries(env)
            rw.combo_entries(env, suggested)
            rw.player_entries(env)
            rw.button_actions(env)
            rw.tokens_to_place(env)
            rw.last_die_in_log(env)
            rw.phase_hint(env.phase)
            rw.pass_label(env.phase)
            for seat in range(env.n_players):
                text = rw.coin_text(env, seat, env.pov_player)
                if seat != env.pov_player:
                    assert text.endswith('coin token(s)'), text
            for region in env.board.regions:
                ax, ay = region.static.anchor
                assert rw.nearest_region_index(env, ax, ay) == region.index
            env.step(suggested)
            steps += 1
        assert env.done, 'the game must finish inside the step budget'
        phases_seen.add(env.phase.name)
        rw.board_overlay_svg(env)
        rw.status_entries(env)
        assert len(rw.standings(env)) == env.n_players
    assert {rw.MODE_NORMAL, rw.MODE_ONE, rw.MODE_ALL} <= modes_seen, modes_seen
    assert {rw.TARGET_CONQUER, rw.TARGET_ABANDON, rw.TARGET_REDEPLOY} <= kinds_seen
    for name in ('PICK_COMBO', 'CONQUER', 'REDEPLOY', 'VICTIM_REDEPLOY', 'DONE'):
        assert name in phases_seen, (name, phases_seen)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def run_all() -> bool:
    """Run every `test_*` function of this module; print a summary.

    Returns True if everything passed.
    """
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception:                                   # noqa: BLE001
            failed.append(name)
            print(f'FAIL {name}')
            print(''.join('    ' + line for line in
                          traceback.format_exc().splitlines(keepends=True)))
        else:
            print(f'ok   {name}')
    print(f'\n{__name__}: {len(tests) - len(failed)}/{len(tests)} passed'
          + (f', FAILED: {", ".join(failed)}' if failed else ''))
    return not failed


if __name__ == '__main__':
    import sys
    sys.exit(0 if run_all() else 1)


def test_new_log_lines_only_returns_what_was_not_toasted():
    """Toasts: new lines since the last call, everything after a reset."""
    log = ['a', 'b']
    lines, seen = rw.new_log_lines(log, None)
    assert lines == ['a', 'b']
    assert rw.new_log_lines(log, seen)[0] == [], 'nothing new'
    log += ['c', '', 'd']
    lines, seen = rw.new_log_lines(log, seen)
    assert lines == ['c', 'd'], 'blank lines are not toasted'
    fresh = ['new game']                          # env.reset() replaces the list
    assert rw.new_log_lines(fresh, seen)[0] == ['new game']


def test_toast_messages_caps_a_burst_of_lines():
    """A long burst keeps its newest lines and sums up the older ones."""
    assert rw.toast_messages(['x', 'y'], limit=3) == ['x', 'y']
    lines = [str(i) for i in range(10)]
    messages = rw.toast_messages(lines, limit=4)
    assert messages == ['… 7 earlier event(s), see the game log', '7', '8', '9']
