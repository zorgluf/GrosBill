"""Headless checks of the drawings and of the web renderer's pure helpers."""

import random
import xml.etree.ElementTree as ET

from ..envs import art, render_web as rw
from ..envs.constants import (A_DRAW, A_PLACE, A_SKIP, A_TARGET, A_TYPE, BLUE, N_TYPES, Phase, RED,
                              Troop, place_action)
from ..envs.terrains import TERRAIN_KEYS, TERRAINS
from .helpers import blank, give, make_env, put, random_play, slot


def _parse(svg: str):
    return ET.fromstring(svg)


def _overlay(env, **kwargs):
    return _parse(f'<svg xmlns="http://www.w3.org/2000/svg">{rw.overlay_svg(env, **kwargs)}</svg>')


def test_board_svgs():
    for key, T in TERRAINS.items():
        for flip in (False, True):
            root = _parse(art.board_svg(T, flip))
            ids = [el.get('id') for el in root.iter() if (el.get('id') or '').startswith('tb-')]
            assert len([i for i in ids if i.startswith('tb-base-')]) == T.n_bases, key
            assert len([i for i in ids if i.startswith('tb-hq-')]) == sum(len(s) for s in T.hqs), key
            assert root.get('width') == str(art.BOARD_W) and root.get('height') == str(art.BOARD_H)
        assert key in art.THEMES
    assert set(art.THEMES) == set(TERRAIN_KEYS)


def test_no_published_art_is_embedded():
    for T in TERRAINS.values():
        svg = art.board_svg(T)
        assert '<image' not in svg and 'href' not in svg and len(svg) < 30000


def test_small_svgs_parse():
    for owner in (BLUE, RED):
        for troop in range(N_TYPES):
            _parse(art.troop_svg(owner, troop))
            _parse(art.troop_svg(owner, troop, 22, dim=True))
            assert art.troop_title(troop)
        _parse(art.back_svg(owner))
    _parse(art.medal_svg())


def test_the_viewer_sits_at_the_bottom():
    env = make_env('castle')
    for pov, flip in ((BLUE, False), (RED, True), (-1, False)):
        env.pov_player = pov
        assert rw.bottom_seat(env) == max(pov, 0)
        ys = {node: y for node, _, y in rw.nodes(env, rw.bottom_seat(env) == RED)}
        mine = ys[('hq', rw.bottom_seat(env), 0)]
        assert mine > art.BOARD_H / 2 > ys[('hq', 1 - rw.bottom_seat(env), 0)], (pov, flip)


def test_tap_geometry():
    for key in TERRAIN_KEYS:
        env = make_env(key)
        for flip in (False, True):
            found = rw.nodes(env, flip)
            assert len(found) == env.terrain.n_bases + sum(len(s) for s in env.terrain.hqs)
            for node, x, y in found:
                assert rw.node_at(env, x, y, flip) == node
                assert rw.node_at(env, x + 14, y - 14, flip) == node
        assert rw.node_at(env, -80, -80, False) is None


def test_taps_are_the_legal_actions():
    """Tapping a troop then a node, a target, or a button reaches every legal action and nothing else."""
    phases = set()
    for g, key in enumerate(TERRAIN_KEYS * 3):
        env = make_env(key, seed=700 + g)
        rng = random.Random(g)
        while not env.done:
            env.pov_player = env.current_player
            phases.add(env.phase)
            reachable = {b['action'] for b in rw.buttons(env)}
            troops = rw.selectable_troops(env) if env.phase in (Phase.MAIN, Phase.EXTRA) else [None]
            for troop in troops:
                for node in rw.legal_nodes(env, troop):
                    reachable.add(rw.action_for_click(env, node, troop))
            assert reachable == set(env.legal_actions()), (key, env.phase)
            _overlay(env, selected=troops[0] if troops else None, playable=True,
                     suggested=rng.choice(env.legal_actions()))
            assert rw.status_text(env) and rw.PROMPTS[env.phase]
            env.step(rng.choice(env.legal_actions()))
        env.pov_player = -1
        _overlay(env)
        assert 'wins' in rw.status_text(env) or 'Draw' in rw.status_text(env)
        assert rw.buttons(env) == [] and rw.selectable_troops(env) == []
    assert phases == set(Phase)


def test_tap_needs_a_selected_troop():
    env = blank('castle')
    give(env, BLUE, Troop.ROXY, Troop.HOOK)
    env.pov_player = BLUE
    d1 = ('base', env.terrain.bases.index(next(b for b in env.terrain.bases if b.name == 'd1')))
    assert rw.action_for_click(env, d1, None) is None
    assert rw.action_for_click(env, d1, Troop.ROXY) == place_action(Troop.ROXY, slot(env, 'd1'))
    assert rw.action_for_click(env, ('hq', BLUE, 0), Troop.ROXY) is None        # my own H.Q.
    assert rw.action_for_click(env, ('hq', RED, 0), Troop.ROXY) is None         # out of reach
    assert rw.selectable_troops(env) == [Troop.HOOK, Troop.ROXY]
    assert len(rw.legal_nodes(env, Troop.HOOK)) == 15 and len(rw.legal_nodes(env, Troop.ROXY)) == 2
    assert [b['label'] for b in rw.buttons(env)] == ['Draw 2']


def test_red_taps_go_through_its_own_view():
    env = blank('castle', current=RED)
    give(env, RED, Troop.ROXY)
    env.pov_player = RED
    flip = rw.bottom_seat(env) == RED
    a1 = next(i for i, b in enumerate(env.terrain.bases) if b.name == 'a1')
    x, y = art.pt(env.terrain.bases[a1].x, env.terrain.bases[a1].y, flip)
    assert y > art.BOARD_H / 2                  # Red's corner is drawn at the bottom
    node = rw.node_at(env, x, y, flip)
    assert node == ('base', a1)
    action = rw.action_for_click(env, node, Troop.ROXY)
    env.step(action)
    assert env.stacks[a1] == [(RED, Troop.ROXY)]


def test_suggested_parts():
    env = blank('castle')
    give(env, BLUE, Troop.ROXY)
    assert rw.suggested_parts(env, None) == (None, None)
    assert rw.suggested_parts(env, A_DRAW) == (None, None) == rw.suggested_parts(env, A_SKIP)
    assert rw.suggested_parts(env, A_PLACE + Troop.ROXY * 17 + 13) == (Troop.ROXY, ('base', 13))
    assert rw.suggested_parts(env, A_PLACE + Troop.ROXY * 17 + 15) == (Troop.ROXY, ('hq', RED, 0))
    assert rw.suggested_parts(env, A_TARGET + 4) == (None, ('base', 4))
    assert rw.suggested_parts(env, A_TYPE + Troop.STAR) == (Troop.STAR, None)


def test_overlay_shows_stacks_medals_and_capture():
    env = blank('castle')
    env.pov_player = BLUE
    medals = len(_overlay(env).findall('.//{*}polygon'))
    assert medals == env.terrain.n_medals       # one star per medal
    put(env, 'd1', BLUE, Troop.SKULLY, Troop.ROXY)
    put(env, 'd1', RED, Troop.KWAK)
    texts = [t.text for t in _overlay(env).iter() if t.tag.endswith('text')]
    assert '3' in texts                         # the height of the stack
    assert 'top first: Red Kwak, Blue Roxy, Blue Skully' in rw.stack_text(env, 13)
    random_play(env, random.Random(0))
    _overlay(env)


def test_is_human_decision():
    env = make_env(seed=1)
    cb = lambda a: None                         # noqa: E731
    env.pov_player = env.current_player
    assert rw.is_human_decision(env, cb) and not rw.is_human_decision(env, None)
    env.pov_player = 1 - env.current_player
    assert not rw.is_human_decision(env, cb)
    env.pov_player = -1
    assert rw.is_human_decision(env, cb) and rw.viewer(env) == -1
    random_play(env, random.Random(1))
    assert not rw.is_human_decision(env, cb)
