"""Headless checks of the web renderer's pure helpers."""

import copy
import random
import xml.etree.ElementTree as ET

from ..envs import art, render_web as rw
from ..envs.constants import A_DEST, N_CODES, Phase, Res, STAR
from .helpers import make_env, set_ship


def _parse(svg: str):
    return ET.fromstring(svg)


def _states(n_games=4, every=5, pause=True):
    for g in range(n_games):
        env = make_env(3 + g % 4, seed=90 + g, pause=pause)
        rng = random.Random(g)
        step = 0
        while not env.done:
            if step % every == 0:
                yield env
            env.step(-1 if env.current_player == -1 else rng.choice(env.legal_actions()))
            step += 1
        yield env


def test_board_svg():
    env = make_env(4)
    root = _parse(art.board_svg(env.track))
    ids = [el.get('id') for el in root.iter() if (el.get('id') or '').startswith('jm-space-')]
    assert len(ids) == env.track.n
    assert root.get('width') == str(art.BOARD_W)


def test_small_svgs_parse():
    for code in range(N_CODES):
        _parse(art.card_svg(code, (3, 5)))
    _parse(art.card_back_svg())
    for v in range(1, 7):
        _parse(art.die_svg(v))
    for f in (2, 4, 6, 8, 10, STAR):
        _parse(art.combat_face_svg(f))
    for res in Res:
        _parse(art.hold_svg(res, 3 if res else 0, sixth=True))


def test_overlay_and_buttons_in_random_states():
    seen = set()
    for env in _states():
        env.pov_player = -1
        _parse(f'<svg xmlns="http://www.w3.org/2000/svg">{rw.board_overlay_svg(env, callback=print)}</svg>')
        assert rw.status_text(env)
        buttons = rw.decision_buttons(env)
        assert sorted(b['action'] for b in buttons) == sorted(env.legal_actions())
        assert all(b['label'] for b in buttons)
        if env.pending is not None and env.pending.kind in (Phase.MOVE_DEST, Phase.RETREAT_DEST):
            seen.add(env.pending.kind)
            for k, (node, _) in enumerate(env.pending.data):
                assert rw.action_for_board_click(env, node) == A_DEST + k
            other = next(n for n in range(env.track.n) if n not in [p[0] for p in env.pending.data])
            assert rw.action_for_board_click(env, other) is None
        if env.done:
            rows = rw.standings(env)
            assert rows[0]['seat'] == env.winner_player
    assert Phase.MOVE_DEST in seen


def test_click_geometry():
    env = make_env(4)
    for node in range(env.track.n):
        x, y = rw.board_xy(env, node)
        assert rw.nearest_space(env, x, y) == node
    assert rw.nearest_space(env, -100, -100) is None
    slots = rw.ship_slots(env)
    assert sorted(slots) == [0, 1, 2, 3]
    assert len(set(slots.values())) == 4           # fanned out at Port Royal


def test_treasure_visibility():
    env = make_env(4)
    set_ship(env, 1, hidden=[4, 10])               # +3 and -3, known to seat 1 only
    env.public_cursed[10] = True
    assert rw.visible_treasures(env, 1, 1) == ['+3', '-3']
    assert rw.visible_treasures(env, 0, 1) == ['?', 'cursed ?']
    assert rw.visible_treasures(env, -1, 1) == ['+3', '-3']


def test_human_decision_gate():
    env = make_env(4)
    env.pov_player = env.current_player
    assert rw.is_human_decision(env, callback=print)
    assert not rw.is_human_decision(env, callback=None)
    env.pov_player = (env.current_player + 1) % 4
    assert not rw.is_human_decision(env, callback=print)
    env.pov_player = -1
    assert rw.is_human_decision(env, callback=print)


def test_buttons_do_not_leak():
    checked = 0
    for env in _states(n_games=3, every=9, pause=False):
        if env.done or env.current_player < 0:
            continue
        pov = env.current_player
        clone = copy.deepcopy(env)
        clone.redeterminize(pov)
        assert rw.decision_buttons(env) == rw.decision_buttons(clone)
        checked += 1
    assert checked > 10


def test_toast_lines():
    env = make_env(4, seed=7)
    rng = random.Random(0)
    seen = None
    toasted = {p: [] for p in (-1, 0, 1)}
    marks = {p: None for p in toasted}
    while not env.done:
        env.step(-1 if env.current_player == -1 else rng.choice(env.legal_actions()))
        for p in toasted:
            lines, marks[p] = rw.new_log_lines(env.event_log, p, marks[p])
            toasted[p] += lines
    for p in toasted:   # every visible line toasted exactly once, in order
        assert toasted[p] == [t for _, t in env.log_lines(p) if t.strip()]
    assert rw.new_log_lines(env.event_log, -1, marks[-1])[0] == []
    lines, seen = rw.new_log_lines(env.event_log, -1, marks[-1])
    env.reset(seed=8)   # a reset replaces the log: everything is new again
    assert rw.new_log_lines(env.event_log, -1, seen)[0] == [t for _, t in env.log_lines(-1) if t.strip()]
    msgs = rw.toast_messages([str(i) for i in range(20)], limit=8)
    assert len(msgs) == 8 and msgs[0].startswith('… 13 earlier') and msgs[-1] == '19'


def test_active_seat():
    for env in _states(n_games=3, every=3):
        seat = rw.active_seat(env)
        if env.done:
            assert seat == -1
        elif env.current_player >= 0:
            assert seat == env.current_player
        else:   # waiting for "next": the ship that just played its turn
            assert 0 <= seat < env.n_players
            assert seat == env.resolving or env.resolving < 0


def test_move_animation():
    env = make_env(4, seed=11, pause=True)
    rng = random.Random(3)
    seen, total, animated, battles = None, 0, 0, 0
    while not env.done:
        env.step(-1 if env.current_player == -1 else rng.choice(env.legal_actions()))
        moves, seen = rw.new_moves(env.move_log, seen)
        total += len(moves)
        for m in moves:   # every route is a chain of single steps from src to dst
            route = env.track.path(m.src, m.dst, m.d, m.steps)
            assert route[0] == m.src and route[-1] == m.dst
            for a, b in zip(route, route[1:]):
                assert b in env.track.step1(*a, m.d)
            if m.steps is not None and m.dst != (env.track.pr, 1):
                assert len(route) - 1 == m.steps
        if not moves:
            continue
        frames = rw.move_keyframes(env, moves, 2.0)
        assert set(frames) == {m.seat for m in moves}
        for out in frames.values():
            times = [t for t, _, _ in out]
            assert times[0] == 0 and times[-1] == 2.0 * len(moves)
            assert all(a < b for a, b in zip(times, times[1:]))   # no duplicate keyframe
            assert out[-1][1:] == (0.0, 0.0)          # ends where the ship is drawn
        for m in moves:
            first = frames[m.seat][0]
            cx, cy = rw.board_xy(env, m.src[0])
            if m == next(x for x in moves if x.seat == m.seat):
                fx, fy = rw.ship_slots(env)[m.seat]
                assert (round(fx + first[1], 1), round(fy + first[2], 1)) == \
                    (round(cx + rw.SHIP_OFFSET[0], 1), round(cy + rw.SHIP_OFFSET[1], 1))
        svg = rw.board_overlay_svg(env, moves=moves, tag='7')
        root = _parse(f'<svg xmlns="http://www.w3.org/2000/svg">{svg}</svg>')
        groups = [g for g in root.iter('{http://www.w3.org/2000/svg}g') if g.get('class') == 'jm-moving']
        assert len(groups) == len(frames)
        assert all(f'@keyframes jm-mv-7-{seat}' in svg for seat in frames)
        assert 'class="jm-moving"' not in rw.board_overlay_svg(env, moves=moves, duration=0)
        smokes = [g for g in root.iter('{http://www.w3.org/2000/svg}g') if g.get('class') == 'jm-smoke']
        assert len(smokes) == sum(m.battle for m in moves)
        assert 'class="jm-smoke"' not in rw.board_overlay_svg(env, moves=moves, flash=0)
        battles += sum(m.battle for m in moves)
        animated += 1
    assert total == len(env.move_log) > 20 and animated > 10 and battles > 0
    assert rw.new_moves(env.move_log, seen)[0] == []
    assert 'jm-moving' not in rw.board_overlay_svg(env)


def test_battle_flash_timing():
    from ..envs.jamaica import Battle, Move
    moves = [Move(0, (0, 0), (3, 0), 1, 3), Move(1, (0, 0), (5, 0), 1, 5, battle=True),
             Move(1, (5, 0), (7, 0), 1, 2, battle=True)]
    assert rw.battle_arrivals(moves, 2.0) == [(5, 4.0), (7, 6.0)]
    assert rw.battle_flash_delay(moves, Battle(attacker=1, node=5), 2.0) == 4.0
    assert rw.battle_flash_delay(moves, Battle(attacker=1, node=7), 2.0) == 6.0
    assert rw.battle_flash_delay(moves, Battle(attacker=2, node=7), 2.0) == 0.0   # not animated
    assert rw.battle_flash_delay(moves, None) == 0.0
    assert rw.flash_animation('x', 4.0, 2.0) == 'x 0.400s ease-in-out 4.000s 5 both'
