"""What a player must not see: the other rack, the reserves, the tiles in the box."""

import copy
import random

import numpy as np

from ..envs.constants import A_DRAW, BLUE, RED, Troop
from ..envs.terrains import TERRAIN_KEYS
from .helpers import blank, give, make_env, place, put, tiles


def _same(a, b) -> bool:
    return all(np.array_equal(a[k], b[k]) for k in a)


def test_observation_is_invariant_under_redeterminization():
    random.seed(11)
    changed = 0
    for g, key in enumerate(TERRAIN_KEYS * 2):
        env = make_env(key, seed=50 + g)
        rng = random.Random(g)
        step = 0
        while not env.done:
            if step % 4 == 0:
                for pov in (BLUE, RED):
                    before = env.observe(pov)
                    clone = copy.deepcopy(env)
                    clone.redeterminize(pov)
                    assert _same(before, clone.observe(pov)), (key, step, pov)
                    opp = 1 - pov
                    assert clone.rack_size(opp) == env.rack_size(opp)
                    assert len(clone.reserve[opp]) == len(env.reserve[opp])
                    assert sum(clone.blocked[opp]) == sum(env.blocked[opp])
                    assert all(b <= r for b, r in zip(clone.blocked[opp], clone.racks[opp]))
                    assert clone.racks[pov] == env.racks[pov] and clone.blocked[pov] == env.blocked[pov]
                    assert tiles(clone, BLUE) == 24 and tiles(clone, RED) == 24
                    changed += clone.racks[opp] != env.racks[opp] or clone.reserve[pov] != env.reserve[pov]
                    assert clone.stacks == env.stacks and clone.discard == env.discard
            env.step(rng.choice(env.legal_actions()))
            step += 1
    assert changed > 50                         # it does shuffle something


def test_redeterminized_games_stay_playable():
    random.seed(3)
    for g, key in enumerate(TERRAIN_KEYS):
        env = make_env(key, seed=g)
        rng = random.Random(g)
        while not env.done:
            env.redeterminize(env.current_player)
            assert env.action_masks().any()
            env.step(rng.choice(env.legal_actions()))
            assert tiles(env, BLUE) == 24 and tiles(env, RED) == 24


def test_the_other_rack_is_only_a_count():
    a = blank('caribbean')
    b = blank('caribbean')
    give(a, RED, Troop.ROXY, Troop.ROXY)
    give(b, RED, Troop.SKULLY, Troop.KWAK)
    give(a, BLUE, Troop.HOOK)
    give(b, BLUE, Troop.HOOK)
    assert _same(a.observe(BLUE), b.observe(BLUE))
    assert not _same(a.observe(RED), b.observe(RED))


def test_pinned_troop_is_known_to_its_owner_only():
    env = blank('battlefield')
    give(env, BLUE, Troop.ROXY)
    give(env, RED, Troop.SKULLY, Troop.HOOK)
    put(env, 'd1', BLUE, Troop.SKULLY)
    put(env, 'c1', BLUE, Troop.XB42)
    place(env, Troop.ROXY, 'sb')
    pinned = env.blocked[RED].index(1)
    other = copy.deepcopy(env)
    other.blocked[RED] = [0] * 8
    other.blocked[RED][Troop.SKULLY if pinned == Troop.HOOK else Troop.HOOK] = 1
    assert _same(env.observe(BLUE), other.observe(BLUE))
    assert not _same(env.observe(RED), other.observe(RED))
    public = [t for _, t in env.log_lines(BLUE)]
    private = [t for _, t in env.log_lines(RED)]
    name = ('Skully', 'Hook')[pinned == Troop.HOOK]
    assert not any(name in t for t in public) and any(name in t and 'pinned' in t for t in private)


def test_draws_are_logged_face_down():
    env = blank('castle')
    env.reserve[BLUE][-2:] = [Troop.JUMBO, Troop.HOOK]
    env.step(A_DRAW)
    assert [t for _, t in env.log_lines(RED)][-1].endswith('draws 2 troops.')
    for pov in (BLUE, -1):
        assert 'Jumbo' in env.log_lines(pov)[-1][1] and 'Hook' in env.log_lines(pov)[-1][1]


def test_mask_is_only_for_the_player_to_move():
    env = make_env(seed=2)
    p = env.current_player
    assert env.observe(p)['mask'].any() and not env.observe(1 - p)['mask'].any()
