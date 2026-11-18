"""Random games on every terrain: invariants after each step, returns, the half-turn view."""

import copy
import random

import numpy as np

from ..envs.constants import (A_SKIP, BLUE, N_PHASES, N_TYPES, Phase, RACK_MAX, RED)
from ..envs.terrains import N_TERRAINS, TERRAIN_KEYS
from ..envs.toybattle import ToyBattleAllTerrainsEnv
from .helpers import make_env, random_play, tiles


def _check(env):
    T = env.terrain
    for p in (BLUE, RED):
        assert tiles(env, p) == 24
        assert 0 <= env.rack_size(p) <= RACK_MAX
        assert all(0 <= b <= r for b, r in zip(env.blocked[p], env.racks[p]))
        assert all(c >= 0 for c in env.racks[p] + env.discard[p])
        for t in range(N_TYPES):
            total = (env.racks[p][t] + env.on_board(p)[t] + env.discard[p][t]
                     + env.reserve[p].count(t) + env.removed[p].count(t))
            assert total == 3
    assert sum(env.medals) == sum(r.medals for r, o in zip(T.regions, env.region_owner) if o >= 0)
    for p in (BLUE, RED):
        assert env.medals[p] == sum(r.medals for r, o in zip(T.regions, env.region_owner) if o == p)
    if env.done:
        assert not env.action_masks().any()
        return
    mask = env.action_masks()
    assert mask.any()
    assert sum(env.blocked[1 - env.current_player]) == 0 or T.key == 'battlefield'
    if env.phase != Phase.MAIN:
        assert mask[A_SKIP] or env.phase == Phase.ERUPT_DST
    # a region still holding its medals is not fully held by one player
    for region, owner in zip(T.regions, env.region_owner):
        if owner < 0:
            assert len({env.owner(b) for b in region.bases}) > 1 or env.owner(region.bases[0]) < 0


def test_invariants_and_returns():
    outcomes = set()
    for g, key in enumerate(TERRAIN_KEYS * 12):
        env = make_env(key, seed=1000 + g)
        _check(env)
        totals, steps = random_play(env, random.Random(g), check=_check)
        assert env.done and steps < 400
        w = env.winner_player
        assert w is not None                    # the turn limit is never reached by random play
        assert abs(totals[w] - 1) < 1e-9 and abs(totals[1 - w] + 1) < 1e-9
        outcomes.add((key, w))
    assert len(outcomes) == 2 * len(TERRAIN_KEYS)       # both colours win on every terrain


def test_the_turn_limit_ends_long_games():
    env = make_env('castle', seed=1, max_turns=12)
    totals, _ = random_play(env, random.Random(1))
    assert env.done and env.turns_taken == 12
    assert totals == [0.0, 0.0] if env.winner_player is None else abs(abs(totals[0]) - 1) < 1e-9


def _mirrored(env):
    """The same position seen from the other side of a symmetric terrain: the
    board turned by a half-turn and the two players swapped."""
    T = env.terrain
    turn = T.views[RED]
    m = copy.deepcopy(env)
    m.stacks = [[(1 - o, t) for o, t in env.stacks[turn[b]]] for b in range(T.n_bases)]
    for name in ('racks', 'blocked', 'discard', 'reserve', 'removed', 'medals'):
        setattr(m, name, copy.deepcopy(getattr(env, name))[::-1])
    regions = T.region_views[RED]
    m.region_owner = [o if o < 0 else 1 - o for o in (env.region_owner[regions[r]] for r in range(len(regions)))]
    m.current_player, m.first_player = 1 - env.current_player, 1 - env.first_player
    m.pending_base = None if env.pending_base is None else turn[env.pending_base]
    m.erupt_src = None if env.erupt_src is None else turn[env.erupt_src]
    m.base_queue = [turn[b] for b in env.base_queue]
    return m


def test_both_colours_see_the_same_game():
    """On a symmetric terrain the observation and the mask are the same for
    Blue and for Red in the mirrored position (but for the colour bit)."""
    colour_bit = N_TERRAINS + N_PHASES
    compared = 0
    for g, key in enumerate(TERRAIN_KEYS * 2):
        env = make_env(key, seed=300 + g)
        if not env.terrain.symmetric:
            continue
        rng = random.Random(g)
        while not env.done:
            p = env.current_player
            mirror = _mirrored(env)
            a, b = env.observe(p), mirror.observe(1 - p)
            assert a['glob'][colour_bit] != b['glob'][colour_bit]
            b['glob'][colour_bit] = a['glob'][colour_bit]
            for name in a:
                assert np.array_equal(a[name], b[name]), (key, name)
            compared += 1
            env.step(rng.choice(env.legal_actions()))
        assert env.done
    assert compared > 300


def test_all_terrains_env_fuzz():
    env = ToyBattleAllTerrainsEnv()
    seen = set()
    for seed in range(40):
        env.reset(seed=seed)
        seen.add(env.terrain_key)
        random_play(env, random.Random(seed), check=_check)
        assert env.done
    assert len(seen) == len(TERRAIN_KEYS)
