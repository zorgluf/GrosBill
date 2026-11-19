"""The helper columns of the observation (the layout is in `constants.py`)."""

import numpy as np

from ..envs.constants import (B_MY_DIST, B_MY_GAIN, B_QUEUED, B_THEIR_DIST, B_THEIR_GAIN, BASE_F, BLUE,
                              G_CAN_DRAW, G_PHASE, G_RACK, G_SCALARS, GLOB_F, HQ_F, MAX_DIST, Phase,
                              RED, REGION_F, Troop)
from ..envs.terrains import TERRAINS
from .helpers import blank, give, make_env, place, put, slot


def test_shapes_follow_the_layout():
    obs = make_env('pool', seed=1).observation
    assert {k: v.shape for k, v in obs.items()} == {
        'glob': (GLOB_F,), 'bases': (15, BASE_F), 'adj': (15, 15), 'regions': (14, REGION_F),
        'hqs': (4, HQ_F), 'mask': (161,)}
    assert (GLOB_F, BASE_F, REGION_F, HQ_F) == (73, 41, 22, 25)
    assert all(v.dtype == np.float32 for v in obs.values())


def test_medal_gain_columns():
    env = blank('castle')
    for name, troop in zip(('d1', 'c1', 'm1', 'm2'), (Troop.SKULLY, Troop.SKULLY, Troop.CAPN, Troop.CAPN)):
        put(env, name, BLUE, troop)
    give(env, BLUE, Troop.ROXY)
    bases = env.observe(BLUE)['bases']
    b1, s3 = slot(env, 'b1'), slot(env, 's3')
    assert abs(bases[b1, B_MY_GAIN] - 2 / 7) < 1e-6           # b1 m1 c1 m2 is one base short
    assert abs(bases[s3, B_MY_GAIN] - 1 / 7) < 1e-6           # s3 c1 m1 too
    assert abs(bases[:, B_MY_GAIN].sum() - 3 / 7) < 1e-6 and not bases[:, B_THEIR_GAIN].any()
    # the opponent sees the same thing as a threat, in its own view
    theirs = env.observe(RED)['bases']
    red_b1 = env.terrain.slots[RED][env.terrain.slots[BLUE].index(b1)]
    assert abs(theirs[red_b1, B_THEIR_GAIN] - 2 / 7) < 1e-6 and not theirs[:, B_MY_GAIN].any()
    place(env, Troop.ROXY, 'b1')                              # taken: nothing left to gain there
    assert env.medals == [2, 0] and env.observe(BLUE)['bases'][b1, B_MY_GAIN] == 0


def test_gain_counts_a_base_held_by_the_enemy():
    env = blank('castle')
    for name in ('d1', 'c1', 'm1', 'm2'):
        put(env, name, BLUE, Troop.SKULLY if name in ('d1', 'c1') else Troop.CAPN)
    put(env, 'b1', RED, Troop.HOOK)                           # Red sits on the missing base
    bases = env.observe(BLUE)['bases']
    assert abs(bases[slot(env, 'b1'), B_MY_GAIN] - 2 / 7) < 1e-6


def test_queued_special_base():
    env = blank('castle')
    put(env, 'd1', BLUE, Troop.SKULLY)
    put(env, 'c1', BLUE, Troop.HOOK)
    give(env, BLUE, Troop.CAPN, Troop.ROXY)
    assert not env.observation['bases'][:, B_QUEUED].any()
    place(env, Troop.CAPN, 's3')                              # the recall waits for the extra troop
    obs = env.observation
    assert env.phase == Phase.EXTRA and obs['glob'][G_PHASE + Phase.EXTRA] == 1
    assert obs['bases'][slot(env, 's3'), B_QUEUED] == 1 and obs['bases'][:, B_QUEUED].sum() == 1
    place(env, Troop.ROXY, 'm1')
    assert env.phase == Phase.RECALL and not env.observation['bases'][:, B_QUEUED].any()


def test_distances_to_the_hq():
    for key, T in TERRAINS.items():
        env = make_env(key)
        for me in (BLUE, RED):
            bases = env.observe(me)['bases']
            for b in range(T.n_bases):
                row = bases[T.slots[me][b]]
                assert abs(row[B_MY_DIST] * MAX_DIST - T.hq_dist[me][b]) < 1e-5
                assert abs(row[B_THEIR_DIST] * MAX_DIST - T.hq_dist[1 - me][b]) < 1e-5
            near = [b for hq in T.hqs[me] for b in hq.bases]
            assert all(T.hq_dist[me][b] == 1 for b in near) and min(T.hq_dist[me]) == 1
        # one path more per step away
        for b in range(T.n_bases):
            for n in T.adj[b]:
                assert abs(T.hq_dist[BLUE][b] - T.hq_dist[BLUE][n]) <= 1


def test_glob_slices():
    env = blank('caribbean', current=RED)
    give(env, RED, Troop.ROXY, Troop.ROXY, Troop.KWAK)
    g = env.observation['glob']
    assert g[G_SCALARS] == 0 and env.observe(BLUE)['glob'][G_SCALARS] == 1        # the colour bit
    assert abs(g[G_RACK + Troop.ROXY] - 2 / 3) < 1e-6 and abs(g[G_RACK + Troop.KWAK] - 1 / 3) < 1e-6
    assert g[G_CAN_DRAW] == 1 and g[G_PHASE + Phase.MAIN] == 1
