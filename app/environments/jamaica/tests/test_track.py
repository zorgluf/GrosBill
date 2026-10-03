"""Track graph: forks, merges, the start line, retreats, scores."""

import copy
import pickle

from ..envs.constants import FLOOR_SCORE, Kind
from ..envs.track import Track, default_track
from .helpers import mini_track


def test_real_board_shape():
    t = default_track()
    assert t.n == 50
    assert t.kind.count(Kind.PORT) == 11
    assert t.kind.count(Kind.SEA) == 29
    assert t.kind.count(Kind.LAIR) == 9
    assert t.splits == (6, 19, 32)
    assert t.merges == (15, 28, 41)
    assert t.L == 32


def test_forward_through_a_fork():
    t = default_track()
    assert t.destinations((6, 0), 1, 1) == ((13, 0), (7, 0))
    assert t.destinations((6, 0), 1, 3) == ((15, 0), (9, 0))
    # both lanes of the top-right fork, 6 spaces from the split
    assert set(t.destinations((32, 0), 1, 6)) == {(38, 0), (44, 0)}


def test_overshoot_stops_at_port_royal():
    t = default_track()
    assert t.destinations((47, 0), 1, 6) == ((0, 1),)
    assert t.destinations((49, 0), 1, 1) == ((0, 1),)


def test_start_line_both_ways():
    t = default_track()
    assert t.destinations((0, 0), -1, 1) == ((49, -1),)
    assert t.destinations((49, -1), 1, 1) == ((0, 0),)     # back on the start line: no finish
    assert t.destinations((49, -1), 1, 2) == ((1, 0),)
    assert t.destinations((1, 0), -1, 1) == ((0, 0),)


def test_backward_through_a_merge():
    t = default_track()
    assert set(t.destinations((16, 0), -1, 2)) == {(12, 0), (14, 0)}


def test_retreat_targets_per_branch():
    t = default_track()
    free = lambda n: t.kind[n] in (Kind.LAIR, Kind.PORT_ROYAL)
    assert t.retreat_targets((17, 0), free) == ((11, 0), (4, 0))
    assert t.retreat_targets((2, 0), free) == ((0, 0),)      # PR on lap 0 is free: no crossing
    assert t.retreat_targets((48, -1), free) == ((44, -1),)


def test_remaining_and_scores():
    t = default_track()
    assert t.remaining(0, 0) == 32
    assert t.remaining(0, 1) == 0
    assert t.remaining(49, -1) == 33
    assert t.remaining(49, 0) == 1
    assert t.pos_score(0, 1) == 15
    assert t.pos_score(0, 0) == FLOOR_SCORE
    assert t.pos_score(49, -1) == FLOOR_SCORE
    assert [t.pos_score(i, 0) for i in range(41, 50)] == list(range(2, 11))
    assert all(t.pos_score(i, 0) == FLOOR_SCORE for i in range(1, 41))
    assert t.in_floor_zone(40, 0) and not t.in_floor_zone(41, 0) and not t.in_floor_zone(0, 1)


def test_mini_track():
    t = mini_track()
    assert t.splits == (3,) and t.merges == (8,)
    assert t.L == 9
    assert t.destinations((3, 0), 1, 2) == ((8, 0), (5, 0))


def test_validation_errors():
    good = mini_track()
    spaces = [(i, good.kind[i], good.cost[i], good.score[i], 0, 0) for i in range(good.n)]
    edges = [(a, b) for a in range(good.n) for b in good.succ[a]]
    for bad_edges in (edges + [(5, 4)],                 # forward cycle avoiding PR
                      [e for e in edges if e != (6, 8)]):   # dead end
        try:
            Track(spaces, bad_edges, 0)
        except ValueError:
            continue
        raise AssertionError(f'accepted {bad_edges}')


def test_shared_and_picklable():
    t = default_track()
    assert copy.deepcopy(t) is t
    t2 = pickle.loads(pickle.dumps(t))
    assert t2.succ == t.succ and t2.dist == t.dist
