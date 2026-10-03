"""The transcription of the physical game (board, deck, die, treasures)."""

from collections import Counter

from ..envs import data
from ..envs.constants import (FLOOR_SCORE, Kind, MAX_DEST, Power, STAR, Sym, card_syms)
from ..envs.track import default_track


def test_space_counts_and_costs():
    kinds = Counter(k for _, k, _, _, _, _ in data.SPACES)
    assert kinds == {Kind.PORT_ROYAL: 1, Kind.SEA: 29, Kind.PORT: 11, Kind.LAIR: 9}
    ports = sorted(c for _, k, c, _, _, _ in data.SPACES if k == Kind.PORT)
    assert ports == [3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 7]
    seas = Counter(c for _, k, c, _, _, _ in data.SPACES if k == Kind.SEA)
    assert seas == {1: 3, 2: 9, 3: 12, 4: 5}
    assert all(c == 0 for _, k, c, _, _, _ in data.SPACES if k in (Kind.LAIR, Kind.PORT_ROYAL))


def test_scores():
    scores = {i: s for i, _, _, s, _, _ in data.SPACES}
    assert scores[0] == 15
    assert [scores[i] for i in range(41, 50)] == list(range(2, 11))
    assert all(scores[i] is None for i in range(1, 41))
    t = default_track()
    assert t.pos_score(40, 0) == FLOOR_SCORE and t.pos_score(38, 0) == FLOOR_SCORE


def test_forks():
    t = default_track()
    for split, merge in ((6, 15), (19, 28), (32, 41)):
        outer, inner = sorted(t.succ[split], key=lambda s: -t.dist[s])
        lanes = []
        for start in (outer, inner):
            node, length = start, 0
            while node != merge:
                assert len(t.succ[node]) == 1
                node = t.succ[node][0]
                length += 1
            lanes.append(length)
        assert lanes == [6, 2], (split, lanes)
        lane_nodes = []
        node = outer
        while node != merge:
            lane_nodes.append(node)
            node = t.succ[node][0]
        assert sum(t.kind[x] == Kind.LAIR for x in lane_nodes) == 2


def test_deck():
    assert len(data.DECK) == 11 and len(set(data.DECK)) == 11
    morning = Counter(card_syms(c)[0] for c in data.DECK)
    evening = Counter(card_syms(c)[1] for c in data.DECK)
    assert morning == evening == {Sym.FWD: 4, Sym.FOOD: 2, Sym.POWDER: 2, Sym.GOLD: 2, Sym.BACK: 1}


def test_die_and_treasures():
    assert sorted(f for f in data.COMBAT_FACES if f != STAR) == [2, 4, 6, 8, 10]
    assert data.COMBAT_FACES.count(STAR) == 1
    powers = [p for p, _ in data.TREASURES if p is not None]
    assert sorted(powers) == sorted(Power)
    values = sorted(v for p, v in data.TREASURES if p is None)
    assert values == [-4, -3, -2, 3, 3, 5, 7, 7]


def test_drawing_data():
    w, h = data.BOARD_SIZE
    assert all(0 <= x <= w and 0 <= y <= h for _, _, _, _, x, y in data.SPACES)
    assert data.N_LAIRS == 9 == len(default_track().lairs)


def test_destination_slots_suffice():
    t = default_track()
    free = lambda n: t.kind[n] in (Kind.LAIR, Kind.PORT_ROYAL)
    worst = 0
    for node in range(t.n):
        for lap in (-1, 0):
            if node == t.pr and lap == -1:
                continue
            for d in (1, -1):
                for k in range(1, 7):
                    worst = max(worst, len(t.destinations((node, lap), d, k)))
            worst = max(worst, len(t.retreat_targets((node, lap), free)))
    assert worst <= MAX_DEST, worst
