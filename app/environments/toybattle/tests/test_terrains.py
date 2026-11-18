"""The transcription of the 8 terrains."""

from ..envs.constants import BLUE, Effect, MAX_BASES, MAX_HQ, MAX_REGIONS, RED
from ..envs.terrains import H, TERRAIN_KEYS, TERRAINS, W, faces, medal_spot

#: key: (bases, special bases, paths, regions, medals, objective, blue H.Q., red H.Q.)
EXPECTED = {
    'castle':      (15, 4, 24, 8, 14, 7, 1, 1),
    'pool':        (13, 5, 24, 8, 12, 6, 2, 2),
    'clouds':      (14, 4, 28, 13, 16, 8, 1, 1),
    'jungle':      (13, 2, 22, 6, 14, 7, 1, 1),
    'cemetery':    (15, 4, 28, 10, 14, 7, 1, 1),
    'caribbean':   (12, 0, 20, 6, 11, 5, 2, 1),
    'station':     (13, 3, 28, 14, 14, 7, 1, 1),
    'battlefield': (14, 4, 24, 9, 16, 8, 1, 1),
}


def _xy(T):
    xy = {b.name: (b.x, b.y) for b in T.bases}
    xy.update({h.name: (h.x, h.y) for side in T.hqs for h in side})
    return xy


def test_the_eight_terrains():
    assert TERRAIN_KEYS == tuple(EXPECTED)
    for key, T in TERRAINS.items():
        got = (T.n_bases, sum(b.special for b in T.bases), len(T.edges), len(T.regions),
               T.n_medals, T.objective, len(T.hqs[BLUE]), len(T.hqs[RED]))
        assert got == EXPECTED[key], (key, got)
        assert T.index == TERRAIN_KEYS.index(key)


def test_sizes_fit_the_spaces():
    assert max(T.n_bases for T in TERRAINS.values()) == MAX_BASES
    assert max(len(T.regions) for T in TERRAINS.values()) == MAX_REGIONS
    assert max(len(side) for T in TERRAINS.values() for side in T.hqs) == MAX_HQ


def test_objective_is_half_the_medals():
    for key, T in TERRAINS.items():
        assert T.objective == T.n_medals // 2, key
        assert all(1 <= r.medals <= 3 for r in T.regions), key


def test_regions_are_faces_of_the_board():
    for key, T in TERRAINS.items():
        xy = _xy(T)
        found = faces(xy, T.edges)
        assert len(found) == len(T.edges) - len(xy) + 1, key          # Euler: connected and planar
        names = {b.name for b in T.bases}
        face_sets = [frozenset(n for n in cycle if n in names) for cycle in found]
        for region in T.regions:
            members = frozenset(T.bases[b].name for b in region.bases)
            assert face_sets.count(members) == 1, (key, sorted(members))
            # every base around a region is on a path with another one of it
            for b in region.bases:
                assert any(n in region.bases for n in T.adj[b]) or len(region.bases) == 1, key
        assert len(set(frozenset(r.bases) for r in T.regions)) == len(T.regions), key


def test_medal_spots_are_clear_of_the_bases():
    for key, T in TERRAINS.items():
        for region in T.regions:
            x, y = medal_spot(region)
            assert 0 < x < W and 0 < y < H
            assert len(region.outline) >= len(region.bases)
            for b in T.bases:
                assert abs(b.x - x) > 5 or abs(b.y - y) > 5, (key, b.name)


def test_nodes_are_apart_and_inside_the_frame():
    for key, T in TERRAINS.items():
        pts = list(_xy(T).items())
        for i, (a, (ax, ay)) in enumerate(pts):
            assert 8 <= ax <= W - 8 and 8 <= ay <= H - 8, (key, a)
            for b, (bx, by) in pts[i + 1:]:
                assert (ax - bx) ** 2 + (ay - by) ** 2 >= 20 ** 2, (key, a, b)


def test_adjacency_is_symmetric_and_connected():
    for key, T in TERRAINS.items():
        for b, around in enumerate(T.adj):
            assert b not in around and len(set(around)) == len(around)
            assert all(b in T.adj[n] for n in around), key
        seen, todo = set(), [0]
        while todo:
            b = todo.pop()
            if b not in seen:
                seen.add(b)
                todo.extend(T.adj[b])
        assert len(seen) == T.n_bases, key
        for side in T.hqs:
            for hq in side:
                assert hq.bases, (key, hq.name)


def test_blue_is_at_the_bottom():
    for key, T in TERRAINS.items():
        assert all(hq.y > H / 2 and hq.owner == BLUE for hq in T.hqs[BLUE]), key
        assert all(hq.y < H / 2 and hq.owner == RED for hq in T.hqs[RED]), key


def test_half_turn_views():
    for key, T in TERRAINS.items():
        blue, red = T.views
        assert blue == tuple(range(T.n_bases))
        assert sorted(red) == list(blue)
        for p in (BLUE, RED):
            assert all(T.views[p][T.slots[p][b]] == b for b in range(T.n_bases))
        assert T.symmetric == (key != 'caribbean')
        if not T.symmetric:
            assert red == blue and T.region_views[RED] == T.region_views[BLUE]
            continue
        for b in range(T.n_bases):
            twin = red[b]
            assert red[twin] == b and twin != b or (T.bases[b].x, T.bases[b].y) == (W / 2, H / 2)
            assert T.bases[b].special == T.bases[twin].special and T.bases[b].gate == T.bases[twin].gate
            assert sorted(red[n] for n in T.adj[b]) == list(T.adj[twin])
        # both colours have their own H.Q. next to the same slots
        for k, (mine, theirs) in enumerate(zip(T.hqs[BLUE], T.hqs[RED])):
            assert sorted(T.slots[RED][b] for b in theirs.bases) == sorted(mine.bases), (key, k)
            assert mine.gate == theirs.gate
        for k, r in enumerate(T.region_views[RED]):
            assert sorted(red[b] for b in T.regions[k].bases) == list(T.regions[r].bases)
            assert T.regions[k].medals == T.regions[r].medals


def test_special_rules_of_each_terrain():
    effects = {key: T.effect for key, T in TERRAINS.items()}
    assert effects == {'castle': Effect.RECALL, 'pool': Effect.GATE, 'clouds': Effect.DRAW,
                       'jungle': Effect.ERUPT, 'cemetery': Effect.EXHUME, 'caribbean': Effect.NONE,
                       'station': Effect.SHIELD, 'battlefield': Effect.SNIPER}
    pool = TERRAINS['pool']
    gates = sorted(b.gate for b in pool.bases if b.special)
    assert gates == [(1, 2), (1, 2), (3, 4, 5), (3, 4, 5), (6, 7)]
    assert all(b.gate is None for b in pool.bases if not b.special)
    # one gated and one open H.Q. per colour
    for side in pool.hqs:
        assert sorted(str(hq.gate) for hq in side) == ['(6, 7)', 'None']
    for key, T in TERRAINS.items():
        if key != 'pool':
            assert all(b.gate is None for b in T.bases) and all(h.gate is None for s in T.hqs for h in s)
