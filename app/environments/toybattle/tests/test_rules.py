"""Setup, the two actions of a turn and the placement rules (Castle Field unless said)."""

from ..envs.constants import A_DRAW, A_SKIP, BLUE, FIRST_HAND, Phase, RED, Troop
from .helpers import (base, blank, can_place, capture, give, legal_bases, make_env, place, put, tiles, top)

K, SK, CA, JU, HO, XB, ST, RO = Troop.KWAK, Troop.SKULLY, Troop.CAPN, Troop.JUMBO, Troop.HOOK, \
    Troop.XB42, Troop.STAR, Troop.ROXY


def test_setup():
    firsts = set()
    for seed in range(20):
        env = make_env(seed=seed)
        first = env.first_player
        firsts.add(first)
        assert env.current_player == first and env.phase == Phase.MAIN
        assert (env.rack_size(first), env.rack_size(1 - first)) == FIRST_HAND
        assert (len(env.reserve[first]), len(env.reserve[1 - first])) == (17, 16)
        for p in range(2):
            assert len(env.removed[p]) == 4 and tiles(env, p) == 24
            everything = env.reserve[p] + env.removed[p] + [t for t in range(8) for _ in range(env.racks[p][t])]
            assert sorted(everything) == sorted(list(range(8)) * 3)
        assert env.medals == [0, 0] and not any(env.stacks) and not env.done
    assert firsts == {BLUE, RED}


def test_first_troop_goes_next_to_the_hq():
    env = blank()
    give(env, BLUE, RO)
    assert legal_bases(env, RO) == {'d1', 'd2'}
    env = blank(current=RED)
    give(env, RED, RO)
    assert legal_bases(env, RO) == {'a1', 'a2'}


def test_connection_goes_through_own_bases():
    env = blank()
    give(env, BLUE, RO)
    put(env, 'd1', BLUE, SK)
    assert legal_bases(env, RO) == {'d1', 'd2', 'c1'}
    put(env, 'c1', BLUE, SK)
    assert legal_bases(env, RO) == {'d1', 'd2', 'c1', 's3', 'm1', 'm2'}


def test_enemy_or_empty_base_cuts_the_connection():
    env = blank()
    give(env, BLUE, RO)
    put(env, 'd1', BLUE, SK)
    put(env, 'c1', BLUE, SK)
    put(env, 'm1', BLUE, SK)                    # d1 - c1 - m1, all connected
    assert 'b1' in legal_bases(env, RO)
    put(env, 'c1', RED, HO)                     # Red covers c1: m1 is cut off
    assert env.reach(BLUE) == {base(env, 'd1')}
    assert legal_bases(env, RO) == {'d1', 'd2', 'c1'}
    # an own base that is not connected is not a slot either
    assert not can_place(env, RO, 'm1')


def test_covering_needs_a_strictly_higher_strength():
    env = blank()
    put(env, 'd1', RED, HO)
    give(env, BLUE, JU, HO, XB, K)
    assert not can_place(env, JU, 'd1')         # 3 < 4
    assert not can_place(env, HO, 'd1')         # 4 = 4
    assert can_place(env, XB, 'd1')             # 5 > 4
    assert can_place(env, K, 'd1')              # the joker covers anything
    place(env, XB, 'd1')
    assert top(env, 'd1') == (BLUE, XB) and len(env.stacks[base(env, 'd1')]) == 2


def test_own_troops_may_always_be_covered():
    env = blank()
    put(env, 'd1', BLUE, RO)
    give(env, BLUE, SK)
    assert can_place(env, SK, 'd1')


def test_kwak_covers_and_is_covered_by_anything():
    env = blank()
    put(env, 'd1', RED, RO)
    put(env, 'd2', RED, K)
    give(env, BLUE, K, SK)
    assert can_place(env, K, 'd1') and not can_place(env, SK, 'd1')
    assert can_place(env, SK, 'd2') and can_place(env, K, 'd2')     # even a 1, even another Kwak


def test_only_the_visible_troop_counts():
    env = blank()
    put(env, 'd1', BLUE, RO)
    put(env, 'd1', RED, K)                      # Red's joker over Blue's 7
    give(env, BLUE, SK)
    assert env.owner(base(env, 'd1')) == RED
    assert can_place(env, SK, 'd1')
    assert legal_bases(env, SK) == {'d1', 'd2'}     # d1 is not Blue's any more: c1 is out of reach


def test_hook_goes_anywhere_but_not_on_the_hq():
    env = blank()
    give(env, BLUE, HO)
    assert legal_bases(env, HO) == {b.name for b in env.terrain.bases}
    put(env, 'a1', RED, XB)
    put(env, 'a2', RED, JU)
    assert not can_place(env, HO, 'a1') and can_place(env, HO, 'a2')    # still a cover
    assert not env.action_masks()[2 + HO * 17 + 15]                     # the enemy H.Q.
    place(env, HO, 'a2')
    assert top(env, 'a2') == (BLUE, HO) and env.current_player == RED


def test_hq_capture_needs_a_connected_neighbour():
    env = blank()
    give(env, BLUE, SK)
    put(env, 'a1', BLUE, HO)                    # next to Red's H.Q. but not connected
    assert not env.action_masks()[2 + SK * 17 + 15]
    for name in ('d1', 'c1', 'm1', 'b1'):
        put(env, name, BLUE, RO if name != 'b1' else K)
    assert base(env, 'a1') in env.reach(BLUE)
    assert env.action_masks()[2 + SK * 17 + 15]
    _, reward, done, _, _ = capture(env, SK)
    assert done and env.winner_player == BLUE and env.end_reason == 'H.Q. captured'
    assert reward == [1.0, -1.0]
    assert env.captured == (RED, 0, SK) and tiles(env, BLUE) == 24


def test_own_hq_is_never_a_slot():
    env = make_env(seed=1)
    assert env.action_space.n == 161
    # the two H.Q. slots are the enemy's: nothing is legal there at the start
    assert not env.action_masks()[2:138].reshape(8, 17)[:, 15:].any()


def test_draw_two():
    env = blank()
    give(env, BLUE, RO)
    env.reserve[BLUE][-2:] = [SK, JU]
    _, reward, done, _, info = env.step(A_DRAW)
    assert env.racks[BLUE][SK] == 1 and env.racks[BLUE][JU] == 1 and env.rack_size(BLUE) == 3
    assert env.current_player == RED and reward == [0.0, 0.0] and not done
    assert info == {'next_step_no_action': False}


def test_draw_is_limited_by_the_rack_and_the_reserve():
    env = blank()
    give(env, BLUE, RO, RO, RO, ST, ST, ST, XB)             # 7 troops: draws only one
    env.step(A_DRAW)
    assert env.rack_size(BLUE) == 8
    env = blank()
    give(env, BLUE, RO, RO, RO, ST, ST, ST, XB, XB)         # 8: no draw
    assert not env.action_masks()[A_DRAW]
    env = blank()
    give(env, BLUE, RO)
    del env.reserve[BLUE][1:]                               # a single tile left
    env.step(A_DRAW)
    assert env.rack_size(BLUE) == 2 and not env.reserve[BLUE]
    env = blank()
    give(env, BLUE, RO)
    env.removed[BLUE] += env.reserve[BLUE]
    env.reserve[BLUE] = []
    assert not env.action_masks()[A_DRAW] and env.action_masks().any()


def test_a_turn_is_one_action():
    env = blank()
    give(env, BLUE, RO, RO)
    give(env, RED, RO)
    place(env, RO, 'd1')
    assert env.current_player == RED and env.turns_taken == 1 and env.phase == Phase.MAIN
    assert not env.action_masks()[A_SKIP]
    place(env, RO, 'a1')
    assert env.current_player == BLUE and env.turns_taken == 2


def test_illegal_actions_raise():
    env = blank()
    give(env, BLUE, RO)
    for action in (A_SKIP, 2 + RO * 17 + 0, 2 + SK * 17 + 13, 138, 160, 161, -1):
        try:
            env.step(action)
        except Exception:
            continue
        raise AssertionError(f'action {action} should be illegal')
    assert env.current_player == BLUE and env.turns_taken == 0


def test_two_hq_on_the_caribbean_sea():
    env = blank('caribbean')
    give(env, BLUE, RO)
    assert legal_bases(env, RO) == {'i', 'l', 'h'}          # next to either blue H.Q.
    env = blank('caribbean', current=RED)
    give(env, RED, RO)
    assert legal_bases(env, RO) == {'a', 'd'}
    put(env, 'd', RED, SK)
    put(env, 'b', RED, SK)
    put(env, 'e', RED, SK)
    put(env, 'h', RED, ST)                                  # d - b - e - h reaches the second blue H.Q.
    mask = env.action_masks()[2 + RO * 17:2 + RO * 17 + 17]
    assert not mask[15] and mask[16]
    capture(env, RO, 1)
    assert env.done and env.winner_player == RED
