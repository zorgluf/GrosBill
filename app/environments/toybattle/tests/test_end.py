"""Regions, medals, the ways the game ends and the rewards."""

from ..envs.constants import A_DRAW, A_SKIP, BLUE, Phase, RED, SHAPING, Troop
from .helpers import blank, give, place, put, target, top

K, SK, CA, JU, HO, XB, ST, RO = Troop.KWAK, Troop.SKULLY, Troop.CAPN, Troop.JUMBO, Troop.HOOK, \
    Troop.XB42, Troop.STAR, Troop.ROXY


def _castle(*names, current=BLUE, **kwargs):
    """Castle Field with a blue troop on each of `names` (d1 - c1 - m1 ... are connected)."""
    env = blank('castle', current=current, **kwargs)
    troops = [SK, SK, CA, CA, JU, JU, HO, HO, XB, XB, ST, ST]
    for name, troop in zip(names, troops):
        put(env, name, BLUE, troop)
    return env


def test_surrounding_a_region_gives_its_medals():
    env = _castle('d1', 'c1', 'm1', 'm2')
    give(env, BLUE, RO)
    _, reward, done, _, _ = place(env, RO, 'b1')            # b1 m1 c1 m2: 2 medals
    assert env.medals == [2, 0] and not done
    assert env.region_owner.count(BLUE) == 1 and env.region_owner.count(-1) == 7
    gain = SHAPING * 2 / 7
    assert abs(reward[BLUE] - gain) < 1e-9 and abs(reward[RED] + gain) < 1e-9


def test_medals_are_kept_and_given_once():
    env = _castle('d1', 'c1', 'm1', 'm2')
    give(env, BLUE, RO, RO)
    give(env, RED, K)
    put(env, 'a1', RED, RO)
    place(env, RO, 'b1')
    place(env, K, 'b1')                                     # Red takes b1 back
    assert env.medals == [2, 0] and top(env, 'b1') == (RED, K)
    _, reward, _, _, _ = place(env, RO, 'b1')               # control again: nothing more
    assert env.medals == [2, 0] and reward == [0.0, 0.0]


def test_several_regions_in_one_turn():
    env = _castle('d1', 'c1', 'm1', 'b1', 'b2', 'a1', 'a2')
    give(env, BLUE, RO, RO)
    place(env, RO, 'm2')                                    # 3 (top) + 2 (b1 m1 c1 m2)
    assert env.medals == [5, 0] and not env.done


def test_medals_objective_wins_at_once():
    env = _castle('d1', 'c1', 'm1', 'b1', 'b2', 'a1', 'a2', 'c2', 'm3')
    give(env, BLUE, CA, RO)
    _, reward, done, _, _ = place(env, CA, 'm2')            # 3 + 2 + 2 = 7: no extra troop, the game is over
    assert done and env.winner_player == BLUE and env.end_reason == 'medals objective'
    assert env.medals == [7, 0] and reward == [1.0, -1.0]
    assert env.final_ranking() == [BLUE, RED] and not env.action_masks().any()


def test_the_opponent_may_gain_medals_on_my_turn():
    env = blank('castle')
    put(env, 'd1', BLUE, SK)
    put(env, 'c1', RED, HO)
    put(env, 'c1', BLUE, XB)                                # Blue over Red on c1
    for name in ('b1', 'm1', 'm2'):
        put(env, name, RED, RO)
    give(env, BLUE, ST)
    place(env, ST, 's3')
    assert env.phase == Phase.RECALL
    _, reward, done, _, _ = target(env, 'c1')               # shows Red's Hook: b1 m1 c1 m2 is Red's
    assert env.medals == [0, 2] and not done and env.current_player == RED
    assert abs(reward[BLUE] + SHAPING * 2 / 7) < 1e-9 and reward[RED] == -reward[BLUE]


def test_the_opponent_may_even_win_on_my_turn():
    env = blank('castle')
    put(env, 'd1', BLUE, SK)
    put(env, 'c1', RED, HO)
    put(env, 'c1', BLUE, XB)
    for name in ('b1', 'm1', 'm2'):
        put(env, name, RED, RO)
    env.medals = [0, 5]
    give(env, BLUE, ST)
    place(env, ST, 's3')
    _, reward, done, _, _ = target(env, 'c1')
    assert done and env.winner_player == RED
    assert abs(reward[RED] - (1 - SHAPING * 5 / 7)) < 1e-9          # the potential it already had is not paid twice


def test_jumbo_may_reveal_a_region():
    env = _castle('d1', 'c1', 'm1', 'b1')
    put(env, 'm2', BLUE, RO)
    put(env, 'm2', RED, K)                                  # Red's joker sits on Blue's 7
    give(env, BLUE, JU)
    place(env, JU, 'c1')
    target(env, 'm2')
    assert top(env, 'm2') == (BLUE, RO) and env.medals == [2, 0]


def _out_of_tiles(env, p):
    env.removed[p] += env.reserve[p]
    env.reserve[p] = []


def test_no_action_left_compares_the_medals():
    for medals, winner in (([2, 1], BLUE), ([1, 2], RED), ([1, 1], RED), ([0, 0], RED)):
        env = blank('castle', current=RED)
        give(env, RED, RO)
        _out_of_tiles(env, BLUE)                            # Blue: empty rack, empty reserve
        env.medals = list(medals)
        _, reward, done, _, _ = place(env, RO, 'a1')
        assert done and env.winner_player == winner, medals  # a tie is lost by the stuck player
        assert 'no action left' in env.end_reason
        assert reward[winner] == 1 - SHAPING * (medals[winner] - medals[1 - winner]) / 7


def test_stuck_with_troops_on_the_rack():
    env = blank('castle', current=RED)
    give(env, RED, RO)
    put(env, 'd1', RED, RO)
    give(env, BLUE, SK, CA)                                 # too weak for d1, and d2 is about to be taken
    _out_of_tiles(env, BLUE)
    put(env, 'c2', RED, XB)
    put(env, 'm2', RED, XB)
    put(env, 'b2', RED, ST)
    put(env, 'a2', RED, ST)                                 # a2 - b2 - m2 - c2 - d2: connected
    place(env, RO, 'd2')
    assert env.done and env.winner_player == RED and 'medals tied' in env.end_reason


def test_a_full_rack_that_cannot_be_placed():
    env = blank('castle', current=RED)
    give(env, BLUE, SK, SK, SK, CA, CA, CA, JU, JU)         # 8 troops: no draw
    put(env, 'd1', RED, RO)
    put(env, 'c2', RED, XB)
    put(env, 'm2', RED, XB)
    put(env, 'b2', RED, ST)
    put(env, 'a2', RED, ST)
    give(env, RED, RO)
    env.medals = [1, 0]
    place(env, RO, 'd2')
    assert env.done and env.winner_player == BLUE           # stuck, but ahead


def test_turn_limit():
    env = blank('castle', max_turns=4)
    for _ in range(3):
        _, reward, done, _, _ = env.step(A_DRAW)
        assert not done
    _, reward, done, truncated, _ = env.step(A_DRAW)
    assert done and not truncated and env.winner_player is None and reward == [0.0, 0.0]
    assert env.final_ranking() == [] and 'draw' in env.end_reason
    env = blank('castle', max_turns=2)
    env.medals = [0, 3]
    env.step(A_DRAW)
    _, reward, done, _, _ = env.step(A_DRAW)
    assert done and env.winner_player == RED


def test_capn_extra_troop_after_medals():
    env = _castle('d1', 'c1', 'm1', 'm2')
    give(env, BLUE, CA, RO)
    place(env, CA, 'b1')
    assert env.medals == [2, 0] and env.phase == Phase.EXTRA
    env.step(A_SKIP)
    assert env.current_player == RED
