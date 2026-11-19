"""The troop effects and the special bases of each terrain."""

from ..envs.constants import A_DRAW, A_SKIP, A_TYPE, BLUE, Phase, RED, Troop
from .helpers import (base, blank, can_place, give, legal_bases, place, put, target, targets, tiles, top)

K, SK, CA, JU, HO, XB, ST, RO = Troop.KWAK, Troop.SKULLY, Troop.CAPN, Troop.JUMBO, Troop.HOOK, \
    Troop.XB42, Troop.STAR, Troop.ROXY


# --------------------------------------------------------------------------- #
# troops (on the Caribbean Sea: no special base)
# --------------------------------------------------------------------------- #

def test_skully_draws_two():
    env = blank('caribbean')
    give(env, BLUE, SK, RO)
    env.reserve[BLUE][-2:] = [JU, HO]
    place(env, SK, 'l')
    assert env.racks[BLUE][JU] == 1 and env.racks[BLUE][HO] == 1 and env.rack_size(BLUE) == 3
    assert env.current_player == RED


def test_skully_draws_one_with_seven_troops_left():
    env = blank('caribbean')
    give(env, BLUE, SK, RO, RO, RO, ST, ST, ST, XB)     # 8: 7 left once Skully is placed
    place(env, SK, 'l')
    assert env.rack_size(BLUE) == 8


def test_star_draws_one():
    env = blank('caribbean')
    give(env, BLUE, ST)
    env.reserve[BLUE][-1] = JU
    place(env, ST, 'l')
    assert env.racks[BLUE] == [0, 0, 0, 1, 0, 0, 0, 0]


def test_xb42_discards_a_random_troop_of_the_opponent_rack():
    seen = set()
    for seed in range(30):
        env = blank('caribbean', seed=seed)
        give(env, BLUE, XB)
        give(env, RED, SK, RO, RO)
        place(env, XB, 'l')
        assert env.rack_size(RED) == 2 and sum(env.discard[RED]) == 1 and not any(env.discard[BLUE])
        seen.add(env.discard[RED].index(1))
        assert tiles(env, RED) == 24
    assert seen == {SK, RO}
    env = blank('caribbean')
    give(env, BLUE, XB)
    place(env, XB, 'l')                                 # empty rack: nothing happens
    assert env.current_player == RED and not any(env.discard[RED])


def test_capn_places_an_extra_troop():
    env = blank('caribbean')
    give(env, BLUE, CA, SK, RO)
    place(env, CA, 'l')
    assert env.phase == Phase.EXTRA and env.current_player == BLUE
    mask = env.action_masks()
    assert mask[A_SKIP] and not mask[A_DRAW]
    assert legal_bases(env, RO) == {'l', 'i', 'h', 'j', 'k'}        # through the Cap'n just placed
    env.reserve[BLUE][-2:] = [XB, XB]
    place(env, SK, 'j')                                             # its effect applies too
    assert env.racks[BLUE][XB] == 2 and env.current_player == RED and env.phase == Phase.MAIN


def test_capn_may_be_skipped_and_chained():
    env = blank('caribbean')
    give(env, BLUE, CA, CA, RO)
    place(env, CA, 'l')
    place(env, CA, 'j')
    assert env.phase == Phase.EXTRA and env.current_player == BLUE
    env.step(A_SKIP)
    assert env.current_player == RED and env.racks[BLUE][RO] == 1
    assert top(env, 'l') == (BLUE, CA) and top(env, 'j') == (BLUE, CA)


def test_capn_without_a_troop_to_place():
    env = blank('caribbean')
    give(env, BLUE, CA)
    place(env, CA, 'l')
    assert env.current_player == RED and env.phase == Phase.MAIN


def test_capn_extra_troop_may_capture_the_hq():
    env = blank('caribbean', current=RED)
    give(env, RED, CA, SK)
    for name in ('d', 'b', 'e'):
        put(env, name, RED, RO)
    place(env, CA, 'h')
    _, reward, done, _, _ = env.step(2 + SK * 17 + 16)
    assert done and env.winner_player == RED and reward == [-1.0, 1.0]


def test_jumbo_discards_an_adjacent_visible_enemy():
    env = blank('caribbean')
    give(env, BLUE, JU)
    put(env, 'j', RED, SK, RO)          # Roxy over Skully
    put(env, 'k', RED, HO)
    put(env, 'i', BLUE, RO)
    put(env, 'f', RED, RO)              # not adjacent to l
    place(env, JU, 'l')
    assert env.phase == Phase.JUMBO and env.pending_base == base(env, 'l')
    assert targets(env) == {'j', 'k'}
    target(env, 'j')
    assert top(env, 'j') == (RED, SK)   # the troop underneath shows again
    assert env.discard[RED][RO] == 1 and env.current_player == RED and tiles(env, RED) == 24


def test_jumbo_may_be_skipped_or_have_no_target():
    env = blank('caribbean')
    give(env, BLUE, JU, JU)
    put(env, 'j', RED, RO)
    place(env, JU, 'l')
    env.step(A_SKIP)
    assert top(env, 'j') == (RED, RO) and env.current_player == RED
    env = blank('caribbean')
    give(env, BLUE, JU)
    place(env, JU, 'l')
    assert env.current_player == RED


def test_kwak_and_roxy_have_no_effect():
    for troop in (K, RO):
        env = blank('caribbean')
        give(env, BLUE, troop)
        give(env, RED, RO)
        put(env, 'j', RED, SK)
        place(env, troop, 'l')
        assert env.current_player == RED and env.rack_size(BLUE) == 0 and env.rack_size(RED) == 1


# --------------------------------------------------------------------------- #
# terrains
# --------------------------------------------------------------------------- #

def _castle_with_tower():
    """Blue holds d1 - c1 and may place on the special base s3."""
    env = blank('castle')
    put(env, 'd1', BLUE, SK)
    put(env, 'c1', BLUE, HO)
    return env


def test_castle_takes_another_troop_back():
    env = _castle_with_tower()
    give(env, BLUE, RO)
    put(env, 'a1', BLUE, XB)                    # anywhere on the terrain, connected or not
    place(env, RO, 's3')
    assert env.phase == Phase.RECALL and targets(env) == {'d1', 'c1', 'a1'}     # not s3 itself
    target(env, 'a1')
    assert top(env, 'a1') is None and env.racks[BLUE][XB] == 1 and env.current_player == RED


def test_castle_recall_shows_what_was_covered():
    env = _castle_with_tower()
    give(env, BLUE, RO)
    put(env, 'd2', RED, SK)
    put(env, 'd2', BLUE, ST)
    place(env, RO, 's3')
    target(env, 'd2')
    assert top(env, 'd2') == (RED, SK) and env.racks[BLUE][ST] == 1


def test_castle_recall_is_optional_and_needs_room():
    env = _castle_with_tower()
    give(env, BLUE, RO)
    place(env, RO, 's3')
    env.step(A_SKIP)
    assert env.current_player == RED and top(env, 'c1') == (BLUE, HO)
    env = blank('castle')
    give(env, BLUE, RO)
    put(env, 's3', RED, SK)                     # covered troops do not count: nothing to take back
    put(env, 'd1', BLUE, SK)
    put(env, 'c1', BLUE, HO)
    put(env, 'd1', RED, RO)
    put(env, 'c1', RED, RO)
    give(env, BLUE, HO)
    place(env, HO, 's3')
    assert env.current_player == RED


def test_castle_recall_comes_after_the_troop_effects():
    env = _castle_with_tower()
    give(env, BLUE, CA, SK)
    place(env, CA, 's3')
    assert env.phase == Phase.EXTRA
    env.reserve[BLUE][-2:] = [XB, XB]
    place(env, SK, 'm1')
    assert env.phase == Phase.RECALL and env.racks[BLUE][XB] == 2
    assert 'm1' in targets(env) and 's3' not in targets(env)
    target(env, 'm1')
    assert env.racks[BLUE][SK] == 1 and env.current_player == RED


def test_pool_gates():
    env = blank('pool')
    give(env, BLUE, K, SK, CA, JU, HO, XB, ST, RO)
    # b is next to the open blue H.Q.; l4 (1/2) and c3 are next to b
    put(env, 'b', BLUE, SK)
    assert {t for t in Troop if can_place(env, t, 'l4')} == {K, SK, CA}
    put(env, 'c3', BLUE, CA)
    assert {t for t in Troop if can_place(env, t, 'c2')} == {K, ST, RO}         # 6/7
    assert not can_place(env, HO, 'l4') and can_place(env, HO, 'l2') and can_place(env, HO, 'r3')   # 3/4/5
    assert {t for t in Troop if can_place(env, t, 'l2')} == {HO}                # the others are not connected
    assert all(can_place(env, t, 'r4') for t in Troop)


def test_pool_gated_hq():
    env = blank('pool', current=RED)
    give(env, RED, K, SK, RO, ST)
    put(env, 't', RED, HO)                      # next to Red's open H.Q., leads to c1 and r1
    put(env, 'c1', RED, HO)
    put(env, 'c2', RED, RO)
    put(env, 'c3', RED, HO)                     # c3 touches the gated blue H.Q. (B2), b the open one
    hq = {t: env.action_masks()[2 + t * 17 + 15:2 + t * 17 + 17].tolist() for t in (K, SK, RO, ST)}
    assert hq == {K: [False, True], SK: [False, False], RO: [False, True], ST: [False, True]}
    put(env, 'b', RED, XB)
    assert env.action_masks()[2 + SK * 17 + 15]


def test_clouds_special_base_draws_one():
    env = blank('clouds')
    give(env, BLUE, RO, ST)
    put(env, 'l5', BLUE, SK)
    env.reserve[BLUE][-1] = JU
    place(env, RO, 's4')
    assert env.racks[BLUE][JU] == 1 and env.current_player == RED
    place_blue = blank('clouds')
    give(place_blue, BLUE, ST)
    put(place_blue, 'l5', BLUE, SK)
    place(place_blue, ST, 's4')                 # Star then the base: two troops
    assert place_blue.rack_size(BLUE) == 2


def test_jungle_eruption_moves_an_adjacent_enemy():
    env = blank('jungle')
    give(env, BLUE, RO)
    put(env, 'j', BLUE, ST)                     # B - j - k, and k is next to v2 (special)
    put(env, 'k', BLUE, SK)
    put(env, 'g', RED, XB)
    put(env, 'i', RED, SK, HO)
    put(env, 'f', RED, RO)                      # not adjacent to v2
    place(env, RO, 'v2')
    assert env.phase == Phase.ERUPT_SRC and targets(env) == {'g', 'i'}
    target(env, 'i')
    assert env.phase == Phase.ERUPT_DST and targets(env) == {'f', 'j', 'v2'}
    assert not env.action_masks()[A_SKIP]
    target(env, 'j')
    assert top(env, 'j') == (RED, HO) and top(env, 'i') == (RED, SK)
    assert env.current_player == RED and tiles(env, RED) == 24


def test_jungle_eruption_ignores_the_placement_rules():
    env = blank('jungle')
    give(env, BLUE, RO)
    put(env, 'j', BLUE, ST)
    put(env, 'k', BLUE, SK)
    put(env, 'g', RED, SK)
    put(env, 'f', BLUE, RO)
    place(env, RO, 'v2')
    target(env, 'g')
    target(env, 'f')                            # a 1 lands on a 7
    assert env.stacks[base(env, 'f')] == [(BLUE, RO), (RED, SK)] and top(env, 'g') is None


def test_jungle_eruption_is_optional():
    env = blank('jungle')
    give(env, BLUE, RO, RO)
    put(env, 'j', BLUE, ST)
    put(env, 'k', BLUE, SK)
    put(env, 'g', RED, SK)
    place(env, RO, 'v2')
    env.step(A_SKIP)
    assert top(env, 'g') == (RED, SK) and env.current_player == RED
    env = blank('jungle')
    give(env, BLUE, RO)
    put(env, 'j', BLUE, ST)
    put(env, 'k', BLUE, SK)
    place(env, RO, 'v2')                        # no enemy around
    assert env.current_player == RED


def test_cemetery_takes_a_discarded_troop_back():
    env = blank('cemetery')
    give(env, BLUE, RO)
    put(env, 'k', BLUE, SK)
    for troop in (XB, JU):
        env.reserve[BLUE].remove(troop)
        env.discard[BLUE][troop] += 1
    env.reserve[RED].remove(RO)
    env.discard[RED][RO] += 1
    place(env, RO, 'p4')
    assert env.phase == Phase.EXHUME
    assert [a - A_TYPE for a in env.legal_actions() if a >= A_TYPE] == [JU, XB]
    env.step(A_TYPE + XB)
    assert env.racks[BLUE][XB] == 1 and env.discard[BLUE] == [0, 0, 0, 1, 0, 0, 0, 0]
    assert env.current_player == RED and tiles(env, BLUE) == 24


def test_cemetery_needs_a_discarded_troop():
    env = blank('cemetery')
    give(env, BLUE, RO)
    put(env, 'k', BLUE, SK)
    env.reserve[RED].remove(RO)
    env.discard[RED][RO] += 1                   # only an enemy troop in the discard
    place(env, RO, 'p4')
    assert env.current_player == RED


def test_station_shield_cancels_the_troop_effect():
    for troop in (SK, ST, CA, JU, XB):
        env = blank('station')
        give(env, BLUE, troop, RO)
        give(env, RED, RO)
        put(env, 'i', BLUE, HO)                 # B - i - g, and g is next to the special base s1
        put(env, 'g', BLUE, SK)
        put(env, 'e', RED, SK)                  # an enemy next to s1, for Jumbo
        place(env, troop, 's1')
        assert env.current_player == RED and env.phase == Phase.MAIN, troop
        assert env.racks[BLUE][RO] == 1 and env.rack_size(BLUE) == 1 and env.rack_size(RED) == 1
        assert top(env, 'e') == (RED, SK)
    env = blank('station')
    give(env, BLUE, SK)
    place(env, SK, 'j')                         # a plain base: the effect applies
    assert env.rack_size(BLUE) == 2


def test_station_hook_needs_a_connection_to_the_special_bases():
    env = blank('station')
    give(env, BLUE, HO)
    assert legal_bases(env, HO) == {b.name for b in env.terrain.bases if not b.special}
    put(env, 'i', BLUE, SK)
    put(env, 'g', BLUE, SK)
    assert legal_bases(env, HO) == {b.name for b in env.terrain.bases if b.name not in ('s2', 's3')}


def test_battlefield_pins_a_troop_down():
    env = blank('battlefield')
    give(env, BLUE, RO, RO)
    give(env, RED, SK)
    put(env, 'd1', BLUE, SK)
    put(env, 'c1', BLUE, SK)
    place(env, RO, 'sb')
    assert env.blocked[RED] == [0, 1, 0, 0, 0, 0, 0, 0] and env.current_player == RED
    assert not legal_bases(env, SK) and env.legal_actions() == [A_DRAW]
    env.step(A_DRAW)
    assert env.blocked[RED] == [0] * 8          # back at the end of Red's turn
    env.step(A_DRAW)
    assert legal_bases(env, SK) == {'a1', 'a2'}


def test_battlefield_pins_one_tile_only():
    env = blank('battlefield')
    give(env, BLUE, RO)
    give(env, RED, SK, SK)
    put(env, 'd1', BLUE, SK)
    put(env, 'c1', BLUE, XB)
    place(env, RO, 'sb')
    assert env.blocked[RED][SK] == 1 and legal_bases(env, SK) == {'a1', 'a2'}
    seen = set()
    for seed in range(30):
        env = blank('battlefield', seed=seed)
        give(env, BLUE, RO)
        give(env, RED, SK, HO)
        put(env, 'd1', BLUE, SK)
        put(env, 'c1', BLUE, XB)
        place(env, RO, 'sb')
        seen.add(env.blocked[RED].index(1))
    assert seen == {SK, HO}
