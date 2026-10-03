"""Game flow: rounds, the Captain, loading, moving, paying, shortages, the end."""

import random

from ..envs import rules
from ..envs.constants import (A_DEST, A_LOAD_HOLD, A_PAY_HOLD, FLOOR_SCORE, MAX_ROUNDS, Phase,
                              Power, Res, Sym, card_code, N_CODES)
from ..envs.data import DECK
from ..envs.jamaica import JamaicaEnv, Task, TK
from .helpers import (act, at_pause, captain_action, card_action, land, make_env, random_play,
                      run_tasks, set_ship)

G, F, P, E = Res.GOLD, Res.FOOD, Res.POWDER, Res.EMPTY


def test_reset_state():
    env = make_env(4, seed=3)
    assert env.pending.kind == Phase.CAPTAIN and env.current_player == env.captain
    assert len(env.pile) == 9 and len(env.removed) == 3
    for s in env.ships:
        assert len(s.hand) == 3 and len(s.deck) == 8 and not s.discard
        assert sorted(s.hand + s.deck) == sorted(DECK)
        assert rules.total(s, F) == 3 and rules.total(s, G) == 3
        assert (s.node, s.lap) == (0, 0)


def test_player_counts():
    assert JamaicaEnv().n_players == 4
    assert JamaicaEnv(player_names=['a', 'b', 'c', 'd', 'e']).n_players == 5
    for n in (2, 7):
        try:
            JamaicaEnv(n_players=n)
        except ValueError:
            continue
        raise AssertionError(f'{n} players accepted')


def test_captain_decision_and_doubles():
    env = make_env(4, seed=1)
    codes = sorted(set(env.ships[env.captain].hand))
    env.raw_dice = (4, 4)
    assert env.legal_actions() == [captain_action(env, c) for c in codes]
    env.raw_dice = (5, 2)
    assert len(env.legal_actions()) == 2 * len(codes)
    env.step(captain_action(env, codes[0], order=1))
    assert env.dice == (2, 5)
    assert env.ships[env.captain].chosen == codes[0]


def test_card_choice_is_always_asked():
    env = make_env(4, seed=2)
    nxt = (env.captain + 1) % 4
    same = card_code(Sym.FWD, Sym.FWD)
    set_ship(env, nxt, hand=[same, same, same])
    env.step(env.legal_actions()[0])
    assert env.pending.kind == Phase.CARD and env.current_player == nxt
    assert env.legal_actions() == [card_action(same)]


def test_load_into_empty_hold():
    env = make_env(4)
    act(env, 0, 0, Sym.GOLD, Sym.FWD, dice=(4, 1))
    assert at_pause(env)
    assert [G, 4] in env.ships[0].holds and rules.total(env.ships[0], G) == 7


def test_gold_gold_makes_two_holds():
    env = make_env(4)
    run_tasks(env, Task(TK.ACTION, 0, 0), Task(TK.ACTION, 0, 1), dice=(2, 5),
              card=card_code(Sym.GOLD, Sym.GOLD), seat=0)
    holds = [h for h in env.ships[0].holds if h[0] == G]
    assert sorted(h[1] for h in holds) == [2, 3, 5]


def test_full_holds_ask_which_to_empty():
    env = make_env(4)
    set_ship(env, 0, holds=[(G, 1), (F, 2), (F, 2), (P, 3), (P, 3)])
    act(env, 0, 0, Sym.FOOD, Sym.FWD, dice=(4, 1))
    assert env.pending.kind == Phase.LOAD_HOLD
    s = env.ships[0]
    gold_slot = next(i for i, h in enumerate(s.holds) if h[0] == G)
    powder_slots = [i for i, h in enumerate(s.holds) if h[0] == P]
    assert env.legal_actions() == sorted([A_LOAD_HOLD + gold_slot, A_LOAD_HOLD + powder_slots[0]])
    env.step(A_LOAD_HOLD + gold_slot)
    assert rules.total(s, G) == 0 and rules.total(s, F) == 8 and at_pause(env)


def test_load_ignored_and_single_class():
    env = make_env(4)
    set_ship(env, 0, holds=[(F, 1)] * 5)
    act(env, 0, 0, Sym.FOOD, Sym.FWD, dice=(6, 1))
    assert at_pause(env) and rules.total(env.ships[0], F) == 5
    set_ship(env, 0, holds=[(F, 1)] * 4 + [(G, 2)])
    act(env, 0, 0, Sym.FOOD, Sym.FWD, dice=(6, 1))
    assert at_pause(env) and rules.total(env.ships[0], F) == 10 and rules.total(env.ships[0], G) == 0


def test_paying_a_sea_space():
    env = make_env(4)
    land(env, 0, 1)
    assert at_pause(env) and rules.total(env.ships[0], F) == 1


def test_choosing_the_hold_to_pay_from():
    env = make_env(4)
    set_ship(env, 0, holds=[(F, 3), (F, 1), (G, 3)])
    land(env, 0, 2)                                  # sea, 3 food
    assert env.pending.kind == Phase.PAY_HOLD
    s = env.ships[0]
    small = next(i for i, h in enumerate(s.holds) if h == [F, 1])
    env.step(A_PAY_HOLD + small)
    assert at_pause(env)
    assert rules.total(s, F) == 1 and rules.n_empty(s) == 3


def test_shortage_falls_back_to_a_port():
    env = make_env(4)
    set_ship(env, 0, node=6, holds=[(F, 1), (G, 3)])
    land(env, 0, 7)                                  # sea 2 with 1 food
    s = env.ships[0]
    assert at_pause(env) and (s.node, s.lap) == (5, 0)   # 6 (port 5) unaffordable, 5 (port 3) ok
    assert rules.total(s, F) == 0 and rules.total(s, G) == 0


def test_shortage_falls_back_to_a_lair():
    env = make_env(4)
    set_ship(env, 0, node=6, holds=[(F, 1)])
    land(env, 0, 7)
    s = env.ships[0]
    assert at_pause(env) and s.node == 4
    assert not env.lair_token[4] and len(env.pile) == 8
    assert s.hidden or s.powers


def test_retreat_choice_at_a_merge():
    env = make_env(4)
    set_ship(env, 0, node=16, holds=[(F, 1)])
    land(env, 0, 17)
    assert env.pending.kind == Phase.RETREAT_DEST
    assert env.pending.data == ((11, 0), (4, 0))
    env.step(A_DEST + 1)
    assert env.ships[0].node == 4 and at_pause(env)


def test_fork_choice():
    env = make_env(4)
    set_ship(env, 0, node=6, holds=[(F, 6), (G, 6)])
    act(env, 0, 0, Sym.FWD, Sym.FWD, dice=(1, 1))
    assert env.pending.kind == Phase.MOVE_DEST
    assert env.pending.data == ((13, 0), (7, 0))
    env.step(A_DEST + 1)
    assert env.ships[0].node == 7


def test_backward_first_move_goes_behind_the_start():
    env = make_env(4)
    act(env, 0, 0, Sym.BACK, Sym.FWD, dice=(2, 1))
    s = env.ships[0]
    assert (s.node, s.lap) == (48, -1) and rules.total(s, F) == 0
    assert env.track.pos_score(s.node, s.lap) == FLOOR_SCORE


def test_lair_token_is_taken_once():
    env = make_env(4)
    land(env, 0, 4)
    land(env, 1, 4)                                  # battle-free? ship 0 is there: battle first
    assert len(env.pile) == 8


def test_sixth_hold_from_a_lair():
    env = make_env(4)
    env.pile[-1] = 3                                 # the 6th Hold card on top
    land(env, 0, 4)
    s = env.ships[0]
    assert s.has(Power.SIXTH) and s.sixth == [E, 0]
    act(env, 0, 0, Sym.GOLD, Sym.FWD, dice=(3, 1))
    act(env, 0, 0, Sym.GOLD, Sym.FWD, dice=(4, 1))
    act(env, 0, 0, Sym.GOLD, Sym.FWD, dice=(5, 1))
    act(env, 0, 0, Sym.GOLD, Sym.FWD, dice=(6, 1))   # 5 regular holds full: into the 6th
    assert s.sixth == [G, 6]


def test_morgans_map_hand_of_four():
    env = make_env(4, seed=5)
    env.ships[1].powers |= 1 << int(Power.MAP)
    rng = random.Random(0)
    while env.round == 1:
        env.step(-1 if env.current_player == -1 else rng.choice(env.legal_actions()))
    assert len(env.ships[1].hand) == 4
    assert all(len(s.hand) == 3 for s in env.ships if s.seat != 1)


def _play_one_round(env, rng):
    r = env.round
    while env.round == r and not env.done:
        env.step(-1 if env.current_player == -1 else rng.choice(env.legal_actions()))


def test_finish_ends_the_game_after_the_round():
    env = make_env(4, seed=7)
    cap = env.captain
    ff = card_code(Sym.FWD, Sym.FOOD)
    set_ship(env, cap, node=48, hand=[ff, ff, ff])
    env.raw_dice = (6, 3)
    env.step(captain_action(env, ff, order=0))
    rng = random.Random(1)
    _play_one_round(env, rng)
    s = env.ships[cap]
    assert env.done and env.round == 1
    assert s.finished and (s.node, s.lap) == (0, 1)
    assert rules.total(s, F) == 3                    # the evening load was ignored
    assert env.final_scores[cap] == 15 + rules.total(s, G) + sum(env.treasure_value(t) for t in s.hidden)


def test_round_cap():
    env = make_env(4, seed=8)
    env.round = MAX_ROUNDS
    _play_one_round(env, random.Random(2))
    assert env.done


def test_reshuffle_keeps_every_card():
    env = make_env(3, seed=9)
    rng = random.Random(3)
    for _ in range(6):
        _play_one_round(env, rng)
        if env.done:
            break
        for s in env.ships:
            cards = s.hand + s.deck + s.discard
            assert sorted(cards) == sorted(DECK) and len(s.hand) == 3


def test_returns_equal_terminal_rewards():
    for seed in range(6):
        env = make_env(4, seed=seed)
        totals, _ = random_play(env, random.Random(seed))
        assert env.done
        assert all(abs(a - b) < 1e-9 for a, b in zip(totals, env.terminal_rewards))
        assert abs(sum(env.terminal_rewards)) < 1e-9


def test_ties_share_rewards():
    env = make_env(4, seed=4)
    for s in env.ships:
        set_ship(env, s.seat, node=0, holds=[(G, 3)], hidden=[])
    env.round = MAX_ROUNDS
    for s in env.ships:
        s.hand = [card_code(Sym.GOLD, Sym.GOLD)] * 3
    # everybody loads the same: identical scores and positions at the end
    env.raw_dice = (2, 2)
    while not env.done:
        env.step(-1 if env.current_player == -1 else env.legal_actions()[0])
    assert len(set(env.final_scores)) == 1
    assert all(abs(r) < 1e-9 for r in env.terminal_rewards)


def test_pauses():
    env = make_env(4, seed=6, pause=True)
    assert env.pending.kind == Phase.CAPTAIN
    rng = random.Random(4)
    while env.pending.kind != Phase.TURN_PAUSE:
        env.step(rng.choice(env.legal_actions()))
    assert env.current_player == -1 and env._get_info()['next_step_no_action']
    assert not env.action_masks().any()
    try:
        env.step(0)
    except Exception:
        pass
    else:
        raise AssertionError('an action was accepted during a pause')
    env.step(-1)
