"""Battles: stars, ties, Lady Beth, the Sabre, powder masks and the loot."""

from ..envs import rules
from ..envs.constants import (A_GIVE_CURSED, A_LOAD_HOLD, A_NOTHING, A_POWDER, A_SABRE,
                              A_STEAL_HIDDEN, A_STEAL_HOLD, A_STEAL_POWER, A_TARGET, Phase, Power,
                              Res, SABRE_KEEP, SABRE_REROLL, STAR)
from .helpers import at_pause, land, make_env, set_ship

G, F, P, E = Res.GOLD, Res.FOOD, Res.POWDER, Res.EMPTY


def _battle(env, rig, att=0, dfn=1, node=1):
    set_ship(env, dfn, node=node)
    env._rig_combat = list(rig)
    land(env, att, node)


def test_star_wins_before_paying():
    env = make_env(4)
    _battle(env, [STAR])
    b = env.battle
    assert env.pending.kind == Phase.REWARD and env.current_player == 0
    assert b.winner == 0 and not b.def_rolled
    assert rules.total(env.ships[0], F) == 3          # not paid yet
    env.step(A_NOTHING)
    assert at_pause(env) and rules.total(env.ships[0], F) == 1


def test_defender_star():
    env = make_env(4)
    _battle(env, [10, STAR])
    assert env.pending.kind == Phase.REWARD and env.current_player == 1
    assert env.battle.winner == 1


def test_tie_loses_the_powder():
    env = make_env(4)
    set_ship(env, 0, holds=[(F, 3), (P, 3)])
    _battle(env, [4, 6])
    assert env.pending.kind == Phase.ATTACK_POWDER
    env.step(A_POWDER + 2)
    assert at_pause(env)                              # 4+2 = 6 against 6: nothing happens
    assert rules.total(env.ships[0], P) == 1
    assert env.last_battle.winner == -1


def test_lady_beth():
    env = make_env(4)
    set_ship(env, 1, powers=1 << int(Power.BETH))
    _battle(env, [6, 4])
    assert at_pause(env) and env.last_battle.winner == -1   # 6 against 4+2


def test_no_battle_at_port_royal():
    env = make_env(4)
    land(env, 0, 0)
    assert at_pause(env) and env.last_battle is None


def test_target_choice():
    env = make_env(4)
    set_ship(env, 2, node=1)
    _battle(env, [STAR])
    assert env.pending.kind == Phase.TARGET
    assert env.legal_actions() == [A_TARGET + 0, A_TARGET + 1]
    env.step(A_TARGET + 1)
    assert env.battle.defender == 2


def test_powder_masks():
    env = make_env(4)
    set_ship(env, 0, holds=[(P, 6), (P, 6), (P, 6), (P, 2)])
    set_ship(env, 1, holds=[(P, 3)])
    _battle(env, [8, 4])
    assert env.legal_actions() == [A_POWDER + k for k in range(13)]   # 3 + 10 - 2 + 1
    env.step(A_POWDER + 1)                            # attacker: 8 + 1 = 9
    # defender with 3 powder: 9 - 10 = -1 -> can tie from 0; 9 - 2 + 1 = 8 > 3
    assert env.pending.kind == Phase.DEFENSE_POWDER
    assert env.legal_actions() == [A_POWDER + k for k in range(4)]


def test_sabre_own_reroll():
    env = make_env(4)
    set_ship(env, 0, powers=1 << int(Power.SABRE))
    _battle(env, [2, STAR])
    assert env.pending.kind == Phase.SABRE and env.current_player == 0
    env.step(A_SABRE + SABRE_REROLL)
    assert env.pending.kind == Phase.REWARD and env.battle.winner == 0


def test_sabre_forces_a_star_reroll():
    env = make_env(4)
    set_ship(env, 1, powers=1 << int(Power.SABRE))
    _battle(env, [STAR, 4, 6])
    assert env.pending.kind == Phase.SABRE and env.current_player == 1
    env.step(A_SABRE + SABRE_REROLL)
    assert env.pending.kind == Phase.REWARD and env.battle.winner == 1   # 4 against 6


def test_sabre_once_per_battle():
    env = make_env(4)
    set_ship(env, 0, powers=1 << int(Power.SABRE))
    _battle(env, [2, 2, 8])
    env.step(A_SABRE + SABRE_REROLL)                  # 2 again, no second window
    assert env.pending.kind == Phase.REWARD and env.battle.winner == 1


def test_sabre_keep():
    env = make_env(4)
    set_ship(env, 0, powers=1 << int(Power.SABRE))
    _battle(env, [8, 2])                              # no window on the defender's worst face
    env.step(A_SABRE + SABRE_KEEP)
    assert env.pending.kind == Phase.REWARD and env.battle.winner == 0


def test_no_sabre_window_after_own_star():
    env = make_env(4)
    set_ship(env, 0, powers=1 << int(Power.SABRE))
    _battle(env, [STAR])
    assert env.pending.kind == Phase.REWARD


def test_steal_a_hold_with_a_dump_choice():
    env = make_env(4)
    set_ship(env, 0, holds=[(F, 1), (F, 1), (P, 2), (P, 2), (F, 3)])
    set_ship(env, 1, holds=[(G, 5)])
    _battle(env, [STAR])
    env.step(A_POWDER + 0)                            # the attacker has powder: keep it
    loser = env.ships[1]
    gold_slot = next(i for i, h in enumerate(loser.holds) if h[0] == G)
    env.step(A_STEAL_HOLD + gold_slot)
    assert env.pending.kind == Phase.LOAD_HOLD and env.current_player == 0
    w = env.ships[0]
    powder_slot = next(i for i, h in enumerate(w.holds) if h[0] == P)
    env.step(A_LOAD_HOLD + powder_slot)
    assert rules.total(w, G) == 5 and rules.total(w, P) == 2 and rules.total(loser, G) == 0


def test_steal_masked_when_the_winner_cannot_load():
    env = make_env(4)
    set_ship(env, 0, holds=[(G, 1)] * 5)
    set_ship(env, 1, holds=[(G, 3), (F, 3)])
    _battle(env, [STAR])
    loser = env.ships[1]
    gold_slot = next(i for i, h in enumerate(loser.holds) if h[0] == G)
    food_slot = next(i for i, h in enumerate(loser.holds) if h[0] == F)
    legal = env.legal_actions()
    assert A_STEAL_HOLD + gold_slot not in legal and A_STEAL_HOLD + food_slot in legal


def test_steal_the_sixth_hold_with_its_contents():
    env = make_env(4)
    set_ship(env, 1, sixth=(G, 4))
    _battle(env, [STAR])
    env.step(A_STEAL_POWER + int(Power.SIXTH))
    assert env.ships[0].sixth == [G, 4] and env.ships[1].sixth is None
    assert env.ships[0].has(Power.SIXTH) and not env.ships[1].has(Power.SIXTH)


def test_steal_a_hidden_treasure():
    env = make_env(4)
    set_ship(env, 1, hidden=[4])
    _battle(env, [STAR])
    env.step(A_STEAL_HIDDEN)
    assert env.ships[0].hidden == [4] and env.known[4] & 1


def test_give_the_worst_curse():
    env = make_env(4)
    set_ship(env, 0, hidden=[9, 11, 6])               # -2, -4, +5
    _battle(env, [STAR])
    assert A_GIVE_CURSED in env.legal_actions()
    env.step(A_GIVE_CURSED)
    assert sorted(env.ships[0].hidden) == [6, 9] and env.ships[1].hidden == [11]
    assert env.public_cursed[11] and env.known[11] & 2


def test_attacker_losing_can_run_short():
    env = make_env(4)
    set_ship(env, 0, holds=[(F, 2)])
    _battle(env, [2, 8])                              # defender wins and takes the food
    food_slot = next(i for i, h in enumerate(env.ships[0].holds) if h[0] == F)
    env.step(A_STEAL_HOLD + food_slot)
    assert at_pause(env)
    assert env.ships[0].node == 0                     # nothing left to pay: back to Port Royal
