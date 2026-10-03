"""The consequence simulator agrees with the engine; battle odds are sane."""

import copy
import random

from ..envs import rules, sim
from ..envs.constants import A_CAPTAIN, A_PAY_HOLD, N_CODES, Phase, Res
from ..envs.rules import SIXTH_SLOT
from ..envs.data import COMBAT_FACES
from .helpers import make_env


def _captain_states(n_games=12):
    for g in range(n_games):
        env = make_env(4, seed=70 + g, pause=True)
        rng = random.Random(g)
        while not env.done:
            if env.current_player >= 0 and env.pending.kind == Phase.CAPTAIN:
                yield env
            env.step(-1 if env.current_player == -1 else rng.choice(env.legal_actions()))


def test_simulated_card_matches_the_engine():
    compared = 0
    for env in _captain_states():
        cap = env.captain
        hi, lo = max(env.raw_dice), min(env.raw_dice)
        for a in env.legal_actions():
            order, code = divmod(a - A_CAPTAIN, N_CODES)
            dice = (hi, lo) if order == 0 else (lo, hi)
            o = sim.play(env, cap, code, dice)
            if o.combats or o.lairs:
                continue
            e = copy.deepcopy(env)
            e.step(a)
            ok = True
            rng = random.Random(0)
            while not (e.pending is not None and e.pending.kind == Phase.TURN_PAUSE):
                if e.current_player == cap and e.pending.kind == Phase.PAY_HOLD:
                    # the simulator pays from the smallest hold first (the 6th on ties)
                    ship = e.ships[cap]
                    e.step(min(e.legal_actions(), key=lambda a: (
                        rules.hold_at(ship, a - A_PAY_HOLD)[1], a - A_PAY_HOLD != SIXTH_SLOT)))
                elif e.current_player == cap and e.resolving == cap:
                    ok = False                 # a fork or a dump choice: the sim is greedy there
                    break
                else:
                    e.step(rng.choice(e.legal_actions()))
            if not ok:
                continue
            s, start = e.ships[cap], env.ships[cap]
            got = (rules.total(s, Res.GOLD) - rules.total(start, Res.GOLD),
                   rules.total(s, Res.FOOD) - rules.total(start, Res.FOOD),
                   rules.total(s, Res.POWDER) - rules.total(start, Res.POWDER),
                   env.track.remaining(start.node, start.lap) - env.track.remaining(s.node, s.lap),
                   s.finished)
            want = (o.d_gold, o.d_food, o.d_powder, o.progress, o.finished)
            assert got == want, (env.card_name(code), dice, got, want)
            compared += 1
    assert compared > 30


def test_battle_odds():
    w, t = sim.p_battle(0, 0, 0, 0, COMBAT_FACES)
    assert 0 < w < 1 and 0 < t < 1 and w + t <= 1
    w, t = sim.p_battle(100, 0, 0, 0, COMBAT_FACES)
    assert abs(w - (1 / 6 + 25 / 36)) < 1e-9 and t == 0     # only the defender's star beats it
    w, t = sim.p_defend(12, 0, 0, COMBAT_FACES)
    assert abs(w - 1 / 6) < 1e-9 and t == 0
    w, t = sim.p_defend(10, 0, 0, COMBAT_FACES)
    assert abs(w - 1 / 6) < 1e-9 and abs(t - 1 / 6) < 1e-9
