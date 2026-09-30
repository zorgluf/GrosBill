"""Random games for 3-6 players with the invariants checked after every step."""

import random

import numpy as np

from ..envs.constants import A_STEAL_HOLD, MAX_HOLD, N_HOLDS, Phase, Power, Res
from ..envs.data import DECK, TREASURES
from ..envs.jamaica import JamaicaEnv

POWER_TID = {p: i for i, (p, _) in enumerate(TREASURES) if p is not None}


def check_invariants(env) -> None:
    n = env.n_players
    tr = env.track
    acting = not env.done and env.current_player >= 0
    mask = env.action_masks()
    assert mask.any() == acting, (env.pending, env.current_player)
    if acting:
        assert env.current_player == env.pending.seat
    assert len(env.stack) < 64
    held_tids = []
    for s in env.ships:
        assert len(s.holds) == N_HOLDS
        for h in s.holds + ([s.sixth] if s.sixth is not None else []):
            assert 0 <= h[1] <= MAX_HOLD
            assert (h[1] == 0) == (h[0] == Res.EMPTY), h
        assert (s.sixth is not None) == s.has(Power.SIXTH)
        cards = s.hand + s.deck + s.discard + ([s.chosen] if s.chosen >= 0 and not s.revealed else [])
        assert sorted(cards) == sorted(DECK)
        assert s.lap in (-1, 0, 1)
        assert s.finished == ((s.node, s.lap) == (tr.pr, 1))
        held_tids += s.hidden + [POWER_TID[p] for p in Power if s.has(p)]
        for t in s.hidden:
            assert TREASURES[t][0] is None and env.known[t] >> s.seat & 1
    assert len(env.pile) == sum(env.lair_token.values())
    assert sorted(env.pile + env.removed + held_tids) == list(range(len(TREASURES)))
    if env.done:
        assert any(s.finished for s in env.ships) or env.round >= 50
    obs = env.observation
    assert env.observation_space.contains(obs), 'observation outside its space'
    if acting:
        assert np.array_equal(obs['mask'] > 0, mask)


def _biased_choice(env, rng):
    """Random, but seek battles and loot so that rare rules get exercised."""
    legal = env.legal_actions()
    if env.pending.kind == Phase.REWARD and rng.random() < 0.8:
        loot = [a for a in legal if a >= A_STEAL_HOLD]
        return rng.choice(loot[:-1] or loot)
    return rng.choice(legal)


def _play(env, rng, biased=False):
    seen = set()
    totals = [0.0] * env.n_players
    while not env.done:
        if env.current_player == -1:
            a = -1
        else:
            seen.add(Phase(env.pending.kind))
            a = _biased_choice(env, rng) if biased else rng.choice(env.legal_actions())
        _, r, _, _, _ = env.step(a)
        totals = [x + y for x, y in zip(totals, r)]
        check_invariants(env)
    assert all(abs(a - b) < 1e-9 for a, b in zip(totals, env.terminal_rewards))
    return seen


def test_fuzz_all_player_counts():
    seen = set()
    for n in (3, 4, 5, 6):
        for seed in range(8):
            env = JamaicaEnv(n_players=n, pause_between_turns=seed % 2 == 0)
            env.reset(seed=seed)
            check_invariants(env)
            seen |= _play(env, random.Random(100 * n + seed), biased=seed % 3 == 0)
    missing = set(Phase) - seen - {Phase.TURN_PAUSE, Phase.DONE}
    assert not missing, f'phases never reached: {missing}'


def test_fuzz_with_powers_everywhere():
    """Deal the four powers to players at the start: Sabre / Beth / Map / 6th Hold paths."""
    for seed in range(6):
        env = JamaicaEnv(n_players=4, pause_between_turns=False)
        env.reset(seed=seed)
        for p, tid in POWER_TID.items():
            if tid in env.pile:
                env.pile.remove(tid)
            elif tid in env.removed:
                env.removed.remove(tid)
            else:
                continue
            seat = int(p) % 4
            env.ships[seat].powers |= 1 << int(p)
            if p == Power.SIXTH:
                env.ships[seat].sixth = [Res.EMPTY, 0]
        # keep pile size == lair tokens: remove tokens from the last lairs
        extra = sum(env.lair_token.values()) - len(env.pile)
        for node in reversed(env.track.lairs):
            if extra <= 0:
                break
            if env.lair_token[node]:
                env.lair_token[node] = False
                extra -= 1
        check_invariants(env)
        _play(env, random.Random(seed), biased=True)
