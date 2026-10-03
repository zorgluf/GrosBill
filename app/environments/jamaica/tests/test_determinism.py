"""Seeded replays, deepcopy independence and cost."""

import copy
import random
import time

import numpy as np

from .helpers import make_env


def _trajectory(seed: int, n: int = 4):
    env = make_env(n, seed=seed)
    rng = random.Random(seed)
    out = []
    while not env.done:
        a = -1 if env.current_player == -1 else rng.choice(env.legal_actions())
        obs, r, _, _, _ = env.step(a)
        out.append((a, tuple(r), obs['glob'].tobytes(), obs['action_feats'].tobytes()))
    return out, env.final_scores


def test_same_seed_same_game():
    assert _trajectory(11) == _trajectory(11)
    assert _trajectory(11)[0] != _trajectory(12)[0]


def test_deepcopy_is_independent():
    env = make_env(4, seed=13)
    rng = random.Random(13)
    for _ in range(60):
        env.step(-1 if env.current_player == -1 else rng.choice(env.legal_actions()))
    snapshot = copy.deepcopy(env.observation)
    state = [(s.node, s.lap, [list(h) for h in s.holds], list(s.hand)) for s in env.ships]
    clone = copy.deepcopy(env)
    crng = random.Random(99)
    while not clone.done:
        clone.step(-1 if clone.current_player == -1 else crng.choice(clone.legal_actions()))
    assert all(np.array_equal(snapshot[k], env.observation[k]) for k in snapshot)
    assert state == [(s.node, s.lap, [list(h) for h in s.holds], list(s.hand)) for s in env.ships]
    assert clone.track is env.track
    env.step(-1 if env.current_player == -1 else rng.choice(env.legal_actions()))


def test_deepcopy_is_cheap():
    env = make_env(6, seed=14)
    rng = random.Random(14)
    for _ in range(200):
        if env.done:
            break
        env.step(-1 if env.current_player == -1 else rng.choice(env.legal_actions()))
    t0 = time.perf_counter()
    for _ in range(50):
        copy.deepcopy(env)
    per_copy = (time.perf_counter() - t0) / 50
    assert per_copy < 0.02, f'deepcopy takes {per_copy * 1000:.1f} ms'
