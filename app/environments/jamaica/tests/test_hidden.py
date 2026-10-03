"""Hidden information never reaches the observation (or the log) of a seat."""

import random

import numpy as np

from ..envs.constants import Phase
from .helpers import make_env


def _same_obs(a, b) -> bool:
    return all(np.array_equal(a[k], b[k]) for k in a)


def _states(n_games=6, every=7):
    """Random mid-game states where a real player is to act."""
    for g in range(n_games):
        env = make_env(3 + g % 4, seed=40 + g)
        rng = random.Random(g)
        step = 0
        while not env.done:
            if env.current_player >= 0 and step % every == 0:
                yield env
            a = -1 if env.current_player == -1 else rng.choice(env.legal_actions())
            env.step(a)
            step += 1


def test_observation_invariant_under_redeterminize():
    import copy
    checked = 0
    for env in _states():
        pov = env.current_player
        before = copy.deepcopy(env.observation)
        clone = copy.deepcopy(env)
        clone.redeterminize(pov)
        assert _same_obs(before, clone.observation), f'leak in phase {Phase(env.pending.kind).name}'
        checked += 1
    assert checked > 50


def test_redeterminize_changes_what_is_hidden():
    import copy
    changed = 0
    for env in _states(n_games=3, every=11):
        pov = env.current_player
        clone = copy.deepcopy(env)
        clone.redeterminize(pov)
        for s, c in zip(env.ships, clone.ships):
            if s.seat != pov and sorted(s.hand) != sorted(c.hand):
                changed += 1
        # what pov knows is untouched
        assert env.ships[pov].hand == clone.ships[pov].hand
        assert env.ships[pov].hidden == clone.ships[pov].hidden
        assert [s.discard for s in env.ships] == [s.discard for s in clone.ships]
        assert [s.powers for s in env.ships] == [s.powers for s in clone.ships]
        assert [len(s.hidden) for s in env.ships] == [len(s.hidden) for s in clone.ships]
    assert changed > 0


def test_unrevealed_card_is_not_seen():
    import copy
    env = make_env(4, seed=5)
    rng = random.Random(5)
    env.step(rng.choice(env.legal_actions()))       # Captain decides
    nxt = env.current_player
    env.step(rng.choice(env.legal_actions()))       # next seat picks
    viewer = env.current_player
    obs = copy.deepcopy(env.observation)
    other = copy.deepcopy(env)
    s = other.ships[nxt]
    if s.hand:
        s.hand[0], s.chosen = s.chosen, s.hand[0]
        other._version += 1
    assert other.current_player == viewer
    assert _same_obs(obs, other.observation)


def test_private_log_lines():
    env = make_env(4, seed=6)
    rng = random.Random(6)
    while env.round < 4 and not env.done:
        env.step(-1 if env.current_player == -1 else rng.choice(env.legal_actions()))
    everyone = (1 << 4) - 1
    private = [(r, t, m) for r, t, m in env.event_log if m != everyone]
    assert private, 'the card picks should be private lines'
    for pov in range(4):
        lines = env.log_lines(pov)
        assert lines == [(r, t) for r, t, m in env.event_log if m >> pov & 1]
        assert len(lines) < len(env.event_log)
    assert len(env.log_lines(-1)) == len(env.event_log)
