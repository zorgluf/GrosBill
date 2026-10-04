"""One network for every player count: the count drawn at every reset.

`JamaicaEnv(player_counts=...)` draws the count of each new game (or takes
``reset(options={'n_players': n})``); `JamaicaAllCountsEnv`, what `-e jamaica`
builds for training, does it for 3-6 players unless a count is given.
"""

import random

from ..envs.constants import DEFAULT_PLAYERS, N_ACTIONS, PLAYER_COUNTS
from ..envs.jamaica import JamaicaAllCountsEnv, JamaicaEnv
from .test_fuzz import _play, check_invariants


def _raises(call) -> bool:
    try:
        call()
    except ValueError:
        return True
    return False


def test_every_count_is_drawn_and_played():
    """Successive resets cover every count; each game runs to the end with the
    invariants of the fuzz tests, the spaces never change."""
    env = JamaicaEnv(player_counts=PLAYER_COUNTS, pause_between_turns=False)
    assert env.player_counts == (3, 4, 5, 6) and env.n_players == DEFAULT_PLAYERS
    spaces = (env.action_space, env.observation_space)
    counts = []
    env.reset(seed=11)
    for game in range(16):
        if game:
            env.reset()
        n = env.n_players
        counts.append(n)
        assert env.player_names == [f'Player {i + 1}' for i in range(n)]
        assert len(env.ships) == n and len(env._step_rewards) == n
        assert abs(env.observation['glob'][63] - n / 6) < 1e-6      # the network sees the count
        assert (env.action_space, env.observation_space) == spaces
        assert env.action_space.n == N_ACTIONS
        check_invariants(env)
        _play(env, random.Random(game))
        assert len(env.terminal_rewards) == n
    assert set(counts) == set(PLAYER_COUNTS), counts


def test_requested_count_and_errors():
    env = JamaicaEnv(player_counts=(3, 5))
    env.reset(seed=1, options={'n_players': 5})
    assert env.n_players == 5 and len(env.ships) == 5
    env.reset(seed=1, options={'n_players': 3})
    assert env.n_players == 3 and env.player_names == ['Player 1', 'Player 2', 'Player 3']
    assert _raises(lambda: env.reset(options={'n_players': 4}))         # not in the set
    # a fixed-count env keeps its count
    fixed = JamaicaEnv(n_players=4)
    fixed.reset(seed=0, options={'n_players': 4})
    assert _raises(lambda: fixed.reset(options={'n_players': 5}))
    # the count is either fixed or drawn, never both
    assert _raises(lambda: JamaicaEnv(n_players=4, player_counts=PLAYER_COUNTS))
    assert _raises(lambda: JamaicaEnv(player_names=['a', 'b', 'c'], player_counts=(3,)))
    assert _raises(lambda: JamaicaEnv(player_counts=(3, 7)))


def test_seeded_draws_are_reproducible():
    def counts(seed):
        env = JamaicaEnv(player_counts=PLAYER_COUNTS)
        env.reset(seed=seed)
        out = [env.n_players]
        for _ in range(9):
            env.reset()
            out.append(env.n_players)
        return out
    assert counts(5) == counts(5)


def test_all_counts_env_is_the_registered_jamaica():
    """`-e jamaica` trains on every count; given a count it plays that one
    (`test.py`, `play.py`)."""
    from utils.register import get_environment
    cls = get_environment('jamaica')
    assert cls is JamaicaAllCountsEnv
    env = cls()
    assert env.name == 'jamaica' and env.player_counts == PLAYER_COUNTS
    for n in PLAYER_COUNTS:
        fixed = cls(n_players=n)
        assert fixed.player_counts is None
        for seed in range(3):
            fixed.reset(seed=seed)
            assert fixed.n_players == n
    assert cls(player_names=['a', 'b', 'c']).player_counts is None
    # the plain env keeps its fixed default
    assert JamaicaEnv().player_counts is None and JamaicaEnv().n_players == 4
