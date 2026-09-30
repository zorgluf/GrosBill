"""The GBEnv contract used by train.py / test.py / play.py / the self-play wrapper."""

import random

from ..envs.constants import N_ACTIONS, Phase
from ..envs.jamaica import JamaicaEnv
from .helpers import make_env


def test_spaces_and_reset():
    env = JamaicaEnv()
    obs, info = env.reset(seed=0)
    assert env.action_space.n == N_ACTIONS
    assert env.observation_space.contains(obs)
    assert info == {'next_step_no_action': False}
    assert env.current_player >= 0 and env.action_masks().any()
    assert env.name == 'jamaica'


def test_every_legal_action_has_a_description():
    env = make_env(5, seed=21, pause=True)
    rng = random.Random(21)
    kinds = set()
    while not env.done:
        if env.current_player >= 0:
            kinds.add(env.pending.kind)
            for a in env.legal_actions():
                assert env.describe_action(a)
            env.step(rng.choice(env.legal_actions()))
        else:
            env.step(-1)
    assert Phase.CAPTAIN in kinds and Phase.CARD in kinds
    assert env.phase == Phase.DONE


def test_step_after_the_end_raises():
    env = make_env(3, seed=22)
    rng = random.Random(22)
    while not env.done:
        env.step(-1 if env.current_player == -1 else rng.choice(env.legal_actions()))
    assert env.winner_player is not None and 0 <= env.current_player < 3
    try:
        env.step(0)
    except Exception:
        return
    raise AssertionError('stepping a finished game should raise')


def test_names_in_the_log():
    env = JamaicaEnv(player_names=['Anne', 'Mary', 'John'])
    env.reset(seed=1)
    assert any('Captain' in t and env.player_names[env.captain] in t for _, t in env.log_lines(-1))
