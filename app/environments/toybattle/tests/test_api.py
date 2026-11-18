"""The GBEnv contract used by train.py / test.py / play.py / the self-play wrapper."""

import copy
import random

import numpy as np

from utils.register import get_environment

from ..envs.constants import A_DRAW, N_ACTIONS, Phase
from ..envs.terrains import TERRAIN_KEYS, TERRAINS
from ..envs.toybattle import BASE_F, GLOB_F, ToyBattleAllTerrainsEnv, ToyBattleEnv
from .helpers import make_env, random_play


def test_spaces_and_reset():
    env = ToyBattleEnv()
    obs, info = env.reset(seed=0)
    assert env.name == 'toybattle' and env.n_players == 2 and env.terrain_key == 'castle'
    assert env.action_space.n == N_ACTIONS == 161
    assert env.observation_space.contains(obs)
    assert obs['glob'].shape == (GLOB_F,) and obs['bases'].shape == (15, BASE_F)
    assert info == {'next_step_no_action': False}
    assert env.current_player in (0, 1) and env.action_masks().any()
    assert (obs['mask'] == env.action_masks()).all()


def test_every_terrain_plays_to_the_end():
    phases = set()
    reasons = set()
    for g, key in enumerate(TERRAIN_KEYS * 6):
        env = make_env(key, seed=g)
        rng = random.Random(g)
        while not env.done:
            phases.add(env.phase)
            actions = env.legal_actions()
            assert actions, (key, g)
            for a in actions:
                assert env.describe_action(a)
            obs, reward, done, truncated, info = env.step(rng.choice(actions))
            assert env.observation_space.contains(obs), (key, g)
            assert info == {'next_step_no_action': False} and not truncated
            assert len(reward) == 2 and reward[0] == -reward[1]
        reasons.add(env.end_reason.split(',')[0])
        assert env.turns_taken > 0 and 0 <= env.current_player < 2
    assert phases == set(Phase)
    assert reasons == {'H.Q. captured', 'medals objective', 'no action left'}


def test_step_after_the_end_raises():
    env = make_env(seed=3)
    random_play(env, random.Random(3))
    assert env.done and not env.action_masks().any()
    try:
        env.step(A_DRAW)
    except Exception:
        return
    raise AssertionError('stepping a finished game should raise')


def test_terrain_choice():
    assert ToyBattleEnv(terrain='pool').reset(seed=1) and ToyBattleEnv(terrain='pool').terrain_key == 'pool'
    env = ToyBattleEnv(terrains=('pool', 'jungle'))
    seen = set()
    for seed in range(20):
        env.reset(seed=seed)
        seen.add(env.terrain_key)
    assert seen == {'pool', 'jungle'}
    env.reset(seed=0, options={'terrain': 'station'})
    assert env.terrain is TERRAINS['station']
    for bad in (dict(terrain='moon'), dict(terrains=('castle', 'moon'))):
        try:
            ToyBattleEnv(**bad)
        except ValueError:
            continue
        raise AssertionError('unknown terrain accepted')


def test_registered_env_draws_every_terrain():
    cls = get_environment('toybattle')
    assert cls is ToyBattleAllTerrainsEnv
    env = cls()
    seen = set()
    for seed in range(80):
        obs, _ = env.reset(seed=seed)
        seen.add(env.terrain_key)
        assert obs['glob'][:8].sum() == 1 and obs['glob'][env.terrain.index] == 1
    assert seen == set(TERRAIN_KEYS)
    assert cls(terrain='clouds').terrain_pool == ('clouds',)
    assert cls(['Ann', 'Bob']).player_names == ['Ann', 'Bob']


def test_same_seed_same_game():
    def trace(seed):
        env = ToyBattleAllTerrainsEnv()
        env.reset(seed=seed)
        rng = random.Random(seed)
        out = [env.terrain_key, env.first_player]
        while not env.done:
            a = rng.choice(env.legal_actions())
            _, r, _, _, _ = env.step(a)
            out.append((a, tuple(r), tuple(map(tuple, env.racks))))
        return out
    assert trace(5) == trace(5)
    assert trace(5) != trace(6)


def test_save_and_restore():
    env = make_env('cemetery', seed=4)
    rng = random.Random(4)
    for _ in range(25):
        env.step(rng.choice(env.legal_actions()))
    data = env.save_state()
    other = ToyBattleEnv()                      # another terrain: the state brings its own
    other.restore_state(data)
    assert other.terrain_key == 'cemetery' and other.stacks == env.stacks
    for key, value in env.observation.items():
        assert np.array_equal(value, other.observation[key]), key
    while not env.done:                         # the generator state is saved too
        a = rng.choice(env.legal_actions())
        assert env.step(a)[1] == other.step(a)[1]
    assert other.done and other.racks == env.racks and other.event_log == env.event_log


def test_deepcopy_is_independent():
    env = make_env('jungle', seed=8)
    rng = random.Random(8)
    for _ in range(12):
        env.step(rng.choice(env.legal_actions()))
    clone = copy.deepcopy(env)
    assert clone.terrain is env.terrain
    before = copy.deepcopy(env.stacks), copy.deepcopy(env.racks), list(env.event_log)
    random_play(clone, random.Random(1))
    assert clone.done and not env.done
    assert (env.stacks, env.racks, env.event_log) == before


def test_names_in_the_log():
    env = ToyBattleEnv(player_names=['Ann', 'Bob'])
    env.reset(seed=1)
    text = env.log_lines(-1)[0][1]
    assert 'Castle Field' in text and ('Ann (Blue)' in text or 'Bob (Red)' in text)
