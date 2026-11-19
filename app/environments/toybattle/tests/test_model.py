"""The policy network (`models/toybattle/models.py`): shapes, start, invariances."""

import random

import numpy as np
import torch as th

from models.toybattle.models import (CustomPolicy, MAX_HOPS, N_TOKENS, ToyBattleNet, hop_distances)

from ..envs.constants import A_PLACE, A_TARGET, A_TYPE, MAX_BASES, N_ACTIONS, N_SLOTS, N_TYPES
from ..envs.terrains import TERRAIN_KEYS, TERRAINS
from ..envs.toybattle import ToyBattleAllTerrainsEnv
from .helpers import make_env


def _batch(n_games=3, seed=0):
    env = ToyBattleAllTerrainsEnv()
    rng = random.Random(seed)
    rows = []
    for g in range(n_games):
        env.reset(seed=seed + g)
        while not env.done:
            rows.append(env.observation)
            env.step(rng.choice(env.legal_actions()))
    return {k: th.tensor(np.stack([r[k] for r in rows])) for k in rows[0]}


def _awake(seed=0):
    """A network whose zero-initialised layers are randomised (so outputs differ)."""
    th.manual_seed(seed)
    net = ToyBattleNet()
    for p in net.parameters():
        if float(p.detach().abs().sum()) == 0:
            th.nn.init.normal_(p, std=0.1)
    return net.eval()


def test_shapes_and_uniform_start():
    th.manual_seed(0)
    net = ToyBattleNet()
    batch = _batch()
    with th.no_grad():
        logits, value = net(batch)
    n = batch['glob'].shape[0]
    assert logits.shape == (n, N_ACTIONS) and value.shape == (n,)
    assert float(logits.abs().max()) == 0           # uniform over the legal actions
    assert N_TOKENS == 42 and 500_000 < sum(p.numel() for p in net.parameters()) < 1_000_000


def test_hop_distances():
    for key, T in TERRAINS.items():
        adj = th.zeros(1, MAX_BASES, MAX_BASES)
        for b, around in enumerate(T.adj):
            adj[0, b, list(around)] = 1
        found = hop_distances(adj)[0]
        for start in range(T.n_bases):
            dist, todo = {start: 0}, [start]
            while todo:
                b = todo.pop(0)
                for n in T.adj[b]:
                    if n not in dist:
                        dist[n] = dist[b] + 1
                        todo.append(n)
            for b in range(T.n_bases):
                assert int(found[start, b]) == min(dist[b], MAX_HOPS), (key, start, b)
        assert (found[T.n_bases:, :T.n_bases] == MAX_HOPS).all()         # padding is far from everything


def _shuffled(obs, perm):
    """The same positions with the base slots renumbered by `perm`."""
    out = dict(obs)
    out['bases'] = obs['bases'][:, perm]
    out['adj'] = obs['adj'][:, perm][:, :, perm]
    regions = obs['regions'].clone()
    regions[:, :, :MAX_BASES] = obs['regions'][:, :, :MAX_BASES][:, :, perm]
    out['regions'] = regions
    hqs = obs['hqs'].clone()
    hqs[:, :, 2 + N_TYPES:] = obs['hqs'][:, :, 2 + N_TYPES:][:, :, perm]
    out['hqs'] = hqs
    mask = obs['mask'].clone()
    place = obs['mask'][:, A_PLACE:A_TARGET].view(-1, N_TYPES, N_SLOTS).clone()
    place[:, :, :MAX_BASES] = place[:, :, :MAX_BASES][:, :, perm]
    mask[:, A_PLACE:A_TARGET] = place.reshape(-1, N_TYPES * N_SLOTS)
    mask[:, A_TARGET:A_TYPE] = obs['mask'][:, A_TARGET:A_TYPE][:, perm]
    out['mask'] = mask
    return out


def test_base_numbering_does_not_matter():
    """Nothing is tied to a slot: renumbering the bases renumbers the logits."""
    net, batch = _awake(), _batch(seed=5)
    perm = th.randperm(MAX_BASES, generator=th.Generator().manual_seed(3))
    with th.no_grad():
        a, va = net(batch)
        b, vb = net(_shuffled(batch, perm))
    pa, pb = (x[:, A_PLACE:A_TARGET].view(-1, N_TYPES, N_SLOTS) for x in (a, b))
    assert float(a.abs().mean()) > 0.05
    assert th.allclose(pa[:, :, :MAX_BASES][:, :, perm], pb[:, :, :MAX_BASES], atol=1e-4)
    assert th.allclose(pa[:, :, MAX_BASES:], pb[:, :, MAX_BASES:], atol=1e-4)      # the H.Q. slots
    assert th.allclose(a[:, A_TARGET:A_TYPE][:, perm], b[:, A_TARGET:A_TYPE], atol=1e-4)
    assert th.allclose(a[:, :A_PLACE], b[:, :A_PLACE], atol=1e-4)
    assert th.allclose(a[:, A_TYPE:], b[:, A_TYPE:], atol=1e-4) and th.allclose(va, vb, atol=1e-4)


def test_padding_is_ignored():
    net = _awake(1)
    env = make_env('caribbean', seed=2)                 # 12 bases, 6 regions, 3 H.Q.: plenty of padding
    obs = {k: th.tensor(v).unsqueeze(0) for k, v in env.observation.items()}
    noisy = {k: v.clone() for k, v in obs.items()}
    n = env.terrain.n_bases
    noisy['bases'][:, n:, 1:] = 0.7                     # features of bases that do not exist
    noisy['regions'][:, len(env.terrain.regions):, MAX_BASES + 1:] = 0.7
    noisy['hqs'][:, 1, 1:2 + N_TYPES] = 0.7             # Red's second H.Q. does not exist
    with th.no_grad():
        a, va = net(obs)
        b, vb = net(noisy)
    legal = obs['mask'][0] > 0.5
    assert th.allclose(a[0, legal], b[0, legal], atol=1e-4) and th.allclose(va, vb, atol=1e-4)


def test_every_parameter_learns():
    th.manual_seed(2)
    net = ToyBattleNet()
    opt = th.optim.SGD(net.parameters(), lr=0.1)
    batch = _batch(n_games=1, seed=9)
    weights = th.randn(N_ACTIONS, generator=th.Generator().manual_seed(4))
    for _ in range(2):                                  # the second step reaches behind the zero layers
        logits, value = net(batch)
        loss = (logits * weights).sum(1).mean() + value.pow(2).mean() + value.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    missing = [name for name, p in net.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0]
    assert not missing, missing


def test_all_terrains_in_one_batch():
    net = _awake(3)
    rows = [make_env(key, seed=1).observation for key in TERRAIN_KEYS]
    batch = {k: th.tensor(np.stack([r[k] for r in rows])) for k in rows[0]}
    with th.no_grad():
        together, _ = net(batch)
        alone = th.cat([net({k: v[i:i + 1] for k, v in batch.items()})[0] for i in range(len(rows))])
    assert th.allclose(together, alone, atol=1e-4)


def test_sb3_policy():
    """The network as `train.py` builds it: legal actions only, uniform before any training."""
    from sb3_contrib import MaskablePPO
    from utils.register import get_network_arch
    assert get_network_arch('toybattle') is CustomPolicy
    env = ToyBattleAllTerrainsEnv()
    env.reset(seed=3)
    model = MaskablePPO(CustomPolicy, env, device='cpu', seed=0)
    plain = ToyBattleNet()
    assert sum(p.numel() for p in model.policy.parameters()) == sum(p.numel() for p in plain.parameters())
    rng = random.Random(3)
    for step in range(60):
        if env.done:
            env.reset(seed=step)
        mask = env.action_masks()
        action, _ = model.predict(env.observation, action_masks=mask, deterministic=False)
        assert mask[int(action)]
        if step % 20 == 0:
            obs, _ = model.policy.obs_to_tensor(env.observation)
            with th.no_grad():
                dist = model.policy.get_distribution(obs, action_masks=mask[None])
                value = model.policy.predict_values(obs)
            probs = dist.distribution.probs[0].numpy()
            assert np.allclose(probs[mask], 1 / mask.sum(), atol=1e-5) and probs[~mask].sum() < 1e-6
            assert value.shape == (1, 1)
        env.step(rng.choice(env.legal_actions()))
