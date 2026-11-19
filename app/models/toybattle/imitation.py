"""Supervised check of the Toy Battle network before any PPO run ("gate 1").

The teacher is the one-ply greedy player (`utils.agents.GreedyAgent`): in every
position it knows which legal actions pay most at once (a win, medals). The
positions come from games between two such players that play at random 30% of
the time. Two questions, each answered against a flat MLP on the same data:

* does the network find the teacher's moves in the *decisive* positions (those
  where some legal actions are better than others) and predict who wins?
* does what it learned carry over to a terrain it never saw?

    python3 -m models.toybattle.imitation generate --out /tmp/tb_data --games 330
    python3 -m models.toybattle.imitation train --data /tmp/tb_data --model graph
    python3 -m models.toybattle.imitation train --data /tmp/tb_data --model mlp --no-helpers

`train` prints one JSON line of metrics per evaluation set:
`seen` = held-out games of the training terrains, `unseen` = the held-out terrain.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time

import numpy as np
import torch as th

from environments.toybattle.envs.constants import (B_MY_GAIN, B_THEIR_GAIN, BASE_F, G_SCALARS, GLOB_F, HQ_F,
                                                   MAX_BASES, MAX_HQ, MAX_REGIONS, N_ACTIONS, REGION_F)
from environments.toybattle.envs.terrains import TERRAIN_KEYS
from environments.toybattle.envs.toybattle import ToyBattleEnv
from utils.agents import GreedyAgent

KEYS = ('glob', 'bases', 'adj', 'regions', 'hqs', 'mask')
EPSILON = 0.3


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def play(job):
    """One game -> rows (observation, teacher's best actions, flags)."""
    key, seed = job
    env = ToyBattleEnv(terrain=key)
    env.reset(seed=seed)
    rng = random.Random(seed)
    rows, movers = [], []
    while not env.done:
        obs = env.observation
        scores = GreedyAgent.scores(env)
        top = max(scores.values())
        best = [a for a, s in scores.items() if s >= top - 1e-9]
        target = np.zeros(N_ACTIONS, dtype=bool)
        target[best] = True
        rows.append((obs, target, len(best) < len(scores), top > 0.45))
        movers.append(env.current_player)
        env.apply(rng.choice(list(scores)) if rng.random() < EPSILON else rng.choice(best))
    outcome = [0 if env.winner_player is None else 1 if env.winner_player == p else -1 for p in movers]
    return key, seed, rows, outcome


def generate(args):
    os.makedirs(args.out, exist_ok=True)
    jobs = [(key, 100000 * i + g) for i, key in enumerate(TERRAIN_KEYS) for g in range(args.games)]
    start = time.time()
    by_key = {key: [] for key in TERRAIN_KEYS}
    with mp.Pool(args.procs) as pool:
        for n, (key, seed, rows, outcome) in enumerate(pool.imap_unordered(play, jobs, chunksize=4)):
            by_key[key].append((seed, rows, outcome))
            if (n + 1) % 200 == 0:
                print(f'{n + 1}/{len(jobs)} games, {time.time() - start:.0f} s', flush=True)
    for key, games in by_key.items():
        games.sort(key=lambda g: g[0])
        data = {k: np.stack([row[0][k] for _, rows, _ in games for row in rows]).astype(np.float16) for k in KEYS}
        data['best'] = np.stack([row[1] for _, rows, _ in games for row in rows])
        data['decisive'] = np.array([row[2] for _, rows, _ in games for row in rows])
        data['win_now'] = np.array([row[3] for _, rows, _ in games for row in rows])
        data['outcome'] = np.array([z for _, _, outcome in games for z in outcome], dtype=np.int8)
        data['game'] = np.array([i for i, (_, rows, _) in enumerate(games) for _ in rows], dtype=np.int32)
        np.savez_compressed(os.path.join(args.out, f'{key}.npz'), **data)
        print(f'{key}: {len(games)} games, {len(data["game"])} positions, '
              f'{data["decisive"].mean():.2f} decisive, {data["win_now"].mean():.3f} with a winning move')


def load(folder, keys):
    parts = [np.load(os.path.join(folder, f'{key}.npz')) for key in keys]
    out = {}
    for name in parts[0].files:
        arrays = [p[name] for p in parts]
        if name == 'game':                              # make the game ids unique across terrains
            arrays = [a + 100000 * i for i, a in enumerate(arrays)]
        out[name] = np.concatenate(arrays)
    return out


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #

class FlatMLP(th.nn.Module):
    """The baseline: every observation entry in one vector, one output per action."""

    def __init__(self, hidden=(512, 512, 256)):
        super().__init__()
        size = GLOB_F + MAX_BASES * (BASE_F + MAX_BASES) + MAX_REGIONS * REGION_F + 2 * MAX_HQ * HQ_F + N_ACTIONS
        layers = []
        for h in hidden:
            layers += [th.nn.Linear(size, h), th.nn.ReLU()]
            size = h
        self.body = th.nn.Sequential(*layers)
        self.policy = th.nn.Linear(size, N_ACTIONS)
        self.value = th.nn.Linear(size, 1)

    def forward(self, obs):
        x = self.body(th.cat([obs[k].float().flatten(1) for k in KEYS], dim=1))
        return self.policy(x), self.value(x).squeeze(-1)


def build(kind):
    if kind == 'graph':
        from models.toybattle.models import ToyBattleNet
        return ToyBattleNet()
    return FlatMLP()


# --------------------------------------------------------------------------- #
# training and metrics
# --------------------------------------------------------------------------- #

def batch_of(data, index, helpers, device):
    obs = {k: th.from_numpy(data[k][index].astype(np.float32)).to(device) for k in KEYS}
    if not helpers:
        obs['bases'][:, :, B_MY_GAIN] = 0
        obs['bases'][:, :, B_THEIR_GAIN] = 0
    best = th.from_numpy(data['best'][index]).to(device)
    outcome = th.from_numpy(data['outcome'][index].astype(np.float32)).to(device)
    return obs, best, outcome


def losses(model, obs, best, outcome):
    logits, value = model(obs)
    logp = th.log_softmax(logits.masked_fill(obs['mask'] < 0.5, -1e9), dim=1)
    policy = -th.logsumexp(logp.masked_fill(~best, -1e9), dim=1).mean()       # mass on the teacher's moves
    return policy, th.nn.functional.mse_loss(value, outcome), logp, value


@th.no_grad()
def evaluate(model, data, helpers, device, batch=1024):
    model.eval()
    hit, lead_hit, value_hit, value_se, chance = [], [], [], [], []
    for start in range(0, len(data['game']), batch):
        index = np.arange(start, min(start + batch, len(data['game'])))
        obs, best, outcome = batch_of(data, index, helpers, device)
        _, _, logp, value = losses(model, obs, best, outcome)
        hit.append(best.gather(1, logp.argmax(dim=1, keepdim=True)).squeeze(1).cpu().numpy())
        chance.append((best.sum(1) / obs['mask'].sum(1)).cpu().numpy())
        value_hit.append(((value > 0) == (outcome > 0)).cpu().numpy())
        value_se.append(((value - outcome) ** 2).cpu().numpy())
        lead = obs['glob'][:, G_SCALARS + 1] - obs['glob'][:, G_SCALARS + 2]
        lead_hit.append(((lead >= 0) == (outcome > 0)).cpu().numpy())
    hit, chance = np.concatenate(hit), np.concatenate(chance)
    decisive, win_now = data['decisive'], data['win_now']
    model.train()
    return {
        'positions': int(len(hit)),
        'decisive': round(float(decisive.mean()), 3),
        'teacher_move_decisive': round(float(hit[decisive].mean()), 4),      # the headline number
        'teacher_move_chance': round(float(chance[decisive].mean()), 4),
        'winning_move_found': round(float(hit[win_now].mean()), 4) if win_now.any() else None,
        'value_sign': round(float(np.concatenate(value_hit).mean()), 4),
        'value_sign_medal_lead': round(float(np.concatenate(lead_hit).mean()), 4),
        'value_mse': round(float(np.concatenate(value_se).mean()), 4),
    }


def train(args):
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    th.set_num_threads(args.threads)
    device = th.device(args.device)
    helpers = not args.no_helpers
    seen_keys = [k for k in TERRAIN_KEYS if k != args.holdout]
    seen, unseen = load(args.data, seen_keys), load(args.data, [args.holdout])
    games = np.unique(seen['game'])
    test_games = set(np.random.permutation(games)[:len(games) // 7].tolist())
    is_test = np.array([g in test_games for g in seen['game']])
    train_set = {k: v[~is_test] for k, v in seen.items()}
    test_set = {k: v[is_test] for k, v in seen.items()}
    if args.limit:
        train_set = {k: v[:args.limit] for k, v in train_set.items()}

    model = build(args.model).to(device)
    n = len(train_set['game'])
    steps = args.epochs * (n // args.batch)
    opt = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = th.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    tag = dict(model=args.model, helpers=helpers, holdout=args.holdout,
               params=sum(p.numel() for p in model.parameters()), train_positions=n, steps=steps)
    print(json.dumps(tag), flush=True)
    start, step = time.time(), 0
    for epoch in range(args.epochs):
        order = np.random.permutation(n)
        for i in range(n // args.batch):
            index = np.sort(order[i * args.batch:(i + 1) * args.batch])
            obs, best, outcome = batch_of(train_set, index, helpers, device)
            policy, value, _, _ = losses(model, obs, best, outcome)
            opt.zero_grad()
            (policy + 0.5 * value).backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f'  step {step}/{steps} policy {policy.item():.3f} value {value.item():.3f} '
                      f'{time.time() - start:.0f} s', flush=True)
        for name, data in (('seen', test_set), ('unseen', unseen)):
            print(json.dumps({**tag, 'epoch': epoch + 1, 'set': name, 'seconds': round(time.time() - start),
                              **evaluate(model, data, helpers, device)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    g = sub.add_parser('generate')
    g.add_argument('--out', required=True)
    g.add_argument('--games', type=int, default=330, help='games per terrain')
    g.add_argument('--procs', type=int, default=os.cpu_count())
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--model', choices=('graph', 'mlp'), default='graph')
    t.add_argument('--holdout', default='caribbean', help='terrain kept out of the training set')
    t.add_argument('--no-helpers', action='store_true', help='hide the medal-gain columns')
    t.add_argument('--epochs', type=int, default=3)
    t.add_argument('--batch', type=int, default=256)
    t.add_argument('--lr', type=float, default=3e-4)
    t.add_argument('--limit', type=int, default=0, help='use only that many training positions')
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--threads', type=int, default=os.cpu_count())
    t.add_argument('--device', default='cpu')
    args = parser.parse_args()
    (generate if args.command == 'generate' else train)(args)


if __name__ == '__main__':
    main()
