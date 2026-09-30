"""Policy for Jamaica (`jamaica`): MLP trunk + pointer-style action scorer.

Design and training recipe in `README.md` next to this file. In short:

* **Trunk**: a shared player encoder (every row of `players`, masked by the
  presence flag) pooled as [me, next seat, previous seat, mean and max of the
  opponents] (so seats 5-6 never carry untrained weights at 4 players), an
  egocentric track encoder, the context vector and my hand -> MLP -> latent.
* **Policy**: every action gets a logit from one shared scorer fed with its
  engine-computed consequence features (`action_feats`), its legality bit and
  an embedding of its identity (family, sub-kind, card symbols), conditioned
  on the latent; plus a per-action linear base logit for the families whose
  slots are not interchangeable. Last layers start at zero -> uniform policy
  over the legal actions.
* **Value**: MLP on the latent.

SB3 plumbing (same pattern as `models/smallw/models.py`): the features
extractor returns [latent | action_feats | mask] flattened, `JamaicaLatent`
replaces the MlpExtractor, `PointerScorer` replaces the default action head.
"""

from typing import Callable

import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environments.jamaica.envs.constants import (ACTION_ESYM, ACTION_FAMILY, ACTION_MSYM,
                                                 ACTION_SUB, MAX_PLAYERS, N_ACTIONS, N_CODES,
                                                 N_FAMILIES, N_SUBS, N_SYMS, SYMMETRIC_FAMILIES)
from environments.jamaica.envs.features import K_FEATS, N_GLOB, N_TRACK, P_COLS, TRACK_COLS

LATENT = 128
HIDDEN = 256
PLAYER_EMB = 32
TRACK_EMB = 8
ID_EMB = 16
SCORER_H1 = 64
SCORER_H2 = 32
VALUE_H = (128, 64)


def _zero(layer: th.nn.Linear) -> th.nn.Linear:
    th.nn.init.zeros_(layer.weight)
    th.nn.init.zeros_(layer.bias)
    return layer


class JamaicaExtractor(BaseFeaturesExtractor):
    """Dict observation -> [latent (LATENT) | action_feats (A*K) | mask (A)]."""

    def __init__(self, observation_space: spaces.Dict):
        assert observation_space['players'].shape == (MAX_PLAYERS, P_COLS)
        assert observation_space['action_feats'].shape == (N_ACTIONS, K_FEATS)
        super().__init__(observation_space, features_dim=LATENT + N_ACTIONS * (K_FEATS + 1))
        self.player_enc = th.nn.Sequential(th.nn.Linear(P_COLS, PLAYER_EMB), th.nn.ReLU())
        self.track_enc = th.nn.Sequential(th.nn.Linear(TRACK_COLS, TRACK_EMB), th.nn.ReLU())
        trunk_in = N_GLOB + N_CODES + 5 * PLAYER_EMB + N_TRACK * TRACK_EMB
        self.trunk = th.nn.Sequential(
            th.nn.Linear(trunk_in, HIDDEN), th.nn.ReLU(),
            th.nn.Linear(HIDDEN, LATENT), th.nn.ReLU(),
        )

    def forward(self, obs) -> th.Tensor:
        players = obs['players'].float()
        batch = players.shape[0]
        present = players[..., 0:1]                                   # (B, P, 1)
        pe = self.player_enc(players) * present                       # (B, P, E)
        n = present.sum(dim=(1, 2)).long().clamp(min=2)              # seats in the game
        prev = pe[th.arange(batch, device=pe.device), n - 1]          # seat before me
        opp, opp_mask = pe[:, 1:], present[:, 1:]
        mean = opp.sum(1) / opp_mask.sum(1).clamp(min=1)
        top = opp.max(1).values                                       # ReLU >= 0: absent rows are 0
        track = obs['track'].float()
        te = self.track_enc(track) * track[..., 0:1]
        x = th.cat([obs['glob'].float(), obs['hand'].float(), pe[:, 0], pe[:, 1], prev, mean, top,
                    te.flatten(1)], dim=1)
        latent = self.trunk(x)
        return th.cat([latent, obs['action_feats'].float().flatten(1), obs['mask'].float()], dim=1)


class JamaicaLatent(th.nn.Module):
    """Replaces SB3's MlpExtractor: actor = identity, critic = MLP(latent)."""

    def __init__(self, features_dim: int):
        super().__init__()
        self.latent_dim_pi = features_dim
        self.latent_dim_vf = VALUE_H[-1]
        self.value = th.nn.Sequential(
            th.nn.Linear(LATENT, VALUE_H[0]), th.nn.ReLU(),
            th.nn.Linear(VALUE_H[0], VALUE_H[1]), th.nn.ReLU(),
        )

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return features

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value(features[:, :LATENT])


class PointerScorer(th.nn.Module):
    """The N_ACTIONS logits from one scorer shared by every action."""

    def __init__(self):
        super().__init__()
        self.fam = th.nn.Embedding(N_FAMILIES, ID_EMB)
        self.sub = th.nn.Embedding(N_SUBS, ID_EMB)
        self.msym = th.nn.Embedding(N_SYMS + 1, ID_EMB)
        self.esym = th.nn.Embedding(N_SYMS + 1, ID_EMB)
        self.register_buffer('fam_idx', th.tensor(ACTION_FAMILY), persistent=False)
        self.register_buffer('sub_idx', th.tensor(ACTION_SUB), persistent=False)
        self.register_buffer('msym_idx', th.tensor(ACTION_MSYM), persistent=False)
        self.register_buffer('esym_idx', th.tensor(ACTION_ESYM), persistent=False)
        base_on = [0.0 if f in [int(x) for x in SYMMETRIC_FAMILIES] else 1.0 for f in ACTION_FAMILY]
        self.register_buffer('base_on', th.tensor(base_on), persistent=False)
        self.w1 = th.nn.Linear(K_FEATS + 1 + ID_EMB, SCORER_H1)
        self.wq = th.nn.Linear(LATENT, SCORER_H1, bias=False)
        self.w2 = th.nn.Linear(SCORER_H1, SCORER_H2)
        self.out = _zero(th.nn.Linear(SCORER_H2, 1))
        self.base = _zero(th.nn.Linear(LATENT, N_ACTIONS))

    def forward(self, latent_pi: th.Tensor) -> th.Tensor:
        a, k = N_ACTIONS, K_FEATS
        lat = latent_pi[:, :LATENT]
        feats = latent_pi[:, LATENT:LATENT + a * k].view(-1, a, k)
        mask = latent_pi[:, LATENT + a * k:].unsqueeze(-1)
        emb = (self.fam(self.fam_idx) + self.sub(self.sub_idx)
               + self.msym(self.msym_idx) + self.esym(self.esym_idx))            # (A, ID_EMB)
        x = th.cat([feats, mask, emb.unsqueeze(0).expand(lat.shape[0], -1, -1)], dim=-1)
        h = th.relu(self.w1(x) + self.wq(lat).unsqueeze(1))
        h = th.relu(self.w2(h))
        return self.out(h).squeeze(-1) + self.base(lat) * self.base_on


class CustomPolicy(MaskableActorCriticPolicy):
    """MLP trunk + pointer scorer (see the module docstring and README.md)."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        kwargs.setdefault('features_extractor_class', JamaicaExtractor)
        kwargs.setdefault('net_arch', [])
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = JamaicaLatent(self.features_extractor.features_dim)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # swap the default Linear(latent_pi, n_actions) head for the pointer
        # scorer, then re-create the optimizer so it covers the new parameters
        assert self.action_space.n == N_ACTIONS
        self.action_net = PointerScorer()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
