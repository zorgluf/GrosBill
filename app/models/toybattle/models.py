"""Policy network for Toy Battle (`toybattle`): a board-agnostic graph transformer.

One network for the 8 terrains. Nothing in it is tied to a board: a base is a
token built from its own features (no learned position), and the board reaches
the network only as relations between tokens, read from the observation
(`adj`, the region membership, the H.Q. adjacency) and turned into a learned
attention bias per head. Every game on every terrain trains the same weights.

* **Tokens (42)**: 1 context (terrain, phase, medals, sizes), 8 troop types
  (my rack, pinned, discards, unseen tiles, what the mask allows), 15 bases,
  14 regions, 4 H.Q. (mine then the enemy's). Padded rows are never attended.
* **Trunk**: `LAYERS` pre-LN blocks, width `D_MODEL`; the attention bias is an
  embedding of the relation between two tokens: base-base hop distance (found
  here from `adj`), base in region, base next to my / their H.Q.
* **Policy**: pointer heads over the tokens. Place (troop, base) from the pair
  (troop token, base token), enemy H.Q. slots from the H.Q. tokens, base picks
  from the base token and the context, type picks from the troop token, draw
  and skip from the context. Last layers start at zero: uniform over the legal
  actions.
* **Value**: attention pooling over the tokens, with the context token.

The three parts (`Trunk`, `PolicyHeads`, `ValueHead`) are plain modules.
`ToyBattleNet` chains them (observation dict -> logits, value) for anything
that is not Stable-Baselines3 (`imitation.py`); `CustomPolicy` plugs the same
three into SB3: the trunk is the features extractor (it returns the tokens,
flattened, and their presence), the value head replaces the MlpExtractor's
critic and the pointer heads replace the action layer.
"""

from __future__ import annotations

from typing import Callable

import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environments.toybattle.envs.constants import (A_PLACE, A_TARGET, A_TYPE, B_MY_HQ, B_THEIR_HQ,
                                                   BASE_F, G_CLOCK, G_MY_DISCARD, G_MY_UNSEEN,
                                                   G_OPP_DISCARD, G_OPP_UNSEEN, G_PINNED, G_RACK,
                                                   GLOB_F, HQ_F, MAX_BASES, MAX_HQ, MAX_REGIONS,
                                                   N_ACTIONS, N_SLOTS, N_TYPES, REGION_F)

D_MODEL = 128
HEADS = 4
LAYERS = 4
FF = 256
HEAD_H = 96
VALUE_H = 256

# token layout
T_CTX = 0
T_TROOPS = 1
T_BASES = T_TROOPS + N_TYPES
T_REGIONS = T_BASES + MAX_BASES
T_HQS = T_REGIONS + MAX_REGIONS
N_TOKENS = T_HQS + 2 * MAX_HQ                     # 42

# relations between two tokens (attention bias)
R_NONE = 0
R_DIST = 1                                        # + min(hops, 5): 0 (itself) .. 5 and more
MAX_HOPS = 5
R_IN_REGION = R_DIST + MAX_HOPS + 1               # base -> its region
R_REGION_OF = R_IN_REGION + 1                     # region -> its base
R_NEAR_MY_HQ = R_REGION_OF + 1                    # base -> my H.Q. next to it (and back)
R_MY_HQ_NEAR = R_NEAR_MY_HQ + 1
R_NEAR_THEIR_HQ = R_MY_HQ_NEAR + 1
R_THEIR_HQ_NEAR = R_NEAR_THEIR_HQ + 1
N_RELATIONS = R_THEIR_HQ_NEAR + 1

CTX_F = G_RACK + 2                                # terrain, phase, scalars, clock, can draw
TROOP_F = 6 + 3                                   # 6 counts + placeable, share of slots, exhumable
BASE_IN = BASE_F + N_TYPES + 1                    # + legal for each troop, legal as a target
HQ_IN = 2 + N_TYPES + N_TYPES + 1                 # exists, capturable, gate, legal per troop, mine


def _zero(layer: th.nn.Linear) -> th.nn.Linear:
    th.nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        th.nn.init.zeros_(layer.bias)
    return layer


class EncoderBlock(th.nn.Module):
    """Pre-LN transformer block taking a (B, H, L, L) additive attention bias."""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = th.nn.LayerNorm(d_model)
        self.qkv = th.nn.Linear(d_model, 3 * d_model)
        self.out = th.nn.Linear(d_model, d_model)
        self.ln2 = th.nn.LayerNorm(d_model)
        self.ff = th.nn.Sequential(th.nn.Linear(d_model, dim_feedforward), th.nn.GELU(),
                                   th.nn.Linear(dim_feedforward, d_model))

    def forward(self, x: th.Tensor, bias: th.Tensor) -> th.Tensor:
        b, l, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, l, 3, self.n_heads, d // self.n_heads).unbind(2)
        att = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                             attn_mask=bias)
        x = x + self.out(att.transpose(1, 2).reshape(b, l, d))
        return x + self.ff(self.ln2(x))


def hop_distances(adj: th.Tensor) -> th.Tensor:
    """(B, N, N) paths -> (B, N, N) hop distances, capped at MAX_HOPS (also for
    two bases that are not connected, i.e. padding)."""
    n = adj.shape[-1]
    reach = th.eye(n, device=adj.device, dtype=adj.dtype).expand_as(adj).clone()
    dist = th.full_like(adj, float(MAX_HOPS)) - MAX_HOPS * reach          # 0 on the diagonal
    step = reach + adj
    for hops in range(1, MAX_HOPS):
        new = (th.bmm(reach, step) > 0).to(adj.dtype)
        dist = th.where((new > reach), th.full_like(dist, float(hops)), dist)
        reach = new
    return dist


def relation_ids(obs) -> th.Tensor:
    """(B, L, L) relation of every token (row) to every token (column), read from the observation."""
    adj, regions, hqs = obs['adj'].float(), obs['regions'].float(), obs['hqs'].float()
    b = adj.shape[0]
    rel = th.zeros(b, N_TOKENS, N_TOKENS, dtype=th.long, device=adj.device)
    bs, rs, hs = slice(T_BASES, T_REGIONS), slice(T_REGIONS, T_HQS), slice(T_HQS, N_TOKENS)
    rel[:, bs, bs] = R_DIST + hop_distances(adj).long()
    member = regions[:, :, :MAX_BASES] > 0.5                                 # (B, region, base)
    rel[:, bs, rs] = member.transpose(1, 2).long() * R_IN_REGION
    rel[:, rs, bs] = member.long() * R_REGION_OF
    near = hqs[:, :, 2 + N_TYPES:] > 0.5                                     # (B, hq, base)
    to_hq = th.tensor([R_NEAR_MY_HQ] * MAX_HQ + [R_NEAR_THEIR_HQ] * MAX_HQ, device=adj.device)
    from_hq = th.tensor([R_MY_HQ_NEAR] * MAX_HQ + [R_THEIR_HQ_NEAR] * MAX_HQ, device=adj.device)
    rel[:, bs, hs] = near.transpose(1, 2).long() * to_hq.view(1, 1, -1)
    rel[:, hs, bs] = near.long() * from_hq.view(1, -1, 1)
    return rel


class Trunk(th.nn.Module):
    """Observation dict (batched tensors) -> ((B, L, d) output tokens, (B, L) presence)."""

    def __init__(self, d_model: int = D_MODEL, layers: int = LAYERS, heads: int = HEADS, ff: int = FF):
        super().__init__()
        d = d_model
        self.d_model = d
        self.ctx_in = th.nn.Linear(CTX_F, d)
        self.troop_emb = th.nn.Parameter(th.randn(N_TYPES, d) * 0.02)
        self.troop_in = th.nn.Linear(TROOP_F, d)
        self.base_in = th.nn.Linear(BASE_IN, d)
        self.region_in = th.nn.Linear(REGION_F - MAX_BASES, d)
        self.hq_in = th.nn.Linear(HQ_IN, d)
        self.kind = th.nn.Parameter(th.randn(5, d) * 0.02)       # ctx / troop / base / region / H.Q.
        self.relation = th.nn.Embedding(N_RELATIONS, heads)
        th.nn.init.zeros_(self.relation.weight)
        self.blocks = th.nn.ModuleList(EncoderBlock(d, heads, ff) for _ in range(layers))
        self.final_ln = th.nn.LayerNorm(d)

    # -- tokens ---------------------------------------------------------- #

    def _tokens(self, obs) -> tuple[th.Tensor, th.Tensor]:
        """((B, L, d) tokens, (B, L) presence)."""
        glob, bases = obs['glob'].float(), obs['bases'].float()
        regions, hqs, mask = obs['regions'].float(), obs['hqs'].float(), obs['mask'].float()
        b = glob.shape[0]
        place = mask[:, A_PLACE:A_TARGET].view(b, N_TYPES, N_SLOTS)        # (B, troop, slot)

        ctx = self.ctx_in(th.cat([glob[:, :G_RACK], glob[:, G_CLOCK:]], dim=1)).unsqueeze(1)

        counts = th.stack([glob[:, o:o + N_TYPES] for o in
                           (G_RACK, G_PINNED, G_MY_DISCARD, G_OPP_DISCARD, G_MY_UNSEEN, G_OPP_UNSEEN)], dim=2)
        legal = th.stack([place.amax(dim=2), place.mean(dim=2), mask[:, A_TYPE:A_TYPE + N_TYPES]], dim=2)
        troops = self.troop_in(th.cat([counts, legal], dim=2)) + self.troop_emb

        base_feats = th.cat([bases, place[:, :, :MAX_BASES].transpose(1, 2),
                             mask[:, A_TARGET:A_TARGET + MAX_BASES].unsqueeze(2)], dim=2)
        base_tok = self.base_in(base_feats)

        region_tok = self.region_in(regions[:, :, MAX_BASES:])

        mine = th.zeros(b, 2 * MAX_HQ, 1, device=glob.device)
        mine[:, :MAX_HQ] = 1
        hq_place = th.zeros(b, 2 * MAX_HQ, N_TYPES, device=glob.device)
        hq_place[:, MAX_HQ:] = place[:, :, MAX_BASES:].transpose(1, 2)     # the enemy H.Q. are slots
        hq_tok = self.hq_in(th.cat([hqs[:, :, :2 + N_TYPES], hq_place, mine], dim=2))

        tokens = th.cat([ctx + self.kind[0], troops + self.kind[1], base_tok + self.kind[2],
                         region_tok + self.kind[3], hq_tok + self.kind[4]], dim=1)
        present = th.cat([th.ones(b, 1 + N_TYPES, device=glob.device), bases[:, :, 0],
                          regions[:, :, MAX_BASES], hqs[:, :, 0]], dim=1)
        return tokens, present

    def _bias(self, obs, present: th.Tensor) -> th.Tensor:
        """(B, H, L, L) attention bias from the relations between the tokens."""
        bias = self.relation(relation_ids(obs)).permute(0, 3, 1, 2)          # (B, H, L, L)
        return bias.masked_fill((present < 0.5)[:, None, None, :], -1e9)

    def forward(self, obs) -> tuple[th.Tensor, th.Tensor]:
        tokens, present = self._tokens(obs)
        bias = self._bias(obs, present)
        for block in self.blocks:
            tokens = block(tokens, bias)
        return self.final_ln(tokens), present


class PolicyHeads(th.nn.Module):
    """(B, L, d) output tokens -> (B, N_ACTIONS) unmasked logits, in the layout of `constants.py`."""

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        d, h = d_model, HEAD_H
        self.pair_slot = th.nn.Linear(d, h)                      # base or enemy H.Q. token
        self.pair_troop = th.nn.Linear(d, h, bias=False)
        self.pair_ctx = th.nn.Linear(d, h, bias=False)
        self.pair_mid = th.nn.Linear(h, h)
        self.pair_out = _zero(th.nn.Linear(h, 1))
        self.bil_slot = th.nn.Linear(d, h, bias=False)
        self.bil_troop = _zero(th.nn.Linear(d, h, bias=False))   # the product starts at zero
        self.target_base = th.nn.Linear(d, h)
        self.target_ctx = th.nn.Linear(d, h, bias=False)
        self.target_out = _zero(th.nn.Linear(h, 1))
        self.type_troop = th.nn.Linear(d, h)
        self.type_ctx = th.nn.Linear(d, h, bias=False)
        self.type_out = _zero(th.nn.Linear(h, 1))
        self.plain = th.nn.Sequential(th.nn.Linear(d, h), th.nn.ReLU(), _zero(th.nn.Linear(h, 2)))

    def forward(self, tokens: th.Tensor) -> th.Tensor:
        b = tokens.shape[0]
        ctx = tokens[:, T_CTX]
        troops = tokens[:, T_TROOPS:T_BASES]                                 # (B, 8, d)
        bases = tokens[:, T_BASES:T_REGIONS]                                 # (B, 15, d)
        enemy_hq = tokens[:, T_HQS + MAX_HQ:]                                # (B, 2, d)
        slots = th.cat([bases, enemy_hq], dim=1)                             # (B, 17, d) = the place slots

        hidden = th.relu(self.pair_slot(slots).unsqueeze(1) + self.pair_troop(troops).unsqueeze(2)
                         + self.pair_ctx(ctx)[:, None, None, :])             # (B, 8, 17, h)
        place = self.pair_out(th.relu(self.pair_mid(hidden))).squeeze(-1)
        place = place + th.einsum('bth,bsh->bts', self.bil_troop(troops), self.bil_slot(slots)) / HEAD_H ** 0.5

        target = self.target_out(th.relu(self.target_base(bases) + self.target_ctx(ctx).unsqueeze(1))).squeeze(-1)
        kind = self.type_out(th.relu(self.type_troop(troops) + self.type_ctx(ctx).unsqueeze(1))).squeeze(-1)
        plain = self.plain(ctx)                                              # draw, skip
        out = th.cat([plain, place.reshape(b, N_TYPES * N_SLOTS), target, kind], dim=1)
        assert out.shape[1] == N_ACTIONS
        return out

class ValueHead(th.nn.Module):
    """Tokens -> (B, VALUE_H) value features: attention pooling, with the context token."""

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.d_model = d_model
        self.pool_query = th.nn.Parameter(th.randn(d_model) * 0.02)
        self.pool_key = th.nn.Linear(d_model, d_model, bias=False)
        self.mlp = th.nn.Sequential(th.nn.Linear(2 * d_model, VALUE_H), th.nn.ReLU())

    def forward(self, tokens: th.Tensor, present: th.Tensor) -> th.Tensor:
        score = (self.pool_key(tokens) @ self.pool_query) / self.d_model ** 0.5
        weight = th.softmax(score.masked_fill(present < 0.5, -1e9), dim=1)
        pooled = (weight.unsqueeze(-1) * tokens).sum(dim=1)
        return self.mlp(th.cat([pooled, tokens[:, T_CTX]], dim=1))


class ToyBattleNet(th.nn.Module):
    """Observation dict (batched tensors) -> (logits (B, N_ACTIONS), value (B,))."""

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.trunk = Trunk(d_model)
        self.heads = PolicyHeads(d_model)
        self.value_head = ValueHead(d_model)
        self.value_out = th.nn.Linear(VALUE_H, 1)

    def forward(self, obs) -> tuple[th.Tensor, th.Tensor]:
        tokens, present = self.trunk(obs)
        return self.heads(tokens), self.value_out(self.value_head(tokens, present)).squeeze(-1)


# --------------------------------------------------------------------------- #
# Stable-Baselines3 plumbing (same pattern as models/smallw and models/jamaica)
# --------------------------------------------------------------------------- #

FEATURES = N_TOKENS * D_MODEL + N_TOKENS          # the tokens, flattened, then their presence


def _split(features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    return features[:, :N_TOKENS * D_MODEL].view(-1, N_TOKENS, D_MODEL), features[:, N_TOKENS * D_MODEL:]


class ToyBattleExtractor(BaseFeaturesExtractor):
    """Dict observation -> [tokens (L * d) | presence (L)]."""

    def __init__(self, observation_space: spaces.Dict):
        assert observation_space['bases'].shape == (MAX_BASES, BASE_F)
        assert observation_space['glob'].shape == (GLOB_F,) and observation_space['mask'].shape == (N_ACTIONS,)
        super().__init__(observation_space, features_dim=FEATURES)
        self.trunk = Trunk()

    def forward(self, obs) -> th.Tensor:
        tokens, present = self.trunk(obs)
        return th.cat([tokens.flatten(1), present], dim=1)


class ToyBattleLatent(th.nn.Module):
    """Replaces SB3's MlpExtractor: the actor passes the tokens on, the critic pools them."""

    def __init__(self):
        super().__init__()
        self.latent_dim_pi = FEATURES
        self.latent_dim_vf = VALUE_H
        self.value_head = ValueHead()

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return features

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_head(*_split(features))


class PointerActionNet(PolicyHeads):
    """The pointer heads, fed with SB3's flat latent."""

    def forward(self, latent_pi: th.Tensor) -> th.Tensor:
        return super().forward(_split(latent_pi)[0])


class CustomPolicy(MaskableActorCriticPolicy):
    """Graph transformer + pointer heads (see the module docstring and README.md)."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # orthogonal init is tuned for MLPs; keep PyTorch defaults for the transformer
        kwargs['ortho_init'] = False
        kwargs.setdefault('features_extractor_class', ToyBattleExtractor)
        kwargs.setdefault('net_arch', [])
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ToyBattleLatent()

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # swap the default Linear(latent_pi, n_actions) head for the pointer heads,
        # then re-create the optimizer so it covers the new parameters
        assert self.action_space.n == N_ACTIONS
        self.action_net = PointerActionNet()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


assert (CTX_F, GLOB_F - G_CLOCK) == (G_RACK + 2, 2) and B_MY_HQ < BASE_F and B_THEIR_HQ < BASE_F
assert HQ_F == 2 + N_TYPES + MAX_BASES
