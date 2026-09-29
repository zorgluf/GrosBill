"""Policy for Small World (`smallw`): entity transformer + map bias + pointer heads.

Design rationale, training recipe and the environment prerequisites (relative
ally action, attack-cost / projected-income columns, the `mask` observation
key) are in `README.md` next to this file. In short:

* **Tokens** (L = 42): 1 context token (phase, turn, die, ...), 30 region
  tokens, 5 player tokens (relative seat order, absent seats masked) and 6
  combo tokens. Every categorical column is an embedding; race and power ids
  go through two tables shared by every token type (`race_emb` /
  `power_emb`), re-projected per role (active race, declined race, combo...).
* **Trunk**: pre-LN transformer blocks with a per-head additive attention
  bias: learned from the shortest-path distance between two regions
  (`map3p.ADJACENCY`) and from the region -> owner-player relation.
* **Policy**: the 133 logits are built from the output tokens (pointer
  heads), in the `smallw.py` action layout: combos from the combo tokens,
  decline / pass from the context token, the four region ranges from the
  region tokens (one FiLM-conditioned head per range), the ally offsets from
  the player tokens. Last layers start at zero -> uniform over legal actions.
* **Value**: attention pooling over every token ++ context token -> MLP.

SB3 plumbing (same pattern as `models/stotten/models.py` `TransformerPolicy`):
the feature extractor returns the flattened output tokens (+ the player
presence flags), `EntityLatentExtractor` replaces the `MlpExtractor` (actor:
tokens untouched, critic: pooled MLP) and `PointerActionNet` replaces the
default `Linear` action head.
"""

from collections import deque
from typing import Callable

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environments.smallw.envs.classes import MAX_PLAYERS, PowerId, RaceId, Terrain
from environments.smallw.envs.map3p import ADJACENCY
from environments.smallw.envs.smallw import (
    A_ALLY,
    A_COMBO,
    A_DECLINE,
    A_DRAGON,
    A_PASS,
    A_REGION,
    A_REGION_ALL,
    A_SORCERER,
    MAX_ATTACK_COST,
    MAX_REGIONS,
    N_ACTIONS,
    Phase,
)

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
DIM_FEEDFORWARD = 256
ID_EMB_DIM = 32          #: shared race / power tables
VF_HIDDEN = 256
MAX_DIST = 6             #: region distances are clipped to this (the bias table has MAX_DIST + 1 rows)
MAX_COUNT = 12           #: token counts are clipped to this for their embedding

N_RACE_CODES = len(RaceId) + 1        # 0 = none
N_POWER_CODES = len(PowerId) + 1      # 0 = none
N_OWNER_CODES = 4 + 2 * (MAX_PLAYERS - 1)
N_PHASES = len(Phase)
N_COMBOS = A_DECLINE - A_COMBO
N_REGION_RANGES = 4                   # region, region_all, sorcerer, dragon
REGION_RANGE_STARTS = (A_REGION, A_REGION_ALL, A_SORCERER, A_DRAGON)

# token layout
N_CTX = 1
TOK_REGIONS = N_CTX
TOK_PLAYERS = TOK_REGIONS + MAX_REGIONS
TOK_COMBOS = TOK_PLAYERS + MAX_PLAYERS
N_TOKENS = TOK_COMBOS + N_COMBOS

#: `regions` columns fed as 0/1 flags: mine, magic, cavern, border, coastal,
#: lair, fortress, hole, hero, dragon
REGION_FLAG_COLS = [1, 2, 3, 4, 5, 9, 10, 12, 13, 14]
#: `global` columns fed as 0/1 flags: dragon used, fortress used, sorcerer used
#: on offsets 1..4, first turn of the race
GLOBAL_FLAG_COLS = [6, 7, 8, 9, 10, 11, 12]


def _distance_matrix() -> np.ndarray:
    """All-pairs shortest-path distance (BFS) between the region indices."""
    n = MAX_REGIONS
    dist = np.full((n, n), MAX_DIST, dtype=np.int64)
    for rid in ADJACENCY:
        src = rid - 1
        dist[src, src] = 0
        seen = {rid}
        queue = deque([(rid, 0)])
        while queue:
            cur, d = queue.popleft()
            for nxt in ADJACENCY[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    dist[src, nxt - 1] = min(d + 1, MAX_DIST)
                    queue.append((nxt, d + 1))
    return dist


def _log1p(x: th.Tensor) -> th.Tensor:
    return th.log1p(x.clamp(min=0))


def _zero_last(module: th.nn.Sequential) -> th.nn.Sequential:
    """Zero the last Linear so the head starts at 0 (uniform masked policy)."""
    th.nn.init.zeros_(module[-1].weight)
    th.nn.init.zeros_(module[-1].bias)
    return module


class EncoderBlock(th.nn.Module):
    """Pre-LN transformer block taking a (B, H, L, L) additive attention bias."""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = th.nn.LayerNorm(d_model)
        self.qkv = th.nn.Linear(d_model, 3 * d_model)
        self.out = th.nn.Linear(d_model, d_model)
        self.ln2 = th.nn.LayerNorm(d_model)
        self.ff = th.nn.Sequential(
            th.nn.Linear(d_model, dim_feedforward),
            th.nn.GELU(),
            th.nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x: th.Tensor, bias: th.Tensor) -> th.Tensor:
        b, l, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, l, 3, self.n_heads, d // self.n_heads).unbind(2)
        att = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=bias)
        x = x + self.out(att.transpose(1, 2).reshape(b, l, d))
        return x + self.ff(self.ln2(x))


class EntityFeatureExtractor(BaseFeaturesExtractor):
    """Tokenises the Dict observation and runs the map-biased transformer.

    Output (flat, as SB3 requires): ``[tokens (N_TOKENS * d) | player present (5)]``.
    """

    def __init__(self, observation_space: spaces.Dict):
        self.d_model = D_MODEL
        super().__init__(observation_space, features_dim=N_TOKENS * D_MODEL + MAX_PLAYERS)
        d, e = D_MODEL, ID_EMB_DIM
        region_cols = observation_space['regions'].shape[1]
        assert observation_space['regions'].shape[0] == MAX_REGIONS
        assert observation_space['mask'].shape == (N_ACTIONS,)
        assert region_cols == 17, 'observation layout changed, see smallw.py'

        # shared identity tables + one projection per role
        self.race_emb = th.nn.Embedding(N_RACE_CODES, e)
        self.power_emb = th.nn.Embedding(N_POWER_CODES, e)

        # regions
        self.terrain_emb = th.nn.Embedding(len(Terrain), d)
        self.owner_emb = th.nn.Embedding(N_OWNER_CODES, d)
        self.conquered_emb = th.nn.Embedding(MAX_PLAYERS + 1, d)
        self.cost_emb = th.nn.Embedding(MAX_ATTACK_COST + 2, d)
        self.count_emb = th.nn.Embedding(MAX_COUNT + 1, d)
        self.region_pos = th.nn.Parameter(th.zeros(MAX_REGIONS, d))
        self.region_race = th.nn.Linear(e, d, bias=False)
        # flags + log1p(tokens, encampments, cost) + 4 legality bits
        self.region_num = th.nn.Linear(len(REGION_FLAG_COLS) + 3 + N_REGION_RANGES, d)

        # players
        self.seat_emb = th.nn.Parameter(th.zeros(MAX_PLAYERS, d))
        self.ally_emb = th.nn.Embedding(MAX_PLAYERS + 1, d)
        self.player_ids = th.nn.Linear(4 * e, d, bias=False)   # active race, power, 2 declined
        # coins, tokens in hand, tray, active / declined regions, coin tokens,
        # projected income (log1p) + must-first-conquest + ally-legal bit
        self.player_num = th.nn.Linear(9, d)

        # combos
        self.combo_pos = th.nn.Parameter(th.zeros(N_COMBOS, d))
        self.combo_ids = th.nn.Sequential(                     # race x power interaction
            th.nn.Linear(2 * e, d), th.nn.GELU(), th.nn.Linear(d, d))
        self.combo_num = th.nn.Linear(3, d)                    # coins, empty, pickable

        # context
        self.ctx_token = th.nn.Parameter(th.zeros(d))
        self.phase_emb = th.nn.Embedding(N_PHASES, d)
        self.turn_emb = th.nn.Embedding(21, d)
        self.die_emb = th.nn.Embedding(5, d)
        # flags + log1p(conquests, to place, amazon, placements) + turn fraction
        # + decline / pass legal
        self.ctx_num = th.nn.Linear(len(GLOBAL_FLAG_COLS) + 4 + 1 + 2, d)

        for p in (self.region_pos, self.seat_emb, self.combo_pos):
            th.nn.init.normal_(p, std=0.02)
        th.nn.init.normal_(self.ctx_token, std=0.02)

        # attention biases
        self.register_buffer('region_dist', th.as_tensor(_distance_matrix()), persistent=False)
        self.dist_bias = th.nn.Embedding(MAX_DIST + 1, N_HEADS)
        self.own_bias = th.nn.Parameter(th.zeros(2, N_HEADS))   # active / declined owner
        th.nn.init.zeros_(self.dist_bias.weight)

        self.blocks = th.nn.ModuleList(
            EncoderBlock(d, N_HEADS, DIM_FEEDFORWARD) for _ in range(N_LAYERS))
        self.final_ln = th.nn.LayerNorm(d)

    # -- tokens ------------------------------------------------------------ #

    def _region_tokens(self, regions: th.Tensor, mask: th.Tensor) -> th.Tensor:
        idx = regions.long()
        legal = th.stack(
            [mask[:, s:s + MAX_REGIONS] for s in REGION_RANGE_STARTS], dim=2)  # (B, R, 4)
        cost = regions[..., 16]
        num = th.cat([
            regions[..., REGION_FLAG_COLS],
            _log1p(regions[..., 8:9]),
            _log1p(regions[..., 11:12]),
            _log1p(cost - 1).unsqueeze(-1) * (cost > 0).unsqueeze(-1),
            legal,
        ], dim=-1)
        return (self.terrain_emb(idx[..., 0].clamp(0, len(Terrain) - 1))
                + self.owner_emb(idx[..., 6].clamp(0, N_OWNER_CODES - 1))
                + self.region_race(self.race_emb(idx[..., 7].clamp(0, N_RACE_CODES - 1)))
                + self.count_emb(idx[..., 8].clamp(0, MAX_COUNT))
                + self.conquered_emb(idx[..., 15].clamp(0, MAX_PLAYERS))
                + self.cost_emb(idx[..., 16].clamp(0, MAX_ATTACK_COST + 1))
                + self.region_num(num)
                + self.region_pos)

    def _player_tokens(self, players: th.Tensor, mask: th.Tensor) -> th.Tensor:
        idx = players.long()
        race = lambda col: self.race_emb(idx[..., col].clamp(0, N_RACE_CODES - 1))
        ids = th.cat([race(2), self.power_emb(idx[..., 3].clamp(0, N_POWER_CODES - 1)),
                      race(6), race(7)], dim=-1)
        num = th.cat([
            _log1p(players[..., [1, 4, 5, 8, 9, 12, 13]]),
            players[..., 11:12],
            mask[:, A_ALLY:A_ALLY + MAX_PLAYERS].unsqueeze(-1),
        ], dim=-1)
        return (self.player_ids(ids) + self.player_num(num)
                + self.ally_emb(idx[..., 10].clamp(0, MAX_PLAYERS)) + self.seat_emb)

    def _combo_tokens(self, combos: th.Tensor, mask: th.Tensor) -> th.Tensor:
        idx = combos.long()
        ids = th.cat([self.race_emb(idx[..., 0].clamp(0, N_RACE_CODES - 1)),
                      self.power_emb(idx[..., 1].clamp(0, N_POWER_CODES - 1))], dim=-1)
        num = th.stack([
            _log1p(combos[..., 2]),
            (idx[..., 0] == 0).float(),
            mask[:, A_COMBO:A_COMBO + N_COMBOS],
        ], dim=-1)
        return self.combo_ids(ids) + self.combo_num(num) + self.combo_pos

    def _ctx_token(self, glob: th.Tensor, mask: th.Tensor) -> th.Tensor:
        idx = glob.long()
        num = th.cat([
            glob[:, GLOBAL_FLAG_COLS],
            _log1p(glob[:, [4, 5, 13, 14]]),
            (glob[:, 0] / glob[:, 1].clamp(min=1)).unsqueeze(-1),
            mask[:, [A_DECLINE, A_PASS]],
        ], dim=-1)
        return (self.ctx_token + self.ctx_num(num)
                + self.phase_emb(idx[:, 2].clamp(0, N_PHASES - 1))
                + self.turn_emb(idx[:, 0].clamp(0, 20))
                + self.die_emb(idx[:, 3].clamp(0, 4)))

    # -- attention bias ---------------------------------------------------- #

    def _attention_bias(self, regions: th.Tensor, present: th.Tensor) -> th.Tensor:
        batch = regions.shape[0]
        bias = regions.new_zeros(batch, N_HEADS, N_TOKENS, N_TOKENS)
        r0, r1 = TOK_REGIONS, TOK_REGIONS + MAX_REGIONS
        p0, p1 = TOK_PLAYERS, TOK_PLAYERS + MAX_PLAYERS
        # region <-> region: shortest-path distance
        bias[:, :, r0:r1, r0:r1] = self.dist_bias(self.region_dist).permute(2, 0, 1)
        # region <-> owner player, one bias per head for an active / declined owner
        owner = regions[..., 6].long()                                  # (B, R)
        owner_k = th.where(owner >= 4, (owner - 4) // 2 + 1, th.zeros_like(owner))
        owned = (owner >= 2)
        declined = th.where(owner >= 4, (owner - 4) % 2, owner - 2).clamp(0, 1)
        onehot = F.one_hot(owner_k.clamp(0, MAX_PLAYERS - 1), MAX_PLAYERS).float()
        onehot = onehot * owned.unsqueeze(-1)                           # (B, R, P)
        per_head = self.own_bias[declined]                              # (B, R, H)
        rp = onehot.unsqueeze(1) * per_head.permute(0, 2, 1).unsqueeze(-1)  # (B, H, R, P)
        bias[:, :, r0:r1, p0:p1] = rp
        bias[:, :, p0:p1, r0:r1] = rp.transpose(2, 3)
        # absent seats are never attended to
        bias[:, :, :, p0:p1] = bias[:, :, :, p0:p1].masked_fill(
            (present < 0.5)[:, None, None, :], -1e9)
        return bias

    def forward(self, obs) -> th.Tensor:
        regions, players = obs['regions'].float(), obs['players'].float()
        combos, glob, mask = obs['combos'].float(), obs['global'].float(), obs['mask'].float()
        present = players[..., 0]
        tokens = th.cat([
            self._ctx_token(glob, mask).unsqueeze(1),
            self._region_tokens(regions, mask),
            self._player_tokens(players, mask),
            self._combo_tokens(combos, mask),
        ], dim=1)                                                       # (B, L, d)
        bias = self._attention_bias(regions, present)
        for block in self.blocks:
            tokens = block(tokens, bias)
        tokens = self.final_ln(tokens)
        return th.cat([tokens.flatten(1), present], dim=1)


class EntityLatentExtractor(th.nn.Module):
    """Replaces SB3's MlpExtractor.

    * actor: the flat features untouched (`PointerActionNet` slices the tokens),
    * critic: attention pooling (learned query) over the present tokens
      ++ context token -> MLP.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.latent_dim_pi = N_TOKENS * d_model + MAX_PLAYERS
        self.latent_dim_vf = VF_HIDDEN
        self.pool_query = th.nn.Parameter(th.zeros(d_model))
        self.pool_key = th.nn.Linear(d_model, d_model, bias=False)
        self.value_mlp = th.nn.Sequential(
            th.nn.Linear(2 * d_model, VF_HIDDEN), th.nn.ReLU(),
            th.nn.Linear(VF_HIDDEN, VF_HIDDEN), th.nn.ReLU(),
        )

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return features

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        tokens = features[:, :N_TOKENS * self.d_model].view(-1, N_TOKENS, self.d_model)
        present = features[:, N_TOKENS * self.d_model:]
        scores = self.pool_key(tokens) @ self.pool_query / self.d_model ** 0.5   # (B, L)
        absent = th.zeros_like(scores, dtype=th.bool)
        absent[:, TOK_PLAYERS:TOK_PLAYERS + MAX_PLAYERS] = present < 0.5
        weights = th.softmax(scores.masked_fill(absent, -1e9), dim=1)
        pooled = (weights.unsqueeze(-1) * tokens).sum(1)
        return self.value_mlp(th.cat([pooled, tokens[:, 0]], dim=1))


class PointerActionNet(th.nn.Module):
    """The 133 logits, each from the output token of the entity it targets."""

    def __init__(self, d_model: int):
        super().__init__()
        d, h = d_model, d_model // 2
        self.d_model = d
        mlp = lambda n_in: _zero_last(th.nn.Sequential(
            th.nn.Linear(n_in, h), th.nn.GELU(), th.nn.Linear(h, 1)))
        self.combo_head = mlp(2 * d)
        self.ally_head = mlp(2 * d)
        self.ctx_head = _zero_last(th.nn.Sequential(
            th.nn.Linear(d, h), th.nn.GELU(), th.nn.Linear(h, 2)))       # decline, pass
        # one FiLM-conditioned head per region range
        self.film = th.nn.Linear(d, N_REGION_RANGES * 2 * d)
        th.nn.init.zeros_(self.film.weight)
        th.nn.init.zeros_(self.film.bias)
        self.region_heads = th.nn.ModuleList(mlp(d) for _ in range(N_REGION_RANGES))

    def forward(self, latent_pi: th.Tensor) -> th.Tensor:
        d = self.d_model
        tokens = latent_pi[:, :N_TOKENS * d].view(-1, N_TOKENS, d)
        ctx = tokens[:, 0]
        regions = tokens[:, TOK_REGIONS:TOK_REGIONS + MAX_REGIONS]
        players = tokens[:, TOK_PLAYERS:TOK_PLAYERS + MAX_PLAYERS]
        combos = tokens[:, TOK_COMBOS:TOK_COMBOS + N_COMBOS]
        with_ctx = lambda t: th.cat([t, ctx.unsqueeze(1).expand(-1, t.shape[1], -1)], dim=-1)

        combo_logits = self.combo_head(with_ctx(combos)).squeeze(-1)        # (B, 6)
        ctx_logits = self.ctx_head(ctx)                                     # (B, 2)
        film = self.film(ctx).view(-1, N_REGION_RANGES, 2, d)
        region_logits = [
            head(regions * (1 + film[:, k, 0:1]) + film[:, k, 1:2]).squeeze(-1)
            for k, head in enumerate(self.region_heads)
        ]                                                                   # 4 x (B, 30)
        ally_logits = self.ally_head(with_ctx(players)).squeeze(-1)         # (B, 5)
        logits = th.cat([combo_logits, ctx_logits, *region_logits, ally_logits], dim=1)
        assert logits.shape[1] == N_ACTIONS
        return logits


class CustomPolicy(MaskableActorCriticPolicy):
    """Entity transformer policy (see the module docstring and README.md)."""

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
        kwargs.setdefault('features_extractor_class', EntityFeatureExtractor)
        kwargs.setdefault('net_arch', [])
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = EntityLatentExtractor(self.features_extractor.d_model)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # swap the default Linear(latent_pi, n_actions) head for the pointer
        # heads, then re-create the optimizer so it covers the new parameters
        assert self.action_space.n == N_ACTIONS
        self.action_net = PointerActionNet(self.features_extractor.d_model)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
