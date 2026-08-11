from typing import Callable

import numpy as np
from gymnasium import spaces
import torch as th
import torch.nn.functional as F
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

CARD_EMBEDDING_DIM = 8
N_STONE_STATES = 3  # 0 = mine, 1 = neutral, 2 = opponent


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    Embedding-based extractor for the (stationary) Schotten Totten observation.

    - current_player_hand : MultiBinary(NB_CARDS) presence vector -> used as-is (already 0/1).
    - stones              : Box(NB_STONES,) in {0,1,2}            -> one-hot.
    - cards_played        : Box(NB_STONES, 2, 3) identity tokens   -> learned embedding.
                            (0 = empty slot, 1..NB_CARDS = card identity + 1)

    Cards are treated as categorical identities (via an embedding / one-hot), NOT as a
    single packed scalar, so the network can actually reason about colors, runs and sets.
    """

    def __init__(self, observation_space: spaces.Dict):
        hand_space = observation_space["current_player_hand"]
        self.hand_dim = int(hand_space.shape[0])

        stones_space = observation_space["stones"]
        self.n_stones = int(stones_space.shape[0])

        played_space = observation_space["cards_played"]
        self.n_played_slots = int(np.prod(played_space.shape))
        self.num_card_tokens = int(played_space.high.max()) + 1  # NB_CARDS + 1

        features_dim = (
            self.hand_dim
            + self.n_stones * N_STONE_STATES
            + self.n_played_slots * CARD_EMBEDDING_DIM
        )
        super().__init__(observation_space, features_dim=features_dim)

        self.card_embedding = th.nn.Embedding(
            num_embeddings=self.num_card_tokens, embedding_dim=CARD_EMBEDDING_DIM
        )

    def forward(self, obs) -> th.Tensor:
        hand = obs["current_player_hand"].float()                          # (B, hand_dim)
        stones = obs["stones"].long().clamp(0, N_STONE_STATES - 1)         # (B, n_stones)
        stones_oh = F.one_hot(stones, num_classes=N_STONE_STATES).flatten(1).float()
        played = obs["cards_played"].long().clamp(0, self.num_card_tokens - 1).flatten(1)
        played_emb = self.card_embedding(played).flatten(1)                # (B, n_played_slots*emb)
        return th.cat([hand, stones_oh, played_emb], dim=1)


class CustomPolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CustomFeatureExtractor,
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
            *args,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Transformer policy (registered as env name 'stottentr').
#
# Stone-level tokens + pointer action head:
#   - each stone is one token built from its 6 card-slot embeddings + stone state,
#   - a hand token summarises the held cards, a CLS token summarises the game,
#   - a 2-layer pre-LN transformer encoder mixes them (cross-stone planning,
#     card counting),
#   - the action logit for "play card c on stone s" is a scaled dot product
#     between a query derived from card c's embedding and a key derived from
#     stone s's output token, so per-stone knowledge is shared across all cards.
#
# Kept separate from CustomPolicy so existing zoo/stotten snapshots still load.
# ---------------------------------------------------------------------------

TR_D_MODEL = 64
TR_N_HEADS = 4
TR_N_LAYERS = 2
TR_DIM_FEEDFORWARD = 128
TR_POINTER_DIM = 32
TR_VF_HIDDEN = 128


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Stone-level transformer encoder for the Schotten Totten observation.

    Tokens (1 CLS + NB_STONES + 1 hand):
      - stone s : Linear([emb(my 3 slots) ++ emb(opp 3 slots) ++ state one-hot]) + pos_emb[s]
                  (learned positional embedding: adjacency matters, 3 adjacent stones win)
      - hand    : Linear(sum of embeddings of held cards)
      - CLS     : learned vector, global summary for the value head.

    SB3 extractors must return one flat vector, so the output is the concatenation
      [ CLS out (d_model) | stone outs (n_stones*d_model) | raw hand multi-hot (hand_dim) ]
    which TransformerPolicy's latent extractor / pointer head slice back apart.
    The raw multi-hot is passed through so the exact hand content is never lost
    (the sum-of-embeddings hand token alone is slightly lossy).
    """

    def __init__(self, observation_space: spaces.Dict):
        hand_space = observation_space["current_player_hand"]
        self.hand_dim = int(hand_space.shape[0])                     # NB_CARDS

        stones_space = observation_space["stones"]
        self.n_stones = int(stones_space.shape[0])                   # NB_STONES

        played_space = observation_space["cards_played"]
        self.slots_per_stone = int(np.prod(played_space.shape[1:]))  # 2 sides * 3 slots
        self.num_card_tokens = int(played_space.high.max()) + 1      # NB_CARDS + 1 (0 = empty)
        self.d_model = TR_D_MODEL

        features_dim = self.d_model + self.n_stones * self.d_model + self.hand_dim
        super().__init__(observation_space, features_dim=features_dim)

        self.card_embedding = th.nn.Embedding(
            num_embeddings=self.num_card_tokens, embedding_dim=CARD_EMBEDDING_DIM
        )
        stone_in_dim = self.slots_per_stone * CARD_EMBEDDING_DIM + N_STONE_STATES
        self.stone_proj = th.nn.Linear(stone_in_dim, self.d_model)
        self.stone_pos = th.nn.Parameter(th.zeros(self.n_stones, self.d_model))
        self.hand_proj = th.nn.Linear(CARD_EMBEDDING_DIM, self.d_model)
        self.cls_token = th.nn.Parameter(th.zeros(1, 1, self.d_model))
        th.nn.init.normal_(self.stone_pos, std=0.02)
        th.nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = th.nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=TR_N_HEADS,
            dim_feedforward=TR_DIM_FEEDFORWARD,
            dropout=0.0,               # no dropout: interacts badly with PPO's ratio clipping
            batch_first=True,
            norm_first=True,           # pre-LN: post-LN transformers routinely diverge in RL
        )
        self.encoder = th.nn.TransformerEncoder(
            encoder_layer,
            num_layers=TR_N_LAYERS,
            norm=th.nn.LayerNorm(self.d_model),
            enable_nested_tensor=False,  # unsupported with norm_first, silences the warning
        )

    def forward(self, obs) -> th.Tensor:
        hand_multihot = obs["current_player_hand"].float()                       # (B, NB_CARDS)
        stones_state = obs["stones"].long().clamp(0, N_STONE_STATES - 1)         # (B, n_stones)
        played = obs["cards_played"].long().clamp(0, self.num_card_tokens - 1)   # (B, n_stones, 2, 3)
        batch = hand_multihot.shape[0]

        # Stone tokens: 6 card-slot embeddings + stone state, projected to d_model.
        played_emb = self.card_embedding(played.flatten(2))                      # (B, n_stones, 6, emb)
        state_oh = F.one_hot(stones_state, num_classes=N_STONE_STATES).float()   # (B, n_stones, 3)
        stone_in = th.cat([played_emb.flatten(2), state_oh], dim=2)              # (B, n_stones, 51)
        stone_tokens = self.stone_proj(stone_in) + self.stone_pos                # (B, n_stones, d)

        # Hand token: permutation-invariant sum of held-card embeddings
        # (rows 1.. of the shared table; row 0 is the 'empty slot' token).
        hand_emb = hand_multihot @ self.card_embedding.weight[1:]                # (B, emb)
        hand_token = self.hand_proj(hand_emb).unsqueeze(1)                       # (B, 1, d)

        cls = self.cls_token.expand(batch, -1, -1)                               # (B, 1, d)
        tokens = th.cat([cls, stone_tokens, hand_token], dim=1)                  # (B, n_stones+2, d)
        out = self.encoder(tokens)

        cls_out = out[:, 0]                                                      # (B, d)
        stones_out = out[:, 1 : 1 + self.n_stones].flatten(1)                    # (B, n_stones*d)
        return th.cat([cls_out, stones_out, hand_multihot], dim=1)


class PointerLatentExtractor(th.nn.Module):
    """
    Replaces SB3's MlpExtractor. Slices the flat feature vector produced by
    TransformerFeatureExtractor:
      - latent_pi : the per-stone output tokens, untouched (the pointer action
                    head needs them individually, not pooled),
      - latent_vf : MLP over [CLS out ++ raw hand multi-hot].
    """

    def __init__(self, d_model: int, n_stones: int, hand_dim: int):
        super().__init__()
        self.d_model = d_model
        self.n_stones = n_stones
        self.hand_dim = hand_dim
        self.latent_dim_pi = n_stones * d_model
        self.latent_dim_vf = TR_VF_HIDDEN
        self.value_net = th.nn.Sequential(
            th.nn.Linear(d_model + hand_dim, TR_VF_HIDDEN),
            th.nn.ReLU(),
            th.nn.Linear(TR_VF_HIDDEN, TR_VF_HIDDEN),
            th.nn.ReLU(),
        )

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return features[:, self.d_model : self.d_model + self.latent_dim_pi]

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        cls_out = features[:, : self.d_model]
        hand_multihot = features[:, self.d_model + self.latent_dim_pi :]
        return self.value_net(th.cat([cls_out, hand_multihot], dim=1))


class PointerActionNet(th.nn.Module):
    """
    Factored action head over action = card_identity * NB_STONES + stone_idx:
        logit(c, s) = < q_proj(emb(c)) , k_proj(stone_out_s) > / sqrt(dim)
    The card embedding table is shared with the feature extractor, so cards live
    in a single representation space. k_proj starts near zero so the initial
    policy is ~uniform over legal actions (PPO-friendly).
    """

    def __init__(self, card_embedding: th.nn.Embedding, d_model: int, n_stones: int):
        super().__init__()
        self.card_embedding = card_embedding      # shared module, params deduplicated
        self.n_stones = n_stones
        self.d_model = d_model
        emb_dim = card_embedding.embedding_dim
        self.q_proj = th.nn.Linear(emb_dim, TR_POINTER_DIM, bias=False)
        self.k_proj = th.nn.Linear(d_model, TR_POINTER_DIM, bias=False)
        self.scale = TR_POINTER_DIM ** -0.5
        th.nn.init.orthogonal_(self.q_proj.weight, gain=1.0)
        th.nn.init.orthogonal_(self.k_proj.weight, gain=0.01)

    def forward(self, latent_pi: th.Tensor) -> th.Tensor:
        stones_out = latent_pi.view(-1, self.n_stones, self.d_model)   # (B, S, d)
        keys = self.k_proj(stones_out)                                 # (B, S, p)
        queries = self.q_proj(self.card_embedding.weight[1:])          # (C, p), row 0 = empty
        logits = th.einsum("cp,bsp->bcs", queries, keys) * self.scale  # (B, C, S)
        return logits.flatten(1)                                       # row-major: c * n_stones + s


class TransformerPolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Orthogonal init is tuned for MLPs; keep PyTorch defaults for the transformer.
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=TransformerFeatureExtractor,
            net_arch=[],
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        fe = self.features_extractor
        self.mlp_extractor = PointerLatentExtractor(fe.d_model, fe.n_stones, fe.hand_dim)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # Swap the default Linear(latent_pi, n_actions) head for the pointer head,
        # then re-create the optimizer so it covers the new parameters.
        fe = self.features_extractor
        assert self.action_space.n == (fe.num_card_tokens - 1) * fe.n_stones
        self.action_net = PointerActionNet(fe.card_embedding, fe.d_model, fe.n_stones)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
