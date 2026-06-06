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
