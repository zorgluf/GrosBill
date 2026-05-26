from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import Tensor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

EMBEDDING_DIM = 16
NHEAD = 1


def init_weights(module):
    if isinstance(module, th.nn.Linear):
        th.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            th.nn.init.constant_(module.bias, 0)
    elif isinstance(module, th.nn.LayerNorm):
        th.nn.init.constant_(module.weight, 1.0)
        th.nn.init.constant_(module.bias, 0)

class CustomFeatureExtractor(BaseFeaturesExtractor):

    NUM_EMBEDDINGS = 64 # 10*6 for cards + 3 for stone positions
    #NUM_EMBEDDINGS = 63 # 10*6 for cards + 3 for stone positions

    def __init__(self, observation_space: spaces.Dict):
        #feature dim
        features_dim = spaces.flatdim(observation_space)
        # Pad features_dim to be a multiple of NHEAD
        if features_dim % NHEAD != 0:
            features_dim += NHEAD - (features_dim % NHEAD)
        features_dim = features_dim * EMBEDDING_DIM
        super().__init__(observation_space, features_dim=features_dim)

        self.embedding = th.nn.Embedding(num_embeddings=self.NUM_EMBEDDINGS, embedding_dim=EMBEDDING_DIM)

    def forward(self, obs) -> th.Tensor:

        stones = obs["stones"] + 60
        merged_tensor = th.cat([ obs["current_player_hand"], stones, th.flatten(obs["cards_played"], start_dim=1) ], dim=1).int()
        merged_tensor = merged_tensor.to(self.embedding.weight.device)
        # Pad merged_tensor to features_dim
        #batch_size, current_dim = merged_tensor.shape
        #pad_size = int(self._features_dim / EMBEDDING_DIM) - current_dim
        #padding = th.zeros((batch_size, pad_size), dtype=merged_tensor.dtype, device=merged_tensor.device)
        #merged_tensor = th.cat([merged_tensor, padding], dim=1)
        
        #return self.embedding(merged_tensor)
        return th.flatten(self.embedding(merged_tensor), start_dim=1)
    
class CustomNetwork(th.nn.Module):
    """
    Custom network for policy and value function using a two-layer Transformer encoder.
    It receives as input the features extracted by the features extractor.

    :param input_feature_dim: dimension of the features extracted with the features_extractor
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        input_feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 1,
        dim_feedforward: int = 512,
    ):
        super().__init__()

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Transformer expects input shape (seq_len, batch, embed_dim)
        # We'll treat the features as a sequence of length 1
        encoder_layer = th.nn.TransformerEncoderLayer(
            d_model=EMBEDDING_DIM,
            nhead=NHEAD,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation="relu"
        )
        self.transformer_encoder = th.nn.TransformerEncoder(encoder_layer, num_layers=4)
        # Appliquer l'initialisation à chaque couche
        for layer in self.transformer_encoder.layers:
            layer.apply(init_weights)

        self.policy_head = th.nn.Sequential(
            th.nn.Flatten(),
            th.nn.Linear(input_feature_dim * EMBEDDING_DIM, self.latent_dim_pi),
            th.nn.ReLU(),
        )
        self.value_head = th.nn.Sequential(
            th.nn.Flatten(),
            th.nn.Linear(input_feature_dim * EMBEDDING_DIM, self.latent_dim_vf),
            th.nn.Tanh(),
        )


    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def _common_forward(self, x: Tensor) -> Tensor:
        x = self.transformer_encoder(x)
        return x

    def forward_actor(self, features: Tensor) -> Tensor:
        extracted_features = self._common_forward(features)
        policy_net = self.policy_head(extracted_features)
        return policy_net

    def forward_critic(self, features: Tensor) -> Tensor:
        extracted_features = self._common_forward(features)
        value_net = self.value_head(extracted_features)
        return value_net


class CustomPolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization (to test if still useful)
        # kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CustomFeatureExtractor,
            net_arch=dict(pi=[ 500, 300, 100 ],vf=[500, 300, 100]),
            *args,
            **kwargs,
        )
    
    #def _build_mlp_extractor(self) -> None:
    #    features_dim = self.features_extractor._features_dim
    #    self.mlp_extractor = CustomNetwork(features_dim, spaces.flatdim(self.action_space)).to(self.device)


