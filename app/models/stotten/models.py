from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import Tensor
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

MAX_CARDS_PER_PLAYER = 6

class HandModule(th.nn.Module):
    def __init__(self, flatdim):
        super().__init__()
        #Common card layer
        self.card_dim = int(flatdim / MAX_CARDS_PER_PLAYER)
        self.card_fe = th.nn.Sequential(
                    th.nn.Linear(self.card_dim, self.card_dim),
                    th.nn.ReLU()
                )
        
    def forward(self, hand_obs: th.Tensor):
        output_list = []
        for card_obs in th.split(hand_obs.flatten(start_dim=1), self.card_dim, dim=1):
            output_list.append(self.card_fe(card_obs))
        return th.cat(output_list, dim=1)
    
    @property
    def output_flatdim(self):
        return self.card_dim * MAX_CARDS_PER_PLAYER

class CardsPlayedModule(th.nn.Module):

    #OUTPUT_DIM_BY_STONE = 16

    def __init__(self, dim_stones, dim_players, flatdim_cards):
        super().__init__()
        self.dim_stones = dim_stones
        self.dim_players = dim_players
        #Common cards played per player
        self.flatdim_cards = flatdim_cards
        self.cards_player_fe = th.nn.Sequential(
                    th.nn.Linear(self.flatdim_cards, self.flatdim_cards),
                    th.nn.ReLU()
                )
        #Common merge on one stone
        self.cards_merge_fe = th.nn.Sequential(
            th.nn.Linear(dim_players * self.flatdim_cards, dim_players * self.flatdim_cards),
            th.nn.ReLU()
        )
        
    def forward(self, cardsplayed_obs: th.Tensor):
        output_list = []
        for stone_obs in th.split(cardsplayed_obs.flatten(start_dim=1), self.dim_players * self.flatdim_cards, dim=1):
            intermediate_output = []
            for stone_player_obs in th.split(stone_obs, self.flatdim_cards, dim=1):
                intermediate_output.append(self.cards_player_fe(stone_player_obs))
            output_list.append(self.cards_merge_fe(th.cat(intermediate_output, dim=1)))
            
        return th.cat(output_list, dim=1)
    
    @property
    def output_flatdim(self):
        return self.dim_stones * self.flatdim_cards * self.dim_players

class CustomFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Dict):
        #dummy feature dim
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == "current_player":
                extractors[key] = th.nn.Flatten()
                total_concat_size += spaces.flatdim(observation_space.spaces["current_player"])
            if key == "current_player_hand":
                extractors[key] = HandModule(spaces.flatdim(observation_space.spaces["current_player_hand"]))
                total_concat_size += extractors[key].output_flatdim
            if key == "stones":
                extractors[key] = th.nn.Flatten()
                total_concat_size += spaces.flatdim(observation_space.spaces["stones"])
            if key == "cards_played":
                extractors[key] = CardsPlayedModule(
                    observation_space.spaces["cards_played"].shape[0],
                    observation_space.spaces["cards_played"].shape[1],
                    observation_space.spaces["cards_played"].shape[2] * observation_space.spaces["cards_played"].shape[3]
                )
                total_concat_size += extractors[key].output_flatdim
        
        self.extractors = th.nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

        #batch norm layer
        self.batch_norm = th.nn.BatchNorm1d(total_concat_size)

    def forward(self, obs) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(obs[key]))
        return self.batch_norm(th.cat(encoded_tensor_list, dim=1))
    
class CustomNetwork(th.nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        input_feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 1,
    ):
        super().__init__()

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.l1 = th.nn.Sequential(
            th.nn.Linear(input_feature_dim, input_feature_dim),
            th.nn.Dropout(p=0.5),
            th.nn.ReLU(),
            th.nn.BatchNorm1d(input_feature_dim),
            th.nn.Linear(input_feature_dim, input_feature_dim),
            th.nn.Dropout(p=0.5),
            th.nn.ReLU(),
            th.nn.BatchNorm1d(input_feature_dim),
        )

        self.policy_head = th.nn.Sequential(
            th.nn.Linear(input_feature_dim, self.latent_dim_pi),
            th.nn.ReLU(),
        )
        self.value_head =  th.nn.Sequential(
            th.nn.Linear(input_feature_dim, self.latent_dim_vf),
            th.nn.Tanh(),
        )

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        return self.forward_actor(features), self.forward_critic(features)
    
    def _common_forward(self, features: Tensor) -> Tensor:

        return th.add(self.l1(features),features)

    def forward_actor(self, features: Tensor) -> Tensor:
        # Policy network
        extracted_features = self._common_forward(features)
        policy_net = self.policy_head(extracted_features)
        return policy_net

    def forward_critic(self, features: Tensor) -> Tensor:
        # Value network
        extracted_features = self._common_forward(features)
        value_net = self.value_head(extracted_features)
        return value_net


class CustomPolicy(MaskableMultiInputActorCriticPolicy):
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
            *args,
            **kwargs,
        )
    
    def _build_mlp_extractor(self) -> None:
        features_dim = self.features_extractor._features_dim
        self.mlp_extractor = CustomNetwork(features_dim, spaces.flatdim(self.action_space))


