from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
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

    OUTPUT_DIM_BY_STONE = 16

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
            th.nn.Linear(dim_players * self.flatdim_cards, self.OUTPUT_DIM_BY_STONE),
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
        return self.dim_stones * self.OUTPUT_DIM_BY_STONE

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

    def forward(self, obs) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(obs[key]))
        return th.cat(encoded_tensor_list, dim=1)


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
            net_arch=dict(pi=[ 500, 64 ], vf=[ 500, 32]),
            features_extractor_class=CustomFeatureExtractor,
            *args,
            **kwargs,
        )


