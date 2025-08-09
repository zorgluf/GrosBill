import gymnasium as gym
import numpy as np

import random
from time import sleep
import logging as logger
from functools import cmp_to_key
from typing import List

from nicegui import app

from .classes import *
from .render_web import render_web, init_web

from utils.env import GBEnv

class SchottenTottenEnv(GBEnv):

    def __init__(self, player_names: list[str] = None):
        super(SchottenTottenEnv,self).__init__("Schotten Totten", 2, player_names)


    def reset(self, seed=None):
        super().reset(seed=seed)
        #Set up board
        self.board = Board()
        #Set up players
        self.board.add_player(Player(PlayerId.PLAYER1, self.player_names[0]))
        self.board.add_player(Player(PlayerId.PLAYER2, self.player_names[1]))

        #Set up observation space
        self.observation_space = gym.spaces.Dict({
            "player1_hand": gym.spaces.MultiBinary( [ 9, len(Color) ]), #multibinary of all possible cards combination
            "player2_hand": gym.spaces.MultiBinary( [ 9, len(Color) ]),
            "main_deck": gym.spaces.MultiBinary( [ 9, len(Color) ]),
            "stones": gym.spaces.MultiDiscrete([3] * 9),  # 3 states for each stone position
            "cards_played": gym.spaces.MultiBinary( [ 9, 2, 9, len(Color) ]), #multibinary of all possible cards combination for the two side of each stone
            "current_player": gym.spaces.Discrete(2)  # 0 or 1 for player turn
        })

        #Set up the action space
        self.action_space = gym.spaces.MultiDiscrete([ MAX_CARDS_PER_PLAYER, NB_STONES ]) #First dimension is the position in hand of the card played, second dimension is the destination stone.

        return self.observation, self._get_info()
    
    @property
    def observation(self):
        player1_hand = self.board.players[PlayerId.PLAYER1].hand.observation
        player2_hand = self.board.players[PlayerId.PLAYER2].hand.observation
        main_deck = self.board.main_deck.observation
        stones = self.board.stones
        #build cards_played obs
        cards_by_stone = list()
        for i in range(NB_STONES):
            cards_by_stone.append(np.stack([ self.board.played_cards[i][PlayerId.PLAYER1], self.board.played_cards[i][PlayerId.PLAYER2] ]))
        cards_played = np.stack(cards_by_stone)

        return dict(
            player1_hand = player1_hand,
            player2_hand = player2_hand,
            main_deck = main_deck,
            stones = stones,
            cards_played = cards_played,
            current_player = self.current_player
        )
    
    def _get_info(self):
        return {
            "next_step_no_action": False
        }
    
    def action_masks(self):
        """
        Returns a list of legal actions for the current player.
        """
        mask = np.zeros((6, 9), dtype=bool)
        player = self.board.players[self.current_player]
        for hand_idx in range(MAX_CARDS_PER_PLAYER):
            if hand_idx >= len(player.hand):
                mask[hand_idx, :] = False
            else:
                for stone_idx in range(NB_STONES):
                    if len(self.board.played_cards[stone_idx][self.current_player]) < 3:
                        mask[hand_idx, stone_idx] = True
                    else:
                        mask[hand_idx, stone_idx] = False
        return mask
    
    def can_claim_stone(self, stone_idx) -> bool:
        """
        Check if the current player can claim a stone.
        """
        if self.board.stones[stone_idx] != StonePosition.NEUTRAL:
            return False
        # if 3 card on each side, check if the current player has the best combination
        #TODO
        # if other side has 2 cards, and 3 on player side, compute all combination to see if opponent can win
        #TODO
        return False

    def step(self, action: List[int]):
        # check move legality
        if (action != -1) and (self.action_masks()[action[0], action[1]] == False):
            raise Exception(f'Illegal action {action} : Legal actions {self.action_masks()}')
        
        #Play card on stone
        card = self.board.players[self.current_player].hand.draw_one_by_index(action[0])
        self.board.played_cards[action[1]][self.current_player].add(card)
        #Check if current player can claim a stone
        for stone_idx in range(NB_STONES):
            if self.can_claim_stone(stone_idx):
                self.board.stones[stone_idx] = self.current_player * 2
        #Compute reward
        #TODO
        #check if we are done
        #TODO
        return self.observation, reward, terminated, False, self._get_info()

    def render(self, pov_player: int = -1, mode:str = 'human_web', **kwargs):
        """
        Update the render of the environment. Superseeded by subclasses to implement specific rendering logic.
        Args:
            pov_player (int, optional): Player number for point of view rendering. -1 set the pov_player as the current player.
            mode (str, optional): Rendering mode. Defaults to 'human_web'.
            suggested_action (int, optional): Suggested action for human players. Defaults to None.
        """
        if pov_player == -1:
            pov_player = self.current_player
        else:
            self.pov_player = pov_player
        return

    def nicegui_page():
        pass  # Implement NiceGUI page rendering