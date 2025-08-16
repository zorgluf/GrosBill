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

    #the card selected through UI to play
    gui_hand_card_selected: int = None

    def __init__(self, player_names: list[str] = None):
        super(SchottenTottenEnv,self).__init__("stotten", 2, player_names)

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

    def reset(self, seed=None):
        super().reset(seed=seed)
        #Set up board
        self.board = Board()
        #Set up players
        self.board.add_player(Player(PlayerId.PLAYER1, self.player_names[0]))
        self.board.add_player(Player(PlayerId.PLAYER2, self.player_names[1]))
        self.current_player = 0
        #add cards to players
        for i in range(MAX_CARDS_PER_PLAYER):
            self.board.players[PlayerId.PLAYER1.value].hand.add(self.board.main_deck.draw(1))
            self.board.players[PlayerId.PLAYER2.value].hand.add(self.board.main_deck.draw(1))

        return self.observation, self._get_info()
    
    @property
    def current_opponent(self):
        """
        Returns the current opponent player id.
        """
        return abs(self.current_player - 1)

    @property
    def observation(self):
        player1_hand = self.board.players[PlayerId.PLAYER1.value].hand.observation
        player2_hand = self.board.players[PlayerId.PLAYER2.value].hand.observation
        main_deck = self.board.main_deck.observation
        stones = [ position.value for position in self.board.stones ]
        #build cards_played obs
        cards_by_stone = list()
        for i in range(NB_STONES):
            cards_by_stone.append(np.stack([ self.board.played_cards[i][PlayerId.PLAYER1.value].observation, self.board.played_cards[i][PlayerId.PLAYER2.value].observation ]))
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
        mask_card = np.zeros((MAX_CARDS_PER_PLAYER,), dtype=bool)
        mask_stone = np.zeros((NB_STONES,), dtype=bool)
        player = self.board.players[self.current_player]
        for hand_idx in range(MAX_CARDS_PER_PLAYER):
            if hand_idx < len(player.hand):
                mask_card[hand_idx] = True
        for stone_idx in range(NB_STONES):
            if len(self.board.played_cards[stone_idx][self.current_player]) < 3:
                mask_stone[stone_idx] = True
        return np.concatenate([mask_card, mask_stone])
    
    def score_stone(self, cards: Deck) -> int:
        """
        Compute the score of the played cards on a stone. Must have 3 cards in deck
        """
        card_sum = sum([card.value for card in cards])
        #three of a kind
        if len(set([card.value for card in cards])) == 1:
            return 300 + card_sum
        if len(set([card.color for card in cards])) == 1:
            if np.all(np.diff(np.sort([card.value for card in cards])) == 1):
                #color run
                return 400 + card_sum
            else:
                #color
                return 200 + card_sum
        if np.all(np.diff(np.sort([card.value for card in cards])) == 1):
            #run
            return 100 + card_sum
        #sum
        return card_sum
        
    
    def can_claim_stone(self, stone_idx) -> bool:
        """
        Check if the current player can claim a stone.
        """
        if self.board.stones[stone_idx] != StonePosition.NEUTRAL:
            return False
        if len(self.board.played_cards[stone_idx][self.current_player]) == 3:
            if len(self.board.played_cards[stone_idx][self.current_opponent]) == 3:
                # if 3 card on each side, check if the current player has the best combination
                score_stone_current = self.score_stone(self.board.played_cards[stone_idx][self.current_player])
                score_stone_opponent = self.score_stone(self.board.played_cards[stone_idx][self.current_opponent])
                if score_stone_current > score_stone_opponent:
                    return True
                else:
                    if score_stone_current == score_stone_opponent:
                        #special case : the first one to play the stone wins, so it is the opponent that can claim the stone
                        self.board.stones[stone_idx] = StonePosition.PLAYER2 if self.current_player == PlayerId.PLAYER1.value else StonePosition.PLAYER1
                    return False
            else:
                if len(self.board.played_cards[stone_idx][self.current_opponent]) == 2:
                    # if other side has 2 cards, and 3 on player side, compute all combination to see if opponent can win
                    all_not_played_cards = Deck()
                    all_not_played_cards.add(self.board.main_deck)
                    all_not_played_cards.add(self.board.players[PlayerId.PLAYER1.value].hand)
                    all_not_played_cards.add(self.board.players[PlayerId.PLAYER2.value].hand)
                    score_stone_current = self.score_stone(self.board.played_cards[stone_idx][self.current_player])
                    for card in all_not_played_cards:
                        # try to play each card on the stone
                        dummy_stone_cards = Deck()
                        dummy_stone_cards.add(self.board.played_cards[stone_idx][self.current_opponent])
                        dummy_stone_cards.add([card])
                        score_stone_opponent = self.score_stone(dummy_stone_cards)
                        if score_stone_current <= score_stone_opponent:
                            return False
                    return True
        return False

    def step(self, action: List[int]|int):
        terminated = False

        # check move legality
        if (type(action) != int) and (self.action_masks()[action[0]] == False or self.action_masks()[action[1] + MAX_CARDS_PER_PLAYER] == False):
            logger.error(self.observation)
            logger.error(self.compute_score(self.current_player))
            raise Exception(f'Illegal action {action} : Legal actions {self.action_masks()}')
        
        #initial score of the current player
        init_score = self.compute_score(self.current_player)
        #Play card on stone
        if (type(action) != int):
            card = self.board.players[self.current_player].hand.draw_one_by_index(action[0])
            self.board.played_cards[action[1]][self.current_player].add([card])
        #Check if current player can claim a stone
        if self.current_player != -1:
            for stone_idx in range(NB_STONES):
                if self.can_claim_stone(stone_idx):
                    if self.current_player == PlayerId.PLAYER1.value:
                        self.board.stones[stone_idx] = StonePosition.PLAYER1
                    else:
                        self.board.stones[stone_idx] = StonePosition.PLAYER2
        #draw a card from the main deck
        if len(self.board.main_deck) > 0:
            card = self.board.main_deck.draw(1)
            self.board.players[self.current_player].hand.add(card)
        #Compute reward for current player
        after_score = self.compute_score(self.current_player)
        reward = [0., 0.]
        #scale reward into [0;1]
        reward[self.current_player] = (after_score - init_score) / 10
        #check if we are done
        after_score_opponent = self.compute_score(self.current_opponent)
        if (after_score >= 10) or (after_score_opponent >= 10):
            terminated = True
            self.done = True
        else:
            #change player
            self.current_player = abs(self.current_player - 1)

        return self.observation, reward, terminated, False, self._get_info()
    
    def compute_score(self, player: int) -> int:
        """
        Compute the current virtual score for one player
        1 point for each stone claimed, 1 point for each continuous stone claimed
        10 point if winning the game (3 continuois stones claimed or 5 stones claimed)
        """
        score = 0
        for stone_idx in range(NB_STONES):
            if self.board.stones[stone_idx].value == player * 2:
                score += 1
                #check if continuous stone
                if (stone_idx > 0) and (self.board.stones[stone_idx - 1].value == player * 2):
                    score += 1
        if score >= 5:
            score = 10
        return score

    def render(self, **kwargs):
        super().render(**kwargs)
        render_web(self, **kwargs)

    def nicegui_page(self):
        init_web(self)