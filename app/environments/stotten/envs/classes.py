import random
from typing import List
from enum import Enum
import numpy as np
import copy

NB_STONES = 9
MAX_CARDS_PER_PLAYER = 6
NB_VALUES = 9   # card values 1..9
NB_COLORS = 6   # see Color enum below
NB_CARDS = NB_VALUES * NB_COLORS  # 54 distinct cards in the game

class PlayerId(Enum):
    PLAYER1 = 0
    PLAYER2 = 1

class Color(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2
    GOLD = 3
    BLACK = 4
    ORANGE = 5

class StonePosition(Enum):
    NEUTRAL = 1
    PLAYER1 = 0
    PLAYER2 = 2

class Card():
    def __init__(self, value, color: Color):
        self._color = color
        self._value = value
        
    @property
    def value(self):
        return self._value

    @property
    def color(self):
        return self._color

    @property
    def identity(self) -> int:
        """
        Stable, unique index in [0, NB_CARDS) for this card.
        Depends only on the card (value, color), never on hand position or draw order.
        This is what the action space and observations key off of so that a given
        action / observation slot always refers to the same physical card.
        """
        return (self._value - 1) + self._color.value * NB_VALUES

    @property
    def token(self) -> int:
        """
        Observation token for embeddings: identity + 1, so that 0 stays free as the
        'no card / empty slot' token (range [1, NB_CARDS]).
        """
        return self.identity + 1

class Deck():
    def __init__(self, cards: List[Card] = None):
        if cards is None:
            self.cards = []
        else:
            self.cards = cards
    
    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self, n) -> List[Card]:
        """
        Draw firt n cards
        """
        drawn = []
        for x in range(n):
            try:
                drawn.append(self.cards.pop())
            except:
                pass
        return drawn
    
    def add(self, cards: List[Card]):
        for card in cards:
            self.cards.append(card)
                
    def __len__(self):
        return len(self.cards)
    
    def __iter__(self):
        return iter(self.cards)
    
    def draw_one_by_index(self, i) -> Card:
        return self.cards.pop(i)
    
    @property
    def observation(self):
        obs = np.zeros([ len(self.cards) ])
        for i, card in enumerate(self.cards):
            obs[i] = card.value + card.color.value * 10
        return obs

class Player():
    def __init__(self, n: PlayerId, name:str = None):
        self.n = n
        if name:
            self.name = name
        else:
            self.name = str(n)
        self._hand = Deck()

    def add_card(self, card: Card):
        """
        Add a card to the player's deck.
        """
        self.deck.add([card])

    @property
    def hand(self) -> Deck:
        """
        Get the player's hand.
        """
        return self._hand


class Board():
    def __init__(self):
        self._players = list()
        self._stones = [ StonePosition.NEUTRAL for _ in range(NB_STONES) ]
        self._played_cards = [ { PlayerId.PLAYER1.value: Deck(), PlayerId.PLAYER2.value: Deck() } for _ in range(NB_STONES) ]
        self._main_deck = copy.deepcopy(ALL_CARDS)
        self._main_deck.shuffle()
    
    def add_player(self,player):
        self._players.append(player)

    @property
    def players(self) -> List[Player]:
        return self._players

    @property
    def stones(self) -> List[StonePosition]:
        return self._stones
    @stones.setter
    def stones(self, val):
        self._stones = val
    
    @property
    def played_cards(self):
        return self._played_cards
    @played_cards.setter
    def played_cards(self, val):
        self._played_cards = val
    
    @property
    def main_deck(self) -> Deck:
        return self._main_deck
        
ALL_CARDS = Deck([ Card(i, c) for i in range(1, 10) for c in Color ])
    
 
