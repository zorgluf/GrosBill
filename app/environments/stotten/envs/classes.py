import random
from typing import List
from enum import Enum
import numpy as np

NB_STONES = 9
MAX_CARDS_PER_PLAYER = 6

class PlayerId(Enum):
    PLAYER1 = 0
    PLAYER2 = 1

class Card():
    def __init__(self, value, color):
        self._color = color
        self._value = value
        
    @property
    def value(self):
        return self._value

    @property
    def color(self):
        return self._color

class Deck():
    def __init__(self, cards: List[Card] = list()):
        self.cards = list(cards)
    
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
    
    def add(self, cards):
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
        obs = np.zeros([ 9, len(Color)])
        for card in self.cards:
            obs[card.value - 1, card.color.value] = 1
        return obs

class Player():
    def __init__(self, n: PlayerId, name:str = None):
        self.n = n
        if name:
            self.name = name
        else:
            self.name = str(n)
        self._hand = Deck()

    def add_card(self, card):
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

class Color(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2
    YELLOW = 3
    BLACK = 4
    ORANGE = 5

class StonePosition(Enum):
    NEUTRAL = 1
    PLAYER1 = 0
    PLAYER2 = 2


class Board():
    def __init__(self):
        self._players = list()
        self._stones = [ StonePosition.NEUTRAL for _ in range(NB_STONES) ]
        self._played_cards = [ { PlayerId.PLAYER1.value: Deck(), PlayerId.PLAYER2.value: Deck() } for _ in range(NB_STONES) ]
        self._main_deck = ALL_CARDS
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
    
    @property
    def main_deck(self) -> Deck:
        return self._main_deck
        
ALL_CARDS = Deck([ Card(i, c) for i in range(1, 9) for c in Color ])
    
 
