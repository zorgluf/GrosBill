import gymnasium as gym
import numpy as np

import logging as logger
from typing import List, Tuple

from .classes import *
from .render_web import RenderWeb

from utils.env import GBEnv

WIN_SCORE = 10

class SchottenTottenEnv(GBEnv):

    #the card selected through UI to play
    gui_hand_card_selected: int = None

    def __init__(self, player_names: list[str] = None):
        super(SchottenTottenEnv,self).__init__("stotten", 2, player_names)

        #Set up observation space
        self.observation_space = gym.spaces.Dict({
            # presence vector over the NB_CARDS distinct cards: 1 if the current player holds that card.
            # Stationary: a card always occupies the same slot (its identity), regardless of draw order.
            "current_player_hand": gym.spaces.MultiBinary(NB_CARDS),
            "stones": gym.spaces.Box(low=0, high=2, dtype=np.int8, shape = (NB_STONES,)),  # 3 states for each stone position, 0 : current player position, 1 : neutral, 2 : opponent position
            # played cards as embedding tokens: 0 = empty slot, 1..NB_CARDS = card identity+1.
            # [x, 0, y] : y-th card played by the current player on stone x ; [x, 1, y] : opponent side.
            "cards_played": gym.spaces.Box(low=0, high=NB_CARDS, dtype=np.int64, shape= ( NB_STONES, 2, 3 )),
        })

        #Set up the action space
        # action = card_identity * NB_STONES + dest stone_index.
        # Stationary: "play card C on stone S" is always the same integer, independent of hand order.
        self.action_space = gym.spaces.Discrete(NB_CARDS * NB_STONES)

    def reset(self, seed=None):
        super().reset(seed=seed)
        #Set up board
        self.board = Board()
        #Set up players
        self.board.add_player(Player(PlayerId.PLAYER1, self.player_names[0]))
        self.board.add_player(Player(PlayerId.PLAYER2, self.player_names[1]))
        self.current_player = 0
        self.winner_player = None
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

    @staticmethod
    def _played_tokens(cards) -> np.ndarray:
        """Encode up to 3 played cards as embedding tokens (0 = empty, identity+1 otherwise)."""
        tokens = np.zeros(3, dtype=np.int64)
        for i, card in enumerate(cards):
            tokens[i] = card.token
        return tokens

    @property
    def observation(self):
        # Hand as a stationary presence vector over card identities.
        current_player_hand = np.zeros(NB_CARDS, dtype=np.int8)
        for card in self.board.players[self.current_player].hand:
            current_player_hand[card.identity] = 1

        stones = list()
        for position in self.board.stones:
            if position == StonePosition.NEUTRAL:
                stones.append(1)
            elif position == (StonePosition.PLAYER1 if self.current_player == PlayerId.PLAYER1.value else StonePosition.PLAYER2):
                stones.append(0)
            else:
                stones.append(2)
        stones = np.array(stones, dtype=np.int8)

        #build cards_played obs (current player side first, opponent side second), as identity tokens
        cards_by_stone = list()
        for i in range(NB_STONES):
            mine = self._played_tokens(self.board.played_cards[i][self.current_player])
            theirs = self._played_tokens(self.board.played_cards[i][self.current_opponent])
            cards_by_stone.append(np.stack([mine, theirs]))
        cards_played = np.stack(cards_by_stone)

        return dict(
            current_player_hand = current_player_hand,
            stones = stones,
            cards_played = cards_played,
        )
    
    def _get_info(self):
        return {
            "next_step_no_action": False
        }
    
    def action_masks(self):
        """
        Returns the boolean legality mask over the identity-based action space.
        action = card_identity * NB_STONES + stone_idx is legal iff the current player
        holds that card and the target stone has fewer than 3 of their cards.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        # which stones still have room on the current player's side
        open_stones = [s for s in range(NB_STONES)
                       if len(self.board.played_cards[s][self.current_player]) < 3]
        if not open_stones:
            return mask
        for card in self.board.players[self.current_player].hand:
            base = card.identity * NB_STONES
            for stone_idx in open_stones:
                mask[base + stone_idx] = True
        return mask
    
    def score_stone(self, cards: Deck) -> int:
        """
        Compute the score of the played cards on a stone.
        """
        card_sum = sum([card.value for card in cards])
        nb = len(cards)
        if nb == 0:
            return 0
        #same value
        if len(set([card.value for card in cards])) == 1:
            return (300 * (nb - 1) / 2) + card_sum
        if len(set([card.color for card in cards])) == 1:
            if np.all(np.diff(np.sort([card.value for card in cards])) == 1):
                #color run
                return (400 * (nb - 1) / 2) + card_sum
            else:
                #color
                return (200 * (nb - 1) / 2) + card_sum
        if np.all(np.diff(np.sort([card.value for card in cards])) == 1):
            #run
            return (100 * (nb - 1) / 2) + card_sum
        #sum
        return card_sum
        
    
    def can_claim_stone(self, stone_idx) -> Tuple[bool,bool]:
        """
        Check if the current player can claim a stone.
        Check also if the opponent can claim it in case of 3 cards on each side.
        return Tuple(bool, bool) : current player can claim, opponent can claim
        """
        if self.board.stones[stone_idx] != StonePosition.NEUTRAL:
            return (False, False)
        if len(self.board.played_cards[stone_idx][self.current_player]) == 3:
            if len(self.board.played_cards[stone_idx][self.current_opponent]) == 3:
                # if 3 card on each side, check if the current player has the best combination
                score_stone_current = self.score_stone(self.board.played_cards[stone_idx][self.current_player])
                score_stone_opponent = self.score_stone(self.board.played_cards[stone_idx][self.current_opponent])
                if score_stone_current > score_stone_opponent:
                    return (True, False)
                else:
                    if score_stone_current == score_stone_opponent:
                        #special case : the first one to play the stone wins, so it is the opponent that can claim the stone
                        return (False, True)
                    return (False, True)
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
                        if score_stone_current < score_stone_opponent:
                            return (False, False)
                    return (True, False)
        return (False, False)

    def _draw_from_hand_by_identity(self, player: int, identity: int) -> Card:
        """Remove and return the card with the given identity from the player's hand."""
        hand = self.board.players[player].hand
        for idx, card in enumerate(hand):
            if card.identity == identity:
                return hand.draw_one_by_index(idx)
        raise Exception(f'Card identity {identity} not in player {player} hand')

    def step(self, action: int):
        terminated = False
        reward = [0., 0.]
        mover = self.current_player          # player taking this action (rewards are attributed by absolute id)
        opponent = self.current_opponent
        current_before_score = self.compute_score(mover)
        opponent_before_score = self.compute_score(opponent)
        # check move legality
        if self.action_masks()[action] == False:
            logger.error(self.observation)
            logger.error(self.compute_score(mover))
            raise Exception(f'Illegal action {action} : Legal actions {self.action_masks()}')

        #Play card on stone (identity-based action: card_identity * NB_STONES + stone_idx)
        card = self._draw_from_hand_by_identity(mover, action // NB_STONES)
        self.board.played_cards[action % NB_STONES][mover].add([card])
        #Check if current player can claim a stone
        for stone_idx in range(NB_STONES):
            current_claim, opponent_claim = self.can_claim_stone(stone_idx)
            if current_claim:
                if mover == PlayerId.PLAYER1.value:
                    self.board.stones[stone_idx] = StonePosition.PLAYER1
                else:
                    self.board.stones[stone_idx] = StonePosition.PLAYER2
            if opponent_claim:
                if opponent == PlayerId.PLAYER1.value:
                    self.board.stones[stone_idx] = StonePosition.PLAYER1
                else:
                    self.board.stones[stone_idx] = StonePosition.PLAYER2
        #draw a card from the main deck
        if len(self.board.main_deck) > 0:
            card = self.board.main_deck.draw(1)
            self.board.players[mover].hand.add(card)

        current_after_score = self.compute_score(mover)
        opponent_after_score = self.compute_score(opponent)

        if (current_after_score >= WIN_SCORE) or (opponent_after_score >= WIN_SCORE):
            # Someone reached the win condition: terminal, zero-sum +1 / -1 outcome.
            self.winner_player = mover if current_after_score >= opponent_after_score else opponent
            reward[self.winner_player] = 1.0
            reward[abs(self.winner_player - 1)] = -1.0
            terminated = True
            self.done = True
            return self.observation, reward, terminated, False, self._get_info()

        # Dense, zero-sum shaping on claimed stones only (no virtual card values):
        # reward the mover for improving their claim position relative to the opponent;
        # mirror it onto the opponent so the game stays zero-sum.
        relative_gain = ((current_after_score - current_before_score)
                         - (opponent_after_score - opponent_before_score)) / WIN_SCORE
        reward[mover] = relative_gain
        reward[opponent] = -relative_gain

        #change player
        self.current_player = abs(self.current_player - 1)

        # Safety: if the player to move has no legal action (e.g. empty hand after the
        # deck is exhausted), the game cannot continue. End it and decide by current score,
        # so MaskablePPO is never handed an all-False action mask.
        if not self.action_masks().any():
            # played-cards combination values break ties between equal claim counts
            s_p0 = self.compute_score(PlayerId.PLAYER1.value, with_virtual=True)
            s_p1 = self.compute_score(PlayerId.PLAYER2.value, with_virtual=True)
            self.winner_player = PlayerId.PLAYER1.value if s_p0 >= s_p1 else PlayerId.PLAYER2.value
            reward[self.winner_player] = 1.0
            reward[abs(self.winner_player - 1)] = -1.0
            terminated = True
            self.done = True

        return self.observation, reward, terminated, False, self._get_info()
    
    def compute_score(self, player: int, with_virtual: bool = False) -> float:
        """
        Compute the current score for one player
        1 point for each stone claimed, 1 point for each continuous stone claimed
        10 point if winning the game (3 continuous stones claimed or 5 stones claimed)
        with_virtual adds the played-cards combination values (/1000) on top: only used
        as a tie-breaker when the game ends by deck/hand exhaustion, NOT in the reward
        shaping (it rewarded building combos even on lost stones — pure noise for PPO).
        """
        score = 0
        stones = 0
        #check if 5 stones owned
        for stone_idx in range(NB_STONES):
            if self.board.stones[stone_idx].value == player * 2:
                stones += 1
        if stones >= 5:
            return WIN_SCORE
        #check if continuous stone
        cont_score = 0
        for stone_idx in range(NB_STONES):
            if self.board.stones[stone_idx].value == player * 2:
                if stone_idx > 0:
                    if self.board.stones[stone_idx - 1].value == player * 2:
                        cont_score += 1
                        score += 1
                        if cont_score >= 2:
                            return WIN_SCORE
                        continue
            cont_score = 0
        if with_virtual:
            # add virtual card values
            for stone_idx in range(NB_STONES):
                score += float(self.score_stone(self.board.played_cards[stone_idx][player])) / 1000

        return score + stones

    def nicegui_page(self):
        self.render_web = RenderWeb()
        self.render_web.init_web(self)

    def render(self, **kwargs):
        super().render(**kwargs)
        if self.render_mode == "human":
            # Print played cards for opponent
            print("Opponent's played cards:")
            for stone_idx in range(NB_STONES):
                opponent_cards = self.board.played_cards[stone_idx][self.current_opponent]
                print(f"  Stone {stone_idx}: {[f"{card.color}/{card.value}" for card in opponent_cards]}")
            
            # Print stone positions
            print("\nStone positions:")
            stone_positions = []
            for stone_idx, position in enumerate(self.board.stones):
                if position == StonePosition.PLAYER1:
                    pos_str = "Player 1"
                elif position == StonePosition.PLAYER2:
                    pos_str = "Player 2"
                else:
                    pos_str = "Neutral"
                stone_positions.append(f"{pos_str}")
            print("  " + " | ".join(stone_positions))
            
            # Print played cards for current player
            print("\nCurrent player's played cards:")
            for stone_idx in range(NB_STONES):
                player_cards = self.board.played_cards[stone_idx][self.current_player]
                print(f"  Stone {stone_idx}: {[f"{card.color}/{card.value}" for card in player_cards]}")
            
            # Print current player hand
            print("\nCurrent player's hand:")
            current_hand = self.board.players[self.current_player].hand
            print(f"  {[f"{card.color}/{card.value}" for card in current_hand]}")
        else:
            self.render_web.render_web(self, **kwargs)