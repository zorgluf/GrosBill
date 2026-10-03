"""Jamaica (`jamaica`): the pirate race around the island, 3 to 6 players.

Rules as modelled, and the rulings, are in `environments/jamaica/README.md`;
the board, deck, combat die and treasures in `data.py`.

Flow. The engine is a LIFO stack of immutable `Task` tuples plus one pending
`Decision` and two context slots (`battle`, `payment`); no generators, so a
deepcopy stays cheap. A task handler that needs a decision pushes its own
continuation first, then calls `_decide`; `_apply` only mutates state and
pushes tasks. A decision with a single legal action is applied at once
(except CAPTAIN / CARD, which are always presented, so that every seat acts
in round 1 before anything changes: the self-play wrapper drops the rewards
earned before the learner's first step).

Round: the Captain rolls, then decides the die order and his card in one
CAPTAIN decision; the other seats pick a card (CARD), hidden until revealed;
then every seat, from the Captain clockwise, resolves morning then evening
(loads, moves, forks, battles, payments, shortages). A TURN_PAUSE no-action
step follows each seat's resolution when `pause_between_turns` (web UI).

Reward: terminal rank reward (`rank_rewards`, ties shared) plus zero-sum
potential shaping on the public score, with phi = 0 at the start and at the
end, so the undiscounted return of every seat equals its rank reward.

Observation / action layout: `features.py` / `constants.py`.
"""

from __future__ import annotations

import logging as logger
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import gymnasium as gym
import numpy as np

from utils.env import GBEnv

from . import data, features, rules
from .constants import (A_CAPTAIN, A_CARD, A_DEST, A_GIVE_CURSED, A_LOAD_HOLD, A_NOTHING,
                        A_PAY_HOLD, A_POWDER, A_SABRE, A_STEAL_HIDDEN, A_STEAL_HOLD,
                        A_STEAL_POWER, A_TARGET, DEFAULT_PLAYERS, HAND_SIZE, Kind,
                        MAP_HAND_SIZE, MAX_DEST, MAX_PLAYERS, MAX_ROUNDS, MIN_PLAYERS,
                        N_ACTIONS, N_CODES, Phase, Power, POWER_NAMES, RES_NAMES, Res,
                        SABRE_REROLL, START_FOOD, START_GOLD, STAR, SYM_NAMES, SYM_RES, Sym,
                        action_family, card_syms, Family)
from .rules import SIXTH_SLOT, Ship, hold_at
from .track import Track, default_track


class TK(IntEnum):
    """Internal task kinds."""
    ROUND_START = 0
    CHOOSE = 1
    RESOLVE = 2
    ACTION = 3
    LOAD = 4       # a=res b=amount c=source (0 card, 1 loot)
    LAND = 5       # a=node b=lap c=via shortage
    PAY = 6
    PAY_MORE = 7
    BATTLE = 8
    ROUND_END = 9
    PAUSE = 10


class Task(NamedTuple):
    kind: int
    seat: int = -1
    a: int = 0
    b: int = 0
    c: int = 0


class Decision(NamedTuple):
    kind: int          #: Phase
    seat: int
    data: tuple = ()


class BS(IntEnum):
    """Battle stages."""
    TARGET = 0
    ATT_K = 1
    ATT_ROLL = 2
    ATT_SABRE = 3
    ATT_CHECK = 4
    DEF_K = 5
    DEF_ROLL = 6
    DEF_SABRE = 7
    COMPARE = 8
    REWARD = 9
    END = 10


@dataclass(slots=True)
class Battle:
    attacker: int
    node: int
    defender: int = -1
    stage: int = BS.TARGET
    att_k: int = 0
    att_face: int = 0
    att_rolled: bool = False
    def_k: int = 0
    def_face: int = 0
    def_rolled: bool = False
    sabre_used: bool = False
    winner: int = -1
    loot: str = ''


@dataclass(slots=True)
class Payment:
    seat: int
    res: int
    remaining: int
    node: int


def rank_rewards(n: int) -> tuple[float, ...]:
    """Terminal reward per final rank, zero-sum (same table as smallw).

    Rank 0 gets +1; rank j > 0 gets -(2j-1)/(n-1)**2, so the losers share
    exactly -1 (4 players: +1, -1/9, -3/9, -5/9).
    """
    losers = n - 1
    return tuple([1.0] + [-(2 * j - 1) / (losers ** 2) for j in range(1, n)])


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


class JamaicaEnv(GBEnv):
    metadata = {'render_modes': ['human_web']}

    SHAPING_SCALE = 40.0   #: public-score lead (points) per 1.0 of potential
    PHI_CLIP = 0.6

    def __init__(self, n_players: int | None = None, player_names: list[str] | None = None,
                 pause_between_turns: bool = True, track: Track | None = None):
        n = len(player_names) if player_names is not None else (n_players or DEFAULT_PLAYERS)
        if not MIN_PLAYERS <= n <= MAX_PLAYERS:
            raise ValueError(f'Jamaica is played by {MIN_PLAYERS} to {MAX_PLAYERS} players, not {n}')
        super().__init__(name='jamaica', n_players=n, player_names=player_names)
        self.track = track if track is not None else default_track()
        self.pause_between_turns = pause_between_turns
        self.faces = data.COMBAT_FACES
        self.treasures = data.TREASURES
        self.deck_cards = data.DECK
        self.mu_hidden = _mean(v for p, v in self.treasures if p is None)
        self.mu_cursed = _mean(v for p, v in self.treasures if p is None and v < 0)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = features.observation_space()
        self.render_web = None
        self.ships: list[Ship] = []
        self.pending = None
        self.stack: list[Task] = []
        self.event_log: list[tuple[int, str, int]] = []

    # ------------------------------------------------------------------ #
    # gym API
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        n = self.n_players
        self._rig_d6: list[int] = []
        self._rig_combat: list[int] = []
        self.round = 0
        self.turns_taken = 0
        self.captain = int(self.np_random.integers(n))
        ids = list(range(len(self.treasures)))
        self.np_random.shuffle(ids)
        self.pile = ids[:data.N_TREASURES_USED]
        self.removed = ids[data.N_TREASURES_USED:]
        self.lair_token = {node: True for node in self.track.lairs}
        self.known = [0] * len(self.treasures)
        self.public_cursed = [False] * len(self.treasures)
        self.ships = []
        for seat in range(n):
            ship = Ship(seat=seat, node=self.track.pr, lap=0)
            ship.holds[0] = [Res.FOOD, START_FOOD]
            ship.holds[1] = [Res.GOLD, START_GOLD]
            rules.sort_holds(ship)
            ship.deck = list(self.deck_cards)
            self.np_random.shuffle(ship.deck)
            self._refill(ship)
            self.ships.append(ship)
        self.raw_dice = (0, 0)
        self.dice = None
        self.order = [(self.captain + k) % n for k in range(n)]
        self.resolving = -1
        self.cur_action = None          # (seat, slot, symbol, die) being resolved
        self.stack = [Task(TK.ROUND_START)]
        self.pending = None
        self.battle = None
        self.payment = None
        self.last_battle = None
        self.done = False
        self.winner_player = None
        self.terminal_rewards = None
        self.final_scores = None
        self.game_ending = False
        self.event_log = []
        self._last_seat = self.captain
        self._step_rewards = [0.0] * n
        self._version = 0
        self._obs_cache = {}
        self._advance(allow_pause=False)
        self._phi = self._potential()
        return self.observation, self._get_info()

    def step(self, action):
        if self.done:
            raise Exception('the game is over')
        d = self.pending
        a = -1 if action is None else int(action)
        self._step_rewards = [0.0] * self.n_players
        if d.kind == Phase.TURN_PAUSE:
            if a != -1:
                raise Exception(f'only step(-1) during a pause, got {a}')
        elif not (0 <= a < N_ACTIONS and self.action_masks()[a]):
            raise Exception(f'Illegal action {a} in phase {Phase(d.kind).name}: legal {self.legal_actions()}')
        self.pending = None
        if d.kind != Phase.TURN_PAUSE:
            self._apply(d, a)
        self._advance()
        self._update_rewards()
        return self.observation, list(self._step_rewards), self.done, False, self._get_info()

    def _get_info(self):
        return {'next_step_no_action': (not self.done) and self.current_player == -1}

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(N_ACTIONS, dtype=bool)
        if self.pending is not None and self.pending.kind not in (Phase.TURN_PAUSE, Phase.DONE):
            mask[self._legal()] = True
        return mask

    def legal_actions(self) -> list[int]:
        if self.pending is None or self.pending.kind in (Phase.TURN_PAUSE, Phase.DONE):
            return []
        return self._legal()

    @property
    def phase(self) -> int:
        if self.done:
            return Phase.DONE
        return self.pending.kind if self.pending is not None else Phase.DONE

    @property
    def pov_seat(self) -> int:
        """Seat whose point of view the observation takes."""
        if 0 <= self.current_player < self.n_players:
            return self.current_player
        return self._last_seat

    @property
    def observation(self):
        seat = self.pov_seat
        key = (self._version, seat)
        obs = self._obs_cache.get(key)
        if obs is None:
            obs = features.build_obs(self, seat)
            self._obs_cache = {key: obs}
        return obs

    # ------------------------------------------------------------------ #
    # flow
    # ------------------------------------------------------------------ #

    def _push(self, *tasks: Task) -> None:
        self.stack.extend(tasks)

    def _advance(self, allow_pause: bool = True) -> None:
        handlers = self._handlers()
        while self.pending is None and not self.done:
            if not self.stack:
                raise RuntimeError('empty task stack')
            t = self.stack.pop()
            if t.kind == TK.PAUSE:
                if allow_pause and self.pause_between_turns:
                    self.pending = Decision(Phase.TURN_PAUSE, -1)
                continue
            handlers[t.kind](t)
        if self.done:
            self.current_player = self._last_seat
        elif self.pending.kind == Phase.TURN_PAUSE:
            self.current_player = -1
        else:
            self.current_player = self.pending.seat
            self._last_seat = self.pending.seat
        self._version += 1

    def _handlers(self):
        return {
            TK.ROUND_START: self._h_round_start,
            TK.CHOOSE: self._h_choose,
            TK.RESOLVE: self._h_resolve,
            TK.ACTION: self._h_action,
            TK.LOAD: self._h_load,
            TK.LAND: self._h_land,
            TK.PAY: self._h_pay,
            TK.PAY_MORE: self._h_pay_more,
            TK.BATTLE: self._h_battle,
            TK.ROUND_END: self._h_round_end,
        }

    def _decide(self, kind: int, seat: int, data: tuple = ()) -> None:
        self.pending = Decision(kind, seat, data)
        if kind in (Phase.CAPTAIN, Phase.CARD):
            return
        legal = self._legal()
        if not legal:
            raise RuntimeError(f'no legal action for {Phase(kind).name}')
        if len(legal) == 1:
            d, self.pending = self.pending, None
            self._apply(d, legal[0])

    # -- task handlers -------------------------------------------------- #

    def _h_round_start(self, t: Task) -> None:
        n = self.n_players
        self.round += 1
        self.raw_dice = (self._roll_d6(), self._roll_d6())
        self.dice = None
        self.order = [(self.captain + k) % n for k in range(n)]
        self.resolving = -1
        self.cur_action = None
        for ship in self.ships:
            ship.chosen = -1
            ship.revealed = False
        self._log(f'--- Round {self.round}: {self._name(self.captain)} is Captain and rolls '
                  f'{self.raw_dice[0]} and {self.raw_dice[1]}')
        self._push(Task(TK.ROUND_END))
        for seat in reversed(self.order):
            self._push(Task(TK.PAUSE, seat), Task(TK.RESOLVE, seat))
        for seat in reversed(self.order[1:]):
            self._push(Task(TK.CHOOSE, seat))
        self._decide(Phase.CAPTAIN, self.captain)

    def _h_choose(self, t: Task) -> None:
        self._decide(Phase.CARD, t.seat)

    def _h_resolve(self, t: Task) -> None:
        ship = self.ships[t.seat]
        ship.revealed = True
        ship.discard.append(ship.chosen)
        self.resolving = t.seat
        m, e = card_syms(ship.chosen)
        self._log(f'{self._name(t.seat)} reveals {SYM_NAMES[m]} (morning {self.dice[0]}) / '
                  f'{SYM_NAMES[e]} (evening {self.dice[1]})')
        self._push(Task(TK.ACTION, t.seat, 1), Task(TK.ACTION, t.seat, 0))

    def _h_action(self, t: Task) -> None:
        ship = self.ships[t.seat]
        if ship.finished:
            self._log(f'{self._name(t.seat)} has reached Port Royal: the evening action is ignored')
            self.cur_action = None
            return
        m, e = card_syms(ship.chosen)
        sym = m if t.a == 0 else e
        die = self.dice[t.a]
        self.cur_action = (t.seat, t.a, sym, die)
        if sym in SYM_RES:
            self._push(Task(TK.LOAD, t.seat, int(SYM_RES[sym]), die, 0))
            return
        d = 1 if sym == Sym.FWD else -1
        dests = self.track.destinations((ship.node, ship.lap), d, die)
        if len(dests) == 1:
            self._push(Task(TK.LAND, t.seat, dests[0][0], dests[0][1], 0))
        else:
            self._decide(Phase.MOVE_DEST, t.seat, dests)

    def _h_load(self, t: Task) -> None:
        ship = self.ships[t.seat]
        res, amount = t.a, t.b
        slot = rules.empty_slot(ship)
        if slot is not None:
            rules.load_into(ship, slot, res, amount)
            self._log(f'{self._name(t.seat)} loads {amount} {RES_NAMES[res]}'
                      + (' into the 6th hold' if slot == SIXTH_SLOT else ''))
            return
        if not rules.dump_classes(ship, res):
            self._log(f'{self._name(t.seat)} cannot load {RES_NAMES[res]}: every hold already holds it')
            return
        self._decide(Phase.LOAD_HOLD, t.seat, (res, amount, t.c))

    def _h_land(self, t: Task) -> None:
        ship = self.ships[t.seat]
        node, lap, via = t.a, t.b, t.c
        ship.node, ship.lap = node, lap
        if node == self.track.pr and lap >= 1:
            ship.finished = True
            self.game_ending = True
            self._log(f'{self._name(t.seat)} reaches Port Royal! The game ends after this round')
            return
        self._log(f'{self._name(t.seat)} ' + ('falls back' if via else 'moves')
                  + f' to {self.space_label(node, lap)}')
        self._push(Task(TK.PAY, t.seat, via))
        if node != self.track.pr and self.foes_at(t.seat, node):
            self.battle = Battle(attacker=t.seat, node=node)
            self._push(Task(TK.BATTLE, t.seat))

    def _h_pay(self, t: Task) -> None:
        ship = self.ships[t.seat]
        node = ship.node
        kind, cost = self.track.kind[node], self.track.cost[node]
        if kind == Kind.LAIR:
            if self.lair_token.get(node):
                self._take_treasure(t.seat, node)
            return
        res = rules.space_res(kind)
        if res == Res.EMPTY or cost <= 0:
            return
        if rules.total(ship, res) >= cost:
            self.payment = Payment(t.seat, res, cost, node)
            self._push(Task(TK.PAY_MORE, t.seat))
            return
        paid = rules.drain_all(ship, res)
        self._log(f'{self._name(t.seat)} is short of {RES_NAMES[res]}: pays {paid} of {cost} '
                  f'and falls back')
        targets = self.track.retreat_targets(
            (node, ship.lap), lambda nd: rules.can_pay(ship, self.track.kind[nd], self.track.cost[nd]))
        if len(targets) == 1:
            self._push(Task(TK.LAND, t.seat, targets[0][0], targets[0][1], 1))
        else:
            self._decide(Phase.RETREAT_DEST, t.seat, targets)

    def _h_pay_more(self, t: Task) -> None:
        p, ship = self.payment, self.ships[t.seat]
        while p.remaining > 0:
            classes = rules.pay_classes(ship, p.res)
            if len(classes) > 1 and rules.total(ship, p.res) > p.remaining:
                self._push(t)
                self._decide(Phase.PAY_HOLD, t.seat, (p.res,))
                return
            p.remaining -= rules.take(ship, classes[0], p.remaining)
        cost = self.track.cost[p.node]
        self._log(f'{self._name(t.seat)} pays {cost} {RES_NAMES[p.res]}')
        self.payment = None

    def _h_battle(self, t: Task) -> None:
        b = self.battle
        n = self.n_players
        while True:
            if b.stage == BS.TARGET:
                foes = self.foes_at(b.attacker, b.node)
                b.stage = BS.ATT_K
                if len(foes) > 1:
                    self._push(t)
                    self._decide(Phase.TARGET, b.attacker, tuple(foes))
                    return
                b.defender = foes[0]
                self._log_battle_start(b)
            elif b.stage == BS.ATT_K:
                b.stage = BS.ATT_ROLL
                self._push(t)
                self._decide(Phase.ATTACK_POWDER, b.attacker)
                return
            elif b.stage == BS.ATT_ROLL:
                b.att_face = self._roll_combat()
                b.att_rolled = True
                b.stage = BS.ATT_SABRE
                self._log(f'{self._name(b.attacker)} spends {b.att_k} gunpowder and rolls '
                          f'{self._face_name(b.att_face)}')
            elif b.stage == BS.ATT_SABRE:
                b.stage = BS.ATT_CHECK
                window = self._sabre_window(b, 0)
                if window is not None:
                    self._push(t)
                    self._decide(*window)
                    return
            elif b.stage == BS.ATT_CHECK:
                if b.att_face == STAR:
                    b.winner = b.attacker
                    b.stage = BS.REWARD
                    self._log(f'Star! {self._name(b.attacker)} wins at once')
                else:
                    b.stage = BS.DEF_K
            elif b.stage == BS.DEF_K:
                b.stage = BS.DEF_ROLL
                self._push(t)
                self._decide(Phase.DEFENSE_POWDER, b.defender)
                return
            elif b.stage == BS.DEF_ROLL:
                b.def_face = self._roll_combat()
                b.def_rolled = True
                b.stage = BS.DEF_SABRE
                self._log(f'{self._name(b.defender)} spends {b.def_k} gunpowder and rolls '
                          f'{self._face_name(b.def_face)}')
            elif b.stage == BS.DEF_SABRE:
                b.stage = BS.COMPARE
                window = self._sabre_window(b, 1)
                if window is not None:
                    self._push(t)
                    self._decide(*window)
                    return
            elif b.stage == BS.COMPARE:
                att, dfn = self.ships[b.attacker], self.ships[b.defender]
                r = rules.battle_winner(b.att_face, b.att_k, rules.beth(att),
                                        b.def_face, b.def_k, rules.beth(dfn))
                if r == 0:
                    self._log(f'Tie ({self.strength(b, 0)} each): nothing happens')
                    b.stage = BS.END
                else:
                    b.winner = b.attacker if r > 0 else b.defender
                    self._log(f'{self._name(b.winner)} wins the battle '
                              f'({self.strength(b, 0)} against {self.strength(b, 1)})')
                    b.stage = BS.REWARD
            elif b.stage == BS.REWARD:
                b.stage = BS.END
                self._push(t)
                self._decide(Phase.REWARD, b.winner)
                return
            else:
                self.last_battle = b
                self.battle = None
                return

    def _h_round_end(self, t: Task) -> None:
        self.resolving = -1
        self.cur_action = None
        self.turns_taken = self.round
        if self.game_ending or self.round >= MAX_ROUNDS:
            self._end_game()
            return
        for ship in self.ships:
            self._refill(ship)
        self.captain = (self.captain + 1) % self.n_players
        self._push(Task(TK.ROUND_START))

    # ------------------------------------------------------------------ #
    # decisions
    # ------------------------------------------------------------------ #

    def _legal(self) -> list[int]:
        d = self.pending
        k = d.kind
        n = self.n_players
        if k == Phase.CAPTAIN:
            ship = self.ships[d.seat]
            codes = sorted(set(ship.hand))
            orders = [0] if self.raw_dice[0] == self.raw_dice[1] else [0, 1]
            return [A_CAPTAIN + o * N_CODES + c for o in orders for c in codes]
        if k == Phase.CARD:
            return [A_CARD + c for c in sorted(set(self.ships[d.seat].hand))]
        if k in (Phase.MOVE_DEST, Phase.RETREAT_DEST):
            return [A_DEST + i for i in range(len(d.data))]
        if k == Phase.LOAD_HOLD:
            return [A_LOAD_HOLD + s for s in rules.dump_classes(self.ships[d.seat], d.data[0])]
        if k == Phase.PAY_HOLD:
            return [A_PAY_HOLD + s for s in rules.pay_classes(self.ships[d.seat], d.data[0])]
        if k == Phase.TARGET:
            return sorted(A_TARGET + (c - d.seat) % n - 1 for c in d.data)
        if k == Phase.ATTACK_POWDER:
            b = self.battle
            att, dfn = self.ships[b.attacker], self.ships[b.defender]
            kmax = rules.attacker_powder_max(rules.total(att, Res.POWDER), rules.total(dfn, Res.POWDER),
                                             rules.beth(att), rules.beth(dfn), self.faces)
            return [A_POWDER + i for i in range(kmax + 1)]
        if k == Phase.DEFENSE_POWDER:
            b = self.battle
            dfn = self.ships[b.defender]
            opts = rules.defender_powder_options(rules.total(dfn, Res.POWDER), self.strength(b, 0),
                                                 rules.beth(dfn), self.faces)
            return [A_POWDER + i for i in opts]
        if k == Phase.SABRE:
            return [A_SABRE, A_SABRE + 1]
        if k == Phase.REWARD:
            return self._reward_options()
        return []

    def _reward_options(self) -> list[int]:
        b = self.battle
        winner = self.ships[b.winner]
        loser = self.ships[self.loser(b)]
        acts = []
        seen = set()
        for slot in rules.slots(loser):
            h = hold_at(loser, slot)
            if h[0] == Res.EMPTY or not rules.can_load(winner, h[0]):
                continue
            key = (h[0], h[1], slot == SIXTH_SLOT)
            if key not in seen:
                seen.add(key)
                acts.append(A_STEAL_HOLD + slot)
        for p in Power:
            if loser.has(p):
                acts.append(A_STEAL_POWER + int(p))
        if loser.hidden:
            acts.append(A_STEAL_HIDDEN)
        if any(self.treasure_value(tid) < 0 for tid in winner.hidden):
            acts.append(A_GIVE_CURSED)
        acts.append(A_NOTHING)
        return acts

    def _apply(self, d: Decision, a: int) -> None:
        k = d.kind
        ship = self.ships[d.seat] if d.seat >= 0 else None
        if k == Phase.CAPTAIN:
            order, code = divmod(a - A_CAPTAIN, N_CODES)
            hi, lo = max(self.raw_dice), min(self.raw_dice)
            self.dice = (hi, lo) if order == 0 else (lo, hi)
            ship.chosen = code
            ship.hand.remove(code)
            self._log(f'{self._name(d.seat)} places the dice: morning {self.dice[0]}, '
                      f'evening {self.dice[1]}')
            self._log(f'you pick {self.card_name(code)}', seat=d.seat)
        elif k == Phase.CARD:
            code = a - A_CARD
            ship.chosen = code
            ship.hand.remove(code)
            self._log(f'you pick {self.card_name(code)}', seat=d.seat)
        elif k in (Phase.MOVE_DEST, Phase.RETREAT_DEST):
            node, lap = d.data[a - A_DEST]
            self._push(Task(TK.LAND, d.seat, node, lap, 1 if k == Phase.RETREAT_DEST else 0))
        elif k == Phase.LOAD_HOLD:
            slot = a - A_LOAD_HOLD
            res, amount = d.data[0], d.data[1]
            dumped = rules.load_into(ship, slot, res, amount)
            self._log(f'{self._name(d.seat)} throws {dumped[1]} {RES_NAMES[dumped[0]]} overboard '
                      f'and loads {amount} {RES_NAMES[res]}')
        elif k == Phase.PAY_HOLD:
            slot = a - A_PAY_HOLD
            self.payment.remaining -= rules.take(ship, slot, self.payment.remaining)
        elif k == Phase.TARGET:
            b = self.battle
            b.defender = (d.seat + a - A_TARGET + 1) % self.n_players
            self._log_battle_start(b)
        elif k == Phase.ATTACK_POWDER:
            b = self.battle
            b.att_k = a - A_POWDER
            rules.spend_powder(ship, b.att_k)
        elif k == Phase.DEFENSE_POWDER:
            b = self.battle
            b.def_k = a - A_POWDER
            rules.spend_powder(ship, b.def_k)
        elif k == Phase.SABRE:
            b = self.battle
            if a - A_SABRE == SABRE_REROLL:
                b.sabre_used = True
                which = d.data[0]
                face = self._roll_combat()
                roller = b.attacker if which == 0 else b.defender
                if which == 0:
                    b.att_face = face
                else:
                    b.def_face = face
                self._log(f"{self._name(d.seat)} uses Saran's Sabre: {self._name(roller)} rerolls "
                          f'{self._face_name(face)}')
        elif k == Phase.REWARD:
            self._apply_reward(a)
        else:
            raise RuntimeError(f'cannot apply {k}')

    def _apply_reward(self, a: int) -> None:
        b = self.battle
        w, l = b.winner, self.loser(b)
        winner, loser = self.ships[w], self.ships[l]
        if A_STEAL_HOLD <= a < A_STEAL_POWER:
            slot = a - A_STEAL_HOLD
            h = hold_at(loser, slot)
            res, cnt = int(h[0]), int(h[1])
            h[0], h[1] = Res.EMPTY, 0
            rules.sort_holds(loser)
            b.loot = f'{cnt} {RES_NAMES[res]}'
            self._log(f'{self._name(w)} takes the {cnt} {RES_NAMES[res]} of a hold of {self._name(l)}')
            self._push(Task(TK.LOAD, w, res, cnt, 1))
        elif A_STEAL_POWER <= a < A_STEAL_HIDDEN:
            p = Power(a - A_STEAL_POWER)
            loser.powers &= ~(1 << int(p))
            winner.powers |= 1 << int(p)
            if p == Power.SIXTH:
                winner.sixth, loser.sixth = loser.sixth, None
            b.loot = POWER_NAMES[p]
            self._log(f'{self._name(w)} takes {POWER_NAMES[p]} from {self._name(l)}')
        elif a == A_STEAL_HIDDEN:
            i = int(self.np_random.integers(len(loser.hidden)))
            tid = loser.hidden.pop(i)
            winner.hidden.append(tid)
            self.known[tid] |= 1 << w
            b.loot = 'a face-down treasure'
            self._log(f'{self._name(w)} takes a face-down treasure from {self._name(l)}')
            self._log(f'it is worth {self.treasure_value(tid):+d}', seat=w)
        elif a == A_GIVE_CURSED:
            tid = min((t for t in winner.hidden if self.treasure_value(t) < 0), key=self.treasure_value)
            winner.hidden.remove(tid)
            loser.hidden.append(tid)
            self.known[tid] |= 1 << l
            self.public_cursed[tid] = True
            b.loot = 'a cursed treasure (given)'
            self._log(f'{self._name(w)} gives a cursed treasure to {self._name(l)}')
            self._log(f'you receive a cursed treasure worth {self.treasure_value(tid):+d}', seat=l)
        else:
            b.loot = 'nothing'
            self._log(f'{self._name(w)} takes nothing')

    def _sabre_window(self, b: Battle, which: int):
        """(Phase.SABRE, holder, (which,)) when the Sabre can matter, else None.

        `which` is the roll just made: 0 attacker, 1 defender.
        """
        if b.sabre_used:
            return None
        holder = next((s for s in (b.attacker, b.defender) if self.ships[s].has(Power.SABRE)), None)
        if holder is None:
            return None
        roller = b.attacker if which == 0 else b.defender
        face = b.att_face if which == 0 else b.def_face
        if holder == roller:
            if face == STAR:
                return None
        elif face != STAR and face == min(rules.numeric_faces(self.faces)):
            return None
        return (Phase.SABRE, holder, (which,))

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _roll_d6(self) -> int:
        if self._rig_d6:
            return self._rig_d6.pop(0)
        return int(self.np_random.integers(1, 7))

    def _roll_combat(self) -> int:
        if self._rig_combat:
            return self._rig_combat.pop(0)
        return int(self.faces[int(self.np_random.integers(len(self.faces)))])

    def _refill(self, ship: Ship) -> None:
        limit = MAP_HAND_SIZE if ship.has(Power.MAP) else HAND_SIZE
        while len(ship.hand) < limit:
            if not ship.deck:
                if not ship.discard:
                    break
                ship.deck, ship.discard = ship.discard, []
                self.np_random.shuffle(ship.deck)
            ship.hand.append(ship.deck.pop())

    def _take_treasure(self, seat: int, node: int) -> None:
        self.lair_token[node] = False
        tid = self.pile.pop()
        power, value = self.treasures[tid]
        ship = self.ships[seat]
        if power is not None:
            ship.powers |= 1 << int(power)
            if power == Power.SIXTH:
                ship.sixth = [Res.EMPTY, 0]
            self._log(f'{self._name(seat)} plunders the lair: {POWER_NAMES[power]}!')
        else:
            ship.hidden.append(tid)
            self.known[tid] |= 1 << seat
            self._log(f'{self._name(seat)} plunders the lair: a face-down treasure')
            self._log(f'it is worth {value:+d}', seat=seat)

    def foes_at(self, seat: int, node: int) -> list[int]:
        """Other ships on `node` (never at Port Royal)."""
        if node == self.track.pr:
            return []
        return [o.seat for o in self.ships if o.seat != seat and o.node == node and not o.finished]

    def loser(self, b: Battle) -> int:
        return b.defender if b.winner == b.attacker else b.attacker

    def strength(self, b: Battle, side: int) -> int:
        """Combat strength of the attacker (0) / defender (1), star excluded."""
        if side == 0:
            return b.att_face + b.att_k + rules.beth(self.ships[b.attacker])
        return b.def_face + b.def_k + rules.beth(self.ships[b.defender])

    def treasure_value(self, tid: int) -> int:
        return int(self.treasures[tid][1])

    def final_score(self, seat: int) -> int:
        ship = self.ships[seat]
        return (self.track.pos_score(ship.node, ship.lap) + rules.total(ship, Res.GOLD)
                + sum(self.treasure_value(t) for t in ship.hidden))

    def public_score(self, seat: int) -> float:
        """Score as everybody sees it (face-down cards at their mean value)."""
        ship = self.ships[seat]
        hidden = sum(self.mu_cursed if self.public_cursed[t] else self.mu_hidden for t in ship.hidden)
        return self.track.pos_score(ship.node, ship.lap) + rules.total(ship, Res.GOLD) + hidden

    def _potential(self) -> list[float]:
        n = self.n_players
        s = [self.public_score(i) for i in range(n)]
        tot = sum(s)
        return [float(np.clip((s[i] - (tot - s[i]) / (n - 1)) / self.SHAPING_SCALE,
                              -self.PHI_CLIP, self.PHI_CLIP)) for i in range(n)]

    def _update_rewards(self) -> None:
        if self.done:
            for i in range(self.n_players):
                self._step_rewards[i] += self.terminal_rewards[i] - self._phi[i]
            self._phi = [0.0] * self.n_players
            return
        phi = self._potential()
        for i in range(self.n_players):
            self._step_rewards[i] += phi[i] - self._phi[i]
        self._phi = phi

    def _end_game(self) -> None:
        n = self.n_players
        self.done = True
        self.pending = None
        scores = [self.final_score(i) for i in range(n)]
        rem = [self.track.remaining(s.node, s.lap) for s in self.ships]
        order = sorted(range(n), key=lambda i: (-scores[i], rem[i], i))
        self.winner_player = order[0]
        table = rank_rewards(n)
        rewards = [0.0] * n
        rank = 0
        while rank < n:
            group = [rank]
            while (rank + len(group) < n
                   and (scores[order[rank + len(group)]], rem[order[rank + len(group)]])
                   == (scores[order[rank]], rem[order[rank]])):
                group.append(rank + len(group))
            share = sum(table[r] for r in group) / len(group)
            for r in group:
                rewards[order[r]] = share
            rank += len(group)
        self.terminal_rewards = rewards
        self.final_scores = scores
        standings = ', '.join(f'#{r + 1} {self._name(s)}: {scores[s]}' for r, s in enumerate(order))
        self._log(f'---- GAME OVER after {self.round} rounds: {standings} ----')
        logger.info(f'jamaica game over: {standings}')

    # -- hidden information -------------------------------------------- #

    def redeterminize(self, pov_player: int) -> None:
        """Resample everything `pov_player` cannot see, keeping what it knows.

        Reshuffles its own draw pile; each opponent's hand + draw pile (+ the
        chosen card while unrevealed) jointly; the treasure pile, the removed
        cards and every face-down card whose value `pov_player` does not know
        (a card publicly known as cursed stays cursed).
        """
        rng = self.np_random
        me = self.ships[pov_player]
        rng.shuffle(me.deck)
        for ship in self.ships:
            if ship.seat == pov_player:
                continue
            hidden_chosen = ship.chosen >= 0 and not ship.revealed
            pool = ship.hand + ship.deck + ([ship.chosen] if hidden_chosen else [])
            rng.shuffle(pool)
            nh, nd = len(ship.hand), len(ship.deck)
            ship.hand = pool[:nh]
            ship.deck = pool[nh:nh + nd]
            if hidden_chosen:
                ship.chosen = pool[nh + nd]
        bit = 1 << pov_player
        slots_ = [(ship, i) for ship in self.ships if ship.seat != pov_player
                  for i, tid in enumerate(ship.hidden) if not self.known[tid] & bit]
        free = self.pile + self.removed + [ship.hidden[i] for ship, i in slots_]
        info = {tid: (self.known[tid], self.public_cursed[tid]) for tid in free}
        is_cursed = lambda t: self.treasures[t][0] is None and self.treasures[t][1] < 0
        cursed = [t for t in free if is_cursed(t)]
        rng.shuffle(cursed)
        new_ids = {}
        for ship, i in slots_:
            if info[ship.hidden[i]][1]:          # publicly known to be cursed
                new_ids[(ship.seat, i)] = cursed.pop()
        taken = set(new_ids.values())
        facedown = [t for t in free if self.treasures[t][0] is None and t not in taken]
        rng.shuffle(facedown)
        for ship, i in slots_:
            if (ship.seat, i) not in new_ids:
                new_ids[(ship.seat, i)] = facedown.pop()
        used = set(new_ids.values())
        leftover = [t for t in free if t not in used]
        rng.shuffle(leftover)
        for tid in free:
            self.known[tid], self.public_cursed[tid] = 0, False
        for (seat, i), tid in new_ids.items():
            old = self.ships[seat].hidden[i]
            self.known[tid], self.public_cursed[tid] = info[old]
        for (seat, i), tid in new_ids.items():
            self.ships[seat].hidden[i] = tid
        n_pile = len(self.pile)
        self.pile = leftover[:n_pile]
        self.removed = leftover[n_pile:]
        self._version += 1

    # -- text ----------------------------------------------------------- #

    def _name(self, seat: int) -> str:
        return self.player_names[seat]

    def _face_name(self, face: int) -> str:
        return 'a star' if face == STAR else str(face)

    def card_name(self, code: int) -> str:
        m, e = card_syms(code)
        return f'{SYM_NAMES[m]} / {SYM_NAMES[e]}'

    def space_label(self, node: int, lap: int = 0) -> str:
        kind, cost = self.track.kind[node], self.track.cost[node]
        if kind == Kind.PORT_ROYAL:
            text = 'Port Royal'
        elif kind == Kind.PORT:
            text = f'a port ({cost} gold)'
        elif kind == Kind.SEA:
            text = f'the sea ({cost} food)'
        else:
            text = 'a pirate lair'
        return f'{text} [space {node}{", behind the start" if lap < 0 else ""}]'

    def _log_battle_start(self, b: Battle) -> None:
        self._log(f'Battle! {self._name(b.attacker)} attacks {self._name(b.defender)}')

    def _log(self, text: str, seat: int | None = None) -> None:
        """Public line, or a private one visible to `seat` only."""
        mask = (1 << self.n_players) - 1 if seat is None else 1 << seat
        self.event_log.append((self.round, text, mask))
        logger.debug(text)

    def log_lines(self, pov: int | None) -> list[tuple[int, str]]:
        """Log lines visible to `pov` (every line when pov is None / -1)."""
        if pov is None or pov < 0:
            return [(r, t) for r, t, _ in self.event_log]
        return [(r, t) for r, t, m in self.event_log if m >> pov & 1]

    def describe_action(self, a: int) -> str:
        fam, i = action_family(a)
        d = self.pending
        if fam == Family.CAPTAIN:
            order, code = divmod(i, N_CODES)
            hi, lo = max(self.raw_dice), min(self.raw_dice)
            m_die, e_die = (hi, lo) if order == 0 else (lo, hi)
            return f'dice {m_die}/{e_die}, play {self.card_name(code)}'
        if fam == Family.CARD:
            return f'play {self.card_name(i)}'
        if fam == Family.DEST and d is not None and i < len(d.data):
            return f'go to {self.space_label(*d.data[i])}'
        if fam == Family.LOAD_HOLD:
            ship = self.ships[d.seat]
            h = hold_at(ship, i)
            return f'empty the hold of {h[1]} {RES_NAMES[h[0]]}'
        if fam == Family.PAY_HOLD:
            ship = self.ships[d.seat]
            h = hold_at(ship, i)
            return f'pay from the hold of {h[1]} {RES_NAMES[h[0]]}'
        if fam == Family.POWDER:
            return f'spend {i} gunpowder'
        if fam == Family.TARGET:
            return f'attack {self._name((d.seat + i + 1) % self.n_players)}'
        if fam == Family.SABRE:
            return 'use the Sabre: reroll' if i == SABRE_REROLL else 'keep the roll'
        if fam == Family.STEAL_HOLD:
            loser = self.ships[self.loser(self.battle)]
            h = hold_at(loser, i)
            return f'take the hold of {h[1]} {RES_NAMES[h[0]]}'
        if fam == Family.STEAL_POWER:
            return f'take {POWER_NAMES[Power(i)]}'
        if fam == Family.STEAL_HIDDEN:
            return 'take a face-down treasure'
        if fam == Family.GIVE_CURSED:
            return 'give a cursed treasure'
        return 'take nothing'

    # -- rendering ------------------------------------------------------ #

    def nicegui_page(self):
        from .render_web import RenderWeb
        self.render_web = RenderWeb()
        self.render_web.init_web(self)

    def render(self, **kwargs):
        super().render(**kwargs)
        if self.render_web is not None:
            self.render_web.render_web(self, **kwargs)


assert MAX_DEST >= 2
