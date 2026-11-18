"""Toy Battle (Paolo Mori & Alessandro Zucchini, Repos Production 2025) — 2 players.

`ToyBattleEnv` plays one terrain (Castle Field by default, `terrain=<key>`),
or draws the terrain of every game among `terrains` at `reset()` (or takes
`options={'terrain': key}`). The spaces are the same for the 8 terrains
(padded to the largest one), so one network can play them all: that is what
`ToyBattleAllTerrainsEnv`, registered as `toybattle`, is for.

Seat 0 plays Blue and seat 1 Red; the starting player is drawn at `reset()`.
Rules as modelled, rulings and the observation layout: `../README.md`.
"""

from __future__ import annotations

import random

import gymnasium as gym
import numpy as np

from utils.env import GBEnv

from .constants import (A_DRAW, A_PLACE, A_SKIP, A_TARGET, A_TYPE, COLOR_NAMES, COPIES, Effect,
                        FIRST_HAND, MAX_BASES, MAX_HQ, MAX_REGIONS, MAX_TURNS, N_ACTIONS,
                        N_PHASES, N_SLOTS, N_TYPES, Phase, RACK_MAX, REMOVED, SHAPING, TRIGGERED,
                        TROOP_NAMES, Troop, place_action)
from .terrains import N_TERRAINS, TERRAIN_KEYS, TERRAINS, Terrain

# observation sizes
BASE_F = 36
REGION_F = MAX_BASES + 7
HQ_F = 2 + N_TYPES + MAX_BASES
GLOB_F = N_TERRAINS + N_PHASES + 1 + 3 + 4 + N_TYPES + N_TYPES + 1 + 4 * N_TYPES + 2


class ToyBattleEnv(GBEnv):

    render_web = None

    def __init__(self, player_names: list[str] | None = None, terrain: str | None = None,
                 terrains=None, shaping: float = SHAPING, max_turns: int = MAX_TURNS):
        super().__init__('toybattle', 2, player_names)
        pool = (terrain,) if terrain is not None else tuple(terrains) if terrains else ('castle',)
        for key in pool:
            if key not in TERRAINS:
                raise ValueError(f'unknown terrain {key!r}: one of {", ".join(TERRAIN_KEYS)}')
        self.terrain_pool = pool
        self.shaping = shaping
        self.max_turns = max_turns

        def box(*shape):
            return gym.spaces.Box(0.0, 1.0, shape=shape, dtype=np.float32)

        self.observation_space = gym.spaces.Dict({
            'glob': box(GLOB_F),                        # terrain, phase, medals, racks, discards
            'bases': box(MAX_BASES, BASE_F),            # one row per base slot (canonical view)
            'adj': box(MAX_BASES, MAX_BASES),           # paths between the base slots
            'regions': box(MAX_REGIONS, REGION_F),      # bases around each region, medals, shares
            'hqs': box(2 * MAX_HQ, HQ_F),               # my H.Q. then the enemy's
            'mask': box(N_ACTIONS),                     # the legal actions
        })
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.terrain_key = pool[0]
        self.rng = random.Random()

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #

    @property
    def terrain(self) -> Terrain:
        return TERRAINS[self.terrain_key]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = random.Random(seed)
        key = (options or {}).get('terrain')
        if key is None:
            key = self.terrain_pool[0] if len(self.terrain_pool) == 1 else self.rng.choice(self.terrain_pool)
        if key not in TERRAINS:
            raise ValueError(f'unknown terrain {key!r}')
        self.terrain_key = key
        T = self.terrain

        #: stacks[base] = [(owner, troop), ...] bottom first; only the last one is visible
        self.stacks = [[] for _ in T.bases]
        self.racks = [[0] * N_TYPES for _ in range(2)]
        #: troops of the rack laid face down by a Battlefield special base
        self.blocked = [[0] * N_TYPES for _ in range(2)]
        #: face-up discards, by owner (Jumbo, XB-42)
        self.discard = [[0] * N_TYPES for _ in range(2)]
        self.reserve = [[], []]             #: face down, drawn from the end
        self.removed = [[], []]             #: the 4 tiles put back in the box unseen
        for p in range(2):
            tiles = [t for t in range(N_TYPES) for _ in range(COPIES)]
            self.rng.shuffle(tiles)
            self.removed[p], self.reserve[p] = tiles[:REMOVED], tiles[REMOVED:]
        self.medals = [0, 0]
        self.region_owner = [-1] * len(T.regions)   #: who took the medals of each region
        self.first_player = self.rng.randrange(2)
        for p, n in zip((self.first_player, 1 - self.first_player), FIRST_HAND):
            self._draw(p, n, log=False)

        self.current_player = self.first_player
        self.phase = Phase.MAIN
        self.pending_base = None            #: base the pending effect comes from
        self.erupt_src = None               #: Volcanic Jungle: the base of the troop being moved
        self.base_queue = []                #: special bases whose effect is still to apply this turn
        self.turns_taken = 0
        self.winner_player = None
        self.end_reason = ''
        self.captured = None                #: (seat whose H.Q. fell, H.Q. index, troop)
        self.marks = []                     #: bases touched since the start of the last turn (UI)
        self.event_log = []                 #: (turn, public text, seat, text for that seat)
        self._log(f'{T.name}: {T.objective} medals to win. {self._who(self.first_player)} starts.')
        return self.observation, self._info()

    @staticmethod
    def _info():
        return {'next_step_no_action': False}

    # ------------------------------------------------------------------ #
    # board queries
    # ------------------------------------------------------------------ #

    def owner(self, base: int) -> int:
        """Seat occupying `base` (its visible troop), -1 if empty."""
        stack = self.stacks[base]
        return stack[-1][0] if stack else -1

    def rack_size(self, p: int) -> int:
        return sum(self.racks[p])

    def reach(self, p: int) -> set[int]:
        """Bases occupied by `p` and connected to one of their H.Q."""
        T = self.terrain
        seen: set[int] = set()
        todo = [b for hq in T.hqs[p] for b in hq.bases if self.owner(b) == p]
        while todo:
            b = todo.pop()
            if b in seen:
                continue
            seen.add(b)
            todo.extend(n for n in T.adj[b] if n not in seen and self.owner(n) == p)
        return seen

    def frontier(self, p: int, reach: set[int] | None = None) -> set[int]:
        """Bases `p` may place on as far as the connection rule goes."""
        T = self.terrain
        reach = self.reach(p) if reach is None else reach
        out = set(reach)
        for hq in T.hqs[p]:
            out.update(hq.bases)
        for b in reach:
            out.update(T.adj[b])
        return out

    def can_cover(self, p: int, troop: int, base: int) -> bool:
        stack = self.stacks[base]
        if not stack:
            return True
        owner, top = stack[-1]
        return owner == p or troop == Troop.KWAK or top == Troop.KWAK or troop > top

    @staticmethod
    def _gate_ok(gate, troop: int) -> bool:
        return gate is None or troop == Troop.KWAK or troop in gate

    def placements(self, p: int) -> list[tuple[int, int]]:
        """Legal (troop, slot) placements of `p`, slots in `p`'s view."""
        T = self.terrain
        troops = [t for t in range(N_TYPES) if self.racks[p][t] - self.blocked[p][t] > 0]
        if not troops:
            return []
        reach = self.reach(p)
        frontier = self.frontier(p, reach)
        shielded = T.effect == Effect.SHIELD
        out = []
        for b, base in enumerate(T.bases):
            slot = T.slots[p][b]
            for t in troops:
                if not self._gate_ok(base.gate, t):
                    continue
                if b not in frontier and not (t == Troop.HOOK and not (shielded and base.special)):
                    continue
                if self.can_cover(p, t, b):
                    out.append((t, slot))
        for k, hq in enumerate(T.hqs[1 - p]):
            if any(b in reach for b in hq.bases):
                out.extend((t, MAX_BASES + k) for t in troops if self._gate_ok(hq.gate, t))
        return out

    def jumbo_targets(self, p: int, base: int) -> list[int]:
        return [b for b in self.terrain.adj[base] if self.owner(b) == 1 - p]

    def recall_targets(self, p: int, origin: int) -> list[int]:
        return [b for b in range(len(self.stacks)) if b != origin and self.owner(b) == p]

    def can_draw(self, p: int) -> bool:
        return self.rack_size(p) < RACK_MAX and bool(self.reserve[p])

    def can_act(self, p: int) -> bool:
        return self.can_draw(p) or bool(self.placements(p))

    # ------------------------------------------------------------------ #
    # actions
    # ------------------------------------------------------------------ #

    def target_bases(self) -> list[int]:
        """Bases the current TARGET decision may pick (empty in the other phases)."""
        p, origin = self.current_player, self.pending_base
        if self.phase == Phase.JUMBO:
            return self.jumbo_targets(p, origin)
        if self.phase == Phase.RECALL:
            return self.recall_targets(p, origin)
        if self.phase == Phase.ERUPT_SRC:
            return self.jumbo_targets(p, origin)
        if self.phase == Phase.ERUPT_DST:
            return list(self.terrain.adj[self.erupt_src])
        return []

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(N_ACTIONS, dtype=bool)
        if self.done:
            return mask
        p, T = self.current_player, self.terrain
        if self.phase in (Phase.MAIN, Phase.EXTRA):
            for t, slot in self.placements(p):
                mask[place_action(t, slot)] = True
            if self.phase == Phase.MAIN:
                mask[A_DRAW] = self.can_draw(p)
            else:
                mask[A_SKIP] = True
        elif self.phase == Phase.EXHUME:
            for t in range(N_TYPES):
                mask[A_TYPE + t] = self.discard[p][t] > 0
            mask[A_SKIP] = True
        else:
            for b in self.target_bases():
                mask[A_TARGET + T.slots[p][b]] = True
            mask[A_SKIP] = self.phase != Phase.ERUPT_DST
        return mask

    def legal_actions(self) -> list[int]:
        return [int(a) for a in np.flatnonzero(self.action_masks())]

    def describe_action(self, action: int) -> str:
        T, p = self.terrain, self.current_player
        if action == A_DRAW:
            return 'draw 2 troops'
        if action == A_SKIP:
            return 'skip the effect'
        if action < A_TARGET:
            t, slot = divmod(action - A_PLACE, N_SLOTS)
            if slot >= MAX_BASES:
                return f'capture the enemy H.Q. with {TROOP_NAMES[t]}'
            return f'place {TROOP_NAMES[t]} on {self._base_name(T.views[p][slot])}' \
                if slot < T.n_bases else f'place {TROOP_NAMES[t]} on slot {slot}'
        if action < A_TYPE:
            slot = action - A_TARGET
            name = self._base_name(T.views[p][slot]) if slot < T.n_bases else f'slot {slot}'
            verb = {Phase.JUMBO: 'discard the troop on', Phase.RECALL: 'take back the troop on',
                    Phase.ERUPT_SRC: 'move the troop on', Phase.ERUPT_DST: 'move it to'}
            return f'{verb.get(self.phase, "pick")} {name}'
        return f'take {TROOP_NAMES[action - A_TYPE]} back from the discard'

    def step(self, action):
        if self.done:
            raise Exception('the game is over: reset() first')
        action = int(action)
        if not 0 <= action < N_ACTIONS or not self.action_masks()[action]:
            raise Exception(f'Illegal action {action} ({self.phase.name}): legal actions {self.legal_actions()}')
        p = self.current_player
        phi = self._potential(p)
        if self.phase == Phase.MAIN:
            self.marks = []

        if action == A_DRAW:
            self._draw(p, 2)
            self._end_turn()
        elif action == A_SKIP:
            self._continue()
        elif action < A_TARGET:
            troop, slot = divmod(action - A_PLACE, N_SLOTS)
            self._place(p, troop, slot)
        elif action < A_TYPE:
            self._target(p, self.terrain.views[p][action - A_TARGET])
        else:
            troop = action - A_TYPE
            self.discard[p][troop] -= 1
            self.racks[p][troop] += 1
            self._log(f'{self._who(p)} takes {TROOP_NAMES[troop]} back from the discard.')
            self._continue()

        reward = [0.0, 0.0]
        gain = (0.0 if self.done else self._potential(p)) - phi
        if self.done and self.winner_player is not None:
            gain += 1.0 if self.winner_player == p else -1.0
        reward[p], reward[1 - p] = gain, -gain
        return self.observation, reward, self.done, False, self._info()

    def _potential(self, p: int) -> float:
        return self.shaping * (self.medals[p] - self.medals[1 - p]) / self.terrain.objective

    # -- the pieces of a turn -------------------------------------------- #

    def _draw(self, p: int, n: int, log: bool = True) -> int:
        n = min(n, RACK_MAX - self.rack_size(p), len(self.reserve[p]))
        drawn = [self.reserve[p].pop() for _ in range(n)]
        for t in drawn:
            self.racks[p][t] += 1
        if log and n:
            plural = 's' if n > 1 else ''
            self._log(f'{self._who(p)} draws {n} troop{plural}.', p,
                      f'{self._who(p)} draws {", ".join(TROOP_NAMES[t] for t in drawn)}.')
        return n

    def _place(self, p: int, troop: int, slot: int) -> None:
        T = self.terrain
        self.racks[p][troop] -= 1
        if slot >= MAX_BASES:
            self.captured = (1 - p, slot - MAX_BASES, troop)
            self._log(f'{self._who(p)} places {TROOP_NAMES[troop]} on the enemy H.Q.!')
            self._win(p, 'H.Q. captured')
            return
        b = T.views[p][slot]
        base = T.bases[b]
        covered = self.stacks[b][-1] if self.stacks[b] else None
        self.stacks[b].append((p, troop))
        self.marks.append(b)
        over = ''
        if covered is not None:
            over = f', over {"their own" if covered[0] == p else "the enemy"} {TROOP_NAMES[covered[1]]}'
        self._log(f'{self._who(p)} places {TROOP_NAMES[troop]} on {self._base_name(b)}{over}.')
        if self._check_regions():
            return
        if base.special and T.effect in TRIGGERED:
            self.base_queue.append(b)
        if not (base.special and T.effect == Effect.SHIELD):
            if troop == Troop.SKULLY:
                self._draw(p, 2)
            elif troop == Troop.STAR:
                self._draw(p, 1)
            elif troop == Troop.XB42:
                self._blast(p)
            elif troop == Troop.CAPN and self.placements(p):
                self.phase = Phase.EXTRA
                return
            elif troop == Troop.JUMBO and self.jumbo_targets(p, b):
                self.phase, self.pending_base = Phase.JUMBO, b
                return
        self._continue()

    def _rack_pick(self, q: int, free_only: bool) -> int | None:
        """A random tile of `q`'s rack (among the ones not laid down if `free_only`)."""
        tiles = [t for t in range(N_TYPES)
                 for _ in range(self.racks[q][t] - (self.blocked[q][t] if free_only else 0))]
        return self.rng.choice(tiles) if tiles else None

    def _blast(self, p: int) -> None:
        """XB-42: a random troop of the opponent's rack is discarded face up."""
        q = 1 - p
        troop = self._rack_pick(q, free_only=False)
        if troop is None:
            return
        # the tile hit may be one a Battlefield special base laid down
        if self.rng.random() < self.blocked[q][troop] / self.racks[q][troop]:
            self.blocked[q][troop] -= 1
        self.racks[q][troop] -= 1
        self.discard[q][troop] += 1
        self._log(f"{self._who(p)}'s XB-42 discards {TROOP_NAMES[troop]} from {self._who(q)}'s rack.")

    def _snipe(self, p: int) -> None:
        """Battlefield: a random troop of the opponent's rack is laid face down."""
        q = 1 - p
        troop = self._rack_pick(q, free_only=True)
        if troop is None:
            return
        self.blocked[q][troop] += 1
        self._log(f"A troop of {self._who(q)}'s rack is pinned down for their next turn.", q,
                  f"{self._who(q)}'s {TROOP_NAMES[troop]} is pinned down for their next turn.")

    def _target(self, p: int, b: int) -> None:
        if self.phase == Phase.JUMBO:
            owner, troop = self.stacks[b].pop()
            self.discard[owner][troop] += 1
            self.marks.append(b)
            self._log(f"{self._who(p)}'s Jumbo discards {TROOP_NAMES[troop]} from {self._base_name(b)}.")
        elif self.phase == Phase.RECALL:
            _, troop = self.stacks[b].pop()
            self.racks[p][troop] += 1
            self.marks.append(b)
            self._log(f'{self._who(p)} takes {TROOP_NAMES[troop]} back from {self._base_name(b)}.')
        elif self.phase == Phase.ERUPT_SRC:
            self.erupt_src = b
            self.phase = Phase.ERUPT_DST
            return
        else:                                           # ERUPT_DST
            src = self.erupt_src
            tile = self.stacks[src].pop()
            self.stacks[b].append(tile)
            self.marks += [src, b]
            self._log(f"The eruption moves {self._who(tile[0])}'s {TROOP_NAMES[tile[1]]} "
                      f'from {self._base_name(src)} to {self._base_name(b)}.')
        if not self._check_regions():
            self._continue()

    def _continue(self) -> None:
        """Every troop effect of the turn is done: apply the special bases, then end the turn."""
        p, T = self.current_player, self.terrain
        self.phase, self.pending_base, self.erupt_src = Phase.MAIN, None, None
        while self.base_queue:
            b = self.base_queue.pop(0)
            if T.effect == Effect.DRAW:
                self._draw(p, 1)
            elif T.effect == Effect.SNIPER:
                self._snipe(p)
            elif T.effect == Effect.RECALL:
                if self.rack_size(p) < RACK_MAX and self.recall_targets(p, b):
                    self.phase, self.pending_base = Phase.RECALL, b
                    return
            elif T.effect == Effect.EXHUME:
                if self.rack_size(p) < RACK_MAX and any(self.discard[p]):
                    self.phase, self.pending_base = Phase.EXHUME, b
                    return
            elif T.effect == Effect.ERUPT:
                if self.jumbo_targets(p, b):
                    self.phase, self.pending_base = Phase.ERUPT_SRC, b
                    return
        self._end_turn()

    def _end_turn(self) -> None:
        p = self.current_player
        self.blocked[p] = [0] * N_TYPES         # the troops laid down are available again
        self.phase, self.pending_base, self.erupt_src = Phase.MAIN, None, None
        self.turns_taken += 1
        self.current_player = 1 - p
        if self.turns_taken >= self.max_turns:
            self._log(f'Turn limit ({self.max_turns}) reached.')
            self._compare_medals(None, 'turn limit')
        elif not self.can_act(1 - p):
            self._log(f'{self._who(1 - p)} can neither draw nor place a troop.')
            self._compare_medals(1 - p, 'no action left')

    def _check_regions(self) -> bool:
        """Give the medals of the regions now surrounded; True if that ends the game."""
        T, cur = self.terrain, self.current_player
        for p in (cur, 1 - cur):
            for r, region in enumerate(T.regions):
                if self.region_owner[r] < 0 and all(self.owner(b) == p for b in region.bases):
                    self.region_owner[r] = p
                    self.medals[p] += region.medals
                    plural = 's' if region.medals > 1 else ''
                    self._log(f'{self._who(p)} controls a region: +{region.medals} medal{plural} '
                              f'({self.medals[p]}/{T.objective}).')
        for p in (cur, 1 - cur):
            if self.medals[p] >= T.objective:
                self._win(p, 'medals objective')
                return True
        return False

    def _compare_medals(self, stuck: int | None, reason: str) -> None:
        if self.medals[0] != self.medals[1]:
            self._win(0 if self.medals[0] > self.medals[1] else 1, f'{reason}, more medals')
        elif stuck is not None:
            self._win(1 - stuck, f'{reason}, medals tied')
        else:
            self.done, self.end_reason = True, f'{reason}, medals tied: draw'
            self._log('The game is a draw.')

    def _win(self, p: int, reason: str) -> None:
        self.done, self.winner_player, self.end_reason = True, p, reason
        self.base_queue = []
        self._log(f'{self._who(p)} wins ({reason}).')

    # ------------------------------------------------------------------ #
    # log
    # ------------------------------------------------------------------ #

    def _who(self, p: int) -> str:
        return f'{self.player_names[p]} ({COLOR_NAMES[p]})'

    def _base_name(self, b: int) -> str:
        base = self.terrain.bases[b]
        return f'{"special base" if base.special else "base"} {base.name}'

    def _log(self, text: str, seat: int | None = None, private: str | None = None) -> None:
        self.event_log.append((self.turns_taken, text, seat, private))

    def log_lines(self, pov: int = -1) -> list[tuple[int, str]]:
        """The log as `pov` may read it (-1: everything)."""
        return [(turn, private if private is not None and pov in (-1, seat) else text)
                for turn, text, seat, private in self.event_log]

    # ------------------------------------------------------------------ #
    # observation
    # ------------------------------------------------------------------ #

    @property
    def observation(self):
        return self.observe(self.current_player)

    def on_board(self, p: int) -> list[int]:
        """Troops of `p` on the terrain, by type (covered ones included: stacks are public)."""
        counts = [0] * N_TYPES
        for stack in self.stacks:
            for owner, troop in stack:
                if owner == p:
                    counts[troop] += 1
        if self.captured is not None and self.captured[0] != p:
            counts[self.captured[2]] += 1
        return counts

    def observe(self, pov: int) -> dict[str, np.ndarray]:
        """What seat `pov` knows, in its own view of the board (see `Terrain.views`)."""
        T, me, opp = self.terrain, pov, 1 - pov
        view = T.views[me]
        reach = (self.reach(me), self.reach(opp))
        front = (self.frontier(me, reach[0]), self.frontier(opp, reach[1]))
        my_hq = {b for hq in T.hqs[me] for b in hq.bases}
        opp_hq = {b for hq in T.hqs[opp] for b in hq.bases}

        bases = np.zeros((MAX_BASES, BASE_F), dtype=np.float32)
        adj = np.zeros((MAX_BASES, MAX_BASES), dtype=np.float32)
        for slot, b in enumerate(view):
            base, stack = T.bases[b], self.stacks[b]
            row = bases[slot]
            row[0] = 1
            row[1] = base.special
            if stack:
                owner, top = stack[-1]
                row[2 if owner == me else 3] = 1
                row[5 + top] = 1
                row[13] = min(len(stack), 4) / 4
                row[14] = min(sum(o == me for o, _ in stack), 3) / 3
                row[15] = min(sum(o == opp for o, _ in stack), 3) / 3
                if len(stack) > 1:
                    owner, under = stack[-2]
                    row[16 if owner == me else 17] = 1
                    row[18] = under / 7
                    row[19] = under == Troop.KWAK
            else:
                row[4] = 1
            row[20] = b in reach[0]
            row[21] = b in reach[1]
            row[22] = b in front[0]
            row[23] = b in front[1]
            row[24] = b in my_hq
            row[25] = b in opp_hq
            for t in range(N_TYPES):
                row[26 + t] = self._gate_ok(base.gate, t)
            row[34] = b == self.pending_base
            row[35] = b == self.erupt_src
            for n in T.adj[b]:
                adj[slot, T.slots[me][n]] = 1

        regions = np.zeros((MAX_REGIONS, REGION_F), dtype=np.float32)
        for k, r in enumerate(T.region_views[me]):
            region, taker = T.regions[r], self.region_owner[r]
            row = regions[k]
            for b in region.bases:
                row[T.slots[me][b]] = 1
            row[MAX_BASES] = 1
            row[MAX_BASES + 1] = region.medals / 3 if taker < 0 else 0
            row[MAX_BASES + 2] = taker == me
            row[MAX_BASES + 3] = taker == opp
            row[MAX_BASES + 4] = sum(self.owner(b) == me for b in region.bases) / len(region.bases)
            row[MAX_BASES + 5] = sum(self.owner(b) == opp for b in region.bases) / len(region.bases)
            row[MAX_BASES + 6] = len(region.bases) / 5

        hqs = np.zeros((2 * MAX_HQ, HQ_F), dtype=np.float32)
        for side, (owner, attacker_reach) in enumerate(((me, reach[1]), (opp, reach[0]))):
            for k, hq in enumerate(T.hqs[owner]):
                row = hqs[side * MAX_HQ + k]
                row[0] = 1
                row[1] = any(b in attacker_reach for b in hq.bases)     # may be captured now
                for t in range(N_TYPES):
                    row[2 + t] = self._gate_ok(hq.gate, t)
                for b in hq.bases:
                    row[2 + N_TYPES + T.slots[me][b]] = 1

        g = np.zeros(GLOB_F, dtype=np.float32)
        g[T.index] = 1
        i = N_TERRAINS
        g[i + self.phase] = 1
        i += N_PHASES
        g[i] = me == 0                                                  # I play Blue
        g[i + 1] = min(self.medals[me] / T.objective, 1)
        g[i + 2] = min(self.medals[opp] / T.objective, 1)
        g[i + 3] = T.objective / 8
        g[i + 4] = self.rack_size(me) / RACK_MAX
        g[i + 5] = self.rack_size(opp) / RACK_MAX
        g[i + 6] = len(self.reserve[me]) / (N_TYPES * COPIES - REMOVED)
        g[i + 7] = len(self.reserve[opp]) / (N_TYPES * COPIES - REMOVED)
        i += 8
        mine, theirs = self.on_board(me), self.on_board(opp)
        for t in range(N_TYPES):
            g[i + t] = self.racks[me][t] / COPIES
            g[i + N_TYPES + t] = self.blocked[me][t] / COPIES
            g[i + 2 * N_TYPES + 1 + t] = self.discard[me][t] / COPIES
            g[i + 3 * N_TYPES + 1 + t] = self.discard[opp][t] / COPIES
            # my tiles still in my reserve or in the box / theirs not seen yet
            g[i + 4 * N_TYPES + 1 + t] = (COPIES - self.racks[me][t] - mine[t] - self.discard[me][t]) / COPIES
            g[i + 5 * N_TYPES + 1 + t] = (COPIES - theirs[t] - self.discard[opp][t]) / COPIES
        g[i + 2 * N_TYPES] = sum(self.blocked[opp]) / RACK_MAX
        i += 6 * N_TYPES + 1
        g[i] = min(self.turns_taken / self.max_turns, 1)
        g[i + 1] = self.can_draw(me)

        mask = self.action_masks().astype(np.float32) if pov == self.current_player \
            else np.zeros(N_ACTIONS, dtype=np.float32)
        return {'glob': g, 'bases': bases, 'adj': adj, 'regions': regions, 'hqs': hqs, 'mask': mask}

    def redeterminize(self, pov_player: int) -> None:
        """Re-deal everything hidden from `pov_player` (determinized search): the
        opponent's rack, reserve and removed tiles are pooled and dealt again, and
        so are `pov_player`'s own reserve and removed tiles. Sizes are kept; what
        `pov_player` observes does not change."""
        opp = 1 - pov_player
        rack = [t for t in range(N_TYPES) for _ in range(self.racks[opp][t])]
        pool = rack + self.reserve[opp] + self.removed[opp]
        random.shuffle(pool)
        n_rack, n_reserve = len(rack), len(self.reserve[opp])
        rack = pool[:n_rack]
        self.reserve[opp] = pool[n_rack:n_rack + n_reserve]
        self.removed[opp] = pool[n_rack + n_reserve:]
        self.racks[opp] = [rack.count(t) for t in range(N_TYPES)]
        laid = rack[:sum(self.blocked[opp])]            # the rack is in random order
        self.blocked[opp] = [laid.count(t) for t in range(N_TYPES)]
        pool = self.reserve[pov_player] + self.removed[pov_player]
        random.shuffle(pool)
        n_reserve = len(self.reserve[pov_player])
        self.reserve[pov_player], self.removed[pov_player] = pool[:n_reserve], pool[n_reserve:]

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #

    def nicegui_page(self):
        from .render_web import RenderWeb
        self.render_web = RenderWeb()
        self.render_web.init_web(self)

    def render(self, **kwargs):
        super().render(**kwargs)
        if self.render_web is not None:
            self.render_web.render_web(self, **kwargs)


class ToyBattleAllTerrainsEnv(ToyBattleEnv):
    """`ToyBattleEnv` that, built without a terrain, draws the terrain of every
    game among the 8 — what `train.py` and the self-play wrapper build for
    `-e toybattle`, so that the single `zoo/toybattle` network learns them all.
    Given a `terrain` (`play.py`) it plays that one only."""

    def __init__(self, player_names: list[str] | None = None, terrain: str | None = None,
                 terrains=None, shaping: float = SHAPING, max_turns: int = MAX_TURNS):
        if terrain is None and not terrains:
            terrains = TERRAIN_KEYS
        super().__init__(player_names, terrain, terrains, shaping, max_turns)
