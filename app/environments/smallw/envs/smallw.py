"""`SmallWorldEnv` — the Small World game engine (task T3).

Vanilla rules only: every race behaves like the Ratmen and every power is a
no-op. Everything race- or power-specific goes through `hooks.py`
(`races.py` / `powers.py` register the real behaviours in T4 / T5); the only
structural pieces wired here are the ones that need engine support:

* the GHOUL_CONQUER / GHOUL_REDEPLOY phases of a declined race that still
  conquers (`conquers_in_decline`),
* the Sorcerer and Dragon action ranges and their substitution / dragon-move
  mechanics (`sorcerer_targets`, `dragon_available`),
* the Berserk die drawn before every conquest (`die_before_each_conquest`),
* the ENCAMPMENTS / FORTRESS / HEROES / ALLY / STOUT_DECLINE sub-phases,
* the attack-only tokens of `RaceDef.attack_bonus_tokens` (through
  `redeployable_tokens`) and the Spirit flag `RaceInPlay.is_spirit`.

Action space (`Discrete(133)`, laid out for 5 players and 30 regions)
---------------------------------------------------------------------
=========  ==========================================================
0-5        `A_COMBO + i`: pick the visible combo `i` (costs `i` coins)
6          `A_DECLINE`: go in decline (start of turn, or STOUT_DECLINE)
7          `A_PASS`: end conquests / skip a placement / no ally
8-37       `A_REGION + index`: conquer, abandon, redeploy 1 token, place an
           encampment / a fortress / a hero on that region
38-67      `A_REGION_ALL + index`: redeploy every remaining token there
68-97      `A_SORCERER + index`: Sorcerer substitution on that region
98-127     `A_DRAGON + index`: Dragon Master conquest on that region
128-132    `A_ALLY + offset`: Diplomat ally (seat at relative offset 1..4
           from the acting player — the observation is egocentric too)
=========  ==========================================================

`index` is the 0-based region index (`region.id - 1`).

Observation (`gym.spaces.Dict` of int64 boxes)
----------------------------------------------
`regions` (30, 16) — one row per region, columns:

===  ==========================================================
0    terrain id (`Terrain`, 0..5)
1    mine symbol
2    magic-source symbol
3    cavern symbol
4    border region (may host a first conquest)
5    coastal (adjacent to a sea or the lake)
6    owner code: 0 empty, 1 Lost Tribe, 2 me active, 3 me declined,
     `4 + 2k` / `5 + 2k` opponent `k` active / declined
     (`k` = relative seat order, 1..4 → codes 4..11)
7    race id + 1 (0 = none)
8    race tokens in the region
9    Troll's Lair
10   fortress
11   encampments
12   Halfling hole
13   hero
14   dragon
15   conquered this turn: 0 = no, else relative seat + 1
16   attack cost for the acting race + 1 (0 = not attackable / no acting
     race): `_conquest_cost`, every race / power modifier and the die
     included, whatever the tokens in hand
===  ==========================================================

`players` (5, 13) — row `k` = the seat at relative offset `k` from the current
player (row 0 = current player, unused rows are zero), columns:

===  ==========================================================
0    present
1    coins: **value** of the stack for row 0 (the current player), **number of
     coin tokens** for an opponent — the rules hide the value but the physical
     stacks are public (plan T3b); `OBS_OPPONENT_COINS` exposes the value of
     the opponents here too
2    active race id + 1
3    active power id + 1
4    tokens in hand of the active race
5    tokens left in the tray for the active race
6    declined race id + 1
7    second declined (Spirit) race id + 1
8    regions held by the active race
9    regions held by the declined race(s)
10   Diplomat ally: relative offset + 1 (0 = none)
11   must make a first conquest next
12   number of coin tokens held (public for every player)
13   projected income: the coins this player would score if his turn ended
     now (regions + race / power bonuses, public information)
===  ==========================================================

`combos` (6, 3) — race id + 1, power id + 1, coins lying on it (always "1"
tokens, so their count is their value; 0, 0, 0 = empty slot; row 0 is the free
top of the column).

`global` (16,):

=====  ==========================================================
0      current turn (1-based; `turns_total + 1` once the game is over)
1      total turns of the game
2      phase id (`Phase`)
3      reinforcement die of the acting race + 1 (0 = none)
4      conquests made this turn by the acting race
5      tokens the acting race still has to place
6      dragon already used this turn
7      fortress already placed this turn
8-11   Sorcerer substitution already used on relative opponent 1..4
12     first turn of the acting race
13     attack-only tokens still in hand (Amazons)
14     placements left in the current sub-phase (encampments / heroes)
15     number of visible combos
=====  ==========================================================

`mask` (133,) — `action_masks()` as 0/1, so the network sees the legal targets
(all zeros outside a decision point).

Reward (plan 3.7, zero-sum)
---------------------------
* shaping, at every scoring event: `(coins gained - mean coins gained by the
  others) / SHAPING_SCALE` with `SHAPING_SCALE = 200`;
* terminal, by final rank: rank 0 gets `+1.0`, rank `j` gets
  `-(2j-1)/(n-1)**2` (3 players: `+1.0 / -0.25 / -0.75`), which sums to 0;
  players tied on (coins, tokens on board) share the average of their rank
  rewards.
"""

from __future__ import annotations

import logging as logger
from enum import IntEnum

import gymnasium as gym
import numpy as np

from utils.env import GBEnv

from . import powers as _powers  # noqa: F401  (registers the power hooks)
from . import races as _races  # noqa: F401  (registers the race hooks)
from .classes import (
    MAX_PLAYERS,
    N_VISIBLE_COMBOS,
    POWERS,
    RACES,
    Board,
    ComboColumn,
    Dice,
    PlayerState,
    PowerId,
    RaceId,
    RaceInPlay,
    Region,
    Symbol,
    Terrain,
    Tray,
    map_for,
    turns_for,
)
from .hooks import get_power_hooks, get_race_hooks
from .render_web import RenderWeb

__all__ = [
    'SmallWorldEnv', 'Phase',
    'A_COMBO', 'A_DECLINE', 'A_PASS', 'A_REGION', 'A_REGION_ALL',
    'A_SORCERER', 'A_DRAGON', 'A_ALLY', 'N_ACTIONS', 'MAX_REGIONS',
    'region_action', 'region_all_action', 'sorcerer_action', 'dragon_action',
    'ally_action', 'action_kind',
]


# --------------------------------------------------------------------------- #
# Action space layout
# --------------------------------------------------------------------------- #

#: Region slots reserved in the action space and in the observation (the
#: 3-player board has 30 regions; a bigger board would need a bigger layout).
MAX_REGIONS = 30

A_COMBO = 0          #: 0..5   pick visible combo i
A_DECLINE = 6        #: go in decline
A_PASS = 7           #: end conquests / skip / no ally
A_REGION = 8         #: 8..37   region index (conquer / abandon / place)
A_REGION_ALL = 38    #: 38..67  region index (redeploy everything there)
A_SORCERER = 68      #: 68..97  region index (Sorcerer substitution)
A_DRAGON = 98        #: 98..127 region index (Dragon Master conquest)
A_ALLY = 128         #: 128..132 relative seat offset (Diplomat)
N_ACTIONS = 133      #: size of the action space

KIND_COMBO = 'combo'
KIND_DECLINE = 'decline'
KIND_PASS = 'pass'
KIND_REGION = 'region'
KIND_REGION_ALL = 'region_all'
KIND_SORCERER = 'sorcerer'
KIND_DRAGON = 'dragon'
KIND_ALLY = 'ally'


def region_action(index: int) -> int:
    """Action that targets the region at 0-based `index` (conquer / place)."""
    return A_REGION + int(index)


def region_all_action(index: int) -> int:
    """Action that redeploys every remaining token on region `index`."""
    return A_REGION_ALL + int(index)


def sorcerer_action(index: int) -> int:
    """Sorcerer substitution action on region `index`."""
    return A_SORCERER + int(index)


def dragon_action(index: int) -> int:
    """Dragon Master conquest action on region `index`."""
    return A_DRAGON + int(index)


def ally_action(offset: int) -> int:
    """Diplomat action choosing as ally the seat at relative `offset` (1..4)
    from the acting player."""
    return A_ALLY + int(offset)


def action_kind(action: int) -> tuple[str, int]:
    """Split an action into ``(kind, argument)``.

    Kinds: ``'combo'`` (combo index), ``'decline'``, ``'pass'`` (argument 0),
    ``'region'`` / ``'region_all'`` / ``'sorcerer'`` / ``'dragon'`` (0-based
    region index) and ``'ally'`` (relative seat offset).

    Raises:
        ValueError: if `action` is outside the action space.
    """
    a = int(action)
    if A_COMBO <= a < A_DECLINE:
        return KIND_COMBO, a - A_COMBO
    if a == A_DECLINE:
        return KIND_DECLINE, 0
    if a == A_PASS:
        return KIND_PASS, 0
    if A_REGION <= a < A_REGION_ALL:
        return KIND_REGION, a - A_REGION
    if A_REGION_ALL <= a < A_SORCERER:
        return KIND_REGION_ALL, a - A_REGION_ALL
    if A_SORCERER <= a < A_DRAGON:
        return KIND_SORCERER, a - A_SORCERER
    if A_DRAGON <= a < A_ALLY:
        return KIND_DRAGON, a - A_DRAGON
    if A_ALLY <= a < N_ACTIONS:
        return KIND_ALLY, a - A_ALLY
    raise ValueError(f'action {action} outside the smallw action space (0..{N_ACTIONS - 1})')


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #

class Phase(IntEnum):
    """The decision points of a turn (plan 3.3).

    TURN_PAUSE is the one exception: it has no legal action at all, the
    step entering it returns ``info["next_step_no_action"] = True`` and
    `current_player = -1`, and ``step(-1)`` starts the next turn. It exists so
    the web UI can show each player's move before the next one plays.

    Every other phase is only *entered* when the acting player has a choice:
    the engine auto-advances through everything else (a redeployment with no
    token to place, a victim who holds at most one region, a sub-phase without
    a legal placement), so `action_masks()` is never all-False while the game
    is running.
    """

    PICK_COMBO = 0       #: first turn of the game or of a new race
    CONQUER = 1          #: active race: conquer / abandon / decline / end
    GHOUL_CONQUER = 2    #: a declined race that still conquers (Ghouls)
    REDEPLOY = 3         #: place the tokens in hand
    GHOUL_REDEPLOY = 4   #: idem for the declined race
    ENCAMPMENTS = 5      #: Bivouacking
    FORTRESS = 6         #: Fortified
    HEROES = 7           #: Heroic
    ALLY = 8             #: Diplomat
    STOUT_DECLINE = 9    #: Stout
    VICTIM_REDEPLOY = 10  #: an opponent redeploys the tokens he lost
    DONE = 11            #: the game is over
    TURN_PAUSE = 12      #: between two turns, nobody moves: `step(-1)` goes on


#: Redeployment flavours, stored in `_redeploy_return` to know what to do once
#: the redeployment is over (plain strings: the env must stay deep-copyable).
_RD_ACTIVE = 'active'    # the active race's own redeployment
_RD_GHOUL = 'ghoul'      # a declined race acting before the active one
_RD_DECLINE = 'decline'  # a race that just declined and kept its tokens
_RD_VICTIM = 'victim'    # an opponent redeploying his losses


# --------------------------------------------------------------------------- #
# Observation layout
# --------------------------------------------------------------------------- #

REGION_COLS = 17
PLAYER_COLS = 14
COMBO_COLS = 3
GLOBAL_COLS = 16

#: Attack costs are clipped to this in the observation (`regions` column 16).
MAX_ATTACK_COST = 30

#: Upper bound of every `regions` column (see the module docstring).
_REGION_HIGH = np.array(
    [len(Terrain) - 1, 1, 1, 1, 1, 1, 11, len(RaceId), 30, 1, 1, 5, 1, 1, 1, MAX_PLAYERS,
     MAX_ATTACK_COST + 1],
    dtype=np.int64,
)
#: Upper bound of every `players` column.
_PLAYER_HIGH = np.array(
    [1, 999, len(RaceId), len(PowerId), 30, 30, len(RaceId), len(RaceId), 30, 30,
     MAX_PLAYERS, 1, 999, 99],
    dtype=np.int64,
)
#: Upper bound of every `combos` column.
_COMBO_HIGH = np.array([len(RaceId), len(PowerId), 99], dtype=np.int64)
#: Upper bound of every `global` entry.
_GLOBAL_HIGH = np.array(
    [20, 20, max(int(p) for p in Phase), 4, 30, 30, 1, 1, 1, 1, 1, 1, 1, 4, 5, N_VISIBLE_COMBOS],
    dtype=np.int64,
)


class SmallWorldEnv(GBEnv):
    """Small World, base game, 3 players (2/4/5 need their map table first)."""

    metadata = {'render_modes': ['human_web']}

    #: The rules hide the coin *values* of the other players (only the number
    #: of tokens is public — `players` column 12, and it is what column 1 holds
    #: for an opponent row); flip this to True to expose their value in
    #: `players` column 1 as well (debugging / perfect-information runs).
    OBS_OPPONENT_COINS = False

    #: Divider of the dense shaping reward (plan 3.7): the cumulated shaping of
    #: a whole game stays well below the terminal magnitudes.
    SHAPING_SCALE = 200.0

    def __init__(self, n_players: int = 3, player_names: list[str] = None,
                 pause_between_turns: bool = True):
        super(SmallWorldEnv, self).__init__('smallw', n_players, player_names)
        #: stop in TURN_PAUSE after each player's turn (a no-action step, see
        #: `Phase.TURN_PAUSE`); the engine unit tests turn it off
        self.pause_between_turns = pause_between_turns

        # raises NotImplementedError for a player count whose board is missing
        self.n_regions = len(map_for(n_players))
        if self.n_regions > MAX_REGIONS:
            raise NotImplementedError(
                f'the action space is laid out for {MAX_REGIONS} regions, '
                f'the {n_players}-player map has {self.n_regions}'
            )
        self.turns_total = turns_for(n_players)

        self.observation_space = gym.spaces.Dict({
            'regions': gym.spaces.Box(
                low=0, high=np.tile(_REGION_HIGH, (MAX_REGIONS, 1)),
                shape=(MAX_REGIONS, REGION_COLS), dtype=np.int64),
            'players': gym.spaces.Box(
                low=0, high=np.tile(_PLAYER_HIGH, (MAX_PLAYERS, 1)),
                shape=(MAX_PLAYERS, PLAYER_COLS), dtype=np.int64),
            'combos': gym.spaces.Box(
                low=0, high=np.tile(_COMBO_HIGH, (N_VISIBLE_COMBOS, 1)),
                shape=(N_VISIBLE_COMBOS, COMBO_COLS), dtype=np.int64),
            'global': gym.spaces.Box(
                low=0, high=_GLOBAL_HIGH, shape=(GLOBAL_COLS,), dtype=np.int64),
            'mask': gym.spaces.Box(
                low=0, high=1, shape=(N_ACTIONS,), dtype=np.int64),
        })
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

        # -- state, all (re)built by reset() -------------------------------- #
        self.board: Board = None
        self.players: list[PlayerState] = []
        self.combo_column: ComboColumn = None
        self.tray: Tray = None
        self.dice: Dice = None
        self.turn = 0
        self.turns_taken = 0
        self.phase = Phase.DONE
        self.event_log: list[str] = []
        self.turn_events: list[str] = []
        self._turn_seat = 0
        self._acting_rip: RaceInPlay | None = None
        self._acting_this_turn: list[RaceInPlay] = []
        self._victims: list[RaceInPlay] = []
        self._encampments_left = 0
        self._heroes_left = 0
        self._turn_acted = False
        self._turn_abandoned = False
        self._redeploy_return = _RD_ACTIVE
        self._decline_score_after = False
        self._step_rewards = [0.0] * n_players
        #: the pure ranked part of the terminal reward, set by `_end_game`
        self.terminal_rewards: list[float] | None = None

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        """Start a new game and enter the first player's turn."""
        super().reset(seed=seed)

        self.board = Board(self.n_players)
        self.board.setup()
        self.players = [PlayerState(seat, self.player_names[seat])
                        for seat in range(self.n_players)]
        self.combo_column = ComboColumn(self.np_random)
        self.tray = Tray()
        self.dice = Dice(self.np_random)

        self.turn = 1
        self.turns_taken = 0
        self.current_player = 0
        self.winner_player = None
        self.done = False
        self.event_log = []
        self.turn_events = []
        self.phase = Phase.PICK_COMBO
        self._turn_seat = 0
        self._acting_rip = None
        self._acting_this_turn = []
        self._victims = []
        self._encampments_left = 0
        self._heroes_left = 0
        self._turn_acted = False
        self._turn_abandoned = False
        self._redeploy_return = _RD_ACTIVE
        self._decline_score_after = False
        self._step_rewards = [0.0] * self.n_players
        self.terminal_rewards = None

        self._log(f'---- NEW GAME: {self.n_players} players, '
                  f'{self.turns_total} turns ----')
        self._begin_turn(0)
        return self.observation, self._get_info()

    def _get_info(self):
        """`next_step_no_action` is True only in TURN_PAUSE (between two
        turns): the dice are pre-rolled and every other phase the engine stops
        in needs a real decision."""
        return {'next_step_no_action': self.phase == Phase.TURN_PAUSE}

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #

    def _log(self, message: str) -> None:
        """Append `message` to the game log and to the current turn's log."""
        self.event_log.append(message)
        self.turn_events.append(message)

    @property
    def current_race(self) -> RaceInPlay | None:
        """The `RaceInPlay` currently acting (None in PICK_COMBO / DONE)."""
        return self._acting_rip

    def _label(self, rip: RaceInPlay) -> str:
        """``'Player 1 (Elves)'`` — used by every log line."""
        suffix = ' decline' if rip.in_decline else ''
        return f'{self.players[rip.owner].name} ({RACES[rip.race].name}{suffix})'

    def _describe_races(self, player: PlayerState) -> str:
        """Short description of a player's races, for the turn header."""
        bits = []
        if player.active is not None:
            power = ('-' if player.active.power is None
                     else POWERS[player.active.power].name)
            bits.append(f'{RACES[player.active.race].name}/{power}')
        else:
            bits.append('no race')
        for dec in player.declined:
            bits.append(f'{RACES[dec.race].name} in decline')
        return ', '.join(bits)

    def _holds(self, rip: RaceInPlay, region: Region) -> bool:
        """True if `region` is occupied by exactly that race of that player."""
        return (region.owner == rip.owner and region.race == rip.race
                and region.in_decline == rip.in_decline)

    # -- hook plumbing (see hooks.py for the combination rules) ---------- #

    def _hooks(self, rip: RaceInPlay):
        return get_race_hooks(rip.race), get_power_hooks(rip.power)

    def _rip_flag(self, name: str, rip: RaceInPlay) -> bool:
        """Logical OR of a ``(rip)`` boolean hook."""
        race_h, power_h = self._hooks(rip)
        return bool(getattr(race_h, name)(rip)) or bool(getattr(power_h, name)(rip))

    def _env_flag(self, name: str, rip: RaceInPlay) -> bool:
        """Logical OR of an ``(env, rip)`` boolean hook."""
        race_h, power_h = self._hooks(rip)
        return (bool(getattr(race_h, name)(self, rip))
                or bool(getattr(power_h, name)(self, rip)))

    def _void(self, name: str, rip: RaceInPlay, *args) -> None:
        """Call a side-effect hook on the race then on the power."""
        race_h, power_h = self._hooks(rip)
        getattr(race_h, name)(self, rip, *args)
        getattr(power_h, name)(self, rip, *args)

    def _three_valued(self, name: str, rip: RaceInPlay, region: Region,
                      default: bool) -> bool:
        """Resolve a `bool | None` hook: any True wins, then any False."""
        race_h, power_h = self._hooks(rip)
        opinions = [getattr(race_h, name)(self, rip, region),
                    getattr(power_h, name)(self, rip, region)]
        if any(o is True for o in opinions):
            return True
        if any(o is False for o in opinions):
            return False
        return default

    def _score_bonus(self, rip: RaceInPlay) -> int:
        race_h, power_h = self._hooks(rip)
        return int(race_h.score_bonus(self, rip)) + int(power_h.score_bonus(self, rip))

    def _defender_loss(self, victim: RaceInPlay, region: Region) -> int:
        race_h, power_h = self._hooks(victim)
        return min(int(race_h.defender_loss(self, victim, region)),
                   int(power_h.defender_loss(self, victim, region)))

    def _redeployable(self, rip: RaceInPlay) -> int:
        """Tokens `rip` still has to put on the board (0 = redeployment over)."""
        race_h, power_h = self._hooks(rip)
        return max(0, min(int(race_h.redeployable_tokens(self, rip)),
                          int(power_h.redeployable_tokens(self, rip)),
                          rip.tokens_in_hand))

    def _sorcerer_targets(self, rip: RaceInPlay) -> set[int]:
        """Region **ids** on which a Sorcerer substitution is legal."""
        race_h, power_h = self._hooks(rip)
        targets = set(race_h.sorcerer_targets(self, rip))
        targets |= set(power_h.sorcerer_targets(self, rip))
        return {rid for rid in targets if 1 <= rid <= self.n_regions}

    # ------------------------------------------------------------------ #
    # Conquest legality
    # ------------------------------------------------------------------ #

    def _can_attack(self, rip: RaceInPlay, region: Region) -> bool:
        """Target legality of `region` for `rip`, cost aside (dragon included)."""
        if self._holds(rip, region):
            return False
        if region.is_immune:                      # hole / hero / dragon
            return False
        if not self._three_valued('can_target', rip, region, not region.is_water):
            return False
        # Diplomat: an ally may not attack the *active* race of his protector
        if region.owner is not None and not region.in_decline:
            if self.players[region.owner].ally == rip.owner:
                return False
        held = self.board.regions_of_race(rip)
        if not held or (rip is self.players[rip.owner].active
                        and self.players[rip.owner].must_first_conquest):
            return bool(region.border) or self._rip_flag('ignores_border_rule', rip)
        default = bool(set(region.adjacent) & {r.id for r in held})
        return self._three_valued('is_adjacent', rip, region, default)

    def _conquest_cost(self, rip: RaceInPlay, region: Region) -> int:
        """Tokens needed for `rip` to conquer `region` (>= 1, die included)."""
        race_h, power_h = self._hooks(rip)
        cost = region.base_conquest_cost()
        cost = int(race_h.conquest_cost_modifier(self, rip, region, cost))
        cost = int(power_h.conquest_cost_modifier(self, rip, region, cost))
        cost = max(1, cost)
        if rip.die is not None:
            cost = max(1, cost - int(rip.die))
        return cost

    def _conquest_options(self, rip: RaceInPlay) -> dict[int, tuple[int, bool]]:
        """``{region index: (cost, is_final_attempt)}`` for `rip`.

        A *final attempt* is a region short by at most 3 tokens: it is only
        offered when no Berserk die is already in play (with Berserk the die is
        rolled before the choice and is part of the cost).
        """
        options: dict[int, tuple[int, bool]] = {}
        hand = rip.tokens_in_hand
        if hand <= 0:
            return options
        for region in self.board.regions:
            if not self._can_attack(rip, region):
                continue
            cost = self._conquest_cost(rip, region)
            if cost <= hand:
                options[region.index] = (cost, False)
            elif rip.die is None and cost <= hand + 3:
                options[region.index] = (cost, True)
        return options

    def _dragon_targets(self, rip: RaceInPlay) -> list[int]:
        """Region indices a Dragon Master conquest may take (1 token, any
        defence)."""
        if rip.tokens_in_hand < 1 or not self._env_flag('dragon_available', rip):
            return []
        return [r.index for r in self.board.regions if self._can_attack(rip, r)]

    def _ally_options(self) -> list[int]:
        """Opponents the Diplomat may choose (not attacked this turn).

        The union over every race of the current player is used, so a declined
        race that attacked someone (Ghouls) also blocks the alliance.
        """
        seat = self._turn_seat
        attacked: set[int] = set()
        for rip in self.players[seat].all_races():
            attacked |= rip.attacked_players
        return [p for p in range(self.n_players) if p != seat and p not in attacked]

    # ------------------------------------------------------------------ #
    # Action masks
    # ------------------------------------------------------------------ #

    def action_masks(self):
        """Boolean legality mask over the 133 actions (all-False only once the
        game is over and in TURN_PAUSE, where the only move is ``step(-1)``)."""
        mask = np.zeros(N_ACTIONS, dtype=bool)
        if self.done:
            return mask
        phase = self.phase
        rip = self._acting_rip

        if phase == Phase.PICK_COMBO:
            coins = self.players[self._turn_seat].coins
            for i in range(len(self.combo_column.visible)):
                if coins >= self.combo_column.cost(i):
                    mask[A_COMBO + i] = True

        elif phase in (Phase.CONQUER, Phase.GHOUL_CONQUER):
            mask[A_PASS] = True
            if (phase == Phase.CONQUER and not self._turn_acted
                    and not self._turn_abandoned):
                # allowed on any turn while nothing has been done yet, even on
                # the turn the race was picked (user ruling, plan section 5)
                mask[A_DECLINE] = True
            if not self._turn_acted:                      # abandon own regions
                for region in self.board.regions_of_race(rip):
                    mask[A_REGION + region.index] = True
            for index in self._conquest_options(rip):     # conquer
                mask[A_REGION + index] = True
            for rid in self._sorcerer_targets(rip):
                mask[A_SORCERER + rid - 1] = True
            for index in self._dragon_targets(rip):
                mask[A_DRAGON + index] = True

        elif phase in (Phase.REDEPLOY, Phase.GHOUL_REDEPLOY, Phase.VICTIM_REDEPLOY):
            if self._redeployable(rip) > 0:
                for region in self.board.regions_of_race(rip):
                    mask[A_REGION + region.index] = True
                    mask[A_REGION_ALL + region.index] = True

        elif phase == Phase.ENCAMPMENTS:
            if self._encampments_left > 0:
                for region in self.board.regions_of_race(rip):
                    mask[A_REGION + region.index] = True

        elif phase == Phase.FORTRESS:
            mask[A_PASS] = True
            if self.tray.available_marker('fortress') > 0:
                for region in self.board.regions_of_race(rip):
                    if not region.fortress:
                        mask[A_REGION + region.index] = True

        elif phase == Phase.HEROES:
            if self._heroes_left > 0:
                for region in self.board.regions_of_race(rip):
                    if not region.hero:
                        mask[A_REGION + region.index] = True

        elif phase == Phase.ALLY:
            mask[A_PASS] = True
            for seat in self._ally_options():
                mask[A_ALLY + (seat - self._turn_seat) % self.n_players] = True

        elif phase == Phase.STOUT_DECLINE:
            mask[A_DECLINE] = True
            mask[A_PASS] = True

        return mask

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step(self, action):
        """Apply one action; returns `(obs, rewards, terminated, False, info)`.

        `rewards` holds one entry per player (shaping at every scoring event,
        the ranked outcome on the terminal step; both zero-sum).
        """
        if self.done:
            raise Exception(f'Illegal action {action} : the game is over')
        try:
            action = int(action)
        except (TypeError, ValueError):
            raise Exception(f'Illegal action {action!r} : not an action index')
        if self.phase == Phase.TURN_PAUSE:
            if action != -1:
                raise Exception(f'Illegal action {action} : between two turns '
                                f'only the no-action step (-1) is legal')
            self._step_rewards = [0.0] * self.n_players
            self._advance_seat()
            return (self.observation, list(self._step_rewards), self.done,
                    False, self._get_info())
        if not 0 <= action < N_ACTIONS:
            raise Exception(f'Illegal action {action} : outside 0..{N_ACTIONS - 1}')
        masks = self.action_masks()
        if masks[action] == False:  # noqa: E712 - same style as the other envs
            raise Exception(f'Illegal action {action} : Legal actions {masks}')

        self._step_rewards = [0.0] * self.n_players
        kind, arg = action_kind(action)
        phase = self.phase

        if phase == Phase.PICK_COMBO:
            self._step_pick_combo(arg)
        elif phase in (Phase.CONQUER, Phase.GHOUL_CONQUER):
            self._step_conquer(kind, arg)
        elif phase in (Phase.REDEPLOY, Phase.GHOUL_REDEPLOY, Phase.VICTIM_REDEPLOY):
            self._step_redeploy(kind, arg)
        elif phase == Phase.ENCAMPMENTS:
            self._step_encampment(arg)
        elif phase == Phase.FORTRESS:
            self._step_fortress(kind, arg)
        elif phase == Phase.HEROES:
            self._step_hero(arg)
        elif phase == Phase.ALLY:
            self._step_ally(kind, arg)
        elif phase == Phase.STOUT_DECLINE:
            self._step_stout(kind)
        else:  # pragma: no cover - unreachable, the masks are empty in DONE
            raise Exception(f'Illegal action {action} : nothing to do in {phase.name}')

        return (self.observation, list(self._step_rewards), self.done, False,
                self._get_info())

    # ------------------------------------------------------------------ #
    # Turn flow
    # ------------------------------------------------------------------ #

    def _begin_turn(self, seat: int) -> None:
        """Start the turn of `seat` (plan 3.3, step 1)."""
        self._turn_seat = seat
        self.current_player = seat
        self.turn_events = []
        self._turn_acted = False
        self._turn_abandoned = False
        player = self.players[seat]
        player.ally = None                      # the protection he gave expires
        for region in self.board.regions:
            region.conquered_this_turn_by = None

        acting: list[RaceInPlay] = []
        if player.active is not None:
            acting.append(player.active)
        ghouls = [rip for rip in player.declined
                  if self._rip_flag('conquers_in_decline', rip)
                  and (self.board.regions_of_race(rip) or rip.tokens_in_hand)]
        acting.extend(ghouls)
        for rip in acting:
            rip.new_turn()
        self._log(f'-- turn {self.turn}/{self.turns_total}: {player.name} '
                  f'[{self._describe_races(player)}] --')
        for rip in acting:
            self._void('on_turn_start', rip)
        self._acting_this_turn = list(acting)

        if ghouls:
            rip = ghouls[0]
            if len(ghouls) > 1:  # pragma: no cover - needs two such races
                self._log(f'{player.name}: only {RACES[rip.race].name} conquers '
                          f'in decline this turn')
            self._ready_troops(rip)
            self._enter_conquer(rip, ghoul=True)
        else:
            self._enter_active_turn()

    def _enter_active_turn(self) -> None:
        """PICK_COMBO or CONQUER for the active race (plan 3.3, steps 2 and 3)."""
        player = self.players[self._turn_seat]
        self._turn_acted = False
        self._turn_abandoned = False
        if player.active is None:
            if not self.combo_column.visible:  # pragma: no cover - 14 races, 3 seats
                self._log(f'{player.name}: no combo left to pick, turn skipped')
                self._finish_turn()
                return
            self._acting_rip = None
            self.phase = Phase.PICK_COMBO
            return
        self._ready_troops(player.active)
        self._enter_conquer(player.active, ghoul=False)

    def _ready_troops(self, rip: RaceInPlay) -> None:
        """*Ready your troops*: take every token back except one per region."""
        self._lift_tokens(rip)
        rip.pending_redeploy = False

    def _lift_tokens(self, rip: RaceInPlay) -> int:
        """Take every token of `rip` back in hand except one per region.

        Used when the troops are readied and at the start of a race's own
        redeployment: placing the whole hand again is then equivalent to moving
        the tokens freely between the regions held (>= 1 per region).
        Returns the number of tokens lifted.
        """
        lifted = 0
        for region in self.board.regions_of_race(rip):
            if region.tokens > 1:
                lifted += region.tokens - 1
                region.tokens = 1
        rip.tokens_in_hand += lifted
        return lifted

    def _enter_conquer(self, rip: RaceInPlay, ghoul: bool) -> None:
        """Enter the conquest phase of `rip`."""
        self._acting_rip = rip
        self.phase = Phase.GHOUL_CONQUER if ghoul else Phase.CONQUER
        self._turn_acted = False
        self._turn_abandoned = False
        self._roll_berserk(rip)

    def _roll_berserk(self, rip: RaceInPlay) -> None:
        """Draw the die before a conquest decision when the race asks for it."""
        rip.die = None
        if self._rip_flag('die_before_each_conquest', rip):
            rip.die = self.dice.next()
            self._log(f'{self._label(rip)} rolls the reinforcement die: {rip.die}')

    def _step_pick_combo(self, index: int) -> None:
        """Pick a combo, take the tokens from the tray, then conquer."""
        player = self.players[self._turn_seat]
        combo = self.combo_column.visible[index]
        race, power = combo.race, combo.power
        coins_on_it = combo.coins
        self.combo_column.pick(index, player)
        rip = RaceInPlay(race, power, player.seat)
        wanted = rip.initial_tokens()
        taken = self.tray.take(race, wanted)
        assert taken == wanted, (
            f'tray holds {taken}/{wanted} {RACES[race].name} tokens on a fresh pick')
        rip.tokens_in_hand = taken
        player.active = rip
        player.must_first_conquest = True
        rip.new_turn()
        self._log(f'{player.name} picks combo {index}: {RACES[race].name} / '
                  f'{POWERS[power].name} for {index} coin(s)'
                  + (f', collecting {coins_on_it} coin(s)' if coins_on_it else '')
                  + f' -> {taken} tokens in hand, {player.coins} coins '
                    f'({player.coin_count} coin token(s))')
        self._void('on_turn_start', rip)
        self._acting_this_turn = [rip] + [r for r in self._acting_this_turn
                                          if r is not rip]
        self._enter_conquer(rip, ghoul=False)

    # ------------------------------------------------------------------ #
    # Conquest phase
    # ------------------------------------------------------------------ #

    def _step_conquer(self, kind: str, arg: int) -> None:
        rip = self._acting_rip
        if kind == KIND_DECLINE:
            self._do_decline(score_after=True)
            return
        if kind == KIND_PASS:
            self._log(f'{self._label(rip)} ends its conquests '
                      f'({len(rip.conquests_this_turn)} this turn)')
            self._end_conquests()
            return
        if kind == KIND_SORCERER:
            self._resolve_sorcerer(rip, self.board.by_index(arg))
            return
        if kind == KIND_DRAGON:
            self._resolve_dragon(rip, self.board.by_index(arg))
            return
        # KIND_REGION: abandon an own region, or conquer
        region = self.board.by_index(arg)
        if self._holds(rip, region):
            self._abandon(rip, region)
            return
        cost, final = self._conquest_options(rip)[arg]
        if final:
            self._final_attempt(rip, region, cost)
            return
        self._resolve_conquest(rip, region, cost, 'normal')
        self._roll_berserk(rip)

    def _end_conquests(self) -> None:
        """Leave the conquest phase for the redeployment."""
        rip = self._acting_rip
        ghoul = self.phase == Phase.GHOUL_CONQUER
        self._enter_redeploy(rip, _RD_GHOUL if ghoul else _RD_ACTIVE)

    def _final_attempt(self, rip: RaceInPlay, region: Region, cost: int) -> None:
        """Last conquest of the turn, short by at most 3 tokens: roll the die."""
        hand = rip.tokens_in_hand
        die = self.dice.next()
        rip.die = die
        self._log(f'{self._label(rip)} attempts a final conquest of region '
                  f'{region.id} ({self._terrain(region)}): needs {cost}, has '
                  f'{hand}, die rolls {die}')
        if hand + die >= cost:
            self._resolve_conquest(rip, region, hand, 'final')
        else:
            self._log(f'{self._label(rip)} fails its final conquest')
            self._turn_acted = True
        self._end_conquests()

    def _abandon(self, rip: RaceInPlay, region: Region) -> None:
        """Abandon one of `rip`'s regions (only before its first conquest)."""
        tokens = region.tokens
        rip.tokens_in_hand += tokens
        region.clear_race()
        region.conquered_this_turn_by = None
        self._clear_markers(region)
        self._turn_abandoned = True
        player = self.players[rip.owner]
        if rip is player.active and not self.board.regions_of_race(rip):
            player.must_first_conquest = True
        self._log(f'{self._label(rip)} abandons region {region.id} '
                  f'({self._terrain(region)}), {tokens} token(s) back in hand')
        self._void('on_abandon', rip, region)

    def _terrain(self, region: Region) -> str:
        """``'Farmland'`` — terrain name for the logs."""
        return region.terrain.name.capitalize()

    def _clear_markers(self, region: Region, remove_fortress: bool = True,
                       remove_lair: bool = True, remove_encampments: bool = True,
                       remove_immune: bool = True) -> None:
        """Return the markers of `region` to their supplies.

        Which markers go depends on the event: a conquest takes everything
        (the immune ones cannot be there), a decline keeps the fortress and the
        lair, a declined race leaving the map keeps them as well.
        """
        if remove_fortress and region.fortress:
            region.fortress = False
            self.tray.put_marker('fortress')
        if remove_lair and region.lair:
            region.lair = False
        if remove_encampments:
            while region.encampments > 0:
                region.encampments -= 1
                self.tray.put_marker('encampment')
        if remove_immune:
            if region.hole:
                region.hole = False
                self.tray.put_marker('hole')
            if region.hero:
                region.hero = False
                self.tray.put_marker('hero')
            if region.dragon:
                region.dragon = False
                self.tray.put_marker('dragon')

    def _resolve_conquest(self, rip: RaceInPlay, region: Region, tokens: int,
                          kind: str = 'normal') -> None:
        """Resolve a conquest of `region` by `rip` with `tokens` tokens."""
        player = self.players[rip.owner]
        info = {
            'kind': kind,
            'lost_tribe': region.lost_tribe,
            'owner': region.owner,
            'race': region.race,
            'in_decline': region.in_decline,
            'tokens': region.tokens,
            'was_empty': region.is_empty,
            'victim': None,
            'defender': 'an empty region',
        }
        assert not region.is_immune, (
            f'region {region.id} is immune, it cannot be conquered')

        if region.lost_tribe:
            region.lost_tribe = False
            info['defender'] = 'the Lost Tribe'

        victim = None
        if region.tokens > 0 and region.race is not None:
            victim = self.players[region.owner].race_by(
                region.race, in_decline=region.in_decline)
            assert victim is not None, (
                f'region {region.id} holds {RACES[region.race].name} tokens of '
                f'player {region.owner} but that race is not in play')
            info['victim'] = victim
            info['defender'] = self._label(victim)
            if victim.in_decline and not self._rip_flag('losses_like_active', victim):
                # a declined race is simply wiped out of the region
                self.tray.put(victim.race, region.tokens)
                region.clear_race()
            else:
                loss = max(0, min(self._defender_loss(victim, region), region.tokens))
                keep = region.tokens - loss
                self.tray.put(victim.race, loss)
                victim.tokens_in_hand += keep
                if keep > 0:
                    victim.pending_redeploy = True
                region.clear_race()
                self._log(f'{info["defender"]} loses region {region.id}: '
                          f'{loss} token(s) to the tray, {keep} back in hand')

        self._clear_markers(region, remove_immune=False)
        if victim is not None:
            self._void('on_region_lost', victim, region)
            self._after_loss(victim)

        region.owner = rip.owner
        region.race = rip.race
        region.in_decline = rip.in_decline
        region.tokens = tokens
        rip.tokens_in_hand -= tokens
        assert rip.tokens_in_hand >= 0, 'conquest paid with tokens not in hand'
        region.conquered_this_turn_by = rip.owner
        rip.conquests_this_turn.append(region.id)
        if not info['was_empty']:
            rip.nonempty_conquests += 1
        if victim is not None and not info['in_decline'] and victim.owner != rip.owner:
            rip.attacked_players.add(victim.owner)
        rip.first_conquest_done = True
        if rip is player.active:
            player.must_first_conquest = False
        self._turn_acted = True
        self._log(f'{player.name} ({RACES[rip.race].name}) conquers region '
                  f'{region.id} ({self._terrain(region)}) from {info["defender"]} '
                  f'with {tokens} token(s)'
                  + ('' if kind == 'normal' else f' [{kind}]'))
        self._void('on_conquered', rip, region, info)

    def _after_loss(self, victim: RaceInPlay) -> None:
        """Bookkeeping after `victim` lost a region: vanish or re-enter later."""
        if self.board.regions_of_race(victim):
            return
        owner = self.players[victim.owner]
        if victim.in_decline and victim.tokens_in_hand == 0:
            self._remove_declined_race(victim)
        elif victim is owner.active:
            owner.must_first_conquest = True

    def _resolve_sorcerer(self, rip: RaceInPlay, region: Region) -> None:
        """Sorcerer substitution: replace a lone enemy token by a new one.

        The engine only owns the mechanics; `sorcerer_targets` (T4) decides
        where it is legal. The victim's token goes back to the tray (an Elf
        too), and the new token comes from the tray.
        """
        assert region.race is not None and region.tokens > 0, (
            f'Sorcerer substitution on the empty region {region.id}')
        victim = self.players[region.owner].race_by(
            region.race, in_decline=region.in_decline)
        assert victim is not None, f'no race in play for region {region.id}'
        victim_label = self._label(victim)
        self.tray.put(victim.race, region.tokens)
        region.clear_race()
        self._clear_markers(region, remove_immune=False)
        self._void('on_region_lost', victim, region)
        self._after_loss(victim)

        taken = self.tray.take(rip.race, 1)
        assert taken == 1, f'no {RACES[rip.race].name} token left for a substitution'
        region.owner = rip.owner
        region.race = rip.race
        region.in_decline = rip.in_decline
        region.tokens = 1
        rip.sorcerer_used_on.add(victim.owner)
        if not victim.in_decline and victim.owner != rip.owner:
            rip.attacked_players.add(victim.owner)
        self._turn_acted = True
        self._log(f'{self._label(rip)} replaces the token of {victim_label} in '
                  f'region {region.id} ({self._terrain(region)}) [sorcerer]')

    def _resolve_dragon(self, rip: RaceInPlay, region: Region) -> None:
        """Dragon Master conquest: one token, any defence, dragon settles in."""
        for other in self.board.regions:
            if other.dragon:
                other.dragon = False
                self.tray.put_marker('dragon')
        self._resolve_conquest(rip, region, 1, 'dragon')
        assert self.tray.take_marker('dragon'), 'the dragon marker is missing'
        region.dragon = True
        rip.dragon_used = True
        self._log(f'the dragon settles in region {region.id}')
        self._roll_berserk(rip)

    # ------------------------------------------------------------------ #
    # Redeployment
    # ------------------------------------------------------------------ #

    def _enter_redeploy(self, rip: RaceInPlay, mode: str) -> None:
        """Enter a redeployment phase (auto-advancing when there is nothing
        to place)."""
        self._acting_rip = rip
        self._redeploy_return = mode
        self.phase = (Phase.REDEPLOY if mode == _RD_ACTIVE
                      else Phase.VICTIM_REDEPLOY if mode == _RD_VICTIM
                      else Phase.GHOUL_REDEPLOY)
        if mode in (_RD_ACTIVE, _RD_GHOUL):
            # troop redeployment: the tokens may move freely between the
            # regions held, so everything but one per region goes back in hand
            # (a victim only puts back what it lost, it moves nothing else)
            lifted = self._lift_tokens(rip)
            if lifted:
                self._log(f'{self._label(rip)} lifts {lifted} token(s) to '
                          f'redeploy them ({rip.tokens_in_hand} in hand)')
            self._void('on_redeploy_start', rip)
        self._check_redeploy_done()

    def _check_redeploy_done(self) -> bool:
        """Advance out of the redeployment if nothing is left to place."""
        rip = self._acting_rip
        regions = self.board.regions_of_race(rip)
        if not regions:
            rip.pending_redeploy = False
            owner = self.players[rip.owner]
            if rip is owner.active:
                owner.must_first_conquest = True
            if rip.tokens_in_hand:
                self._log(f'{self._label(rip)} holds no region: '
                          f'{rip.tokens_in_hand} token(s) stay in hand')
            self._after_redeploy()
            return True
        if self._redeployable(rip) <= 0:
            rip.pending_redeploy = False
            self._after_redeploy()
            return True
        return False

    def _step_redeploy(self, kind: str, arg: int) -> None:
        rip = self._acting_rip
        region = self.board.by_index(arg)
        count = 1 if kind == KIND_REGION else self._redeployable(rip)
        self._place_tokens(rip, region, count)
        self._log(f'{self._label(rip)} redeploys {count} token(s) in region '
                  f'{region.id} ({region.tokens} there now)')
        self._check_redeploy_done()

    def _place_tokens(self, rip: RaceInPlay, region: Region, count: int) -> None:
        """Move `count` tokens from `rip`'s hand into one of its regions."""
        assert count > 0 and rip.tokens_in_hand >= count, (
            f'cannot place {count} tokens, {rip.tokens_in_hand} in hand')
        assert self._holds(rip, region), (
            f'region {region.id} does not belong to {self._label(rip)}')
        region.tokens += count
        rip.tokens_in_hand -= count

    def _after_redeploy(self) -> None:
        """Continue the turn once a redeployment is over."""
        mode = self._redeploy_return
        if mode == _RD_GHOUL:
            self._enter_active_turn()
        elif mode == _RD_DECLINE:
            self._finish_decline(self._decline_score_after)
        elif mode == _RD_VICTIM:
            self._next_victim()
        else:
            self._advance_end_phase(Phase.REDEPLOY)

    # ------------------------------------------------------------------ #
    # End-of-turn sub-phases
    # ------------------------------------------------------------------ #

    #: Order of the optional sub-phases after the redeployment.
    _END_PHASES = (Phase.ENCAMPMENTS, Phase.FORTRESS, Phase.HEROES, Phase.ALLY)

    def _advance_end_phase(self, done_phase: Phase) -> None:
        """Enter the next sub-phase after `done_phase`, else score and finish."""
        player = self.players[self._turn_seat]
        rip = player.active
        if rip is None:  # pragma: no cover - the decline path never lands here
            self._finish_turn()
            return
        start = (0 if done_phase == Phase.REDEPLOY
                 else self._END_PHASES.index(done_phase) + 1)
        for phase in self._END_PHASES[start:]:
            if self._try_enter(phase, rip):
                return
        self._score(player)
        if get_power_hooks(rip.power).can_stout_decline(self, rip):
            self.phase = Phase.STOUT_DECLINE
            return
        self._finish_turn()

    def _try_enter(self, phase: Phase, rip: RaceInPlay) -> bool:
        """Enter `phase` if the power asks for it and a placement is legal."""
        hooks = get_power_hooks(rip.power)
        if phase == Phase.ENCAMPMENTS:
            total = int(hooks.encampments_to_place(self, rip))
            if total <= 0:
                return False
            for region in self.board.regions_of_race(rip):   # re-place them all
                while region.encampments > 0:
                    region.encampments -= 1
                    self.tray.put_marker('encampment')
            if not self.board.regions_of_race(rip):
                return False
            self._encampments_left = min(total, self.tray.available_marker('encampment'))
            if self._encampments_left <= 0:
                return False
            self.phase = Phase.ENCAMPMENTS
            return True
        if phase == Phase.FORTRESS:
            if rip.fortress_used or not hooks.can_place_fortress(self, rip):
                return False
            if self.tray.available_marker('fortress') <= 0:
                return False
            if not [r for r in self.board.regions_of_race(rip) if not r.fortress]:
                return False
            self.phase = Phase.FORTRESS
            return True
        if phase == Phase.HEROES:
            total = int(hooks.heroes_to_place(self, rip))
            if total <= 0:
                return False
            for region in self.board.regions:                # move them all
                if region.hero and self._holds(rip, region):
                    region.hero = False
                    self.tray.put_marker('hero')
            free = [r for r in self.board.regions_of_race(rip) if not r.hero]
            self._heroes_left = min(total, self.tray.available_marker('hero'), len(free))
            if self._heroes_left <= 0:
                return False
            self.phase = Phase.HEROES
            return True
        if phase == Phase.ALLY:
            if not hooks.has_ally_choice(self, rip) or not self._ally_options():
                return False
            self.phase = Phase.ALLY
            return True
        return False  # pragma: no cover

    def _step_encampment(self, arg: int) -> None:
        rip = self._acting_rip
        region = self.board.by_index(arg)
        assert self.tray.take_marker('encampment'), 'no encampment left'
        region.encampments += 1
        self._encampments_left -= 1
        self._log(f'{self._label(rip)} places an encampment in region {region.id} '
                  f'({region.encampments} there now)')
        if self._encampments_left <= 0:
            self._advance_end_phase(Phase.ENCAMPMENTS)

    def _step_fortress(self, kind: str, arg: int) -> None:
        rip = self._acting_rip
        if kind == KIND_PASS:
            self._log(f'{self._label(rip)} places no fortress')
        else:
            region = self.board.by_index(arg)
            assert self.tray.take_marker('fortress'), 'no fortress left'
            region.fortress = True
            rip.fortress_used = True
            self._log(f'{self._label(rip)} builds a fortress in region {region.id}')
        self._advance_end_phase(Phase.FORTRESS)

    def _step_hero(self, arg: int) -> None:
        rip = self._acting_rip
        region = self.board.by_index(arg)
        assert self.tray.take_marker('hero'), 'no hero left'
        region.hero = True
        self._heroes_left -= 1
        self._log(f'{self._label(rip)} sends a hero to region {region.id}')
        free = [r for r in self.board.regions_of_race(rip) if not r.hero]
        if self._heroes_left <= 0 or not free:
            self._advance_end_phase(Phase.HEROES)

    def _step_ally(self, kind: str, arg: int) -> None:
        player = self.players[self._turn_seat]
        if kind == KIND_PASS:
            self._log(f'{player.name} chooses no ally')
        else:
            player.ally = (self._turn_seat + arg) % self.n_players
            self._log(f'{player.name} allies with {self.players[player.ally].name} '
                      f'(who cannot attack his active race until his next turn)')
        self._advance_end_phase(Phase.ALLY)

    def _step_stout(self, kind: str) -> None:
        if kind == KIND_DECLINE:
            self._do_decline(score_after=False)
            return
        self._log(f'{self.players[self._turn_seat].name} stays active')
        self._finish_turn()

    # ------------------------------------------------------------------ #
    # Decline
    # ------------------------------------------------------------------ #

    def _do_decline(self, score_after: bool) -> None:
        """Decline the active race, then score (start of turn) or finish."""
        player = self.players[self._turn_seat]
        rip = player.active
        self._decline_score_after = score_after
        if self._decline(player):
            self._enter_redeploy(rip, _RD_DECLINE)
            return
        self._finish_decline(score_after)

    def _decline(self, player: PlayerState) -> bool:
        """Flip the active race of `player` to its decline side.

        Returns True when the race kept tokens in hand and must redeploy them
        (Ghouls) — the caller then runs a redeployment for the declined race.
        """
        rip = player.active
        assert rip is not None, 'nothing to decline'
        for old in list(player.declined):
            if not old.is_spirit:
                self._remove_declined_race(old)

        regions = self.board.regions_of_race(rip)
        rip.is_spirit = self._rip_flag('is_spirit_power', rip)
        keeps = self._rip_flag('keeps_tokens_in_decline', rip)
        self._void('on_decline', rip)
        if rip.power is not None:
            self.combo_column.discard_power(rip.power)
        rip.power = None
        returned = 0
        for region in regions:
            # the fortress and the Troll's Lair stay, everything else goes back
            self._clear_markers(region, remove_fortress=False, remove_lair=False)
            region.in_decline = True
            if not keeps and region.tokens > 1:
                returned += region.tokens - 1
                self.tray.put(rip.race, region.tokens - 1)
                region.tokens = 1
        if not keeps and rip.tokens_in_hand:
            returned += rip.tokens_in_hand
            self.tray.put(rip.race, rip.tokens_in_hand)
            rip.tokens_in_hand = 0
        rip.in_decline = True
        rip.pending_redeploy = False
        player.declined.append(rip)
        player.active = None
        player.must_first_conquest = False
        self._log(f'{player.name} sends the {RACES[rip.race].name} in decline: '
                  f'{len(regions)} region(s) kept, {returned} token(s) to the tray'
                  + (' (Spirit)' if rip.is_spirit else ''))
        if not regions and rip.tokens_in_hand == 0:
            self._remove_declined_race(rip)
            return False
        return keeps and rip.tokens_in_hand > 0

    def _finish_decline(self, score_after: bool) -> None:
        """Scoring (start-of-turn decline only) then end of turn."""
        if score_after:
            self._score(self.players[self._turn_seat])
        self._finish_turn()

    def _remove_declined_race(self, old: RaceInPlay) -> None:
        """A declined race leaves the map; its banner goes under the stack."""
        player = self.players[old.owner]
        freed = 0
        for region in self.board.regions_of_race(old):
            freed += 1
            self.tray.put(old.race, region.tokens)
            region.clear_race()
            # the fortress and the lair stay behind (plan / T3 contract)
            self._clear_markers(region, remove_fortress=False, remove_lair=False)
        if old.tokens_in_hand:
            self.tray.put(old.race, old.tokens_in_hand)
            old.tokens_in_hand = 0
        old.pending_redeploy = False
        player.declined = [d for d in player.declined if d is not old]
        self.combo_column.return_race(old.race)
        self._log(f'{player.name}: the declined {RACES[old.race].name} leave the '
                  f'map ({freed} region(s) freed)')

    # ------------------------------------------------------------------ #
    # Scoring and rewards
    # ------------------------------------------------------------------ #

    def _score(self, player: PlayerState) -> None:
        """One coin per region held plus the race/power bonuses (plan 3.7)."""
        gained = 0
        detail = []
        for rip in player.all_races():
            regions = self.board.regions_of_race(rip)
            bonus = 0
            if not rip.in_decline:
                bonus = self._score_bonus(rip)
            elif self._rip_flag('bonus_in_decline', rip):
                bonus = self._score_bonus(rip)
            gained += len(regions) + bonus
            detail.append(f'{RACES[rip.race].name}'
                          f'{" in decline" if rip.in_decline else ""}: '
                          f'{len(regions)} region(s)'
                          + (f' +{bonus} bonus' if bonus else ''))
        player.coins += gained
        self._log(f'{player.name} scores {gained} coin(s) '
                  f'[{"; ".join(detail) if detail else "no race"}] '
                  f'-> {player.coins} coins in {player.coin_count} token(s)')
        gains = [0] * self.n_players
        gains[player.seat] = gained
        self._shape(gains)

    def _projected_income(self, player: PlayerState) -> int:
        """Coins `player` would score right now (`_score` without its effects)."""
        income = 0
        for rip in player.all_races():
            income += len(self.board.regions_of_race(rip))
            if not rip.in_decline or self._rip_flag('bonus_in_decline', rip):
                income += self._score_bonus(rip)
        return income

    def _shape(self, gains: list[int]) -> None:
        """Add the zero-sum shaping reward of one scoring event."""
        if self.n_players < 2:  # pragma: no cover
            return
        total = sum(gains)
        for seat in range(self.n_players):
            mean_others = (total - gains[seat]) / (self.n_players - 1)
            self._step_rewards[seat] += (gains[seat] - mean_others) / self.SHAPING_SCALE

    def rank_rewards(self) -> tuple[float, ...]:
        """Terminal reward per final rank, zero-sum.

        Rank 0 gets `+1.0`; rank `j > 0` gets `-(2j-1)/(n-1)**2`, so the
        losers share exactly `-1.0` and the penalty grows with the rank
        (3 players: `+1.0, -0.25, -0.75`; 2 players: `+1.0, -1.0`).
        """
        n = self.n_players
        if n < 2:  # pragma: no cover
            return (0.0,)
        losers = n - 1
        return tuple([1.0] + [-(2 * j - 1) / (losers ** 2) for j in range(1, n)])

    def _finish_turn(self) -> None:
        """End-of-turn hooks, then the victims' redeployments."""
        player = self.players[self._turn_seat]
        alive = player.all_races()
        for rip in self._acting_this_turn:
            if any(rip is other for other in alive):
                self._void('on_turn_end', rip)
        self._start_victims()

    def _start_victims(self) -> None:
        """Queue every opponent race that lost tokens this turn (turn order)."""
        self._victims = []
        for k in range(1, self.n_players):
            seat = (self._turn_seat + k) % self.n_players
            for rip in self.players[seat].all_races():
                if rip.pending_redeploy:
                    self._victims.append(rip)
        self._next_victim()

    def _next_victim(self) -> None:
        """Let the next victim redeploy, auto-resolving the forced cases."""
        while self._victims:
            rip = self._victims.pop(0)
            if not rip.pending_redeploy:
                continue
            owner = self.players[rip.owner]
            if not any(rip is other for other in owner.all_races()):
                continue                              # the race vanished meanwhile
            regions = self.board.regions_of_race(rip)
            count = self._redeployable(rip)
            if len(regions) <= 1 or count <= 0:
                if not regions:
                    if rip is owner.active:
                        owner.must_first_conquest = True
                    self._log(f'{self._label(rip)} keeps {rip.tokens_in_hand} '
                              f'token(s) in hand (no region left)')
                elif count > 0:
                    self._place_tokens(rip, regions[0], count)
                    self._log(f'{self._label(rip)} redeploys {count} token(s) in '
                              f'region {regions[0].id} (only region left)')
                rip.pending_redeploy = False
                continue
            self.current_player = rip.owner
            self._acting_rip = rip
            self._redeploy_return = _RD_VICTIM
            self.phase = Phase.VICTIM_REDEPLOY
            return
        self._next_seat()

    def _next_seat(self) -> None:
        """The turn of `_turn_seat` is over: pause (TURN_PAUSE), or go on.

        No pause after the very last turn: the game ends right away.
        """
        last = (self._turn_seat == self.n_players - 1
                and self.turn >= self.turns_total)
        if self.pause_between_turns and not last:
            self.phase = Phase.TURN_PAUSE
            self.current_player = -1
            self._acting_rip = None
            return
        self._advance_seat()

    def _advance_seat(self) -> None:
        """Hand the turn to the next seat, or end the game."""
        seat = self._turn_seat
        if seat == self.n_players - 1:
            self.turn += 1
            self.turns_taken = self.turn - 1
            if self.turn > self.turns_total:
                self._end_game()
                return
        self._begin_turn((seat + 1) % self.n_players)

    def _end_game(self) -> None:
        """Final standings, winner and terminal rewards."""
        self.done = True
        self.phase = Phase.DONE
        self._acting_rip = None
        coins = [p.coins for p in self.players]
        tokens = [self.board.count_tokens_on_board(p.seat) for p in self.players]
        order = sorted(range(self.n_players),
                       key=lambda s: (-coins[s], -tokens[s], s))
        self.winner_player = order[0]

        table = self.rank_rewards()
        rewards = [0.0] * self.n_players
        group: list[int] = []
        for rank, seat in enumerate(order):
            if group and (coins[seat], tokens[seat]) != (coins[order[rank - 1]],
                                                         tokens[order[rank - 1]]):
                self._share(rewards, group, table)
                group = []
            group.append(rank)
        self._share(rewards, group, table)
        self.terminal_rewards = [0.0] * self.n_players
        for rank, seat in enumerate(order):
            self.terminal_rewards[seat] = rewards[rank]
            self._step_rewards[seat] += rewards[rank]

        standings = ', '.join(
            f'#{rank + 1} {self.players[seat].name}: {coins[seat]} coins, '
            f'{tokens[seat]} token(s)' for rank, seat in enumerate(order))
        self._log(f'---- GAME OVER after {self.turns_taken} turns: {standings} ----')
        self._log(f'winner: {self.players[self.winner_player].name}')
        logger.info(f'smallw game over: {standings}')

    @staticmethod
    def _share(rewards: list[float], ranks: list[int], table) -> None:
        """Give every rank of `ranks` the average of their table entries."""
        if not ranks:
            return
        value = sum(table[rank] for rank in ranks) / len(ranks)
        for rank in ranks:
            rewards[rank] = value

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    @property
    def observation(self):
        """The observation of `current_player` (see the module docstring)."""
        n = self.n_players
        seat0 = self.current_player if 0 <= self.current_player < n else 0
        rip = self._acting_rip

        regions = np.zeros((MAX_REGIONS, REGION_COLS), dtype=np.int64)
        for region in self.board.regions:
            row = regions[region.index]
            row[0] = int(region.terrain)
            row[1] = 1 if region.has(Symbol.MINE) else 0
            row[2] = 1 if region.has(Symbol.MAGIC) else 0
            row[3] = 1 if region.has(Symbol.CAVERN) else 0
            row[4] = 1 if region.border else 0
            row[5] = 1 if self.board.is_coastal(region) else 0
            if region.owner is None:
                row[6] = 1 if region.lost_tribe else 0
            else:
                k = (region.owner - seat0) % n
                row[6] = ((3 if region.in_decline else 2) if k == 0
                          else 4 + 2 * (k - 1) + (1 if region.in_decline else 0))
            row[7] = 0 if region.race is None else int(region.race) + 1
            row[8] = region.tokens
            row[9] = int(region.lair)
            row[10] = int(region.fortress)
            row[11] = region.encampments
            row[12] = int(region.hole)
            row[13] = int(region.hero)
            row[14] = int(region.dragon)
            row[15] = (0 if region.conquered_this_turn_by is None
                       else ((region.conquered_this_turn_by - seat0) % n) + 1)
            if rip is not None and self._can_attack(rip, region):
                row[16] = min(self._conquest_cost(rip, region), MAX_ATTACK_COST) + 1

        players = np.zeros((MAX_PLAYERS, PLAYER_COLS), dtype=np.int64)
        for k in range(n):
            seat = (seat0 + k) % n
            player = self.players[seat]
            row = players[k]
            row[0] = 1
            row[1] = (player.coins if (k == 0 or self.OBS_OPPONENT_COINS)
                      else player.coin_count)
            active = player.active
            row[2] = 0 if active is None else int(active.race) + 1
            row[3] = (0 if active is None or active.power is None
                      else int(active.power) + 1)
            row[4] = 0 if active is None else active.tokens_in_hand
            row[5] = 0 if active is None else self.tray.available(active.race)
            row[6] = (int(player.declined[0].race) + 1
                      if len(player.declined) > 0 else 0)
            row[7] = (int(player.declined[1].race) + 1
                      if len(player.declined) > 1 else 0)
            row[8] = len(self.board.regions_of(seat, in_decline=False))
            row[9] = len(self.board.regions_of(seat, in_decline=True))
            row[10] = 0 if player.ally is None else ((player.ally - seat0) % n) + 1
            row[11] = int(player.must_first_conquest)
            row[12] = player.coin_count
            row[13] = min(self._projected_income(player), int(_PLAYER_HIGH[13]))

        combos = np.zeros((N_VISIBLE_COMBOS, COMBO_COLS), dtype=np.int64)
        for i, combo in enumerate(self.combo_column.visible[:N_VISIBLE_COMBOS]):
            combos[i] = (int(combo.race) + 1, int(combo.power) + 1,
                         min(combo.coins, int(_COMBO_HIGH[2])))

        glob = np.zeros(GLOBAL_COLS, dtype=np.int64)
        glob[0] = self.turn
        glob[1] = self.turns_total
        glob[2] = int(self.phase)
        glob[3] = 0 if rip is None or rip.die is None else int(rip.die) + 1
        glob[4] = 0 if rip is None else len(rip.conquests_this_turn)
        glob[5] = 0 if rip is None else self._redeployable(rip)
        glob[6] = 0 if rip is None else int(rip.dragon_used)
        glob[7] = 0 if rip is None else int(rip.fortress_used)
        for k in range(1, min(n, MAX_PLAYERS)):
            seat = (seat0 + k) % n
            glob[7 + k] = 1 if (rip is not None and seat in rip.sorcerer_used_on) else 0
        glob[12] = 1 if (rip is not None and rip.turns_played <= 1) else 0
        glob[13] = (0 if rip is None
                    else min(rip.race_def.attack_bonus_tokens, rip.tokens_in_hand))
        glob[14] = (self._encampments_left if self.phase == Phase.ENCAMPMENTS
                    else self._heroes_left if self.phase == Phase.HEROES else 0)
        glob[15] = len(self.combo_column.visible)

        return {'regions': regions, 'players': players, 'combos': combos,
                'global': glob, 'mask': self.action_masks().astype(np.int64)}

    # ------------------------------------------------------------------ #
    # Imperfect information
    # ------------------------------------------------------------------ #

    def redeterminize(self, pov_player: int):
        """Reshuffle what `pov_player` cannot see (plan 3.8).

        The hidden state is the order of the race / power stacks and of the
        power discard pile, plus the pre-rolled dice that have not been used
        yet. The *value* of the opponents' coins is hidden too but simply not
        observed (only their token count is, plan T3b), so the coins are left
        untouched.
        """
        self.combo_column.redeterminize(self.np_random)
        self.dice.redeterminize(self.np_random)

    # ------------------------------------------------------------------ #
    # Descriptions (GUI / logs / debugging)
    # ------------------------------------------------------------------ #

    def describe_action(self, action: int) -> str:
        """Human-readable description of `action` in the current state."""
        kind, arg = action_kind(int(action))
        rip = self._acting_rip
        if kind == KIND_COMBO:
            if arg < len(self.combo_column.visible):
                combo = self.combo_column.visible[arg]
                coins = (f', +{combo.coins} coin token(s) of value 1'
                         if combo.coins else '')
                return (f'pick combo {arg}: {RACES[combo.race].name} / '
                        f'{POWERS[combo.power].name} (cost {arg}{coins})')
            return f'pick combo {arg} (empty slot)'
        if kind == KIND_DECLINE:
            return 'go in decline'
        if kind == KIND_PASS:
            if self.phase in (Phase.CONQUER, Phase.GHOUL_CONQUER):
                return 'end the conquests'
            if self.phase == Phase.FORTRESS:
                return 'build no fortress'
            if self.phase == Phase.ALLY:
                return 'choose no ally'
            if self.phase == Phase.STOUT_DECLINE:
                return 'stay active'
            return 'pass'
        if kind == KIND_ALLY:
            seat = (self._turn_seat + arg) % self.n_players
            return f'ally with {self.players[seat].name}'
        region = self.board.by_index(arg)
        label = f'region {region.id} ({self._terrain(region)})'
        if kind == KIND_SORCERER:
            return f'replace the enemy token in {label} by a Sorcerer'
        if kind == KIND_DRAGON:
            return f'dragon conquest of {label} with 1 token'
        if kind == KIND_REGION_ALL:
            count = 0 if rip is None else self._redeployable(rip)
            return f'redeploy all {count} remaining token(s) in {label}'
        # KIND_REGION
        if self.phase in (Phase.CONQUER, Phase.GHOUL_CONQUER):
            if rip is not None and self._holds(rip, region):
                return f'abandon {label} ({region.tokens} token(s) back in hand)'
            cost, final = self._conquest_options(rip).get(arg, (0, False))
            return (f'{"final attempt on" if final else "conquer"} {label} '
                    f'for {cost} token(s)'
                    + (f', {rip.tokens_in_hand} in hand' if final else ''))
        if self.phase == Phase.ENCAMPMENTS:
            return f'place an encampment in {label}'
        if self.phase == Phase.FORTRESS:
            return f'build a fortress in {label}'
        if self.phase == Phase.HEROES:
            return f'send a hero to {label}'
        return f'place 1 token in {label}'

    def legal_targets(self) -> dict[int, str]:
        """``{action: description}`` for every legal action (for the GUI)."""
        return {int(action): self.describe_action(int(action))
                for action in np.flatnonzero(self.action_masks())}

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def nicegui_page(self):
        """Build the NiceGUI page (`render_web.RenderWeb`)."""
        self.render_web = RenderWeb()
        self.render_web.init_web(self)

    def render(self, **kwargs):
        """Refresh the NiceGUI page (no terminal rendering for this game)."""
        super().render(**kwargs)
        self.render_web.render_web(self, **kwargs)
