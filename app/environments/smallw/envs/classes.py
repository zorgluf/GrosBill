"""Game data tables and mutable state classes for Small World (`smallw`).

This module is **pure game state**: no gymnasium, no NiceGUI, no I/O. The
engine (`smallw.py`), the race / power hooks (`races.py`, `powers.py`) and the
web renderer (`render_web.py`) are all built on top of it.

Contents:
    * constants: player count, starting coins, visible combos, die faces,
      marker supplies;
    * `RaceId` / `PowerId` enums with the `RACES` / `POWERS` data tables — the
      numeric order of both enums is part of the observation encoding, so
      **never reorder them**;
    * `Region`: one board region = its immutable `RegionDef` + who occupies it;
    * `RaceInPlay`: one race banner held by a player (active or in decline);
    * `PlayerState`: seat, coin tokens (denominations 1/3/5/10, see
      `make_change`), races, Diplomat ally;
    * `Tray`: the supplies of race tokens and of the markers;
    * `Combo` / `ComboColumn`: the column of 6 visible race+power combinations;
    * `Dice`: the pre-rolled reinforcement die;
    * `Board`: the 30 regions plus the queries the engine needs.

`Terrain` and `Symbol` are defined in `map3p.py` (the static map needs them
too) and simply re-exported here.

Two conventions hold everywhere in this file:

* **Randomness** goes through the numpy `Generator` owned by the environment
  (`np_random`, from gymnasium seeding), never through the `random` module, so
  that a seeded game is reproducible. The two classes that need randomness
  later on (`ComboColumn`, `Dice`) keep a reference to it in a public
  `np_random` attribute; the env must reassign it after re-seeding
  (`col.np_random = self.np_random`).
* Everything is plain Python and `copy.deepcopy`-able (the MCTS trainer
  deep-copies the whole env). A deep copy of a `Region` copies its `static`
  definition as well, so engine code must compare regions by `id`, never by
  object identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .map3p import (
    ADJACENCY,
    BORDER_IDS,
    COASTAL_IDS,
    N_REGIONS,
    RegionDef,
    Symbol,
    Terrain,
    map_for,
    turns_for,
)

__all__ = [
    # re-exports
    'Terrain', 'Symbol', 'RegionDef', 'N_REGIONS', 'map_for', 'turns_for',
    # constants
    'MAX_PLAYERS', 'START_COINS', 'N_VISIBLE_COMBOS', 'DIE_FACES',
    'N_FORTRESSES', 'N_ENCAMPMENTS', 'N_HEROES', 'N_HOLES', 'N_DRAGONS',
    'MARKER_SUPPLY', 'MARKER_KINDS', 'MAX_POWER_VALUE',
    'COIN_DENOMINATIONS', 'COIN_SUPPLY', 'make_change',
    # tables
    'RaceId', 'PowerId', 'RaceDef', 'PowerDef', 'RACES', 'POWERS',
    # state
    'Region', 'RaceInPlay', 'PlayerState', 'Tray', 'Combo', 'ComboColumn',
    'Dice', 'Board',
]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Largest supported player count. The action space is sized for it even
#: though only the 3-player board is transcribed so far.
MAX_PLAYERS = 5

#: Coins (of value 1) each player starts with.
START_COINS = 5

#: Coin denominations of the real game, biggest first. Greedy change with this
#: set is provably minimal (checked by the tests against a dynamic program), so
#: `make_change` may simply walk it from the biggest coin down.
COIN_DENOMINATIONS: tuple[int, ...] = (10, 5, 3, 1)

#: Printed coin supply of the base game (109 coins). Documentation / GUI only:
#: the pool is treated as **unlimited** (plan T3b), nothing tracks it.
COIN_SUPPLY: dict[int, int] = {10: 30, 5: 24, 3: 20, 1: 35}

#: Race+power combinations visible in the column at any time (5 face up plus
#: the top of the stack, which the rules also make pickable).
N_VISIBLE_COMBOS = 6

#: The six faces of the reinforcement die: 0, 0, 0, 1, 2, 3.
DIE_FACES: tuple[int, ...] = (0, 0, 0, 1, 2, 3)

#: Marker supplies printed in the rules (components of the base game).
N_FORTRESSES = 6      # Fortified
N_ENCAMPMENTS = 5     # Bivouacking
N_HEROES = 2          # Heroic
N_HOLES = 2           # Halflings
N_DRAGONS = 1         # Dragon Master

#: Supply of each marker kind, keyed by the string the `Tray` methods take.
#: Troll's Lairs are *not* here: a Lair exists in every region the Trolls
#: occupy, so they are unlimited (`Region.lair`).
MARKER_SUPPLY: dict[str, int] = {
    'fortress': N_FORTRESSES,
    'encampment': N_ENCAMPMENTS,
    'hero': N_HEROES,
    'hole': N_HOLES,
    'dragon': N_DRAGONS,
}

#: The valid `kind` arguments of `Tray.take_marker` / `Tray.put_marker`.
MARKER_KINDS: tuple[str, ...] = tuple(MARKER_SUPPLY)


# --------------------------------------------------------------------------- #
# Races and special powers
# --------------------------------------------------------------------------- #

class RaceId(IntEnum):
    """The 14 base-game races.

    The values are observation codes (alphabetical order, as in the plan):
    do not reorder.
    """

    AMAZONS = 0
    DWARVES = 1
    ELVES = 2
    GHOULS = 3
    GIANTS = 4
    HALFLINGS = 5
    HUMANS = 6
    ORCS = 7
    RATMEN = 8
    SKELETONS = 9
    SORCERERS = 10
    TRITONS = 11
    TROLLS = 12
    WIZARDS = 13


class PowerId(IntEnum):
    """The 20 base-game special powers.

    The values are observation codes (alphabetical order, as in the plan):
    do not reorder.
    """

    ALCHEMIST = 0
    BERSERK = 1
    BIVOUACKING = 2
    COMMANDO = 3
    DIPLOMAT = 4
    DRAGON_MASTER = 5
    FLYING = 6
    FOREST = 7
    FORTIFIED = 8
    HEROIC = 9
    HILL = 10
    MERCHANT = 11
    MOUNTED = 12
    PILLAGING = 13
    SEAFARING = 14
    SPIRIT = 15
    STOUT = 16
    SWAMP = 17
    UNDERWORLD = 18
    WEALTHY = 19


@dataclass(frozen=True)
class RaceDef:
    """Immutable description of one race banner.

    Attributes:
        id: the :class:`RaceId`.
        name: display name ("Amazons").
        key: lowercase snake_case name, matching the static assets
            ``static/races/<key>.png`` (banner) and
            ``static/races/<key>_token.jpg`` (token).
        banner_value: number printed on the banner = tokens taken when the race
            is picked (before adding the power value).
        total_tokens: tokens of that race in the box. It caps how many can ever
            be on the map + in hand (Skeletons and Sorcerers can grow up to it).
        attack_bonus_tokens: extra tokens usable only for conquests (Amazons: 4;
            0 for every other race).
    """

    id: RaceId
    name: str
    key: str
    banner_value: int
    total_tokens: int
    attack_bonus_tokens: int = 0


@dataclass(frozen=True)
class PowerDef:
    """Immutable description of one special power badge.

    Attributes:
        id: the :class:`PowerId`.
        name: display name ("Dragon Master").
        key: lowercase snake_case name, matching ``static/powers/<key>.png``.
        value: number printed on the badge = tokens added to the race value
            when the combo is picked.
    """

    id: PowerId
    name: str
    key: str
    value: int


def _race(rid: RaceId, name: str, banner: int, total: int, bonus: int = 0) -> RaceDef:
    """Build a `RaceDef`, deriving `key` from `name`."""
    return RaceDef(
        id=rid,
        name=name,
        key=name.lower().replace(' ', '_'),
        banner_value=banner,
        total_tokens=total,
        attack_bonus_tokens=bonus,
    )


def _power(pid: PowerId, name: str, value: int) -> PowerDef:
    """Build a `PowerDef`, deriving `key` from `name`."""
    return PowerDef(
        id=pid,
        name=name,
        key=name.lower().replace(' ', '_'),
        value=value,
    )


#: The 14 races: banner value, tokens in the box, attack-only bonus tokens.
RACES: dict[RaceId, RaceDef] = {
    RaceId.AMAZONS:   _race(RaceId.AMAZONS,   'Amazons',   6, 15, 4),
    RaceId.DWARVES:   _race(RaceId.DWARVES,   'Dwarves',   3, 8),
    RaceId.ELVES:     _race(RaceId.ELVES,     'Elves',     6, 11),
    RaceId.GHOULS:    _race(RaceId.GHOULS,    'Ghouls',    5, 10),
    RaceId.GIANTS:    _race(RaceId.GIANTS,    'Giants',    6, 11),
    RaceId.HALFLINGS: _race(RaceId.HALFLINGS, 'Halflings', 6, 11),
    RaceId.HUMANS:    _race(RaceId.HUMANS,    'Humans',    5, 10),
    RaceId.ORCS:      _race(RaceId.ORCS,      'Orcs',      5, 10),
    RaceId.RATMEN:    _race(RaceId.RATMEN,    'Ratmen',    8, 13),
    RaceId.SKELETONS: _race(RaceId.SKELETONS, 'Skeletons', 6, 20),
    RaceId.SORCERERS: _race(RaceId.SORCERERS, 'Sorcerers', 5, 18),
    RaceId.TRITONS:   _race(RaceId.TRITONS,   'Tritons',   6, 11),
    RaceId.TROLLS:    _race(RaceId.TROLLS,    'Trolls',    5, 10),
    RaceId.WIZARDS:   _race(RaceId.WIZARDS,   'Wizards',   5, 10),
}

#: The 20 special powers and the tokens they add to the race value.
POWERS: dict[PowerId, PowerDef] = {
    PowerId.ALCHEMIST:     _power(PowerId.ALCHEMIST,     'Alchemist',     4),
    PowerId.BERSERK:       _power(PowerId.BERSERK,       'Berserk',       4),
    PowerId.BIVOUACKING:   _power(PowerId.BIVOUACKING,   'Bivouacking',   5),
    PowerId.COMMANDO:      _power(PowerId.COMMANDO,      'Commando',      4),
    PowerId.DIPLOMAT:      _power(PowerId.DIPLOMAT,      'Diplomat',      5),
    PowerId.DRAGON_MASTER: _power(PowerId.DRAGON_MASTER, 'Dragon Master', 5),
    PowerId.FLYING:        _power(PowerId.FLYING,        'Flying',        5),
    PowerId.FOREST:        _power(PowerId.FOREST,        'Forest',        4),
    PowerId.FORTIFIED:     _power(PowerId.FORTIFIED,     'Fortified',     3),
    PowerId.HEROIC:        _power(PowerId.HEROIC,        'Heroic',        5),
    PowerId.HILL:          _power(PowerId.HILL,          'Hill',          4),
    PowerId.MERCHANT:      _power(PowerId.MERCHANT,      'Merchant',      2),
    PowerId.MOUNTED:       _power(PowerId.MOUNTED,       'Mounted',       5),
    PowerId.PILLAGING:     _power(PowerId.PILLAGING,     'Pillaging',     5),
    PowerId.SEAFARING:     _power(PowerId.SEAFARING,     'Seafaring',     5),
    PowerId.SPIRIT:        _power(PowerId.SPIRIT,        'Spirit',        5),
    PowerId.STOUT:         _power(PowerId.STOUT,         'Stout',         4),
    PowerId.SWAMP:         _power(PowerId.SWAMP,         'Swamp',         4),
    PowerId.UNDERWORLD:    _power(PowerId.UNDERWORLD,    'Underworld',    5),
    PowerId.WEALTHY:       _power(PowerId.WEALTHY,       'Wealthy',       4),
}

#: Largest power value (5). `total_tokens >= banner_value +
#: attack_bonus_tokens + MAX_POWER_VALUE` holds for every race, so the tray
#: can always pay for a freshly picked combo.
MAX_POWER_VALUE = max(p.value for p in POWERS.values())


# --------------------------------------------------------------------------- #
# Region
# --------------------------------------------------------------------------- #

class Region:
    """One board region: its immutable definition plus its current occupation.

    The static part (terrain, symbols, border flag, adjacency, anchor) lives in
    :attr:`static` (a `RegionDef` from `map3p`); everything else is dynamic
    state, all of it plain scalars so that a deep copy is cheap and total.

    Attributes:
        static: the `RegionDef` of this region.
        owner: seat index of the player whose tokens are here, else None.
        race: which of that player's races holds it, else None.
        in_decline: True if the tokens here belong to a declined race.
        tokens: number of race tokens of `race` in the region (0 if empty).
        lost_tribe: True while the Lost Tribe token of the setup is still here.
        lair: Troll's Lair (+1 defence; survives decline, removed on abandon or
            when an enemy conquers the region).
        fortress: Fortified marker (+1 defence, even in decline).
        encampments: number of Bivouacking encampments (+1 defence each).
        hole: Halflings' Hole-in-the-Ground (region immune).
        hero: a Heroic Hero is here (region immune).
        dragon: the Dragon Master's dragon is here (region immune).
        conquered_this_turn_by: seat that conquered the region during the
            current turn, else None (Orcs / Pillaging scoring, and the "a
            region conquered this turn cannot be attacked again" bookkeeping).
    """

    def __init__(self, static: RegionDef):
        self.static = static
        # occupation
        self.owner: int | None = None
        self.race: RaceId | None = None
        self.in_decline: bool = False
        self.tokens: int = 0
        self.lost_tribe: bool = False
        # markers
        self.lair: bool = False
        self.fortress: bool = False
        self.encampments: int = 0
        self.hole: bool = False
        self.hero: bool = False
        self.dragon: bool = False
        # per-turn bookkeeping
        self.conquered_this_turn_by: int | None = None

    # -- static passthroughs ------------------------------------------------ #

    @property
    def id(self) -> int:
        """Region number printed on the board (1..30)."""
        return self.static.id

    @property
    def index(self) -> int:
        """0-based index in `Board.regions` and in the observation (`id - 1`)."""
        return self.static.index

    @property
    def terrain(self) -> Terrain:
        """:class:`Terrain` of the region."""
        return self.static.terrain

    @property
    def symbols(self) -> frozenset[Symbol]:
        """Symbols printed in the region."""
        return self.static.symbols

    @property
    def border(self) -> bool:
        """True if the region may host a first conquest."""
        return self.static.border

    @property
    def adjacent(self) -> tuple[int, ...]:
        """Ids of the regions sharing a border with this one."""
        return self.static.adjacent

    @property
    def is_water(self) -> bool:
        """True for the two seas and the lake."""
        return self.static.is_water

    @property
    def is_mountain(self) -> bool:
        """True for mountain regions (they carry the immovable Mountain token)."""
        return self.static.is_mountain

    def has(self, symbol: Symbol) -> bool:
        """True if `symbol` is printed in this region."""
        return self.static.has(symbol)

    # -- dynamic state ------------------------------------------------------ #

    @property
    def is_empty(self) -> bool:
        """True if the region holds neither a Lost Tribe nor any race token.

        A Mountain token alone leaves the region *empty* (rules p.2), and so do
        the other markers — they only ever sit on an occupied region.
        """
        return not self.lost_tribe and self.tokens <= 0

    @property
    def is_immune(self) -> bool:
        """True if a marker makes the region immune to conquests and powers.

        Halflings' Hole-in-the-Ground, a Hero, or the Dragon.
        """
        return self.hole or self.hero or self.dragon

    @property
    def defence(self) -> int:
        """Defence *above* the base 2 tokens of any conquest.

        +1 per encampment, per fortress, per mountain, per Troll's Lair, for a
        Lost Tribe, and +1 per race token present.
        """
        return (
            self.encampments
            + (1 if self.fortress else 0)
            + (1 if self.is_mountain else 0)
            + (1 if self.lair else 0)
            + (1 if self.lost_tribe else 0)
            + self.tokens
        )

    def base_conquest_cost(self) -> int:
        """Tokens needed to conquer the region, before race/power modifiers.

        ``2 + encampments + fortress + mountain + lair + lost tribe + tokens``
        (rules p.3). Race and power hooks then add their modifiers and the
        engine clamps the result to a minimum of 1.
        """
        return 2 + self.defence

    def clear_race(self) -> None:
        """Remove the race tokens (and their owner) from the region.

        Only the occupation is cleared: the markers are handled by the engine
        and the hooks, because their fate depends on the event (a Troll's Lair
        survives the Trolls' decline but not a conquest, a fortress survives a
        decline, a hole is removed when the Halflings decline, ...).
        """
        self.owner = None
        self.race = None
        self.in_decline = False
        self.tokens = 0

    def reset(self) -> None:
        """Reset the region to its pre-setup state (no tokens, no markers)."""
        self.clear_race()
        self.lost_tribe = False
        self.lair = False
        self.fortress = False
        self.encampments = 0
        self.hole = False
        self.hero = False
        self.dragon = False
        self.conquered_this_turn_by = None

    def __repr__(self) -> str:
        bits = [f'{self.id}', self.terrain.name]
        if self.lost_tribe:
            bits.append('lost_tribe')
        if self.race is not None:
            bits.append(
                f'p{self.owner}:{RACES[self.race].name}'
                f'{"(decline)" if self.in_decline else ""}x{self.tokens}'
            )
        for flag in ('lair', 'fortress', 'hole', 'hero', 'dragon'):
            if getattr(self, flag):
                bits.append(flag)
        if self.encampments:
            bits.append(f'encampments x{self.encampments}')
        return f'<Region {" ".join(bits)}>'


# --------------------------------------------------------------------------- #
# Races in play
# --------------------------------------------------------------------------- #

class RaceInPlay:
    """One race banner held by a player, active or in decline.

    A player has at most one active race and (Spirit aside) one declined race,
    so a `PlayerState` holds 0..3 of these.

    Attributes:
        race: the :class:`RaceId`.
        power: the :class:`PowerId` of the combo while the race is active;
            None once the race has declined (the badge is discarded).
        owner: seat index of the owning player.
        in_decline: True once the banner has been flipped.
        tokens_in_hand: tokens of this race the player holds off the map.
        pending_redeploy: True while `tokens_in_hand` holds tokens taken back
            after losing a region to an attacker; they are redeployed at the
            end of the attacker's turn (rules p.3).
        is_spirit: True for a declined race that had the Spirit power — it does
            not count toward the one-declined-race limit.
        turns_played: how many turns this race has been played (incremented by
            :meth:`new_turn`); 1 during its first turn.
        holes_placed: Halflings' holes placed so far (max 2, first 2 conquests).
        wealthy_paid: True once the Wealthy +7 coins have been paid.
        extra: free dict for hook-specific state (races.py / powers.py).

    Per-turn bookkeeping, reset by :meth:`new_turn`:
        conquests_this_turn: ids of the regions conquered this turn, in order.
        nonempty_conquests: how many of them were non-empty (Orcs, Pillaging,
            Skeletons).
        dragon_used / fortress_used: once-per-turn powers already used.
        sorcerer_used_on: seats already hit by a Sorcerer substitution.
        die: reinforcement die value rolled this turn, else None.
        attacked_players: seats whose *active* race was attacked this turn
            (Diplomat).
        first_conquest_done: True once the first conquest of the turn resolved
            (the border / anywhere restriction only applies before that).
    """

    def __init__(
        self,
        race: RaceId,
        power: PowerId | None,
        owner: int,
        in_decline: bool = False,
        tokens_in_hand: int = 0,
        is_spirit: bool = False,
    ):
        self.race = race
        self.power = power
        self.owner = owner
        self.in_decline = in_decline
        self.tokens_in_hand = tokens_in_hand
        self.pending_redeploy = False
        self.is_spirit = is_spirit
        # persistent bookkeeping
        self.turns_played = 0
        self.holes_placed = 0
        self.wealthy_paid = False
        self.extra: dict = {}
        # per-turn bookkeeping
        self.conquests_this_turn: list[int] = []
        self.nonempty_conquests = 0
        self.dragon_used = False
        self.fortress_used = False
        self.sorcerer_used_on: set[int] = set()
        self.die: int | None = None
        self.attacked_players: set[int] = set()
        self.first_conquest_done = False

    # -- definitions -------------------------------------------------------- #

    @property
    def race_def(self) -> RaceDef:
        """The :class:`RaceDef` of this race."""
        return RACES[self.race]

    @property
    def power_def(self) -> PowerDef | None:
        """The :class:`PowerDef` of the combo, None once declined."""
        return None if self.power is None else POWERS[self.power]

    def initial_tokens(self) -> int:
        """Tokens taken from the tray when the combo is picked.

        Race banner value + power badge value + attack-only bonus tokens
        (Amazons: +4). `RACES[...].total_tokens` is always at least that much.
        """
        race_def = self.race_def
        total = race_def.banner_value + race_def.attack_bonus_tokens
        if self.power is not None:
            total += POWERS[self.power].value
        return total

    # -- bookkeeping -------------------------------------------------------- #

    def new_turn(self) -> None:
        """Start a new turn for this race: reset the per-turn bookkeeping.

        Also increments :attr:`turns_played` (so it is 1 during the race's very
        first turn). The engine calls it once per turn for the active race, and
        once per turn for a declined race that still acts (Ghouls).
        """
        self.turns_played += 1
        self.conquests_this_turn = []
        self.nonempty_conquests = 0
        self.dragon_used = False
        self.fortress_used = False
        self.sorcerer_used_on = set()
        self.die = None
        self.attacked_players = set()
        self.first_conquest_done = False

    def __repr__(self) -> str:
        power = 'declined' if self.power is None else POWERS[self.power].name
        return (
            f'<RaceInPlay p{self.owner} {RACES[self.race].name}/{power}'
            f'{" decline" if self.in_decline else ""}'
            f'{" spirit" if self.is_spirit else ""}'
            f' hand={self.tokens_in_hand}>'
        )


# --------------------------------------------------------------------------- #
# Player
# --------------------------------------------------------------------------- #

def make_change(value: int) -> dict[int, int]:
    """Minimal set of coin tokens worth `value` (greedy 10 / 5 / 3 / 1).

    Args:
        value: total coin value, >= 0.

    Returns:
        `{denomination: count}` with only positive counts (an empty dict for 0),
        holding the **smallest possible number of tokens** — greedy is optimal
        for the 1/3/5/10 set of the real game.

    Raises:
        ValueError: if `value` is negative.
    """
    value = int(value)
    if value < 0:
        raise ValueError(f'cannot make change for a negative value ({value})')
    tokens: dict[int, int] = {}
    for den in COIN_DENOMINATIONS:
        count, value = divmod(value, den)
        if count:
            tokens[den] = count
    return tokens


class PlayerState:
    """Everything about one player except the regions he occupies (see `Board`).

    Coins are modelled as the physical tokens of the real game (plan T3b): the
    *value* of a player's stack is hidden from the others, but the *number of
    tokens* is public, so the denominations matter. Gains are taken from the
    (unlimited) pool as "1" coins and payments are made in "1" coins, breaking a
    bigger coin when needed; after every gain or payment the stack is
    automatically re-made into the minimal number of tokens (`make_change`),
    like a careful player would. A freshly created player holds `coins` coins of
    value 1 (the rules deal five), *not* their minimal change.

    Attributes:
        seat: seat index, 0..n_players-1 (turn order).
        name: display name, for the GUI and the logs.
        coin_tokens: `{denomination: count}`, the actual stack of tokens.
        active: the active `RaceInPlay`, None before the first combo pick and
            during the turn a race declines into the graveyard.
        declined: declined races, at most one plus a Spirit one.
        ally: seat chosen with the Diplomat power (that player may not attack
            this player's active race until his next turn), else None.
        must_first_conquest: True when the next conquest of the active race must
            follow the first-conquest rule (new race, or all regions abandoned
            or lost).
    """

    def __init__(self, seat: int, name: str | None = None, coins: int = START_COINS):
        self.seat = seat
        self.name = name if name is not None else f'player {seat}'
        if int(coins) < 0:
            raise ValueError(f'a player cannot start with {coins} coins')
        self.coin_tokens: dict[int, int] = {1: int(coins)} if coins else {}
        self.active: RaceInPlay | None = None
        self.declined: list[RaceInPlay] = []
        self.ally: int | None = None
        self.must_first_conquest: bool = False

    # -- coins -------------------------------------------------------------- #

    @property
    def coins(self) -> int:
        """Total value of the stack (the victory points; hidden from the others)."""
        return sum(den * count for den, count in self.coin_tokens.items())

    @coins.setter
    def coins(self, value: int) -> None:
        """Set the total value, i.e. :meth:`gain` or :meth:`pay` the difference.

        Setting the value the player already has changes nothing (no coin moved,
        so no change is made either).
        """
        delta = int(value) - self.coins
        if delta > 0:
            self.gain(delta)
        elif delta < 0:
            self.pay(-delta)

    @property
    def coin_count(self) -> int:
        """Number of coin tokens in the stack (public information)."""
        return sum(self.coin_tokens.values())

    def gain(self, n: int) -> None:
        """Take `n` coins of value 1 from the pool, then make change.

        Raises:
            ValueError: if `n` is negative.
        """
        n = int(n)
        if n < 0:
            raise ValueError(f'cannot gain {n} coins (use pay)')
        if n == 0:
            return
        self.coin_tokens[1] = self.coin_tokens.get(1, 0) + n
        self._make_change()

    def pay(self, n: int) -> None:
        """Pay `n` coins of value 1, breaking bigger ones, then make change.

        Raises:
            ValueError: if `n` is negative or larger than the stack's value.
        """
        n = int(n)
        if n < 0:
            raise ValueError(f'cannot pay {n} coins (use gain)')
        if n == 0:
            return
        total = self.coins
        if n > total:
            raise ValueError(
                f'player {self.seat} has {total} coins, cannot pay {n}'
            )
        # break the smallest coins first until `n` coins of value 1 are in hand
        ones = self.coin_tokens.pop(1, 0)
        for den in reversed(COIN_DENOMINATIONS[:-1]):       # 3, then 5, then 10
            while ones < n and self.coin_tokens.get(den, 0):
                self.coin_tokens[den] -= 1
                if not self.coin_tokens[den]:
                    del self.coin_tokens[den]
                ones += den
        self.coin_tokens[1] = ones - n
        self._make_change()

    def _make_change(self) -> None:
        """Re-make the stack into the minimal number of tokens."""
        self.coin_tokens = make_change(self.coins)

    # -- races -------------------------------------------------------------- #

    def all_races(self) -> list[RaceInPlay]:
        """The active race (if any) followed by the declined ones."""
        races = [] if self.active is None else [self.active]
        races.extend(self.declined)
        return races

    def race_by(self, race_id: RaceId, in_decline: bool | None = None) -> RaceInPlay | None:
        """Return this player's `RaceInPlay` for `race_id`, or None.

        Args:
            race_id: the race looked for.
            in_decline: None (default) matches active or declined; True or
                False restricts the search accordingly.
        """
        for rip in self.all_races():
            if rip.race == race_id and (in_decline is None or rip.in_decline == in_decline):
                return rip
        return None

    def __repr__(self) -> str:
        return (
            f'<PlayerState seat={self.seat} coins={self.coins} '
            f'tokens={self.coin_count} '
            f'active={self.active} declined={self.declined}>'
        )


# --------------------------------------------------------------------------- #
# Tray (token and marker supplies)
# --------------------------------------------------------------------------- #

class Tray:
    """The box: race tokens not in play, and the markers not on the board.

    Token conservation invariant, per race: ``tray + hands + board ==
    RACES[race].total_tokens``.
    """

    def __init__(self):
        self.race_tokens: dict[RaceId, int] = {
            rid: RACES[rid].total_tokens for rid in RaceId
        }
        self.markers: dict[str, int] = dict(MARKER_SUPPLY)

    # -- race tokens -------------------------------------------------------- #

    def available(self, race: RaceId) -> int:
        """Tokens of `race` still in the tray."""
        return self.race_tokens[race]

    def take(self, race: RaceId, n: int = 1) -> int:
        """Take up to `n` tokens of `race` out of the tray.

        Returns the number actually taken (0..n), which is smaller than `n`
        when the tray runs out — the Skeletons / Sorcerers limit. Never
        negative, never raises.
        """
        if n <= 0:
            return 0
        taken = min(int(n), self.race_tokens[race])
        self.race_tokens[race] -= taken
        return taken

    def put(self, race: RaceId, n: int = 1) -> None:
        """Put `n` tokens of `race` back into the tray.

        Raises:
            ValueError: if `n` is negative or would push the tray above the
                number of tokens in the box (a token-conservation bug).
        """
        if n == 0:
            return
        if n < 0:
            raise ValueError(f'Tray.put: negative count {n} for {RACES[race].name}')
        total = RACES[race].total_tokens
        if self.race_tokens[race] + n > total:
            raise ValueError(
                f'Tray.put: {RACES[race].name} would hold '
                f'{self.race_tokens[race] + n} tokens > {total} in the box'
            )
        self.race_tokens[race] += n

    # -- markers ------------------------------------------------------------ #

    def available_marker(self, kind: str) -> int:
        """Markers of `kind` still available (see :data:`MARKER_KINDS`)."""
        return self.markers[self._check(kind)]

    def take_marker(self, kind: str) -> bool:
        """Take one marker of `kind`; return False if none is left."""
        kind = self._check(kind)
        if self.markers[kind] <= 0:
            return False
        self.markers[kind] -= 1
        return True

    def put_marker(self, kind: str) -> None:
        """Return one marker of `kind` to the supply.

        Raises:
            ValueError: if that would exceed the printed supply.
        """
        kind = self._check(kind)
        if self.markers[kind] + 1 > MARKER_SUPPLY[kind]:
            raise ValueError(
                f'Tray.put_marker: more than {MARKER_SUPPLY[kind]} {kind}(s)'
            )
        self.markers[kind] += 1

    @staticmethod
    def _check(kind: str) -> str:
        if kind not in MARKER_SUPPLY:
            raise ValueError(
                f'unknown marker kind {kind!r}; expected one of {MARKER_KINDS}'
            )
        return kind

    def __repr__(self) -> str:
        left = {RACES[r].name: n for r, n in self.race_tokens.items() if n}
        return f'<Tray races={left} markers={self.markers}>'


# --------------------------------------------------------------------------- #
# Combos
# --------------------------------------------------------------------------- #

def _shuffled(items: list, np_random: np.random.Generator) -> list:
    """Return a shuffled copy of `items`, preserving the elements' types.

    (`np_random.permutation(items)` would turn `IntEnum` members into numpy
    integers, so only the *indices* are permuted here.)
    """
    order = np_random.permutation(len(items))
    return [items[int(i)] for i in order]


class Combo:
    """One pickable race+power combination, with the coins lying on it.

    Attributes:
        race: the :class:`RaceId` of the banner.
        power: the :class:`PowerId` of the badge.
        coins: number of coins dropped on it by players who skipped it; the
            next player who picks it takes them. They are always tokens of
            value 1 (one per skipped combo), so this count *is* their value.
    """

    def __init__(self, race: RaceId, power: PowerId, coins: int = 0):
        self.race = race
        self.power = power
        self.coins = coins

    def __repr__(self) -> str:
        return (
            f'<Combo {RACES[self.race].name}/{POWERS[self.power].name}'
            f'{f" +{self.coins}c" if self.coins else ""}>'
        )


class ComboColumn:
    """The column of visible race+power combos, plus the two stacks.

    `visible[0]` is the top of the column (free), `visible[i]` costs `i` coins
    — one coin dropped on each combo above it. The stacks are lists whose
    index 0 is the top (next card drawn); `return_race` appends to the end
    (the bottom of the race stack), as the rules ask.

    Attributes:
        visible: the pickable combos, at most :data:`N_VISIBLE_COMBOS`.
        race_stack: unrevealed race banners, index 0 = top.
        power_stack: unrevealed power badges, index 0 = top.
        power_discard: badges discarded by declined races; reshuffled into
            `power_stack` when it runs out.
        np_random: the generator used by :meth:`refill` (reassign it when the
            env is re-seeded).
    """

    def __init__(self, np_random: np.random.Generator):
        self.np_random = np_random
        self.race_stack: list[RaceId] = _shuffled(list(RaceId), np_random)
        self.power_stack: list[PowerId] = _shuffled(list(PowerId), np_random)
        self.power_discard: list[PowerId] = []
        self.visible: list[Combo] = []
        self.refill()

    # -- queries ------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.visible)

    def cost(self, i: int) -> int:
        """Coins needed to pick `visible[i]`: `i` (the top one is free)."""
        return i

    # -- picking ------------------------------------------------------------ #

    def pick(self, i: int, player: PlayerState) -> Combo:
        """Pick `visible[i]` for `player` and refill the column.

        Drops one of the player's coins (a "1" token, broken out of a bigger
        coin if needed) on each combo above `i`, gives him the "1" coins lying
        on the picked combo, removes it from the column and refills (so the
        combos below slide up, keeping their own coins). The player's stack is
        re-made into minimal change by `PlayerState.pay` / `gain`.

        Returns:
            The picked `Combo` (its `coins` reset to 0, they went to `player`).

        Raises:
            IndexError: if `i` is not a visible slot.
            ValueError: if the player cannot pay the `i` coins.
        """
        if not 0 <= i < len(self.visible):
            raise IndexError(f'no visible combo at index {i} ({len(self.visible)} visible)')
        price = self.cost(i)
        if player.coins < price:
            raise ValueError(
                f'player {player.seat} has {player.coins} coins, combo {i} costs {price}'
            )
        for j in range(i):
            self.visible[j].coins += 1
        player.pay(price)
        combo = self.visible.pop(i)
        player.gain(combo.coins)
        combo.coins = 0
        self.refill()
        return combo

    # -- stacks ------------------------------------------------------------- #

    def refill(self) -> None:
        """Reveal combos until :data:`N_VISIBLE_COMBOS` are visible.

        Stops early when the race stack is empty (the column then simply holds
        fewer combos), or when no power is available either — a power badge is
        drawn from `power_stack`, reshuffling `power_discard` into it when it
        runs out.
        """
        while len(self.visible) < N_VISIBLE_COMBOS and self.race_stack:
            power = self._draw_power()
            if power is None:
                break
            race = self.race_stack.pop(0)
            self.visible.append(Combo(race, power))

    def _draw_power(self) -> PowerId | None:
        """Draw the top power badge, reshuffling the discard if needed."""
        if not self.power_stack and self.power_discard:
            self.power_stack = _shuffled(self.power_discard, self.np_random)
            self.power_discard = []
        if not self.power_stack:
            return None
        return self.power_stack.pop(0)

    def return_race(self, race: RaceId) -> None:
        """Put a race banner back at the **bottom** of the race stack.

        Happens when a declined race leaves the map (replaced by a new declined
        race, or its last token conquered).
        """
        self.race_stack.append(race)

    def discard_power(self, power: PowerId) -> None:
        """Discard a power badge (a race went in decline)."""
        self.power_discard.append(power)

    # -- MCTS / imperfect information --------------------------------------- #

    def redeterminize(self, np_random: np.random.Generator) -> None:
        """Reshuffle everything a player cannot see.

        The visible combos (and the coins on them) are untouched; the order of
        the race stack, of the power stack and of the discard pile is
        reshuffled — their contents are public, their order is not. Also
        rebinds :attr:`np_random` so later refills use the same generator.
        """
        self.np_random = np_random
        self.race_stack = _shuffled(self.race_stack, np_random)
        self.power_stack = _shuffled(self.power_stack, np_random)
        self.power_discard = _shuffled(self.power_discard, np_random)

    def __repr__(self) -> str:
        return (
            f'<ComboColumn {self.visible} races_left={len(self.race_stack)} '
            f'powers_left={len(self.power_stack)} discard={len(self.power_discard)}>'
        )


# --------------------------------------------------------------------------- #
# Dice
# --------------------------------------------------------------------------- #

class Dice:
    """The reinforcement die, pre-rolled so that no chance step is needed.

    Faces are :data:`DIE_FACES` = (0, 0, 0, 1, 2, 3): a 0 is three times as
    likely as any other value. The sequence is drawn ahead of time (300 values
    by default, extended on demand) so that `step()` never has to consult the
    generator, which keeps `info["next_step_no_action"]` always False.

    Attributes:
        np_random: generator used to extend the sequence (reassign it when the
            env is re-seeded).
    """

    def __init__(self, np_random: np.random.Generator, n: int = 300):
        self.np_random = np_random
        self._chunk = max(1, int(n))
        self._values: list[int] = []
        self._pos = 0
        self._extend(int(n))

    def _extend(self, n: int) -> None:
        """Append `n` freshly rolled faces to the sequence."""
        if n <= 0:
            return
        rolled = self.np_random.choice(np.asarray(DIE_FACES), size=n)
        self._values.extend(int(v) for v in rolled.tolist())

    @property
    def remaining(self) -> int:
        """Pre-rolled values not consumed yet."""
        return len(self._values) - self._pos

    def peek(self) -> int:
        """Next value, without consuming it."""
        if self.remaining <= 0:
            self._extend(self._chunk)
        return self._values[self._pos]

    def next(self) -> int:
        """Consume and return the next value."""
        value = self.peek()
        self._pos += 1
        return value

    def redeterminize(self, np_random: np.random.Generator) -> None:
        """Reshuffle the values not consumed yet (same multiset, same count).

        The values already rolled are part of the public history and stay put.
        Also rebinds :attr:`np_random`.
        """
        self.np_random = np_random
        rest = self._values[self._pos:]
        if len(rest) > 1:
            self._values[self._pos:] = _shuffled(rest, np_random)

    def __repr__(self) -> str:
        return f'<Dice rolled={self._pos} remaining={self.remaining}>'


# --------------------------------------------------------------------------- #
# Board
# --------------------------------------------------------------------------- #

class Board:
    """The 30 regions of the map plus the queries the engine and GUI need.

    Attributes:
        n_players: player count the map was built for.
        regions: the `Region` objects, ordered by id (`regions[r.index]`).
    """

    def __init__(self, n_players: int = 3):
        self.n_players = n_players
        self.regions: list[Region] = [Region(d) for d in map_for(n_players)]

    # -- access ------------------------------------------------------------- #

    @property
    def n_regions(self) -> int:
        """Number of regions on the map."""
        return len(self.regions)

    def by_id(self, region_id: int) -> Region:
        """Region with the number printed on the board (1-based)."""
        if not 1 <= region_id <= len(self.regions):
            raise IndexError(f'no region with id {region_id}')
        return self.regions[region_id - 1]

    def by_index(self, index: int) -> Region:
        """Region at `index` (0-based, as in the observation)."""
        return self.regions[index]

    def neighbours(self, region: Region | int) -> list[Region]:
        """Regions sharing a border with `region` (a `Region` or a region id).

        Ordered by id. Purely geographic: the Underworld cavern links and the
        Flying power are handled by the hooks, not here.
        """
        region_id = region if isinstance(region, int) else region.id
        return [self.by_id(other) for other in sorted(ADJACENCY[region_id])]

    # -- setup -------------------------------------------------------------- #

    def setup(self) -> None:
        """Reset every region and place the Lost Tribe tokens.

        Nothing else is placed: the Mountain tokens are static
        (`Region.is_mountain`) and every other marker enters play through a
        race or a power.
        """
        for region in self.regions:
            region.reset()
            if region.has(Symbol.LOST_TRIBE):
                region.lost_tribe = True

    # -- queries ------------------------------------------------------------ #

    def regions_of(self, seat: int, in_decline: bool | None = None) -> list[Region]:
        """Regions occupied by `seat`.

        Args:
            seat: the player's seat index.
            in_decline: None (default) for every region he holds, True for the
                regions of his declined race(s) only, False for his active
                race only.
        """
        return [
            r for r in self.regions
            if r.owner == seat and (in_decline is None or r.in_decline == in_decline)
        ]

    def regions_of_race(self, race_in_play: RaceInPlay) -> list[Region]:
        """Regions occupied by that exact race of that exact player."""
        return [
            r for r in self.regions
            if r.owner == race_in_play.owner
            and r.race == race_in_play.race
            and r.in_decline == race_in_play.in_decline
        ]

    def count_tokens_on_board(self, seat: int) -> int:
        """Total race tokens `seat` has on the map (active and declined).

        This is the end-of-game tie-break.
        """
        return sum(r.tokens for r in self.regions if r.owner == seat)

    def is_coastal(self, region: Region | int) -> bool:
        """True if `region` touches a sea or the lake (Tritons)."""
        region_id = region if isinstance(region, int) else region.id
        return region_id in COASTAL_IDS

    def is_border(self, region: Region | int) -> bool:
        """True if `region` may host a first conquest."""
        region_id = region if isinstance(region, int) else region.id
        return region_id in BORDER_IDS

    def caverns(self) -> list[Region]:
        """The regions with a Cavern symbol (all linked by the Underworld)."""
        return [r for r in self.regions if r.has(Symbol.CAVERN)]

    def __repr__(self) -> str:
        occupied = sum(1 for r in self.regions if not r.is_empty)
        return (
            f'<Board {self.n_players}p {len(self.regions)} regions, '
            f'{occupied} non-empty>'
        )
