"""Jamaica constants: enums, sizes and the action layout.

This module must stay free of nicegui / gymnasium imports: the policy
(`models/jamaica/models.py`) imports it on its own.

Action layout (`Discrete(N_ACTIONS)`, no size depends on the board or the deck):

    A_CAPTAIN       0-49   order*25 + card code (order 0 = higher die in the
                           morning, 1 = lower die in the morning; 1 is masked
                           on doubles). The Captain places the dice and picks
                           his card in one decision (no information arrives
                           in between).
    A_CARD         50-74   card code = 5*morning_symbol + evening_symbol
    A_LOAD_HOLD    75-80   hold to empty for a load (regular slots 0-4 in
                           canonical order, 5 = the 6th hold)
    A_PAY_HOLD     81-86   hold to pay from (it pays min(remaining, count))
    A_POWDER      87-123   spend k = 0..36 gunpowder in a battle
    A_DEST       124-127   k-th candidate destination (sorted by distance to
                           Port Royal, then node id): forks and shortage
                           retreats
    A_TARGET     128-132   attack the ship at relative seat offset 1..5
    A_SABRE      133-134   keep / reroll (Saran's Sabre)
    A_STEAL_HOLD 135-140   battle loot: the loser's hold slot
    A_STEAL_POWER 141-144  battle loot: Map / Sabre / Lady Beth / 6th Hold
    A_STEAL_HIDDEN   145   battle loot: one of the loser's face-down cards
    A_GIVE_CURSED    146   battle loot: give my most negative cursed card
    A_NOTHING        147   battle loot: take nothing
"""

from enum import IntEnum

MIN_PLAYERS = 3
MAX_PLAYERS = 6
DEFAULT_PLAYERS = 4
#: every supported player count: one network plays them all (the spaces are sized
#: for MAX_PLAYERS and the observation carries the count)
PLAYER_COUNTS = tuple(range(MIN_PLAYERS, MAX_PLAYERS + 1))
MAX_ROUNDS = 50            #: round cap, then normal scoring
N_HOLDS = 5                #: regular holds per ship
N_SLOTS = N_HOLDS + 1      #: + the 6th Hold treasure card
MAX_HOLD = 6               #: tokens per hold (a load is one die, loot moves as a unit)
HAND_SIZE = 3
MAP_HAND_SIZE = 4          #: with Morgan's Map
START_FOOD = 3
START_GOLD = 3
MAX_DEST = 4               #: destination slots (checked against the real board)
MAX_POWDER = N_SLOTS * MAX_HOLD   #: 36 gunpowder at most
LADY_BETH_BONUS = 2
FLOOR_SCORE = -5           #: score of a ship at or before the "-5" mark


class Res(IntEnum):
    EMPTY = 0
    GOLD = 1
    FOOD = 2
    POWDER = 3


class Sym(IntEnum):
    GOLD = 0
    FOOD = 1
    POWDER = 2
    FWD = 3
    BACK = 4


N_SYMS = len(Sym)
N_CODES = N_SYMS * N_SYMS  #: card code = 5*morning + evening

SYM_RES = {Sym.GOLD: Res.GOLD, Sym.FOOD: Res.FOOD, Sym.POWDER: Res.POWDER}


class Kind(IntEnum):
    PORT_ROYAL = 0
    PORT = 1
    SEA = 2
    LAIR = 3


class Power(IntEnum):
    MAP = 0     #: Morgan's Map: hand of 4
    SABRE = 1   #: Saran's Sabre: one reroll per battle
    BETH = 2    #: Lady Beth: +2 to the combat die
    SIXTH = 3   #: 6th Hold


POWER_NAMES = {
    Power.MAP: "Morgan's Map",
    Power.SABRE: "Saran's Sabre",
    Power.BETH: 'Lady Beth',
    Power.SIXTH: '6th Hold',
}

SYM_NAMES = {
    Sym.GOLD: 'gold',
    Sym.FOOD: 'food',
    Sym.POWDER: 'gunpowder',
    Sym.FWD: 'forward',
    Sym.BACK: 'backward',
}

RES_NAMES = {Res.EMPTY: 'empty', Res.GOLD: 'gold', Res.FOOD: 'food', Res.POWDER: 'gunpowder'}

STAR = -1  #: the star face of the combat die


def is_star(face: int) -> bool:
    return face == STAR


def card_code(morning: int, evening: int) -> int:
    return int(morning) * N_SYMS + int(evening)


def card_syms(code: int) -> tuple[int, int]:
    """(morning symbol, evening symbol) of a card code."""
    return code // N_SYMS, code % N_SYMS


# --- action layout ---------------------------------------------------------
A_CAPTAIN = 0
A_CARD = A_CAPTAIN + 2 * N_CODES          # 50
A_LOAD_HOLD = A_CARD + N_CODES            # 75
A_PAY_HOLD = A_LOAD_HOLD + N_SLOTS        # 81
A_POWDER = A_PAY_HOLD + N_SLOTS           # 87
A_DEST = A_POWDER + MAX_POWDER + 1        # 124
A_TARGET = A_DEST + MAX_DEST              # 128
A_SABRE = A_TARGET + MAX_PLAYERS - 1      # 133
A_STEAL_HOLD = A_SABRE + 2                # 135
A_STEAL_POWER = A_STEAL_HOLD + N_SLOTS    # 141
A_STEAL_HIDDEN = A_STEAL_POWER + len(Power)   # 145
A_GIVE_CURSED = A_STEAL_HIDDEN + 1        # 146
A_NOTHING = A_GIVE_CURSED + 1             # 147
N_ACTIONS = A_NOTHING + 1                 # 148

SABRE_KEEP = 0
SABRE_REROLL = 1


class Family(IntEnum):
    """Action families (the policy embeds them)."""
    CAPTAIN = 0
    CARD = 1
    LOAD_HOLD = 2
    PAY_HOLD = 3
    POWDER = 4
    DEST = 5
    TARGET = 6
    SABRE = 7
    STEAL_HOLD = 8
    STEAL_POWER = 9
    STEAL_HIDDEN = 10
    GIVE_CURSED = 11
    NOTHING = 12


FAMILY_RANGES = (
    (Family.CAPTAIN, A_CAPTAIN, A_CARD),
    (Family.CARD, A_CARD, A_LOAD_HOLD),
    (Family.LOAD_HOLD, A_LOAD_HOLD, A_PAY_HOLD),
    (Family.PAY_HOLD, A_PAY_HOLD, A_POWDER),
    (Family.POWDER, A_POWDER, A_DEST),
    (Family.DEST, A_DEST, A_TARGET),
    (Family.TARGET, A_TARGET, A_SABRE),
    (Family.SABRE, A_SABRE, A_STEAL_HOLD),
    (Family.STEAL_HOLD, A_STEAL_HOLD, A_STEAL_POWER),
    (Family.STEAL_POWER, A_STEAL_POWER, A_STEAL_HIDDEN),
    (Family.STEAL_HIDDEN, A_STEAL_HIDDEN, A_GIVE_CURSED),
    (Family.GIVE_CURSED, A_GIVE_CURSED, A_NOTHING),
    (Family.NOTHING, A_NOTHING, N_ACTIONS),
)


def action_family(action: int) -> tuple[Family, int]:
    """(family, index inside the family) of an action."""
    for fam, lo, hi in FAMILY_RANGES:
        if lo <= action < hi:
            return fam, action - lo
    raise ValueError(f'action {action} out of range')


def _static_tables():
    """Per-action static ids for the policy embeddings.

    Returns (family, sub, morning, evening) lists of length N_ACTIONS:
    `sub` distinguishes the CAPTAIN die order (1/2), SABRE keep/reroll (3/4)
    and the four powers (5-8), 0 otherwise; `morning`/`evening` are the card
    symbols + 1 for CAPTAIN / CARD actions, 0 otherwise.
    """
    fam, sub, msym, esym = [], [], [], []
    for a in range(N_ACTIONS):
        f, i = action_family(a)
        fam.append(int(f))
        s = m = e = 0
        if f == Family.CAPTAIN:
            s = 1 + i // N_CODES
            m, e = (x + 1 for x in card_syms(i % N_CODES))
        elif f == Family.CARD:
            m, e = (x + 1 for x in card_syms(i))
        elif f == Family.SABRE:
            s = 3 + i
        elif f == Family.STEAL_POWER:
            s = 5 + i
        sub.append(s)
        msym.append(m)
        esym.append(e)
    return fam, sub, msym, esym


ACTION_FAMILY, ACTION_SUB, ACTION_MSYM, ACTION_ESYM = _static_tables()
N_FAMILIES = len(Family)
N_SUBS = 9
#: families whose slots are interchangeable (no per-index base logit)
SYMMETRIC_FAMILIES = (Family.LOAD_HOLD, Family.PAY_HOLD, Family.POWDER, Family.DEST,
                      Family.TARGET, Family.STEAL_HOLD)


class Phase(IntEnum):
    """Decision kinds (what the player to act must decide)."""
    CAPTAIN = 0
    CARD = 1
    MOVE_DEST = 2
    RETREAT_DEST = 3
    LOAD_HOLD = 4
    PAY_HOLD = 5
    TARGET = 6
    ATTACK_POWDER = 7
    DEFENSE_POWDER = 8
    SABRE = 9
    REWARD = 10
    TURN_PAUSE = 11
    DONE = 12


N_PHASES = len(Phase)
