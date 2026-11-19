"""Enums, sizes and the action layout of Toy Battle — no nicegui / gym import."""

from __future__ import annotations

from enum import IntEnum


class Troop(IntEnum):
    """The 8 troop types; the value is the strength (Kwak, the joker, is 0)."""
    KWAK = 0
    SKULLY = 1
    CAPN = 2
    JUMBO = 3
    HOOK = 4
    XB42 = 5
    STAR = 6
    ROXY = 7


N_TYPES = len(Troop)
COPIES = 3                  #: copies of each troop per player (24 tiles)
REMOVED = 4                 #: tiles removed unseen from each reserve at setup
RACK_MAX = 8
FIRST_HAND = (3, 4)         #: starting rack of the first / second player

TROOP_NAMES = ('Kwak', 'Skully', "Cap'n", 'Jumbo', 'Hook', 'XB-42', 'Star', 'Roxy')
TROOP_EFFECTS = (
    'Joker: covers any enemy troop, any enemy troop covers it.',
    'Draw 2 troops.',
    'Place 1 extra troop and apply its effect.',
    'Discard 1 visible enemy troop adjacent to Jumbo.',
    'May be placed on any base, connected to your H.Q. or not.',
    "Discard 1 random troop of the opponent's rack.",
    'Draw 1 troop.',
    'No effect.',
)

BLUE, RED = 0, 1            #: seat 0 plays blue, seat 1 plays red
COLOR_NAMES = ('Blue', 'Red')


class Effect(IntEnum):
    """What the special bases of a terrain do."""
    NONE = 0        #: Caribbean Sea: no special base
    RECALL = 1      #: Castle Field: take one of your other visible troops back
    GATE = 2        #: Tropical Pool: only the indicated strengths may be placed
    DRAW = 3        #: City of Clouds: draw 1 troop
    ERUPT = 4       #: Volcanic Jungle: move an adjacent enemy troop
    EXHUME = 5      #: Cursed Cemetery: take one of your discarded troops back
    SHIELD = 6      #: Station Metal-X: troop effects are not applied
    SNIPER = 7      #: Battlefield: one random troop of the opponent's rack is unusable next turn


#: Special-base effects applied after the troop effects of the turn.
TRIGGERED = (Effect.RECALL, Effect.DRAW, Effect.ERUPT, Effect.EXHUME, Effect.SNIPER)

EFFECT_TEXT = {
    Effect.NONE: 'No special base.',
    Effect.RECALL: 'Special base: you may take one of your other visible troops back on your rack.',
    Effect.GATE: 'Special bases and marked H.Q.: only the indicated strengths (or Kwak) may be placed.',
    Effect.DRAW: 'Special base: draw 1 troop.',
    Effect.ERUPT: 'Special base: you may move an adjacent enemy troop to a base adjacent to it.',
    Effect.EXHUME: 'Special base: you may take one of your discarded troops back on your rack.',
    Effect.SHIELD: 'Special base: the effect of the troop placed there is not applied.',
    Effect.SNIPER: "Special base: one random troop of the opponent's rack cannot be placed on their next turn.",
}


class Phase(IntEnum):
    """The decision the current player faces."""
    MAIN = 0        #: draw, or place a troop
    EXTRA = 1       #: Cap'n: place an extra troop, or skip
    JUMBO = 2       #: Jumbo: pick the adjacent enemy troop to discard, or skip
    RECALL = 3      #: Castle Field: pick the troop to take back, or skip
    EXHUME = 4      #: Cursed Cemetery: pick the troop type to take from the discard, or skip
    ERUPT_SRC = 5   #: Volcanic Jungle: pick the enemy troop to move, or skip
    ERUPT_DST = 6   #: Volcanic Jungle: pick where it goes


N_PHASES = len(Phase)

# --------------------------------------------------------------------------- #
# sizes shared by every terrain (the observation and action spaces are padded)
# --------------------------------------------------------------------------- #

MAX_BASES = 15              #: Cursed Cemetery
MAX_HQ = 2                  #: Tropical Pool (2 / 2), Caribbean Sea (2 blue)
MAX_REGIONS = 14            #: Station Metal-X
N_SLOTS = MAX_BASES + MAX_HQ    #: placement targets: the bases, then the enemy H.Q.

# --------------------------------------------------------------------------- #
# action layout — Discrete(N_ACTIONS)
#
# Slots are *canonical*: a base index as seen by the player to move (on the
# terrains with a 180° symmetry Red sees the board turned, so that "my side" is
# the same slots for both colours — see `Terrain.views`).
# --------------------------------------------------------------------------- #

A_DRAW = 0                              #: draw 2 troops
A_SKIP = 1                              #: decline an optional effect
A_PLACE = 2                             #: + troop * N_SLOTS + slot
A_TARGET = A_PLACE + N_TYPES * N_SLOTS  #: + base slot (Jumbo, recall, eruption)
A_TYPE = A_TARGET + MAX_BASES           #: + troop type (exhume)
N_ACTIONS = A_TYPE + N_TYPES            #: 161


def place_action(troop: int, slot: int) -> int:
    return A_PLACE + int(troop) * N_SLOTS + slot


def hq_slot(k: int) -> int:
    """Placement slot of the k-th enemy H.Q."""
    return MAX_BASES + k


# --------------------------------------------------------------------------- #
# observation layout (built by `ToyBattleEnv.observe`, read by the policy)
# --------------------------------------------------------------------------- #

N_TERRAINS = 8
MAX_DIST = 8                #: hop distances are divided by this (the longest one is 7)

# `glob`
G_TERRAIN = 0                           #: one-hot (8)
G_PHASE = G_TERRAIN + N_TERRAINS        #: one-hot (7)
G_SCALARS = G_PHASE + N_PHASES          #: I am Blue, medals mine / theirs / objective, rack mine / theirs, reserve mine / theirs
N_SCALARS = 8
G_RACK = G_SCALARS + N_SCALARS          #: my rack by type
G_PINNED = G_RACK + N_TYPES             #: my troops laid down (Battlefield)
G_MY_DISCARD = G_PINNED + N_TYPES
G_OPP_DISCARD = G_MY_DISCARD + N_TYPES
G_MY_UNSEEN = G_OPP_DISCARD + N_TYPES   #: my tiles still in my reserve or in the box
G_OPP_UNSEEN = G_MY_UNSEEN + N_TYPES    #: their tiles not seen yet (rack, reserve or box)
G_CLOCK = G_OPP_UNSEEN + N_TYPES        #: turns played / turn limit
G_CAN_DRAW = G_CLOCK + 1
GLOB_F = G_CAN_DRAW + 1                 #: 73

# `bases` columns
B_EXISTS, B_SPECIAL, B_MINE, B_THEIRS, B_EMPTY = 0, 1, 2, 3, 4
B_TOP = 5                               #: type of the visible troop, one-hot (8)
B_HEIGHT, B_MY_TILES, B_THEIR_TILES = 13, 14, 15
B_UNDER_MINE, B_UNDER_THEIRS, B_UNDER_STRENGTH, B_UNDER_KWAK = 16, 17, 18, 19
B_MY_REACH, B_THEIR_REACH, B_MY_RANGE, B_THEIR_RANGE = 20, 21, 22, 23
B_MY_HQ, B_THEIR_HQ = 24, 25            #: one path away from an H.Q.
B_GATE = 26                             #: strengths allowed, by type (8)
B_ORIGIN, B_ERUPT_SRC = 34, 35          #: where the pending effect comes from / the troop being moved
B_MY_GAIN, B_THEIR_GAIN = 36, 37        #: medals (share of the objective) won by occupying the base now
B_QUEUED = 38                           #: its special effect is still to apply this turn
B_MY_DIST, B_THEIR_DIST = 39, 40        #: paths to the nearest H.Q. of each side / MAX_DIST
BASE_F = 41

REGION_F = MAX_BASES + 7                #: bases around it, exists, medals left, taken mine / theirs, my / their share, size
HQ_F = 2 + N_TYPES + MAX_BASES          #: exists, can be captured now, strengths allowed, adjacent bases

MAX_TURNS = 300             #: safeguard (not a rule): the game is cut after that many turns
SHAPING = 0.5               #: weight of the medal potential in the reward
