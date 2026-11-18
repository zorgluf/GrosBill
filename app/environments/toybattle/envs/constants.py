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


MAX_TURNS = 300             #: safeguard (not a rule): the game is cut after that many turns
SHAPING = 0.5               #: weight of the medal potential in the reward
