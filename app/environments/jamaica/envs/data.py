"""Jamaica game data, transcribed from photos of the physical game (2026-09-30).

Board: 50 spaces = Port Royal + 29 sea + 11 ports + 9 pirate lairs. The race
goes clockwise on the board photo (Port Royal bottom right, west along the
bottom, north up the left side, east along the top, south down the right
side). Three forks, each a long outer lane (6 spaces, 2 lairs) around an
outer island and a short inner channel (2 spaces, expensive sea):

* bottom left: split at 6 (port 5), outer 7-12, inner 13-14, merge at 15
* top left:    split at 19 (lair), outer 20-25, inner 26-27, merge at 28
* top right:   split at 32 (sea 3), outer 33-38, inner 39-40, merge at 41

Scores: every space before the red "-5" line scores -5 (`score=None`); after
it the printed numbers 2..10, and 15 on Port Royal (at the finish).

Coordinates are pixels of the reference photo (`BOARD_SIZE`, portrait) and are
only used to draw our own board; the renderer does not reuse any artwork.
"""

from .constants import Kind, Power, Sym, STAR, card_code

BOARD_SIZE = (2004, 2996)   #: (width, height) of the coordinate space

PR = 0   #: Port Royal: start and finish

_P, _S, _L, _R = Kind.PORT, Kind.SEA, Kind.LAIR, Kind.PORT_ROYAL

#: (id, kind, cost, score or None, x, y); cost = doubloons (port) or food (sea)
SPACES = (
    (0, _R, 0, 15, 1600, 2780),     # Port Royal
    # bottom, going west
    (1, _S, 2, None, 1400, 2720),
    (2, _S, 3, None, 1221, 2742),
    (3, _S, 2, None, 1075, 2706),
    (4, _L, 0, None, 900, 2830),     # skull rock, bottom centre
    (5, _P, 3, None, 800, 2650),
    (6, _P, 5, None, 660, 2700),     # fork: outer 7 / inner 13
    # bottom-left outer lane (around the big island)
    (7, _S, 2, None, 479, 2804),
    (8, _S, 3, None, 350, 2712),
    (9, _L, 0, None, 140, 2690),     # skull rock, bottom-left corner
    (10, _S, 3, None, 129, 2471),
    (11, _L, 0, None, 120, 2290),    # skull of the big island
    (12, _S, 2, None, 103, 2099),
    # bottom-left inner channel
    (13, _S, 3, None, 581, 2396),
    (14, _S, 4, None, 470, 2175),
    # left side, going north
    (15, _S, 3, None, 307, 1958),    # merge
    (16, _P, 3, None, 330, 1780),
    (17, _S, 2, None, 398, 1585),
    (18, _P, 3, None, 330, 1390),
    (19, _L, 0, None, 190, 1190),    # skull rock, left; fork: outer 20 / inner 26
    # top-left outer lane (around the top-left island)
    (20, _S, 3, None, 125, 945),
    (21, _P, 3, None, 110, 780),
    (22, _L, 0, None, 150, 560),     # skull of the top-left island
    (23, _S, 2, None, 126, 298),
    (24, _S, 3, None, 262, 165),
    (25, _L, 0, None, 510, 125),     # skull island, top
    # top-left inner channel
    (26, _S, 4, None, 505, 965),
    (27, _S, 4, None, 530, 615),
    # top, going east
    (28, _P, 5, None, 715, 420),     # merge
    (29, _S, 1, None, 890, 238),
    (30, _P, 5, None, 1070, 230),
    (31, _P, 3, None, 1270, 230),
    (32, _S, 3, None, 1420, 310),    # fork: outer 33 / inner 39
    # top-right outer lane (around the peninsula)
    (33, _L, 0, None, 1680, 125),    # skull rock of the top-right island
    (34, _P, 3, None, 1880, 250),
    (35, _S, 2, None, 1879, 443),
    (36, _S, 1, None, 1856, 614),
    (37, _L, 0, None, 1870, 820),    # skull island, right
    (38, _S, 1, None, 1817, 1011),
    # top-right inner channel
    (39, _S, 4, None, 1536, 657),
    (40, _S, 4, None, 1541, 945),
    # right side, going south to the finish (after the red -5 line)
    (41, _P, 5, 2, 1720, 1170),      # merge
    (42, _S, 2, 3, 1705, 1322),
    (43, _S, 3, 4, 1672, 1489),
    (44, _L, 0, 5, 1640, 1690),      # skull rock by the Port Royal peninsula
    (45, _P, 7, 6, 1860, 1760),
    (46, _S, 2, 7, 1852, 2040),
    (47, _S, 3, 8, 1830, 2283),
    (48, _S, 3, 9, 1834, 2551),
    (49, _S, 3, 10, 1784, 2781),
)


def _chain(*ids):
    return tuple(zip(ids, ids[1:]))


#: forward edges (from, to)
EDGES = (
    _chain(0, 1, 2, 3, 4, 5, 6)
    + _chain(6, 7, 8, 9, 10, 11, 12, 15)
    + _chain(6, 13, 14, 15)
    + _chain(15, 16, 17, 18, 19)
    + _chain(19, 20, 21, 22, 23, 24, 25, 28)
    + _chain(19, 26, 27, 28)
    + _chain(28, 29, 30, 31, 32)
    + _chain(32, 33, 34, 35, 36, 37, 38, 41)
    + _chain(32, 39, 40, 41)
    + _chain(41, 42, 43, 44, 45, 46, 47, 48, 49, 0)
)

N_LAIRS = 9

#: the 11 action cards of one colour (all colours are identical):
#: (morning symbol, evening symbol)
DECK_CARDS = (
    (Sym.FOOD, Sym.POWDER),
    (Sym.FWD, Sym.BACK),
    (Sym.POWDER, Sym.GOLD),
    (Sym.GOLD, Sym.GOLD),
    (Sym.GOLD, Sym.FWD),
    (Sym.BACK, Sym.FWD),
    (Sym.FWD, Sym.FOOD),
    (Sym.FWD, Sym.FWD),
    (Sym.POWDER, Sym.FOOD),
    (Sym.FOOD, Sym.FWD),
    (Sym.FWD, Sym.POWDER),
)
DECK = tuple(card_code(m, e) for m, e in DECK_CARDS)

#: the combat die (a star wins at once)
COMBAT_FACES = (2, 4, 6, 8, 10, STAR)

#: the 12 treasure cards: (power or None, value); 9 are used per game
TREASURES = (
    (Power.MAP, 0),
    (Power.SABRE, 0),
    (Power.BETH, 0),
    (Power.SIXTH, 0),
    (None, 3),
    (None, 3),
    (None, 5),
    (None, 7),
    (None, 7),
    (None, -2),
    (None, -3),
    (None, -4),
)
N_TREASURES_USED = 9

# --- drawing only: rough outlines of the land (reference-photo pixels) -----

MAIN_ISLAND = (
    (640, 660), (700, 530), (808, 495), (900, 430), (965, 400), (1050, 500),
    (1172, 370), (1250, 420), (1360, 410), (1430, 470), (1470, 560), (1480, 700),
    (1485, 850), (1490, 1000), (1500, 1100), (1560, 1250), (1540, 1400),
    (1510, 1600), (1600, 1720), (1700, 1790), (1790, 1850), (1790, 1960),
    (1740, 2100), (1720, 2300), (1700, 2420), (1680, 2560), (1580, 2560),
    (1440, 2430), (1330, 2380), (1230, 2460), (1120, 2540), (1000, 2530),
    (870, 2530), (780, 2490), (690, 2460), (640, 2400), (570, 2250),
    (530, 2060), (450, 1880), (470, 1700), (560, 1680), (560, 1500),
    (460, 1480), (470, 1300), (560, 1280), (650, 1160), (620, 1000),
    (600, 880), (620, 760),
)

OUTER_ISLANDS = (
    # top-left island
    ((160, 450), (250, 380), (330, 320), (450, 300), (600, 330), (700, 350),
     (640, 390), (560, 470), (470, 600), (420, 760), (460, 900), (480, 1000),
     (410, 1080), (330, 1040), (280, 960), (300, 880), (230, 860), (290, 780),
     (240, 700), (170, 640)),
    # big island, bottom left
    ((190, 2060), (300, 2080), (350, 2150), (420, 2260), (480, 2350),
     (560, 2450), (620, 2560), (650, 2700), (600, 2730), (520, 2680),
     (420, 2620), (300, 2560), (230, 2500), (180, 2400), (150, 2250),
     (140, 2150)),
    # top-right island and its peninsula
    ((1480, 250), (1560, 180), (1680, 170), (1780, 210), (1800, 300),
     (1830, 380), (1790, 420), (1760, 500), (1740, 650), (1700, 800),
     (1720, 950), (1710, 1080), (1660, 1040), (1620, 900), (1600, 800),
     (1640, 650), (1680, 500), (1620, 420), (1540, 360), (1490, 320)),
    # land strip along the left edge
    ((0, 1290), (150, 1280), (250, 1400), (230, 1480), (150, 1560),
     (280, 1620), (300, 1700), (230, 1800), (180, 1880), (60, 1990), (0, 2000)),
)

#: the red "-5" line (drawing only), as a segment
FLOOR_LINE = ((1544, 1090), (1948, 1100))
