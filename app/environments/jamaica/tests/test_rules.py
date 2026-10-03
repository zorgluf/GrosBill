"""Pure rule helpers: holds, loading, paying, battle arithmetic."""

from ..envs import rules
from ..envs.constants import Kind, Power, Res, STAR
from ..envs.data import COMBAT_FACES
from ..envs.rules import SIXTH_SLOT, Ship

G, F, P, E = Res.GOLD, Res.FOOD, Res.POWDER, Res.EMPTY


def _ship(holds, sixth=None, powers=0):
    s = Ship(seat=0)
    s.holds = [list(h) for h in holds] + [[E, 0] for _ in range(5 - len(holds))]
    s.sixth = None if sixth is None else list(sixth)
    s.powers = powers | ((1 << Power.SIXTH) if sixth is not None else 0)
    rules.sort_holds(s)
    return s


def test_canonical_order():
    s = _ship([(F, 2), (E, 0), (G, 3), (P, 1), (G, 5)])
    assert s.holds == [[G, 5], [G, 3], [F, 2], [P, 1], [E, 0]]


def test_empty_slot_prefers_regular_holds():
    s = _ship([(G, 1)] * 4, sixth=(E, 0))
    assert rules.empty_slot(s) == 4
    s = _ship([(G, 1)] * 5, sixth=(E, 0))
    assert rules.empty_slot(s) == SIXTH_SLOT
    s = _ship([(G, 1)] * 5)
    assert rules.empty_slot(s) is None


def test_dump_classes():
    s = _ship([(G, 3), (F, 2), (F, 2), (P, 4), (G, 1)], sixth=(F, 2))
    # loading food: food holds are never emptied; gold 3, gold 1, powder 4 are classes
    assert rules.dump_classes(s, F) == [0, 1, 4]
    # loading gold: the two food-2 holds are one class, the 6th hold another
    assert rules.dump_classes(s, G) == [2, 4, SIXTH_SLOT]
    assert not rules.can_load(_ship([(G, 2)] * 5), G)
    assert rules.can_load(_ship([(G, 2)] * 5), F)


def test_pay_classes_and_take():
    s = _ship([(F, 3), (F, 3), (F, 1), (G, 2)])
    assert s.holds[:4] == [[G, 2], [F, 3], [F, 3], [F, 1]]
    assert rules.pay_classes(s, F) == [1, 3]
    assert rules.take(s, 3, 5) == 1           # drains the 1-food hold
    assert s.holds[-1] == [E, 0] and s.holds[-2] == [E, 0]
    assert rules.total(s, F) == 6


def test_drain_and_load_into():
    s = _ship([(F, 3), (F, 2), (G, 4)], sixth=(F, 1))
    assert rules.drain_all(s, F) == 6
    assert rules.total(s, F) == 0 and s.sixth == [E, 0]
    dumped = rules.load_into(s, 0, P, 5)
    assert dumped == (G, 4) and rules.total(s, P) == 5 and rules.total(s, G) == 0


def test_spend_powder_smallest_first():
    s = _ship([(P, 4), (P, 2), (G, 1)], sixth=(P, 2))
    rules.spend_powder(s, 3)
    # the 6th hold (2) goes first on the tie, then one from the regular 2
    assert s.sixth == [E, 0]
    assert sorted(h[1] for h in s.holds if h[0] == P) == [1, 4]


def test_battle_winner():
    assert rules.battle_winner(STAR, 0, 0, 10, 9, 2) == 1
    assert rules.battle_winner(2, 5, 0, STAR, 0, 0) == -1
    assert rules.battle_winner(6, 1, 0, 4, 3, 0) == 0
    assert rules.battle_winner(4, 0, 2, 6, 0, 0) == 0       # Lady Beth +2
    assert rules.battle_winner(8, 0, 0, 6, 1, 0) == 1


def test_powder_dominance():
    # beating any defence: def 3 powder + 10 - 2 + 1 = 12, capped by own powder
    assert rules.attacker_powder_max(20, 3, 0, 0, COMBAT_FACES) == 12
    assert rules.attacker_powder_max(5, 3, 0, 0, COMBAT_FACES) == 5
    assert rules.attacker_powder_max(0, 3, 0, 0, COMBAT_FACES) == 0
    # attacker strength 14: below 4 only a star helps; 13 guarantees the win
    assert rules.defender_powder_options(20, 14, 0, COMBAT_FACES) == [0] + list(range(4, 14))
    assert rules.defender_powder_options(3, 14, 0, COMBAT_FACES) == [0]
    assert rules.defender_powder_options(20, 4, 2, COMBAT_FACES) == [0, 1]


def test_can_pay():
    s = _ship([(G, 3), (F, 2)])
    assert rules.can_pay(s, Kind.PORT, 3) and not rules.can_pay(s, Kind.PORT, 4)
    assert rules.can_pay(s, Kind.SEA, 2) and not rules.can_pay(s, Kind.SEA, 3)
    assert rules.can_pay(s, Kind.LAIR, 0) and rules.can_pay(s, Kind.PORT_ROYAL, 0)
