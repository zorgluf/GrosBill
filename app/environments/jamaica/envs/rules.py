"""Pure rule helpers shared by the engine and the consequence simulator.

Holds are `[res, count]` lists. A ship has 5 regular holds, kept in canonical
order (non-empty first, by resource then larger count first, empties last), so
slot indices 0-4 are canonical; slot 5 is the 6th Hold card (`ship.sixth`,
None when not owned). `count` is 0 exactly when `res` is EMPTY and never
exceeds 6 (a load is one die; loot moves as a whole hold).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .constants import Kind, LADY_BETH_BONUS, MAX_HOLD, N_HOLDS, Power, Res, STAR

SIXTH_SLOT = N_HOLDS   #: slot index of the 6th Hold


@dataclass(slots=True)
class Ship:
    """One player's ship, holds, cards and treasures."""
    seat: int
    node: int = 0
    lap: int = 0
    holds: list = field(default_factory=lambda: [[Res.EMPTY, 0] for _ in range(N_HOLDS)])
    sixth: list | None = None          #: [res, count] when the 6th Hold is owned
    powers: int = 0                    #: bitmask of Power (face up, public)
    hidden: list = field(default_factory=list)   #: face-down treasure ids
    hand: list = field(default_factory=list)     #: card codes
    deck: list = field(default_factory=list)     #: draw pile, top = end
    discard: list = field(default_factory=list)  #: face up, public
    chosen: int = -1                   #: card picked this round (hidden until revealed)
    revealed: bool = False
    finished: bool = False

    def has(self, power: int) -> bool:
        return bool(self.powers >> int(power) & 1)


def hold_at(ship, slot: int):
    """The hold list of `slot` (5 = the 6th Hold, may be None)."""
    return ship.sixth if slot == SIXTH_SLOT else ship.holds[slot]


def slots(ship) -> list[int]:
    """Slot indices the ship owns (0-4, plus 5 with the 6th Hold)."""
    return list(range(N_HOLDS)) + ([SIXTH_SLOT] if ship.sixth is not None else [])


def sort_holds(ship) -> None:
    ship.holds.sort(key=lambda h: (h[0] == Res.EMPTY, h[0], -h[1]))


def total(ship, res: int) -> int:
    t = sum(h[1] for h in ship.holds if h[0] == res)
    if ship.sixth is not None and ship.sixth[0] == res:
        t += ship.sixth[1]
    return t


def largest(ship, res: int) -> int:
    best = max((h[1] for h in ship.holds if h[0] == res), default=0)
    if ship.sixth is not None and ship.sixth[0] == res:
        best = max(best, ship.sixth[1])
    return best


def n_empty(ship) -> int:
    k = sum(1 for h in ship.holds if h[0] == Res.EMPTY)
    return k + (1 if ship.sixth is not None and ship.sixth[0] == Res.EMPTY else 0)


def empty_slot(ship) -> int | None:
    """Where a load goes without a decision: a regular hold before the 6th."""
    for i, h in enumerate(ship.holds):
        if h[0] == Res.EMPTY:
            return i
    if ship.sixth is not None and ship.sixth[0] == Res.EMPTY:
        return SIXTH_SLOT
    return None


def dump_classes(ship, res: int) -> list[int]:
    """Holds that may be emptied to load `res` (one slot per equivalence class).

    Only meaningful when no hold is empty. Holds of the loaded type are
    excluded (rule: never return the type you are loading).
    """
    out, seen = [], set()
    for slot in slots(ship):
        h = hold_at(ship, slot)
        if h[0] == Res.EMPTY or h[0] == res:
            continue
        key = (h[0], h[1], slot == SIXTH_SLOT)
        if key not in seen:
            seen.add(key)
            out.append(slot)
    return out


def can_load(ship, res: int) -> bool:
    return empty_slot(ship) is not None or bool(dump_classes(ship, res))


def pay_classes(ship, res: int) -> list[int]:
    """Holds `res` can be paid from (one slot per equivalence class)."""
    out, seen = [], set()
    for slot in slots(ship):
        h = hold_at(ship, slot)
        if h[0] != res:
            continue
        key = (h[1], slot == SIXTH_SLOT)
        if key not in seen:
            seen.add(key)
            out.append(slot)
    return out


def take(ship, slot: int, amount: int) -> int:
    """Remove up to `amount` tokens from `slot`; returns how many were taken."""
    h = hold_at(ship, slot)
    k = min(amount, h[1])
    h[1] -= k
    if h[1] == 0:
        h[0] = Res.EMPTY
    if slot != SIXTH_SLOT:
        sort_holds(ship)
    return k


def drain_all(ship, res: int) -> int:
    """Remove every token of `res` (shortage step 1); returns the count."""
    k = 0
    for h in ship.holds + ([ship.sixth] if ship.sixth is not None else []):
        if h[0] == res:
            k += h[1]
            h[0], h[1] = Res.EMPTY, 0
    sort_holds(ship)
    return k


def load_into(ship, slot: int, res: int, amount: int) -> tuple[int, int]:
    """Put `amount` tokens of `res` into `slot`, dumping its content first.

    Returns the dumped (res, count).
    """
    h = hold_at(ship, slot)
    dumped = (int(h[0]), int(h[1]))
    h[0], h[1] = int(res), int(min(amount, MAX_HOLD))
    if h[1] == 0:
        h[0] = Res.EMPTY
    if slot != SIXTH_SLOT:
        sort_holds(ship)
    return dumped


def spend_powder(ship, k: int) -> None:
    """Remove `k` gunpowder, smallest holds first (the 6th Hold first on ties)."""
    order = sorted((slot for slot in slots(ship) if hold_at(ship, slot)[0] == Res.POWDER),
                   key=lambda s: (hold_at(ship, s)[1], s != SIXTH_SLOT))
    for slot in order:
        if k <= 0:
            break
        h = hold_at(ship, slot)
        t = min(k, h[1])
        h[1] -= t
        k -= t
        if h[1] == 0:
            h[0] = Res.EMPTY
    if k > 0:
        raise ValueError('not enough gunpowder')
    sort_holds(ship)


def space_res(kind: int) -> int:
    """Resource a space costs (GOLD for ports, FOOD for sea), EMPTY if free."""
    if kind == Kind.PORT:
        return Res.GOLD
    if kind == Kind.SEA:
        return Res.FOOD
    return Res.EMPTY


def can_pay(ship, kind: int, cost: int) -> bool:
    res = space_res(kind)
    return res == Res.EMPTY or cost <= 0 or total(ship, res) >= cost


# --- combat ---------------------------------------------------------------

def beth(ship) -> int:
    return LADY_BETH_BONUS if ship.has(Power.BETH) else 0


def strength(face: int, powder: int, bonus: int) -> int:
    return face + powder + bonus


def battle_winner(att_face: int, att_k: int, att_bonus: int,
                  def_face: int, def_k: int, def_bonus: int) -> int:
    """+1 attacker wins, -1 defender wins, 0 tie (nothing happens)."""
    if att_face == STAR:
        return 1
    if def_face == STAR:
        return -1
    a = strength(att_face, att_k, att_bonus)
    d = strength(def_face, def_k, def_bonus)
    return (a > d) - (a < d)


def numeric_faces(faces) -> list[int]:
    return [f for f in faces if f != STAR]


def attacker_powder_max(own: int, def_powder: int, att_bonus: int, def_bonus: int, faces) -> int:
    """Largest non-dominated attacker spend.

    Past the amount that beats every non-star defence, more powder is waste.
    """
    num = numeric_faces(faces)
    need = def_powder + max(num) + def_bonus - min(num) - att_bonus + 1
    return max(0, min(own, need))


def defender_powder_options(own: int, att_strength: int, def_bonus: int, faces) -> list[int]:
    """Non-dominated defender spends against a known attacker strength.

    Below the amount that can at least tie with the best face, only a star
    helps, so 0 dominates; past the amount that wins with the worst face, more
    is waste.
    """
    num = numeric_faces(faces)
    lo = max(0, att_strength - max(num) - def_bonus)
    hi = max(0, att_strength - min(num) - def_bonus + 1)
    opts = {0}
    for k in range(max(lo, 1), min(hi, own) + 1):
        opts.add(k)
    return sorted(opts)
