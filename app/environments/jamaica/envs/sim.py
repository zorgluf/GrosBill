"""Deterministic consequence simulator (policy features only, never the rules).

It replays one player's card on a light copy of its ship with the engine's own
`track` / `rules` helpers, enumerating nothing random: a battle is flagged
(resources unchanged), a lair draw is flagged (the pile is never read), a fork
or a shortage retreat takes a fixed greedy choice, a forced dump drops the
least valuable hold. It reads only the acting ship, the public ship positions
and the lair tokens, so it cannot leak hidden information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import rules
from .constants import Kind, Res, STAR, SYM_RES, Sym, card_syms
from .rules import SIXTH_SLOT, hold_at

#: weight of a token when choosing which hold to throw overboard
DUMP_WEIGHT = {Res.GOLD: 3.0, Res.FOOD: 1.2, Res.POWDER: 1.0}


class SimShip:
    """Copy of the parts of a `rules.Ship` the hold helpers use."""
    __slots__ = ('holds', 'sixth', 'powers', 'node', 'lap', 'finished')

    def __init__(self, ship):
        self.holds = [list(h) for h in ship.holds]
        self.sixth = None if ship.sixth is None else list(ship.sixth)
        self.powers = ship.powers
        self.node, self.lap = ship.node, ship.lap
        self.finished = ship.finished

    def copy(self) -> 'SimShip':
        return SimShip(self)

    def has(self, power: int) -> bool:
        return bool(self.powers >> int(power) & 1)


@dataclass(slots=True)
class World:
    occupied: frozenset          #: nodes with another (unfinished) ship
    ships_at: dict               #: node -> number of other ships
    foe_powder: dict             #: node -> max gunpowder among the ships there
    tokens: set                  #: lairs still holding a treasure token


@dataclass(slots=True)
class Half:
    """One action (morning or evening) of a simulated card."""
    sym: int
    die: int
    progress: int = 0            #: spaces gained towards the finish (negative = back)
    combat: bool = False
    lair: bool = False           #: a treasure is drawn
    shortage: bool = False
    retreat: int = 0             #: spaces lost falling back
    dumped: int = 0              #: tokens thrown overboard by a forced load
    ignored: bool = False        #: load impossible, or the ship already finished
    loaded: int = 0
    finished: bool = False


@dataclass(slots=True)
class Outcome:
    halves: list = field(default_factory=list)
    d_gold: int = 0
    d_food: int = 0
    d_powder: int = 0
    d_empty: int = 0
    d_pos_score: int = 0
    progress: int = 0
    combats: int = 0
    lairs: int = 0
    shortages: int = 0
    retreat: int = 0
    dumped: int = 0
    finished: bool = False
    floor_zone: bool = False

    def d_score(self, mu_draw: float) -> float:
        return self.d_pos_score + self.d_gold + self.lairs * mu_draw


def world_of(env, seat: int) -> World:
    occ, count, powder = set(), {}, {}
    for ship in env.ships:
        if ship.seat == seat or ship.finished or ship.node == env.track.pr:
            continue
        occ.add(ship.node)
        count[ship.node] = count.get(ship.node, 0) + 1
        powder[ship.node] = max(powder.get(ship.node, 0), rules.total(ship, Res.POWDER))
    tokens = {n for n, t in env.lair_token.items() if t}
    return World(frozenset(occ), count, powder, tokens)


def _pay_greedy(ship: SimShip, res: int, cost: int) -> None:
    """Pay from the smallest holds first (frees holds)."""
    while cost > 0:
        cands = [s for s in rules.slots(ship) if hold_at(ship, s)[0] == res]
        slot = min(cands, key=lambda s: (hold_at(ship, s)[1], s != SIXTH_SLOT))
        cost -= rules.take(ship, slot, cost)


def _load_greedy(ship: SimShip, res: int, amount: int, half: Half) -> None:
    slot = rules.empty_slot(ship)
    if slot is None:
        classes = rules.dump_classes(ship, res)
        if not classes:
            half.ignored = True
            return
        slot = min(classes, key=lambda s: hold_at(ship, s)[1] * DUMP_WEIGHT[hold_at(ship, s)[0]])
        half.dumped = hold_at(ship, slot)[1]
    rules.load_into(ship, slot, res, amount)
    half.loaded = amount


def _land(track, ship: SimShip, pos, world: World, half: Half, depth: int = 0) -> None:
    node, lap = pos
    ship.node, ship.lap = node, lap
    if node == track.pr and lap >= 1:
        ship.finished = True
        half.finished = True
        return
    if node != track.pr and node in world.occupied:
        half.combat = True
    kind, cost = track.kind[node], track.cost[node]
    if kind == Kind.LAIR:
        if node in world.tokens:
            half.lair = True
            world.tokens.discard(node)
        return
    res = rules.space_res(kind)
    if res == Res.EMPTY or cost <= 0:
        return
    if rules.total(ship, res) >= cost:
        _pay_greedy(ship, res, cost)
        return
    half.shortage = True
    rules.drain_all(ship, res)
    if depth > 8:
        return
    targets = track.retreat_targets(pos, lambda nd: rules.can_pay(ship, track.kind[nd], track.cost[nd]))
    target = targets[0]
    half.retreat += track.remaining(*target) - track.remaining(*pos)
    _land(track, ship, target, world, half, depth + 1)


def _move_options(track, ship: SimShip, d: int, die: int, world: World):
    """Simulated landing for every destination, best first (greedy key)."""
    out = []
    for dest in track.destinations((ship.node, ship.lap), d, die):
        s = ship.copy()
        w = World(world.occupied, world.ships_at, world.foe_powder, set(world.tokens))
        h = Half(Sym.FWD if d > 0 else Sym.BACK, die)
        _land(track, s, dest, w, h)
        gain = track.pos_score(s.node, s.lap) - track.pos_score(ship.node, ship.lap)
        key = (h.shortage, not h.lair, -gain, h.combat, track.remaining(s.node, s.lap), dest[0])
        out.append((key, dest, s, w, h))
    out.sort(key=lambda x: x[0])
    return out


def play(env, seat: int, code: int, dice: tuple[int, int], world: World | None = None) -> Outcome:
    """Simulated outcome of `seat` playing card `code` with (morning, evening) dice."""
    track = env.track
    start = env.ships[seat]
    ship = SimShip(start)
    world = world_of(env, seat) if world is None else World(
        world.occupied, world.ships_at, world.foe_powder, set(world.tokens))
    out = Outcome()
    rem0 = track.remaining(ship.node, ship.lap)
    for slot, sym in enumerate(card_syms(code)):
        half = Half(sym, dice[slot])
        rem_before = track.remaining(ship.node, ship.lap)
        if ship.finished:
            half.ignored = True
        elif sym in SYM_RES:
            _load_greedy(ship, SYM_RES[sym], dice[slot], half)
        else:
            opts = _move_options(track, ship, 1 if sym == Sym.FWD else -1, dice[slot], world)
            _, _, ship, world, half = opts[0]
        half.progress = rem_before - track.remaining(ship.node, ship.lap)
        out.halves.append(half)
    out.d_gold = rules.total(ship, Res.GOLD) - rules.total(start, Res.GOLD)
    out.d_food = rules.total(ship, Res.FOOD) - rules.total(start, Res.FOOD)
    out.d_powder = rules.total(ship, Res.POWDER) - rules.total(start, Res.POWDER)
    out.d_empty = rules.n_empty(ship) - rules.n_empty(start)
    out.d_pos_score = track.pos_score(ship.node, ship.lap) - track.pos_score(start.node, start.lap)
    out.progress = rem0 - track.remaining(ship.node, ship.lap)
    out.combats = sum(h.combat for h in out.halves)
    out.lairs = sum(h.lair for h in out.halves)
    out.shortages = sum(h.shortage for h in out.halves)
    out.retreat = sum(h.retreat for h in out.halves)
    out.dumped = sum(h.dumped for h in out.halves)
    out.finished = ship.finished
    out.floor_zone = track.in_floor_zone(ship.node, ship.lap)
    return out


def land_outcome(env, seat: int, pos) -> tuple[Half, SimShip]:
    """Simulated landing of `seat` on `pos` (for destination features)."""
    ship = SimShip(env.ships[seat])
    half = Half(Sym.FWD, 0)
    _land(env.track, ship, pos, world_of(env, seat), half)
    return half, ship


# --- battle odds -------------------------------------------------------------

def p_battle(att_k: int, att_bonus: int, def_k: int, def_bonus: int, faces) -> tuple[float, float]:
    """(P(attacker wins), P(tie)) with both dice still to roll (no Sabre)."""
    n = len(faces)
    win = tie = 0.0
    for fa in faces:
        if fa == STAR:
            win += 1.0 / n
            continue
        for fd in faces:
            if fd == STAR:
                continue
            a, d = fa + att_k + att_bonus, fd + def_k + def_bonus
            if a > d:
                win += 1.0 / (n * n)
            elif a == d:
                tie += 1.0 / (n * n)
    return win, tie


def p_defend(att_strength: int, def_k: int, def_bonus: int, faces) -> tuple[float, float]:
    """(P(defender wins), P(tie)) against a known attacker strength."""
    n = len(faces)
    win = tie = 0.0
    for fd in faces:
        if fd == STAR:
            win += 1.0 / n
        elif fd + def_k + def_bonus > att_strength:
            win += 1.0 / n
        elif fd + def_k + def_bonus == att_strength:
            tie += 1.0 / n
    return win, tie
