"""Observation of `JamaicaEnv`: a Dict of float32 arrays, normalised by the env.

Everything is from the point of view of one seat (the player to act); rows
of `players` are in relative seat order (row k = seat (me + k) % n), absent
seats are zero. Nothing hidden from that seat is encoded: opponents' hands,
their chosen card until it is revealed, face-down treasure values it does
not know, deck orders. Card counting uses the public discard piles.

Keys:

* `glob` (N_GLOB): context
    0-12 phase one-hot | 13 round/MAX_ROUNDS | 14 I am captain |
    15 my resolution index/(n-1) | 16 ships revealed/n | 17-18 morning /
    evening die/6 (0 until placed) | 19-20 high / low rolled die/6 |
    21-22 I am resolving my morning / evening action | 23 last round |
    24 leader's remaining/L | 25 my remaining/L | 26 treasure pile/9 |
    27 P(next lair draw is a power) | 28 E[unknown face-down value]/7 |
    29-37 lair still has its treasure (track.lairs order) | 38-39 paying
    gold / food | 40 cost left/7 | 41 falling back (retreat choice) |
    42-44 loading gold / food / powder | 45 load amount/6 | 46 move die/6 |
    47 moving backward | 48 I attack | 49 I defend | 50 opponent offset/(n-1)
    | 51-52 my / opponent's powder spent/10 | 53 opponent strength/20 |
    54 opponent strength known | 55 opponent star | 56 my die rolled |
    57 my face/10 | 58 my star | 59-60 my / opponent's Lady Beth |
    61 Sabre used | 62 opponent's powder/20 | 63 n_players/6
* `players` (MAX_PLAYERS, P_COLS): see `_player_row`
* `hand` (25): my cards by code / 2
* `track` (18, 11): egocentric window, rows = spaces exactly +1..+12 then
    -1..-6 steps from my ship (every fork branch): valid, sea / port /
    lair-with-treasure fractions, Port Royal, min food cost/4, min gold
    cost/7, affordable now, most other ships/3, best printed score/15, fork
* `mask` (N_ACTIONS): legal actions (all zero when I am not to act)
* `action_feats` (N_ACTIONS, K_FEATS): consequences of each legal action
    (rows of illegal actions are zero). Columns 0-11 are shared: Δ my
    public score/10, progress/12, Δ gold/6, Δ food/6, Δ powder/6, Δ empty
    holds/3, battles, lair draws, shortages, spaces lost falling back/6,
    reaches Port Royal, Δ opponent's public score/10. Columns 12-23 depend
    on the family (see `_fill_*`).
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from . import rules, sim
from .constants import (A_CAPTAIN, A_CARD, A_DEST, A_GIVE_CURSED, A_LOAD_HOLD, A_NOTHING,
                        A_PAY_HOLD, A_POWDER, A_SABRE, A_STEAL_HIDDEN, A_STEAL_HOLD,
                        A_STEAL_POWER, A_TARGET, Kind, MAX_PLAYERS, MAX_ROUNDS, N_ACTIONS,
                        N_CODES, N_PHASES, N_SYMS, Phase, Power, Res, STAR, Sym, card_syms)
from .rules import SIXTH_SLOT, hold_at

N_GLOB = 64
P_COLS = 46 + N_CODES
TRACK_AHEAD = 12
TRACK_BEHIND = 6
N_TRACK = TRACK_AHEAD + TRACK_BEHIND
TRACK_COLS = 11
K_FEATS = 24
N_LAIR_BITS = 9
CLIP = 5.0

_TRACK_OFFSETS = [(1, k) for k in range(1, TRACK_AHEAD + 1)] + \
                 [(-1, k) for k in range(1, TRACK_BEHIND + 1)]


def observation_space() -> spaces.Dict:
    box = lambda *shape: spaces.Box(-CLIP, CLIP, shape, dtype=np.float32)
    return spaces.Dict({
        'glob': box(N_GLOB),
        'players': box(MAX_PLAYERS, P_COLS),
        'hand': box(N_CODES),
        'track': box(N_TRACK, TRACK_COLS),
        'mask': spaces.Box(0.0, 1.0, (N_ACTIONS,), dtype=np.float32),
        'action_feats': box(N_ACTIONS, K_FEATS),
    })


def _deck_counts(env) -> np.ndarray:
    counts = np.zeros(N_CODES, np.float32)
    for c in env.deck_cards:
        counts[c] += 1
    return counts


def _mu_draw(env) -> float:
    """Expected points of a lair draw (powers count 0)."""
    return float(np.mean([v for _, v in env.treasures]))


def _unknown_mean(env, seat: int) -> float:
    """E[value] of a face-down card whose value `seat` does not know."""
    bit = 1 << seat
    known_ids = {t for t in range(len(env.treasures)) if env.known[t] & bit}
    vals = [v for t, (p, v) in enumerate(env.treasures) if p is None and t not in known_ids]
    return float(np.mean(vals)) if vals else 0.0


def build_obs(env, seat: int) -> dict:
    n = env.n_players
    tr = env.track
    me = env.ships[seat]
    d = env.pending
    acting = (not env.done and d is not None and d.seat == seat
              and d.kind not in (Phase.TURN_PAUSE, Phase.DONE))

    glob = np.zeros(N_GLOB, np.float32)
    glob[int(env.phase)] = 1.0
    glob[13] = env.round / MAX_ROUNDS
    glob[14] = float(env.captain == seat)
    order = env.order
    glob[15] = order.index(seat) / (n - 1)
    glob[16] = sum(s.revealed for s in env.ships) / n
    if env.dice is not None:
        glob[17], glob[18] = env.dice[0] / 6, env.dice[1] / 6
    glob[19], glob[20] = max(env.raw_dice) / 6, min(env.raw_dice) / 6
    if env.cur_action is not None and env.cur_action[0] == seat and env.resolving == seat:
        glob[21 + env.cur_action[1]] = 1.0
    glob[23] = float(env.game_ending)
    glob[24] = min(tr.remaining(s.node, s.lap) for s in env.ships) / tr.L
    glob[25] = tr.remaining(me.node, me.lap) / tr.L
    glob[26] = len(env.pile) / 9
    face_up = sum(bin(s.powers).count('1') for s in env.ships)
    n_powers = sum(1 for p, _ in env.treasures if p is not None)
    unseen = len(env.pile) + len(env.removed)
    glob[27] = (n_powers - face_up) / unseen if unseen else 0.0
    glob[28] = _unknown_mean(env, seat) / 7
    for i, node in enumerate(tr.lairs[:N_LAIR_BITS]):
        glob[29 + i] = float(env.lair_token.get(node, False))
    if acting:
        if env.payment is not None and env.payment.seat == seat:
            glob[38] = float(env.payment.res == Res.GOLD)
            glob[39] = float(env.payment.res == Res.FOOD)
            glob[40] = env.payment.remaining / 7
        glob[41] = float(d.kind == Phase.RETREAT_DEST)
        if d.kind == Phase.LOAD_HOLD:
            glob[42 + int(d.data[0]) - 1] = 1.0
            glob[45] = d.data[1] / 6
        if d.kind == Phase.MOVE_DEST and env.cur_action is not None:
            glob[46] = env.cur_action[3] / 6
            glob[47] = float(env.cur_action[2] == Sym.BACK)
    b = env.battle
    if b is not None and seat in (b.attacker, b.defender) and b.defender >= 0:
        att = seat == b.attacker
        opp = b.defender if att else b.attacker
        glob[48], glob[49] = float(att), float(not att)
        glob[50] = ((opp - seat) % n) / (n - 1)
        glob[51] = (b.att_k if att else b.def_k) / 10
        glob[52] = (b.def_k if att else b.att_k) / 10
        opp_rolled = b.def_rolled if att else b.att_rolled
        opp_face = b.def_face if att else b.att_face
        if opp_rolled:
            if opp_face == STAR:
                glob[55] = 1.0
            else:
                glob[53] = env.strength(b, 1 if att else 0) / 20
                glob[54] = 1.0
        my_rolled = b.att_rolled if att else b.def_rolled
        my_face = b.att_face if att else b.def_face
        if my_rolled:
            glob[56] = 1.0
            if my_face == STAR:
                glob[58] = 1.0
            else:
                glob[57] = my_face / 10
        glob[59] = float(me.has(Power.BETH))
        glob[60] = float(env.ships[opp].has(Power.BETH))
        glob[61] = float(b.sabre_used)
        glob[62] = rules.total(env.ships[opp], Res.POWDER) / 20
    glob[63] = n / MAX_PLAYERS

    players = np.zeros((MAX_PLAYERS, P_COLS), np.float32)
    deck = _deck_counts(env)
    best_other = max((env.public_score(s.seat) for s in env.ships if s.seat != seat), default=0.0)
    for k in range(n):
        other = (seat + k) % n
        players[k] = _player_row(env, seat, other, deck, best_other)

    hand = np.zeros(N_CODES, np.float32)
    for c in me.hand:
        hand[c] += 0.5

    track = np.zeros((N_TRACK, TRACK_COLS), np.float32)
    others = {}
    for s in env.ships:
        if s.seat != seat and not s.finished and s.node != tr.pr:
            others[s.node] = others.get(s.node, 0) + 1
    for row, (dirn, k) in enumerate(_TRACK_OFFSETS):
        ring = tr.destinations((me.node, me.lap), dirn, k)
        if not ring:
            continue
        nodes = [p[0] for p in ring]
        m = len(nodes)
        track[row, 0] = 1.0
        track[row, 1] = sum(tr.kind[x] == Kind.SEA for x in nodes) / m
        track[row, 2] = sum(tr.kind[x] == Kind.PORT for x in nodes) / m
        track[row, 3] = sum(tr.kind[x] == Kind.LAIR and env.lair_token.get(x, False) for x in nodes) / m
        track[row, 4] = float(tr.pr in nodes)
        foods = [tr.cost[x] for x in nodes if tr.kind[x] == Kind.SEA]
        golds = [tr.cost[x] for x in nodes if tr.kind[x] == Kind.PORT]
        track[row, 5] = min(foods) / 4 if foods else 0.0
        track[row, 6] = min(golds) / 7 if golds else 0.0
        track[row, 7] = float(any(rules.can_pay(me, tr.kind[x], tr.cost[x]) for x in nodes))
        track[row, 8] = max(others.get(x, 0) for x in nodes) / 3
        track[row, 9] = max(tr.pos_score(*p) for p in ring) / 15
        track[row, 10] = float(m > 1)

    mask = np.zeros(N_ACTIONS, np.float32)
    feats = np.zeros((N_ACTIONS, K_FEATS), np.float32)
    if acting:
        legal = env.legal_actions()
        mask[legal] = 1.0
        action_feats(env, seat, legal, feats)

    obs = {'glob': glob, 'players': players, 'hand': hand, 'track': track,
           'mask': mask, 'action_feats': feats}
    for key in ('glob', 'players', 'track', 'action_feats'):
        np.clip(obs[key], -CLIP, CLIP, out=obs[key])
    return obs


def _player_row(env, pov: int, seat: int, deck: np.ndarray, best_other: float) -> np.ndarray:
    n = env.n_players
    tr = env.track
    s = env.ships[seat]
    row = np.zeros(P_COLS, np.float32)
    row[0] = 1.0
    row[1] = float(env.captain == seat)
    row[2] = env.order.index(seat) / (n - 1)
    row[3] = float(s.revealed)
    row[4] = float(s.chosen >= 0)
    if s.chosen >= 0 and (s.revealed or seat == pov):
        m, e = card_syms(s.chosen)
        row[5 + m] = 1.0
        row[10 + e] = 1.0
    rem = tr.remaining(s.node, s.lap)
    my_rem = tr.remaining(env.ships[pov].node, env.ships[pov].lap)
    row[15] = rem / tr.L
    row[16] = tr.pos_score(s.node, s.lap) / 15
    row[17] = float(tr.in_floor_zone(s.node, s.lap))
    row[18] = (my_rem - rem) / tr.L
    row[19] = float(s.finished)
    row[20] = float(s.lap < 0)
    row[21] = rules.total(s, Res.GOLD) / 20
    row[22] = rules.total(s, Res.FOOD) / 20
    row[23] = rules.total(s, Res.POWDER) / 20
    row[24] = rules.n_empty(s) / 6
    row[25] = float(s.sixth is not None)
    row[26] = rules.largest(s, Res.GOLD) / 6
    row[27] = rules.largest(s, Res.FOOD) / 6
    row[28] = rules.largest(s, Res.POWDER) / 6
    for p in Power:
        row[29 + int(p)] = float(s.has(p))
    row[33] = len(s.hidden) / 4
    row[34] = sum(env.public_cursed[t] for t in s.hidden) / 3
    if seat == pov:
        row[35] = sum(env.treasure_value(t) for t in s.hidden) / 10
        row[36] = sum(env.treasure_value(t) < 0 for t in s.hidden) / 3
    ps = env.public_score(seat)
    row[37] = ps / 30
    row[38] = (ps - best_other) / 30 if seat == pov else (ps - env.public_score(pov)) / 30
    # lookahead on the shortest route: food, next port, lairs
    food = 0
    port_d, port_c = 12, 0
    lairs = 0
    pos = (s.node, s.lap)
    for k in range(1, TRACK_AHEAD + 1):
        ring = tr.destinations(pos, 1, k)
        nodes = [p[0] for p in ring]
        if k <= 6:
            foods = [tr.cost[x] for x in nodes if tr.kind[x] == Kind.SEA]
            food += min(foods) if foods and len(foods) == len(nodes) else 0
        if port_d == 12 and any(tr.kind[x] == Kind.PORT for x in nodes):
            port_d = k
            port_c = min(tr.cost[x] for x in nodes if tr.kind[x] == Kind.PORT)
        lairs += any(tr.kind[x] == Kind.LAIR and env.lair_token.get(x, False) for x in nodes)
    row[39] = food / 12
    row[40] = port_d / 12
    row[41] = port_c / 7
    row[42] = lairs / 3
    row[43] = len(s.hand) / 4
    row[44] = len(s.deck) / 11
    row[45] = len(s.discard) / 11
    remaining = deck.copy()
    for c in s.discard:
        remaining[c] -= 1
    row[46:46 + N_CODES] = np.divide(remaining, deck, out=np.zeros_like(deck), where=deck > 0)
    return row


# --------------------------------------------------------------------------- #
# per-action consequence features
# --------------------------------------------------------------------------- #

def action_feats(env, seat: int, legal: list[int], out: np.ndarray) -> None:
    d = env.pending
    k = d.kind
    if k in (Phase.CAPTAIN, Phase.CARD):
        _fill_cards(env, seat, legal, out)
    elif k in (Phase.MOVE_DEST, Phase.RETREAT_DEST):
        _fill_dest(env, seat, legal, out)
    elif k == Phase.LOAD_HOLD:
        _fill_load(env, seat, legal, out)
    elif k == Phase.PAY_HOLD:
        _fill_pay(env, seat, legal, out)
    elif k in (Phase.ATTACK_POWDER, Phase.DEFENSE_POWDER):
        _fill_powder(env, seat, legal, out)
    elif k == Phase.TARGET:
        _fill_target(env, seat, legal, out)
    elif k == Phase.SABRE:
        _fill_sabre(env, seat, legal, out)
    elif k == Phase.REWARD:
        _fill_reward(env, seat, legal, out)


def _fill_cards(env, seat, legal, out):
    """12/13 morning/evening progress/6, 14/15 battles, 16/17 lair draws,
    18/19 shortages, 20 tokens dumped/6, 21 copies in hand/2, 22 copies not
    yet discarded/2, 23 ends in the -5 zone."""
    me = env.ships[seat]
    world = sim.world_of(env, seat)
    mu = _mu_draw(env)
    hi, lo = max(env.raw_dice), min(env.raw_dice)
    for a in legal:
        if a < A_CARD:
            order, code = divmod(a - A_CAPTAIN, N_CODES)
            dice = (hi, lo) if order == 0 else (lo, hi)
        else:
            code = a - A_CARD
            dice = env.dice
        o = sim.play(env, seat, code, dice, world)
        r = out[a]
        r[0] = o.d_score(mu) / 10
        r[1] = o.progress / 12
        r[2], r[3], r[4] = o.d_gold / 6, o.d_food / 6, o.d_powder / 6
        r[5] = o.d_empty / 3
        r[6], r[7], r[8] = o.combats, o.lairs, o.shortages
        r[9] = o.retreat / 6
        r[10] = float(o.finished)
        h0, h1 = o.halves
        r[12], r[13] = h0.progress / 6, h1.progress / 6
        r[14], r[15] = float(h0.combat), float(h1.combat)
        r[16], r[17] = float(h0.lair), float(h1.lair)
        r[18], r[19] = float(h0.shortage), float(h1.shortage)
        r[20] = o.dumped / 6
        r[21] = me.hand.count(code) / 2
        r[22] = (env.deck_cards.count(code) - me.discard.count(code)) / 2
        r[23] = float(o.floor_zone)


def _fill_dest(env, seat, legal, out):
    """12-15 kind one-hot (Port Royal, port, sea, lair), 16 cost/7,
    17 affordable, 18 printed score/15, 19 other ships/3, 20 their most
    powder/20, 21 remaining/L, 22 lair treasure, 23 backward."""
    tr = env.track
    me = env.ships[seat]
    d = env.pending
    world = sim.world_of(env, seat)
    mu = _mu_draw(env)
    rem0 = tr.remaining(me.node, me.lap)
    score0 = tr.pos_score(me.node, me.lap)
    backward = d.kind == Phase.RETREAT_DEST or (env.cur_action is not None
                                                 and env.cur_action[2] == Sym.BACK)
    for a in legal:
        pos = d.data[a - A_DEST]
        node = pos[0]
        half, ship = sim.land_outcome(env, seat, pos)
        r = out[a]
        r[0] = (tr.pos_score(ship.node, ship.lap) - score0
                + rules.total(ship, Res.GOLD) - rules.total(me, Res.GOLD) + half.lair * mu) / 10
        r[1] = (rem0 - tr.remaining(ship.node, ship.lap)) / 12
        r[2] = (rules.total(ship, Res.GOLD) - rules.total(me, Res.GOLD)) / 6
        r[3] = (rules.total(ship, Res.FOOD) - rules.total(me, Res.FOOD)) / 6
        r[5] = (rules.n_empty(ship) - rules.n_empty(me)) / 3
        r[6], r[7], r[8] = float(half.combat), float(half.lair), float(half.shortage)
        r[9] = half.retreat / 6
        r[10] = float(half.finished)
        r[12 + tr.kind[node]] = 1.0
        r[16] = tr.cost[node] / 7
        r[17] = float(rules.can_pay(me, tr.kind[node], tr.cost[node]))
        r[18] = tr.pos_score(*pos) / 15
        r[19] = world.ships_at.get(node, 0) / 3
        r[20] = world.foe_powder.get(node, 0) / 20
        r[21] = tr.remaining(*pos) / tr.L
        r[22] = float(node in world.tokens)
        r[23] = float(backward)


def _fill_load(env, seat, legal, out):
    """12-14 dumped type one-hot, 15 dumped amount/6, 16 is the 6th hold,
    17 loaded amount/6."""
    me = env.ships[seat]
    res, amount = env.pending.data[0], env.pending.data[1]
    for a in legal:
        slot = a - A_LOAD_HOLD
        h = hold_at(me, slot)
        r = out[a]
        r[0] = (-h[1] if h[0] == Res.GOLD else 0) / 10 + (amount if res == Res.GOLD else 0) / 10
        r[2 + int(h[0]) - 1] -= h[1] / 6
        r[2 + int(res) - 1] += amount / 6
        r[12 + int(h[0]) - 1] = 1.0
        r[15] = h[1] / 6
        r[16] = float(slot == SIXTH_SLOT)
        r[17] = amount / 6


def _fill_pay(env, seat, legal, out):
    """12 hold count/6, 13 paid from it/6, 14 empties it, 15 cost left/7,
    16 is the 6th hold."""
    me = env.ships[seat]
    res = env.pending.data[0]
    left = env.payment.remaining
    for a in legal:
        slot = a - A_PAY_HOLD
        h = hold_at(me, slot)
        paid = min(left, h[1])
        r = out[a]
        r[2 + int(res) - 1] = -paid / 6
        r[5] = float(paid == h[1]) / 3
        r[12] = h[1] / 6
        r[13] = paid / 6
        r[14] = float(paid == h[1])
        r[15] = (left - paid) / 7
        r[16] = float(slot == SIXTH_SLOT)


def _fill_powder(env, seat, legal, out):
    """12 k/10, 13 powder left/20, 14 (k + Lady Beth)/15; attacker:
    15 P(win | defender adds 0), 16 P(win | defender adds all), 17 P(tie | 0);
    defender: 15 P(win), 16 P(tie), 17 P(lose); 18 only a star can win."""
    b = env.battle
    faces = env.faces
    me = env.ships[seat]
    own = rules.total(me, Res.POWDER)
    att = seat == b.attacker
    opp = env.ships[b.defender if att else b.attacker]
    my_bonus, opp_bonus = rules.beth(me), rules.beth(opp)
    opp_all = rules.total(opp, Res.POWDER)
    s_att = env.strength(b, 0) if not att else 0
    for a in legal:
        kk = a - A_POWDER
        r = out[a]
        r[4] = -kk / 6
        r[12] = kk / 10
        r[13] = (own - kk) / 20
        r[14] = (kk + my_bonus) / 15
        if att:
            w0, t0 = sim.p_battle(kk, my_bonus, 0, opp_bonus, faces)
            w1, _ = sim.p_battle(kk, my_bonus, opp_all, opp_bonus, faces)
            r[15], r[16], r[17] = w0, w1, t0
            star_only = kk + min(rules.numeric_faces(faces)) + my_bonus <= 0
        else:
            w, t = sim.p_defend(s_att, kk, my_bonus, faces)
            r[15], r[16], r[17] = w, t, 1.0 - w - t
            star_only = kk + max(rules.numeric_faces(faces)) + my_bonus < s_att
        r[18] = float(star_only)


def _fill_target(env, seat, legal, out):
    """12 their powder/20, 13 Lady Beth, 14 Sabre, 15 their biggest gold
    hold/6, 16 their biggest hold/6, 17 face-down cards/4, 18 powers/4,
    19 their public lead over me/30, 20 P(win) all-in vs 0, 21 P(win)
    all-in vs all-in, 22 they have not resolved yet this round."""
    n = env.n_players
    me = env.ships[seat]
    own = rules.total(me, Res.POWDER)
    for a in legal:
        o = env.ships[(seat + a - A_TARGET + 1) % n]
        r = out[a]
        opp_powder = rules.total(o, Res.POWDER)
        r[12] = opp_powder / 20
        r[13] = float(o.has(Power.BETH))
        r[14] = float(o.has(Power.SABRE))
        r[15] = rules.largest(o, Res.GOLD) / 6
        r[16] = max((hold_at(o, s)[1] for s in rules.slots(o)), default=0) / 6
        r[17] = len(o.hidden) / 4
        r[18] = bin(o.powers).count('1') / 4
        r[19] = (env.public_score(o.seat) - env.public_score(seat)) / 30
        r[20], _ = sim.p_battle(own, rules.beth(me), 0, rules.beth(o), env.faces)
        r[21], _ = sim.p_battle(own, rules.beth(me), opp_powder, rules.beth(o), env.faces)
        r[22] = float(not o.revealed)


def _sabre_p(env, b, holder, att_face, def_face, def_rolled):
    """(P(holder wins), P(tie)) with the given faces (defender's may be unrolled)."""
    att, dfn = env.ships[b.attacker], env.ships[b.defender]
    ab, db = rules.beth(att), rules.beth(dfn)
    if att_face == STAR:
        p_att, tie = 1.0, 0.0
    elif not def_rolled:
        p_def, tie = sim.p_defend(att_face + b.att_k + ab, 0, db, env.faces)
        p_att = 1.0 - p_def - tie
    else:
        r = rules.battle_winner(att_face, b.att_k, ab, def_face, b.def_k, db)
        p_att, tie = float(r > 0), float(r == 0)
    p = p_att if holder == b.attacker else 1.0 - p_att - tie
    return p, tie


def _fill_sabre(env, seat, legal, out):
    """12 P(win), 13 P(tie) after keeping (row keep) / rerolling (row reroll)."""
    b = env.battle
    which = env.pending.data[0]
    faces = env.faces
    keep = _sabre_p(env, b, seat, b.att_face, b.def_face, b.def_rolled)
    rolls = []
    for f in faces:
        if which == 0:
            rolls.append(_sabre_p(env, b, seat, f, b.def_face, b.def_rolled))
        else:
            rolls.append(_sabre_p(env, b, seat, b.att_face, f, True))
    reroll = (float(np.mean([x[0] for x in rolls])), float(np.mean([x[1] for x in rolls])))
    for a in legal:
        p, t = keep if a == A_SABRE else reroll
        out[a, 12], out[a, 13] = p, t


def _fill_reward(env, seat, legal, out):
    """STEAL_HOLD: 12-14 type one-hot, 15 amount/6, 16 needs a dump,
    17 dumped/6, 18 loser's 6th hold; STEAL_POWER: 12 1; STEAL_HIDDEN:
    12 E[value]/7, 13 loser's face-down cards/4; GIVE_CURSED: 12 |value|/4."""
    b = env.battle
    me = env.ships[seat]
    loser = env.ships[env.loser(b)]
    mu = _unknown_mean(env, seat)
    for a in legal:
        r = out[a]
        if A_STEAL_HOLD <= a < A_STEAL_POWER:
            slot = a - A_STEAL_HOLD
            h = hold_at(loser, slot)
            res, cnt = int(h[0]), int(h[1])
            empty = rules.empty_slot(me)
            dump = 0
            if empty is None:
                classes = rules.dump_classes(me, res)
                dump = min(hold_at(me, s)[1] for s in classes) if classes else 0
            gold_gain = cnt if res == Res.GOLD else 0
            r[0] = gold_gain / 10
            r[2 + res - 1] = cnt / 6
            r[11] = -gold_gain / 10
            r[12 + res - 1] = 1.0
            r[15] = cnt / 6
            r[16] = float(empty is None)
            r[17] = dump / 6
            r[18] = float(slot == SIXTH_SLOT)
        elif A_STEAL_POWER <= a < A_STEAL_HIDDEN:
            r[12] = 1.0
        elif a == A_STEAL_HIDDEN:
            r[0] = mu / 10
            r[11] = -mu / 10
            r[12] = mu / 7
            r[13] = len(loser.hidden) / 4
        elif a == A_GIVE_CURSED:
            v = min(env.treasure_value(t) for t in me.hidden if env.treasure_value(t) < 0)
            r[0] = -v / 10
            r[11] = v / 10
            r[12] = -v / 4
