"""Shared helpers of the Jamaica tests (not a test module)."""

from __future__ import annotations

import random

from ..envs import rules
from ..envs.constants import (A_CARD, A_CAPTAIN, N_CODES, Phase, Power, Res, card_code)
from ..envs.jamaica import JamaicaEnv, Task, TK
from ..envs.track import Track
from ..envs.constants import Kind


def make_env(n: int = 4, seed: int = 0, pause: bool = False, track: Track | None = None) -> JamaicaEnv:
    env = JamaicaEnv(n_players=n, pause_between_turns=pause, track=track)
    env.reset(seed=seed)
    return env


def set_ship(env, seat: int, node: int | None = None, lap: int = 0, holds=None, sixth=None,
             powers: int | None = None, hidden=None, hand=None):
    """Overwrite parts of a ship (holds as a list of (res, count), up to 5)."""
    s = env.ships[seat]
    if node is not None:
        s.node, s.lap = node, lap
    if holds is not None:
        s.holds = [[int(r), int(c)] for r, c in holds]
        while len(s.holds) < 5:
            s.holds.append([Res.EMPTY, 0])
        rules.sort_holds(s)
    if sixth is not None:
        s.sixth = list(sixth) if sixth != 'none' else None
        if s.sixth is not None:
            s.powers |= 1 << int(Power.SIXTH)
    if powers is not None:
        s.powers = powers
    if hidden is not None:
        s.hidden = list(hidden)
        for t in s.hidden:
            env.known[t] |= 1 << seat
    if hand is not None:
        s.hand = list(hand)
    env._version += 1
    return s


def run_tasks(env, *tasks: Task, dice=None, card=None, seat=None):
    """Run `tasks` (first one first) from a clean stack; stops at a decision
    or at the sentinel pause pushed below them."""
    env.pause_between_turns = True
    env.stack = [Task(TK.PAUSE)] + list(reversed(tasks))
    env.pending = None
    env.battle = None
    env.payment = None
    if dice is not None:
        env.dice = dice
    if card is not None and seat is not None:
        env.ships[seat].chosen = card
    env._advance()


def act(env, seat: int, slot: int, morning, evening, dice=(1, 1)):
    """Resolve one action (slot 0 morning / 1 evening) of `seat` with a card."""
    run_tasks(env, Task(TK.ACTION, seat, slot), dice=dice, card=card_code(morning, evening), seat=seat)


def land(env, seat: int, node: int, lap: int = 0, via: int = 0):
    run_tasks(env, Task(TK.LAND, seat, node, lap, via))


def at_pause(env) -> bool:
    return env.pending is not None and env.pending.kind == Phase.TURN_PAUSE


def random_play(env, rng: random.Random, max_steps: int = 20000, check=None):
    """Play random legal actions to the end; `check(env)` after every step."""
    steps = 0
    totals = [0.0] * env.n_players
    while not env.done and steps < max_steps:
        a = -1 if env.current_player == -1 else rng.choice(env.legal_actions())
        _, r, _, _, _ = env.step(a)
        totals = [x + y for x, y in zip(totals, r)]
        steps += 1
        if check is not None:
            check(env)
    return totals, steps


def play_until(env, rng: random.Random, predicate, max_steps: int = 20000) -> bool:
    """Random play until `predicate(env)` holds (True) or the game ends."""
    for _ in range(max_steps):
        if env.done:
            return False
        if predicate(env):
            return True
        a = -1 if env.current_player == -1 else rng.choice(env.legal_actions())
        env.step(a)
    return False


def mini_track() -> Track:
    """A 12-space loop (9 steps round) with one fork (outer 3 spaces / inner 1).

    0 PR -> 1 sea1 -> 2 port2 -> 3 split(sea1) -> outer 4 lair, 5 sea2, 6 sea1 /
    inner 7 sea3 -> 8 merge(sea1) -> 9 port3 -> 10 lair -> 11 sea2 -> 0
    Scores: 9 -> 2, 10 -> 3, 11 -> 4, PR -> 10.
    """
    S, P, L, R = Kind.SEA, Kind.PORT, Kind.LAIR, Kind.PORT_ROYAL
    spaces = [
        (0, R, 0, 10, 0, 0), (1, S, 1, None, 0, 0), (2, P, 2, None, 0, 0),
        (3, S, 1, None, 0, 0), (4, L, 0, None, 0, 0), (5, S, 2, None, 0, 0),
        (6, S, 1, None, 0, 0), (7, S, 3, None, 0, 0), (8, S, 1, None, 0, 0),
        (9, P, 3, 2, 0, 0), (10, L, 0, 3, 0, 0), (11, S, 2, 4, 0, 0),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (3, 7), (7, 8),
             (8, 9), (9, 10), (10, 11), (11, 0)]
    return Track(spaces, edges, 0)


def captain_action(env, code: int, order: int = 0) -> int:
    return A_CAPTAIN + order * N_CODES + code


def card_action(code: int) -> int:
    return A_CARD + code
