"""Shared helpers of the Toy Battle tests (not a test module)."""

from __future__ import annotations

import random

from ..envs.constants import A_TARGET, BLUE, N_TYPES, hq_slot, place_action
from ..envs.toybattle import ToyBattleEnv


def make_env(terrain: str = 'castle', seed: int = 0, **kwargs) -> ToyBattleEnv:
    env = ToyBattleEnv(terrain=terrain, **kwargs)
    env.reset(seed=seed)
    return env


def blank(terrain: str = 'castle', current: int = BLUE, **kwargs) -> ToyBattleEnv:
    """An empty board with empty racks, `current` to play: every tile is back in
    its owner's reserve (20, sorted) or box (4), from where `give`/`put` take them."""
    env = make_env(terrain, **kwargs)
    for p in range(2):
        env.racks[p] = [0] * N_TYPES
        env.removed[p] = [0, 1, 2, 3]
        env.reserve[p] = [t for t in range(N_TYPES) for _ in range(3)]
        for t in env.removed[p]:
            env.reserve[p].remove(t)
    env.current_player = env.first_player = current
    env.event_log.clear()
    return env


def _take(env, p: int, troop: int) -> None:
    for pile in (env.reserve[p], env.removed[p]):
        if troop in pile:
            pile.remove(troop)
            return
    raise AssertionError(f'no tile {troop} left for seat {p}')


def give(env, p: int, *troops: int) -> None:
    """Move `troops` from `p`'s reserve to their rack."""
    for t in troops:
        _take(env, p, t)
        env.racks[p][t] += 1


def put(env, name: str, p: int, *troops: int) -> None:
    """Stack `troops` of `p` on base `name` (the last one on top)."""
    for t in troops:
        _take(env, p, t)
        env.stacks[base(env, name)].append((p, t))


def base(env, name: str) -> int:
    return next(i for i, b in enumerate(env.terrain.bases) if b.name == name)


def slot(env, name: str) -> int:
    return env.terrain.slots[env.current_player][base(env, name)]


def can_place(env, troop: int, name: str) -> bool:
    return bool(env.action_masks()[place_action(troop, slot(env, name))])


def legal_bases(env, troop: int) -> set[str]:
    return {b.name for b in env.terrain.bases if can_place(env, troop, b.name)}


def place(env, troop: int, name: str):
    return env.step(place_action(troop, slot(env, name)))


def capture(env, troop: int, k: int = 0):
    return env.step(place_action(troop, hq_slot(k)))


def target(env, name: str):
    return env.step(A_TARGET + slot(env, name))


def targets(env) -> set[str]:
    return {env.terrain.bases[b].name for b in env.target_bases()}


def top(env, name: str):
    stack = env.stacks[base(env, name)]
    return stack[-1] if stack else None


def tiles(env, p: int) -> int:
    """Every tile of `p`, wherever it is (24 at any time)."""
    return (sum(env.racks[p]) + len(env.reserve[p]) + len(env.removed[p])
            + sum(env.on_board(p)) + sum(env.discard[p]))


def random_play(env, rng: random.Random, max_steps: int = 5000, check=None):
    """Play random legal actions to the end; `check(env)` after every step."""
    totals = [0.0, 0.0]
    steps = 0
    while not env.done and steps < max_steps:
        _, r, _, _, _ = env.step(rng.choice(env.legal_actions()))
        totals = [x + y for x, y in zip(totals, r)]
        steps += 1
        if check is not None:
            check(env)
    return totals, steps


def play_until(env, rng: random.Random, predicate, max_steps: int = 5000) -> bool:
    for _ in range(max_steps):
        if env.done:
            return False
        if predicate(env):
            return True
        env.step(rng.choice(env.legal_actions()))
    return False

