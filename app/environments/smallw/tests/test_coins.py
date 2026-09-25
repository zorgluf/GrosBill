"""Tests for the coin token model (`PlayerState` coins, task T3b).

Plain `assert`-based functions, usable either with pytest
(`pytest environments/smallw/tests/test_coins.py`) or through the bundled
runners:

    cd app
    python -m environments.smallw.tests.run_all        # every test module
    python -m environments.smallw.tests.test_coins     # this module only

What the rules ask (plan T3b): coin *values* are hidden but the *number of
tokens* is public, so the denominations 1 / 3 / 5 / 10 are modelled; gains and
payments are made in "1" coins and the stack is immediately re-made into the
minimal number of tokens.
"""

from __future__ import annotations

import copy
import traceback

import numpy as np

from ..envs.classes import (
    COIN_DENOMINATIONS,
    COIN_SUPPLY,
    START_COINS,
    ComboColumn,
    PlayerState,
    make_change,
)
from ..envs.smallw import PLAYER_COLS, SmallWorldEnv


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _rng(seed: int = 4242) -> np.random.Generator:
    """A fresh seeded generator (tests must never depend on global state)."""
    return np.random.default_rng(seed)


def _with_value(value: int) -> PlayerState:
    """A player whose stack is the minimal change of `value`."""
    player = PlayerState(0, coins=0)
    player.gain(value)
    return player


def _minimal_count(value: int) -> int:
    """Smallest number of 1/3/5/10 tokens worth `value` (dynamic program)."""
    best = [0] + [10 ** 9] * value
    for v in range(1, value + 1):
        best[v] = min(best[v - den] + 1 for den in COIN_DENOMINATIONS if den <= v)
    return best[value]


# --------------------------------------------------------------------------- #
# Change making
# --------------------------------------------------------------------------- #

def test_change_making_examples():
    """The examples of the plan, plus the degenerate values."""
    assert make_change(13) == {10: 1, 3: 1}
    assert make_change(8) == {5: 1, 3: 1}
    assert make_change(7) == {5: 1, 1: 2}
    assert make_change(4) == {3: 1, 1: 1}
    assert make_change(0) == {}
    assert make_change(1) == {1: 1}
    assert make_change(2) == {1: 2}
    assert make_change(5) == {5: 1}
    assert make_change(10) == {10: 1}
    assert make_change(23) == {10: 2, 3: 1}


def test_change_is_well_formed_and_minimal():
    """Only valid denominations, positive counts, right total, fewest tokens."""
    for value in range(0, 301):
        tokens = make_change(value)
        assert set(tokens) <= set(COIN_DENOMINATIONS), value
        assert all(count > 0 for count in tokens.values()), value
        assert sum(den * count for den, count in tokens.items()) == value
        assert sum(tokens.values()) == _minimal_count(value), \
            f'{value}: {tokens} is not minimal'


def test_make_change_rejects_a_negative_value():
    try:
        make_change(-1)
    except ValueError:
        pass
    else:
        raise AssertionError('make_change(-1) must raise ValueError')


def test_coin_supply_table():
    """The printed supply (documentation only: the pool is unlimited)."""
    assert COIN_DENOMINATIONS == (10, 5, 3, 1)
    assert set(COIN_SUPPLY) == set(COIN_DENOMINATIONS)
    assert sum(COIN_SUPPLY.values()) == 109


# --------------------------------------------------------------------------- #
# PlayerState: gain / pay / coins / coin_count
# --------------------------------------------------------------------------- #

def test_player_starts_with_five_coins_of_value_one():
    """Setup deals five "1" coins — not one "5" (the token count is public)."""
    player = PlayerState(2)
    assert player.coins == START_COINS == 5
    assert player.coin_tokens == {1: 5}
    assert player.coin_count == 5
    assert PlayerState(0, coins=0).coin_tokens == {}
    assert PlayerState(0, coins=0).coin_count == 0
    try:
        PlayerState(0, coins=-1)
    except ValueError:
        pass
    else:
        raise AssertionError('a negative start must raise ValueError')


def test_gain_takes_ones_then_makes_change():
    player = PlayerState(0)
    player.gain(0)
    assert player.coin_tokens == {1: 5}, 'gaining nothing changes nothing'
    player.gain(3)                                  # 5 + 3 = 8
    assert player.coins == 8
    assert player.coin_tokens == {5: 1, 3: 1}
    assert player.coin_count == 2
    player.gain(5)                                  # 13
    assert player.coin_tokens == {10: 1, 3: 1}
    assert player.coin_count == 2


def test_pay_breaks_a_bigger_coin():
    """Paying 2 out of a single "5" leaves a single "3"."""
    player = _with_value(5)
    assert player.coin_tokens == {5: 1}
    player.pay(2)
    assert player.coins == 3
    assert player.coin_tokens == {3: 1}
    assert player.coin_count == 1

    player = _with_value(10)
    player.pay(1)                                   # 10 -> 9 = 5 + 3 + 1
    assert player.coin_tokens == {5: 1, 3: 1, 1: 1}
    assert player.coin_count == 3

    player = _with_value(13)                        # {10: 1, 3: 1}
    player.pay(0)
    assert player.coin_tokens == {10: 1, 3: 1}, 'paying nothing changes nothing'
    player.pay(13)
    assert player.coins == 0 and player.coin_tokens == {} and player.coin_count == 0


def test_paying_more_than_owned_raises_and_keeps_the_stack():
    player = _with_value(5)
    for bad in (6, 100):
        try:
            player.pay(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'paying {bad} out of 5 must raise ValueError')
    assert player.coin_tokens == {5: 1}, 'a refused payment must change nothing'
    try:
        player.coins = -1
    except ValueError:
        pass
    else:
        raise AssertionError('a negative total must raise ValueError')
    for bad_call in (lambda: player.gain(-1), lambda: player.pay(-2)):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError('a negative amount must raise ValueError')
    assert player.coin_tokens == {5: 1}


def test_coins_setter_gains_or_pays_the_difference():
    player = PlayerState(0)
    player.coins = START_COINS
    assert player.coin_tokens == {1: 5}, 'setting the same total moves no coin'
    player.coins = 13
    assert player.coins == 13 and player.coin_tokens == {10: 1, 3: 1}
    player.coins += 1                               # 14 = 10 + 3 + 1
    assert player.coin_tokens == {10: 1, 3: 1, 1: 1}
    player.coins -= 10                              # 4 = 3 + 1
    assert player.coin_tokens == {3: 1, 1: 1}
    player.coins = 0
    assert player.coin_tokens == {} and player.coin_count == 0


def test_coin_count_matches_the_stack_after_random_gains_and_payments():
    """`coins` / `coin_count` / `coin_tokens` stay consistent and minimal."""
    rng = _rng()
    player = PlayerState(1)
    player.gain(1)          # leave the initial deal, which is five "1" coins
    for _ in range(2000):
        if rng.random() < 0.5:
            player.gain(int(rng.integers(0, 12)))
        else:
            player.pay(int(rng.integers(0, max(1, player.coins + 1))))
        tokens = player.coin_tokens
        assert set(tokens) <= set(COIN_DENOMINATIONS)
        assert all(count > 0 for count in tokens.values())
        assert player.coins == sum(den * n for den, n in tokens.items())
        assert player.coin_count == sum(tokens.values())
        assert tokens == make_change(player.coins), 'the stack must be minimal'
    assert player.coins >= 0


def test_deepcopy_independence():
    player = _with_value(13)
    clone = copy.deepcopy(player)
    assert clone.coin_tokens == {10: 1, 3: 1}
    assert clone.coin_tokens is not player.coin_tokens
    clone.gain(2)
    clone.pay(5)
    assert clone.coins == 10 and clone.coin_tokens == {10: 1}
    assert player.coins == 13 and player.coin_tokens == {10: 1, 3: 1}


# --------------------------------------------------------------------------- #
# The combo column pays and collects "1" coins
# --------------------------------------------------------------------------- #

def test_combo_pick_pays_and_collects_value_one_coins():
    """End to end: coins dropped on combos come back as "1"s, then re-changed."""
    col = ComboColumn(_rng())
    first = PlayerState(0)                          # {1: 5}
    col.pick(3, first)                              # pays 3, drops one on 0..2
    assert first.coins == 2 and first.coin_tokens == {1: 2}
    assert [c.coins for c in col.visible[:3]] == [1, 1, 1]

    second = PlayerState(1)                         # {1: 5}
    combo = col.visible[2]
    picked = col.pick(2, second)                    # pays 2, collects 1
    assert picked is combo and picked.coins == 0, 'the coins moved to the player'
    assert second.coins == 4, '5 - 2 + 1'
    assert second.coin_tokens == {3: 1, 1: 1}, 'the stack is re-made into change'
    assert second.coin_count == 2
    assert [c.coins for c in col.visible[:2]] == [2, 2]

    third = PlayerState(2)                          # {1: 5}
    picked = col.pick(0, third)                     # free, collects 2 "1" coins
    assert picked.coins == 0
    assert third.coins == 7 and third.coin_tokens == {5: 1, 1: 2}
    assert third.coin_count == 3

    # a payment that has to break a bigger coin: pay 4 out of {5: 1, 1: 2}
    fourth = _with_value(7)                         # {5: 1, 1: 2}
    on_it = col.visible[4].coins
    col.pick(4, fourth)
    assert fourth.coins == 7 - 4 + on_it
    assert fourth.coin_tokens == make_change(fourth.coins)

    # the column never hands out coins it does not hold
    assert all(c.coins >= 0 for c in col.visible)


# --------------------------------------------------------------------------- #
# Observation
# --------------------------------------------------------------------------- #

def test_observation_shows_the_value_for_self_and_the_count_for_opponents():
    env = SmallWorldEnv(3)
    obs, _info = env.reset(seed=7)
    assert PLAYER_COLS == 14
    assert obs['players'].shape == (5, 14)
    assert env.observation_space.contains(obs)

    seat0 = env.current_player
    values = {seat0: 13,                            # {10, 3}       -> 2 tokens
              (seat0 + 1) % 3: 24,                  # {10, 10, 3, 1} -> 4 tokens
              (seat0 + 2) % 3: 7}                   # {5, 1, 1}     -> 3 tokens
    for seat, value in values.items():
        env.players[seat].coins = value
    counts = {seat: env.players[seat].coin_count for seat in values}
    assert [counts[(seat0 + k) % 3] for k in range(3)] == [2, 4, 3]

    players = env.observation['players']
    assert players[0][1] == 13, 'own coins are observed by value'
    assert players[1][1] == 4 and players[2][1] == 3, \
        'an opponent is observed by his number of coin tokens'
    for k in range(3):
        seat = (seat0 + k) % 3
        assert players[k][0] == 1
        assert players[k][12] == counts[seat], 'column 12 = public token count'
    assert not players[3].any() and not players[4].any(), 'unused rows stay zero'
    assert env.observation_space.contains(env.observation)

    # debug flag: the opponents' *value* becomes visible in column 1
    env.OBS_OPPONENT_COINS = True
    players = env.observation['players']
    assert [int(players[k][1]) for k in range(3)] == [13, 24, 7]
    assert [int(players[k][12]) for k in range(3)] == [2, 4, 3]
    assert env.observation_space.contains(env.observation)


def test_random_games_keep_minimal_stacks():
    """A few random games: every stack stays minimal change of its value."""
    env = SmallWorldEnv(3)
    for seed in range(3):
        obs, _info = env.reset(seed=seed)
        rng = _rng(seed)
        steps = 0
        while not env.done:
            legal = np.flatnonzero(env.action_masks())
            action = int(rng.choice(legal)) if legal.size else -1   # pause
            obs = env.step(action)[0]
            steps += 1
            assert steps < 3000, f'seed {seed}: runaway game'
            players = obs['players']
            n = env.n_players
            seat0 = env.current_player if 0 <= env.current_player < n else 0
            for k in range(n):
                player = env.players[(seat0 + k) % n]
                assert player.coins >= 0
                # minimal change, unless the player never touched the five
                # "1" coins dealt at setup
                assert player.coin_tokens in (make_change(player.coins),
                                              {1: START_COINS}), \
                    f'seed {seed}: {player.name} holds {player.coin_tokens}'
                assert players[k][12] == player.coin_count
                expected = (player.coins if k == 0 else player.coin_count)
                assert players[k][1] == expected
            assert env.observation_space.contains(obs)
        assert env.players[env.winner_player].coins == max(
            p.coins for p in env.players)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def run_all() -> bool:
    """Run every `test_*` function of this module; print a summary.

    Returns True if everything passed.
    """
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception:                                   # noqa: BLE001
            failed.append(name)
            print(f'FAIL {name}')
            print(''.join('    ' + line for line in
                          traceback.format_exc().splitlines(keepends=True)))
        else:
            print(f'ok   {name}')
    print(f'\n{__name__}: {len(tests) - len(failed)}/{len(tests)} passed'
          + (f', FAILED: {", ".join(failed)}' if failed else ''))
    return not failed


if __name__ == '__main__':
    import sys
    sys.exit(0 if run_all() else 1)
