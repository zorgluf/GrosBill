"""Tests for the 2, 4 and 5 player games of Small World (issue #7).

Plain `assert`-based functions, usable either with pytest or through the
bundled runners:

    cd app
    python -m environments.smallw.tests.run_all        # every test module
    python -m environments.smallw.tests.test_players   # this module only

Every player count has its own board (`maps.py`), region outlines
(`shapes.py`), action layout and network, under its own environment name
(`smallw2` / `smallw` / `smallw4` / `smallw5`). Checked here:

* the board tables (published region counts, self-checks) and outlines;
* the action / observation sizes of each player count, the 3-player layout
  left untouched, and the registry entries;
* random games on every board with every engine invariant after every step;
* the Diplomat with four opponents and the first-conquest rule on the new
  boards;
* the web renderer helpers (board image, turn track, click → region) and the
  policy network on every board.
"""

from __future__ import annotations

import copy
import traceback

import numpy as np

from ..envs import maps
from ..envs import render_web as rw
from ..envs.classes import PowerId, RaceId
from ..envs.shapes import shape_for
from ..envs.smallw import (
    A_PASS,
    A_REGION,
    LAYOUT_3P,
    N_ACTIONS,
    ActionLayout,
    Phase,
    SmallWorld2Env,
    SmallWorld4Env,
    SmallWorld5Env,
    SmallWorldEnv,
    env_name_for,
    layout_for,
)
from .test_engine import _check_invariants, _env, _legal, _setup_turn

#: Region count / turns / action count of each player count.
EXPECTED = {2: (23, 10, 105), 3: (30, 10, 133), 4: (39, 9, 169), 5: (48, 8, 205)}

#: Random games played per player count by the fuzz test.
FUZZ_GAMES = 4

#: Upper bound on the steps of one random game.
STEP_BUDGET = 6000


# --------------------------------------------------------------------------- #
# Boards
# --------------------------------------------------------------------------- #

def test_every_board_passes_its_self_check():
    """Terrain / symbol / edge counts, symmetric connected adjacency, border
    regions, anchors and turn track of every board."""
    assert maps.self_check()
    assert maps.PLAYER_COUNTS == (2, 3, 4, 5)


def test_boards_have_the_published_sizes():
    """23 / 30 / 39 / 48 regions and 10 / 10 / 9 / 8 turns."""
    for n, (regions, turns, _) in EXPECTED.items():
        board = maps.map_def(n)
        assert board.n_players == n
        assert board.n_regions == regions == len(maps.map_for(n))
        assert board.turns == turns == maps.turns_for(n) == len(board.turn_track)
        assert maps.map_by_region_count(regions) is board
        assert board.board_image == f'board{n}p.svg'
        # two seas on the edge + the lake, and every board has 3 water regions
        assert len(board.water_ids) == 3 and len(board.water_edge_ids) == 2


def test_unknown_player_counts_are_rejected():
    for n in (0, 1, 6):
        for call in (maps.map_def, maps.map_for, maps.turns_for):
            try:
                call(n)
            except ValueError:
                pass
            else:
                raise AssertionError(f'{call.__name__}({n}) should raise')


def test_shapes_match_their_board():
    """Every outline table agrees with its map (adjacency, border regions,
    anchors inside their region, the regions tile the board)."""
    for n in maps.PLAYER_COUNTS:
        assert shape_for(n).check() == [], n
        assert shape_for(n).map is maps.map_def(n)


def test_region_at_finds_every_anchor():
    """A point on a region's anchor is inside that region's outline."""
    for n in maps.PLAYER_COUNTS:
        shape = shape_for(n)
        for region in shape.map.regions:
            assert shape.region_at(*region.anchor) == region.id, (n, region.id)
        width, height = shape.map.board_size
        assert shape.region_at(-5, -5) is None
        assert shape.region_at(width + 5, height / 2) is None


# --------------------------------------------------------------------------- #
# Spaces and registry
# --------------------------------------------------------------------------- #

def test_spaces_and_names_per_player_count():
    """Each player count sizes its spaces from its board and has its own name."""
    for n, (regions, turns, actions) in EXPECTED.items():
        env = SmallWorldEnv(n)
        assert env.name == env_name_for(n) == ('smallw' if n == 3 else f'smallw{n}')
        assert env.n_regions == regions and env.turns_total == turns
        assert env.layout is layout_for(n) and env.layout.N_ACTIONS == actions
        assert env.action_space.n == actions
        assert env.observation_space['regions'].shape == (regions, 17)
        assert env.observation_space['mask'].shape == (actions,)
        assert env.observation_space['players'].shape == (5, 14)


def test_subclasses_default_to_their_player_count():
    """`get_environment(name)()` builds the right game without arguments."""
    from utils.register import get_environment, get_network_arch
    for name, cls, n in (('smallw2', SmallWorld2Env, 2), ('smallw', SmallWorldEnv, 3),
                         ('smallw4', SmallWorld4Env, 4), ('smallw5', SmallWorld5Env, 5)):
        assert get_environment(name) is cls
        env = cls()
        assert env.n_players == n and env.name == name
        assert get_network_arch(name).__name__ == 'CustomPolicy'
    # player names alone also give the count
    assert SmallWorldEnv(player_names=['a', 'b', 'c', 'd']).name == 'smallw4'
    try:
        SmallWorldEnv(4, player_names=['a', 'b'])
    except ValueError:
        pass
    else:
        raise AssertionError('mismatched player names must raise')


def test_three_player_layout_is_unchanged():
    """The 3-player action layout (and the module constants) did not move, so
    the existing 3-player models keep working."""
    assert LAYOUT_3P is layout_for(3) and N_ACTIONS == 133
    assert (LAYOUT_3P.A_REGION_ALL, LAYOUT_3P.A_SORCERER, LAYOUT_3P.A_DRAGON,
            LAYOUT_3P.A_ALLY) == (38, 68, 98, 128)


def test_action_kind_round_trip_on_every_layout():
    """Every action of every layout decodes to a kind and back."""
    for n in maps.PLAYER_COUNTS:
        lay = layout_for(n)
        build = {'region': lay.region_action, 'region_all': lay.region_all_action,
                 'sorcerer': lay.sorcerer_action, 'dragon': lay.dragon_action,
                 'ally': lay.ally_action}
        for action in range(lay.N_ACTIONS):
            kind, arg = lay.action_kind(action)
            if kind in build:
                assert build[kind](arg) == action, (n, action)
        for bad in (-1, lay.N_ACTIONS):
            try:
                lay.action_kind(bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f'{n}p: action {bad} should raise')
    assert copy.deepcopy(LAYOUT_3P) is LAYOUT_3P
    assert ActionLayout(23).N_ACTIONS == 105


# --------------------------------------------------------------------------- #
# Games
# --------------------------------------------------------------------------- #

def test_random_games_on_every_board():
    """Random games on the 2 / 4 / 5 player boards: every engine invariant
    after every step, the right number of turns, zero-sum rewards."""
    for n in (2, 4, 5):
        for seed in range(FUZZ_GAMES):
            env = SmallWorldEnv(n, pause_between_turns=(seed % 2 == 0))
            env.reset(seed=100 + seed)
            rng = np.random.default_rng(seed)
            steps = 0
            while not env.done:
                assert steps < STEP_BUDGET, f'{n}p seed {seed}: over {STEP_BUDGET} steps'
                if env.phase == Phase.TURN_PAUSE:
                    env.step(-1)
                    steps += 1
                    continue
                action = int(rng.choice(np.flatnonzero(env.action_masks())))
                assert env.describe_action(action)
                _, rewards, _, _, _ = env.step(action)
                steps += 1
                assert len(rewards) == n and abs(sum(rewards)) < 1e-9
                _check_invariants(env, f'{n}p seed {seed} step {steps}')
            assert env.turns_taken == maps.turns_for(n), (n, seed, env.turns_taken)
            assert abs(sum(env.terminal_rewards)) < 1e-9


def test_deepcopy_shares_the_static_board():
    """The MCTS trainer deep-copies games: the static board is never copied,
    and the copy plays on independently."""
    env = _env(5, n_players=5)
    clone = copy.deepcopy(env)
    assert clone.board.map is env.board.map and clone.layout is env.layout
    assert env.board.regions[0] is not clone.board.regions[0]
    clone.step(_legal(clone)[0])                     # pick a combo in the copy
    assert env.phase == Phase.PICK_COMBO and clone.phase != Phase.PICK_COMBO
    assert env.players[0].active is None and clone.players[0].active is not None


def test_diplomat_with_four_opponents():
    """5 players: the Diplomat may ally with any of the four other seats, the
    action being the relative seat offset 1..4."""
    env = _env(42, n_players=5)
    lay = env.layout
    border = next(r.id for r in env.board.regions if r.border and not r.is_water)
    _setup_turn(env, 2, RaceId.RATMEN, PowerId.DIPLOMAT, hand=0, held={border: 1},
                turns_played=2)
    env.step(A_PASS)
    assert env.phase == Phase.ALLY
    assert set(_legal(env)) == {A_PASS} | {lay.ally_action(k) for k in (1, 2, 3, 4)}
    assert env.describe_action(lay.ally_action(3)) == f'ally with {env.players[0].name}'
    env.step(lay.ally_action(3))
    assert env.players[2].ally == 0


def test_first_conquest_only_on_border_regions():
    """A fresh race on a new board may only start on a border region."""
    for n in (2, 4, 5):
        env = _env(7, n_players=n)
        _setup_turn(env, 0, RaceId.RATMEN, None, hand=12)
        targets = {a - A_REGION for a in _legal(env)
                   if A_REGION <= a < env.layout.A_REGION_ALL}
        assert targets, n
        for index in targets:
            assert env.board.by_index(index).border, (n, index + 1)
        interior = [r.index for r in env.board.regions if not r.border and not r.is_water]
        assert interior and not targets & set(interior), n


# --------------------------------------------------------------------------- #
# Renderer and network
# --------------------------------------------------------------------------- #

def test_renderer_uses_the_board_of_the_game():
    """Board image, overlay frame, turn marker and click mapping follow the
    player count."""
    assert rw.board_url() == '/smallw_static/board3p.svg'
    for n in maps.PLAYER_COUNTS:
        env = _env(3, n_players=n)
        board = env.board.map
        assert rw.board_url(env) == f'/smallw_static/board{n}p.svg'
        assert (rw.STATIC_DIR / board.board_image).is_file()
        width, height = board.board_size
        svg = rw.board_overlay_svg(env)
        assert svg.startswith(f'<!-- smallw overlay {width}x{height} -->')
        tx, ty = board.turn_track[0]
        assert f'x="{tx - 12:.1f}" y="{ty - 7:.1f}"' in svg
        for region in env.board.regions:
            assert rw.region_index_at(env, *region.static.anchor) == region.index
        # a click on a border line / off the board falls back to an anchor
        assert 0 <= rw.region_index_at(env, -3, -3) < env.n_regions
        # clicks map to legal actions of this layout
        env.step(_legal(env)[0])                     # pick a combo
        if env.phase == Phase.CONQUER:
            targets = rw.region_targets(env)
            assert targets
            for index in targets:
                action, _ = rw.action_for_click(env, index)
                assert action is not None and env.action_masks()[action]


def test_policy_runs_on_every_board():
    """The entity transformer sizes itself from the observation space and gives
    one logit per action of the board, with the board's own distance bias."""
    import torch as th
    from sb3_contrib import MaskablePPO

    from models.smallw.models import CustomPolicy, _distance_matrix
    for n in maps.PLAYER_COUNTS:
        env = _env(9, n_players=n)
        model = MaskablePPO(CustomPolicy, env, device='cpu', n_steps=16, batch_size=8)
        extractor = model.policy.features_extractor
        assert extractor.tok.n_regions == env.n_regions
        dist = extractor.region_dist.numpy()
        assert (dist == _distance_matrix(env.board.map)).all()
        for region in env.board.regions:
            for other in env.board.neighbours(region):
                assert dist[region.index, other.index] == 1
        obs = {k: th.as_tensor(v)[None] for k, v in env.observation.items()}
        with th.no_grad():
            dist_ = model.policy.get_distribution(
                obs, action_masks=env.action_masks()[None])
        assert dist_.distribution.logits.shape == (1, env.layout.N_ACTIONS)
        action, _ = model.predict(env.observation, action_masks=env.action_masks())
        assert env.action_masks()[action]


# --------------------------------------------------------------------------- #
# Stand-alone runner
# --------------------------------------------------------------------------- #

def _run_all() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception:                                   # noqa: BLE001
            failed += 1
            print(f'FAIL {name}')
            traceback.print_exc()
        else:
            print(f'ok   {name}')
    print(f'{len(tests) - failed}/{len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(_run_all())
