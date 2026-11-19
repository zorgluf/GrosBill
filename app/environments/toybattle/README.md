# Toy Battle (`toybattle`)

The two-player duel of toy troops (Paolo Mori & Alessandro Zucchini — Repos Production, 2025):
env, web UI, tests, and a policy (`app/models/toybattle/README.md`), trained model in `app/zoo/pretrained/toybattle/`.

`ToyBattleEnv` plays one terrain (Castle Field by default, `terrain='pool'`…), or draws the terrain
of every game among `terrains=(...)` at `reset()` (or takes `options={'terrain': key}`).
`-e toybattle` registers `ToyBattleAllTerrainsEnv`, which draws among the 8 terrains when built
without one: the observation and action spaces are the same for all of them (padded to the largest),
so a single `zoo/toybattle` network can learn them all.

```
envs/constants.py   troops, phases, sizes, action layout (Discrete(161)) — no nicegui/gym import
envs/terrains.py    the 8 terrains as planar graphs (bases, paths, H.Q., regions, medals) + checks
envs/toybattle.py   ToyBattleEnv (rules, rewards, observation, redeterminize)
envs/art.py         our own SVG drawings (boards, troop tiles, medals)
envs/render_web.py  NiceGUI page (served by play.py at /toybattle), laid out for a phone
tests/              python3 -m environments.toybattle.tests.run_all   (from app/, no pytest)
```

```sh
python3 -m environments.toybattle.tests.run_all            # 107 tests, ~15 s
python3 test.py -g 200 -a random random -e toybattle       # random legal moves on random terrains
python3 test.py -g 200 -r -a greedy random -e toybattle    # the one-ply greedy reference (wins ~87%)
python3 play.py                                            # then /toybattle (pick the terrain on the card)
```

## Rules as modelled

**Troops.** Each player has 24 tiles, 3 of each: Kwak (joker), Skully 1, Cap'n 2, Jumbo 3, Hook 4,
XB-42 5, Star 6, Roxy 7. Four are removed unseen, the 20 others are the face-down reserve. A random
player starts with 3 troops on the rack, the other has 4. Seat 0 is Blue, seat 1 is Red.

**Turn.** Either *draw 2* (1 if a single tile is left or a single space is free; not with 8 troops
on the rack), or *place 1 troop*, then apply its effect, then the effect of its base if it is a
special one.

**Placement.** On an empty base, on a base whose visible troop is yours, or on an enemy troop of
strictly lower strength — Kwak covers anything and anything covers Kwak. The base must be connected
to (one of) your H.Q. by a path going only through bases you occupy; only the visible troop of a
stack occupies its base. Placing a troop on an enemy H.Q. (which is not a base: it must be next to
a connected base of yours) wins at once.

**Troop effects.** Skully: draw 2. Cap'n: place an extra troop and apply its effect (the special
bases of the turn are applied after all the troop effects, in placement order). Jumbo: discard one
visible enemy troop adjacent to it. Hook: goes on any base, connected or not (not on the H.Q.).
XB-42: one random troop of the opponent's rack is discarded face up. Star: draw 1.

**Regions.** As soon as one player occupies every base around a region, they take its medals for
good (checked after every change of the board, for both players — a troop you remove may give the
opponent a region). Reaching the terrain's objective wins at once.

**End.** A player who can neither draw nor place at the start of their turn ends the game: most
medals wins, and that player loses a tie.

| terrain (`key`)             | bases (special) | medals / objective | special bases |
|-----------------------------|-----------------|--------------------|---------------|
| Castle Field `castle`       | 15 (4)          | 14 / 7             | take one of your *other* visible troops back on your rack |
| Tropical Pool `pool`        | 13 (5)          | 12 / 6             | only the printed strengths (1/2, 3/4/5, 6/7) or Kwak may be placed; one H.Q. of each colour is 6/7 too |
| City of Clouds `clouds`     | 14 (4)          | 16 / 8             | draw 1 |
| Volcanic Jungle `jungle`    | 13 (2)          | 14 / 7             | move an adjacent enemy troop to a base adjacent to it |
| Cursed Cemetery `cemetery`  | 15 (4)          | 14 / 7             | take one of your discarded troops back |
| Caribbean Sea `caribbean`   | 12 (0)          | 11 / 5             | none; 2 blue H.Q. against 1 red |
| Station Metal-X `station`   | 13 (3)          | 14 / 7             | the effect of the troop placed there is not applied |
| Battlefield `battlefield`   | 14 (4)          | 16 / 8             | one random troop of the opponent's rack cannot be placed on their next turn |

## Rulings and simplifications

- **Optional effects.** The rules say "you may" for every effect. The ones with a target are a
  decision with a *skip* action (Cap'n, Jumbo, Castle Field, Volcanic Jungle, Cursed Cemetery). The
  ones without (Skully, Star, XB-42, City of Clouds, Battlefield) are always applied: declining
  them is next to never useful and would add a yes/no step to half of the placements.
- **Station Metal-X.** Hook's jump is an effect: on a special base Hook needs the normal connection
  (the player aid says the restriction applies *before* placing the troop).
- **Tropical Pool.** The gates show the Kwak symbol next to the strengths: Kwak goes everywhere.
- **Castle Field.** "Other troops" = any visible troop of yours except the one on the special base
  that triggered the effect; not offered with 8 troops on the rack.
- **Volcanic Jungle.** The troop moved ignores every placement rule: it may land on any base
  adjacent to its own (never an H.Q.), on top of whatever is there — the special base included.
- **Battlefield.** "Point without looking" = a random tile among those not already laid down. The
  owner sees which one, the opponent only that one is down. It still counts in the rack (it may be
  hit by XB-42) and comes back at the end of its owner's turn.
- **Simultaneous regions.** If one action gives regions to both players, the player to move is
  served (and wins) first.
- **Turn limit (not a rule).** Castle Field allows endless placing and taking back, so a game is
  cut after `max_turns` (300) turns: most medals wins, a tie is a draw (`winner_player` is None,
  rewards 0). Random play never gets near it (about 40 turns).

## Where the boards come from

The graphs were read off the published boards and re-drawn from scratch (`envs/art.py`): no
original art is in the repository. Reference pictures are kept untracked in `private/toybattle/`.
`terrains.py` checks itself at import: every region is a bounded face of the drawing, and the 7
terrains with a half-turn symmetry are exactly symmetric (bases, special bases, gates, paths,
regions, H.Q.). The objective is half the medals on every terrain, as printed.

- Paths, bases, special bases and H.Q.: all 8 terrains read on the digital board sheet.
- Medal spaces: read on photos of the physical boards for Castle Field (rulebook), Tropical Pool,
  City of Clouds, Cursed Cemetery, Station Metal-X and Battlefield. For **Volcanic Jungle** and
  **Caribbean Sea** no straight photo was found: the counts come from a third-party transcription,
  consistent with the printed objectives (7 and 5) and with partial photos — worth a look at the
  real boards.

## Spaces

**Actions — `Discrete(161)`** (`constants.py`):

| range     | meaning |
|-----------|---------|
| 0         | draw |
| 1         | skip the optional effect |
| 2..137    | place: `2 + troop * 17 + slot`, slots 0-14 = bases, 15-16 = the enemy H.Q. |
| 138..152  | pick a base (Jumbo target, troop taken back, troop moved by the eruption, where it goes) |
| 153..160  | pick a troop type (Cursed Cemetery) |

Slots are *canonical*: on the symmetric terrains Red sees the board turned by a half-turn
(`Terrain.views`), so "my side" is the same slots for both colours and a position and its mirror
give the same observation and mask (tested). Caribbean Sea is not symmetric: both colours use the
same indices there, and `glob` tells which colour is playing.

**Observation — `Dict` of float32 in [0, 1]**, from the point of view of the player to move
(`env.observe(seat)` for any seat):

| key       | shape     | content |
|-----------|-----------|---------|
| `glob`    | 73        | terrain (8), phase (7), I am Blue, medals mine/theirs/objective, rack and reserve sizes, my rack (8), my pinned troops (8), my discard (8), their discard (8), my unseen tiles (8), their unseen tiles (8), turn clock, can draw |
| `bases`   | 15 x 41   | exists, special, top troop mine/theirs/none, its type (8), stack height, my/their tiles in the stack, the tile under the top (owner, strength, Kwak), connected to my/their H.Q., in my/their placement range, next to my/their H.Q., strengths allowed (8), origin of the pending effect, troop being moved, **medals I / they win by occupying it now**, **special effect still queued this turn**, **paths to my / their nearest H.Q.** |
| `adj`     | 15 x 15   | paths between base slots |
| `regions` | 14 x 22   | bases around it (15), exists, medals left, taken by me/them, share of its bases I/they hold, size |
| `hqs`     | 4 x 25    | my H.Q. then theirs: exists, can be captured now, strengths allowed (8), adjacent bases (15) |
| `mask`    | 161       | the legal actions |

The column and slice names are in `constants.py` (`B_*`, `G_*`), which the policy reads. The three
groups in bold are engine-computed helpers: a region one base short of being surrounded shows up on
that base (as a gain for me, a threat from them), so the network does not have to rediscover the
region arithmetic from sparse rewards.

Hidden: the other rack (only its size), which of their troops is pinned, both reserves' order and
the tiles in the box. `redeterminize(pov)` re-deals all of it without changing `observe(pov)`.

**Reward.** Zero-sum, per seat: +1 / -1 at the end (0 / 0 for a turn-limit draw), plus a potential
on the medal difference, `shaping * (mine - theirs) / objective` (0.5 by default), which is 0 at
both ends — so a game's return is exactly the outcome. `env.apply(action)` is `step()` without the
observation (it returns the rewards only), for searches and scripted players that step copies.

## Web UI

One narrow column, meant for a phone held upright (the whole game fits a 390 x 844 screen): the
opponent's line (medals, face-down rack, reserve, discard), the board with the viewer's H.Q. at the
bottom, then the viewer's line, rack and buttons. Tap a troop of the rack — the bases it may go to
pulse — then a base; Jumbo and the special bases light their targets directly. Bases touched since
the viewer's last turn are ringed in yellow, a tap on any base lists its stack.
