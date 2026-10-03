# Jamaica (`jamaica`)

The pirate race around the island (Malcolm Braff, Bruno Cathala, Sébastien Pauchon — GameWorks,
2007), 3 to 6 players. `-e jamaica` is the 4-player game; `JamaicaEnv(n_players=3..6)` or a
`player_names` list gives other counts. The 2-player Ghost Ship variant is not implemented.

```
envs/constants.py   enums, action layout (Discrete(148)), sizes — no nicegui/gym import
envs/data.py        the transcription: 50 spaces, edges (3 forks), deck, combat die, treasures
envs/track.py       the track graph: forks, the start line, retreats, scores
envs/rules.py       holds, loading, paying, battle arithmetic (shared with the simulator)
envs/jamaica.py     JamaicaEnv (task stack + pending decision), rewards, redeterminize
envs/sim.py         deterministic consequence simulator (policy features only)
envs/features.py    the observation (Dict of float32 arrays, per point of view)
envs/art.py         our own SVG drawings (board, ships, cards, dice, holds)
envs/render_web.py  NiceGUI page (served by play.py at /jamaica)
tests/              python3 -m environments.jamaica.tests.run_all  (from app/, no pytest)
```

The policy and its training recipe: `app/models/jamaica/README.md`.

## Rules as modelled

**Setup.** 9 of the 12 treasure cards are drawn at random into a face-down pile (the other 3 are
removed unseen); a treasure token on each of the 9 pirate lairs. Every ship starts at Port Royal;
each player shuffles their 11 cards, draws 3, and has 3 food and 3 gold in two of their 5 holds.
A random player is the first Captain.

**Round.** The Captain rolls two dice and, after looking at his hand, puts one on *morning* and
one on *evening*; every player secretly picks one card (left symbol = morning action, right symbol
= evening action). From the Captain, clockwise, each player reveals the card and does the morning
action with the morning die, then the evening one. Hands are refilled to 3 (4 with Morgan's Map);
an empty deck is replaced by the shuffled discard. The next player becomes Captain.

**Loading.** The die gives the number of tokens, always put in an *empty* hold. With no empty
hold, one must be emptied — never one of the type being loaded; if every hold holds that type, the
load is ignored. A gold/gold card makes two separate loads.

**Moving.** Exactly the die value, forward or backward, choosing the branch at a fork. A ship pays
for the space it stops on (a lair is free, a port costs the doubloons shown, a sea space one food
per white square) — after a battle if the space is occupied (never in Port Royal). A lair still
holding its token gives a treasure card. Moving backward from the start is allowed but the ship
must still go round the island.

**Shortage.** A ship that cannot pay pays everything it has of that resource, then falls back to
the first space it can pay in full (a lair counts; choosing the route at a merge), fights first
if that space is occupied, then pays.

**Battles.** The ship that stops on an occupied space attacks (choosing its opponent if there are
several). The attacker spends gunpowder and rolls the combat die (2, 4, 6, 8, 10, star), then the
defender does the same; strength = die + gunpowder (+2 with Lady Beth). The higher strength wins,
a tie does nothing, a star wins at once (the defender does not roll after an attacker's star).
Spent gunpowder is lost in every case. The winner takes the contents of one of the loser's holds
(loading rules apply), or one of the loser's treasure cards (face-down ones blind), or gives the
loser one of their own cursed treasures.

**Treasures.** Powers (face up): Morgan's Map (hand of 4), Saran's Sabre (reroll a combat die,
own or the opponent's), Lady Beth (+2 in battle), 6th Hold (an extra hold; stolen with its
contents). Face-down: +3, +3, +5, +7, +7 and the cursed −2, −3, −4.

**End.** When a ship reaches Port Royal (an overshoot stops there) its evening action is ignored,
the round is finished and the game ends. Score = number of the ship's space (every space before
the red line, and behind the start, scores −5; Port Royal 15) + gold in the holds + treasures.
Ties: furthest along wins, then shared.

## Rulings

Interpretations where the rulebook is silent or ambiguous (from the plan, `docs/jamaica_plan.md`):

- R1 The battle winner may also take nothing.
- R2 Saran's Sabre: once per battle, right after either roll (reroll your own die, or force the
  opponent to reroll his — a star included). Not offered after your own star, nor after the
  opponent's worst face. No gunpowder may be added to a reroll.
- R3 A lair's treasure is always taken, also when falling back after a shortage.
- R4 The bank is unlimited. Port Royal is free, never hosts a battle, and reaching it again from
  behind the start line is not a finish.
- R5 Paying: choose a hold, it pays as much as it can of what is due; repeat. Automatic when all
  the holds are equivalent or the whole stock is due. The shortage drain is automatic.
- R6 Gunpowder: choose the amount (dominated amounts are not offered); it comes from the smallest
  gunpowder holds first.
- R7 A shortage from lap 0 never crosses the start line backward (Port Royal is free).
- R8 Round cap: 50 rounds, then normal scoring.
- R9 Forced or equivalent choices are applied automatically, except the card choices.
- R10 The hand size is read when drawing; nobody ever discards.
- R11 A stolen face-down treasure is taken at random; the thief then knows its value.
- R12 Stealing the 6th Hold *card* moves its contents too; stealing a hold's *contents* moves only
  the tokens.
- R13 Tie-break "furthest along" = fewest spaces left to Port Royal on the shortest route.
- R14 The Captain places the dice and picks his card in one decision (nothing is revealed between
  the two, so it is equivalent).
- R15 A load goes into an empty regular hold before the 6th Hold.
- R16 Giving a cursed treasure always gives the winner's most negative one.
- R17 A power gained mid-round works at once (e.g. the 6th Hold for the evening load).
- R18 After an attacker's star the defender neither spends gunpowder nor rolls, unless his Sabre
  forces a reroll; gunpowder spent on a star or a tie is lost.

## Transcription (to proof-read against the board)

Transcribed on 2026-09-30 from photos of the physical game (kept outside the repo, in
`private/jamaica/photos/`). Space numbers follow the race; the web UI draws them at the same place
as on the board. Every space before the red line scores −5.

| # | where | type | cost | score | next |
|---|---|---|---|---|---|
| 0 | start and finish | Port Royal |  | 15 (finish) | 1 |
| 1 | bottom, going west | sea | 2 food | -5 | 2 |
| 2 | bottom, going west | sea | 3 food | -5 | 3 |
| 3 | bottom, going west | sea | 2 food | -5 | 4 |
| 4 | bottom, going west | pirate lair |  | -5 | 5 |
| 5 | bottom, going west | port | 3 gold | -5 | 6 |
| 6 | bottom, going west (fork) | port | 5 gold | -5 | 7, 13 |
| 7 | bottom-left fork, outer lane | sea | 2 food | -5 | 8 |
| 8 | bottom-left fork, outer lane | sea | 3 food | -5 | 9 |
| 9 | bottom-left fork, outer lane | pirate lair |  | -5 | 10 |
| 10 | bottom-left fork, outer lane | sea | 3 food | -5 | 11 |
| 11 | bottom-left fork, outer lane | pirate lair |  | -5 | 12 |
| 12 | bottom-left fork, outer lane | sea | 2 food | -5 | 15 |
| 13 | bottom-left fork, inner channel | sea | 3 food | -5 | 14 |
| 14 | bottom-left fork, inner channel | sea | 4 food | -5 | 15 |
| 15 | left side, going north (merge) | sea | 3 food | -5 | 16 |
| 16 | left side, going north | port | 3 gold | -5 | 17 |
| 17 | left side, going north | sea | 2 food | -5 | 18 |
| 18 | left side, going north | port | 3 gold | -5 | 19 |
| 19 | left side, going north (fork) | pirate lair |  | -5 | 20, 26 |
| 20 | top-left fork, outer lane | sea | 3 food | -5 | 21 |
| 21 | top-left fork, outer lane | port | 3 gold | -5 | 22 |
| 22 | top-left fork, outer lane | pirate lair |  | -5 | 23 |
| 23 | top-left fork, outer lane | sea | 2 food | -5 | 24 |
| 24 | top-left fork, outer lane | sea | 3 food | -5 | 25 |
| 25 | top-left fork, outer lane | pirate lair |  | -5 | 28 |
| 26 | top-left fork, inner channel | sea | 4 food | -5 | 27 |
| 27 | top-left fork, inner channel | sea | 4 food | -5 | 28 |
| 28 | top, going east (merge) | port | 5 gold | -5 | 29 |
| 29 | top, going east | sea | 1 food | -5 | 30 |
| 30 | top, going east | port | 5 gold | -5 | 31 |
| 31 | top, going east | port | 3 gold | -5 | 32 |
| 32 | top, going east (fork) | sea | 3 food | -5 | 33, 39 |
| 33 | top-right fork, outer lane | pirate lair |  | -5 | 34 |
| 34 | top-right fork, outer lane | port | 3 gold | -5 | 35 |
| 35 | top-right fork, outer lane | sea | 2 food | -5 | 36 |
| 36 | top-right fork, outer lane | sea | 1 food | -5 | 37 |
| 37 | top-right fork, outer lane | pirate lair |  | -5 | 38 |
| 38 | top-right fork, outer lane | sea | 1 food | -5 | 41 |
| 39 | top-right fork, inner channel | sea | 4 food | -5 | 40 |
| 40 | top-right fork, inner channel | sea | 4 food | -5 | 41 |
| 41 | right side, finish stretch (merge, after the red -5 line) | port | 5 gold | 2 | 42 |
| 42 | right side, finish stretch | sea | 2 food | 3 | 43 |
| 43 | right side, finish stretch | sea | 3 food | 4 | 44 |
| 44 | right side, finish stretch | pirate lair |  | 5 | 45 |
| 45 | right side, finish stretch | port | 7 gold | 6 | 46 |
| 46 | right side, finish stretch | sea | 2 food | 7 | 47 |
| 47 | right side, finish stretch | sea | 3 food | 8 | 48 |
| 48 | right side, finish stretch | sea | 3 food | 9 | 49 |
| 49 | right side, finish stretch | sea | 3 food | 10 | 0 |

**Action cards** (the same 11 in every colour), morning / evening: food/gunpowder,
forward/backward, gunpowder/gold, gold/gold, gold/forward, backward/forward, forward/food,
forward/forward, gunpowder/food, food/forward, forward/gunpowder.

**Combat die**: 2, 4, 6, 8, 10, star. **Treasures**: Morgan's Map, Saran's Sabre, Lady Beth,
6th Hold, +3, +3, +5, +7, +7, −2, −3, −4.
