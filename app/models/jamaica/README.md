# Jamaica policy (`models/jamaica/models.py`)

A small MLP trunk with a pointer-style action head, about 194k parameters, fast on CPU. The
effort goes into the observation rather than the network: the engine computes the consequences
of every legal action, so the policy does not have to learn the board geometry from positions.

## Observation (`environments/jamaica/envs/features.py`)

It is built from the point of view of the player to act. Rows are in relative seat order, absent
seats are zero, and nothing hidden is encoded.

| key | shape | content |
|---|---|---|
| `glob` | 64 | phase, round, Captain, dice (rolled and placed), last round, lair tokens, the pending payment / load / move, the battle (roles, powder, faces, strengths, Lady Beth, Sabre used) |
| `players` | 6 × 71 | position (spaces to go, printed score, −5 zone, lead over me), holds (totals, empty holds, largest hold per type), powers, face-down count, public score, lookahead (food, next port, lairs), hand / deck / discard sizes, cards not yet discarded per code; my own face-down values on my row |
| `hand` | 25 | my cards by code |
| `track` | 18 × 11 | the spaces exactly +1..+12 and −1..−6 steps from my ship, over every fork branch: kinds, costs, affordability, ships, printed score, fork |
| `mask` | 148 | the legal actions |
| `action_feats` | 148 × 24 | per legal action, its simulated consequences |

`action_feats` has two parts:

- **Columns 0–11 (shared by every family):** Δ public score, progress, Δ gold/food/powder, Δ empty
  holds, battles, lair draws, shortages, spaces lost falling back, reaching Port Royal, Δ opponent
  score.
- **Columns 12–23 (per family):** for a card, both halves of the simulated card (`sim.play`); for
  a destination, its kind, cost and occupants; for gunpowder and targets, the P(win) of the battle;
  for a Sabre decision, P(win) if you keep or reroll; for loot, what you would take.

The simulator reads only the acting ship, the public positions and the lair tokens. A battle is
flagged rather than resolved, and forks, retreats and dumps follow a fixed greedy rule.
`test_sim.py` checks it against the engine, and `test_hidden.py` checks that every observation is
unchanged by `redeterminize(pov)`.

## Network

- **`JamaicaExtractor`:**
  - A shared player encoder (71 → 32, ReLU, multiplied by the presence flag) is pooled as
    `[me, next seat, previous seat, mean(opponents), max(opponents)]`. Rows are not flattened, so
    weights for seats 5–6 never stay untrained in 4-player games.
  - A track encoder (11 → 8).
  - `cat[glob, hand, players, track]` → 256 → 128 gives the latent. The extractor returns
    `[latent | action_feats | mask]`.
- **`JamaicaLatent`:** the actor is the identity; the critic is an MLP 128 → 128 → 64, then SB3's
  value head.
- **`PointerScorer`:** `logit[a] = out(relu(W2 relu(W1 [feat_a, mask_a, emb_a] + Wq latent))) + base(latent)[a]`.
  - `emb_a` is the sum of a family embedding (13 families), a sub-kind embedding (Captain die
    order, Sabre keep/reroll, the four powers) and the morning/evening symbol embeddings (cards).
  - `base` is off for the families whose slots are interchangeable (holds, powder, destinations,
    targets, loot holds).
  - `out` and `base` start at zero, so the initial policy is uniform over the legal actions.

The SB3 plumbing is the same as `models/smallw/models.py`: `_build_mlp_extractor` is replaced,
the action head is swapped in `_build`, then the optimizer is rebuilt.

**Breaking changes.** A change to the observation or action layout invalidates the saved models.
Give it a new policy class and a new registry name, as `stottentr` did.

## Reward (`JamaicaEnv`)

- **Terminal:** the rank reward, same table as smallw: +1, −1/9, −3/9, −5/9 at 4 players, with
  ties (on score, then position) sharing.
- **Shaping:** zero-sum and potential-based, on the public score (face-down cards count at their
  mean value, a card known to be cursed at −3):
  `phi_i = clip((S_i − mean_others)/40, ±0.6)`. phi is 0 at reset and at the end, so the
  undiscounted return of every seat equals its rank reward exactly. The callbacks' `reward > 0`
  therefore means a win.

## Training

```sh
cd app
python3 train.py -r -e jamaica -t 0.15 -g 0.995 -ent 0.005 -lr 3e-4 -os 4096 -ob 512 -oe 5 -n_envs 4 -ne 200
```

- **Threshold:** at 4 players the mean eval reward at parity is 0, and `-t 0.15` is about a 36%
  win rate against three copies of the best model.
- **Once promotions stall:** continue without `-r`, with `-ent 0.002 -lr 1e-4`.
- **What to watch:**
  - `eval/win_rate_vs_base` (above 0.6 against the start model);
  - `rollout/ep_len_mean` (about 25–45 learner decisions per game; random play takes about 24
    rounds);
  - `train/explained_variance`, `train/approx_kl`.
- **Status:** not trained yet. A 12k-step smoke run through `train.py` (self-play, evaluations,
  SubprocVecEnv) completed on 2026-09-30.
