# Small World (`smallw`) policy network — design

Design of the policy in `models.py` (`CustomPolicy`), replacing the original
`CombinedExtractor` + MLP placeholder (which only proved SB3 accepted the spaces).
Implemented as described below; nothing has been trained with it yet.

## Prerequisites on the environment side

These cap performance whatever the network, so they come first.

1. **Relative ally action.** `A_ALLY + seat` used the *absolute* seat, while the whole
   observation is egocentric (relative seat order) and never contains the acting seat:
   the same observation required different actions depending on the seat → aliased,
   unlearnable. `A_ALLY + k` now means *relative offset* `k` (1 = next player).
2. **Engine-computed features.** Small World strategy is mostly attack-cost
   arithmetic, and the costs depend on 34 race / power hooks (mountains +1, encampments,
   lairs, Commando / Mounted discounts, …). `_conquest_cost()` already knows them, so
   the observation exposes, per region (`regions` column 16), the attack cost of the acting race (+ 1,
   0 = not attackable), and per player (`players` column 13) the coins they would score if
   their turn ended now. Without them the network would have to re-learn the rules engine from sparse
   rewards.
3. **The action mask as an input.** The 133-entry legality mask is added to the
   observation (`mask` key); its four region slices become 4 bits per region token.

## Architecture: entity transformer + map-distance bias + pointer heads

### 1. Tokenisation (≈ 42 tokens)

| Token | Built from |
|---|---|
| **30 regions** | Σ embeddings: terrain, owner code (12), race (shared table **R**), conquered-this-turn, region id (learned position) + `Linear` over the flags (mine / magic / cavern / border / coastal / lair / fortress / hole / hero / dragon), the counts (tokens, encampments, attack cost — as `log1p(x)` and an embedding of `min(x, 12)`) and the 4 legality bits of the region |
| **5 players** | relative-seat embedding + R[active] + P[power] (shared table **P**) + R[declined] + R[spirit] + `Linear`(coins, tokens in hand / tray, region counts, ally, must-first-conquest, projected income). Absent rows are masked (`key_padding_mask`) |
| **6 combos** | slot embedding (slot = cost) + **R[race] + P[power] + MLP(R ‖ P)** + coins |
| **1 context (CLS)** | phase emb (13) + turn emb + die / dragon / fortress / sorcerer flags, tokens left to place, placements left |

The shared **R** / **P** tables make "Skeletons in play" and "Skeletons on offer" the
same concept. The race×power MLP captures the non-additive combos
(Flying + Halflings ≠ Flying + Trolls).

### 2. Trunk: transformer with a map bias (Graphormer-style)

* 4 pre-LN layers, `d_model = 128`, 4 heads, FFN 256, no dropout (the choices that work
  in `stottentr`).
* **Per-head attention bias learned from the shortest-path distance between two
  regions** (a 30×30 matrix precomputed from `map3p.ADJACENCY`, stored as a buffer),
  plus a learned bias between a region and the player token that owns it. Unlike a GNN
  limited to k hops, full attention keeps global reasoning ("this combo is strong
  because opponent 2 is spread thin over there") while knowing the topology.
* Later option: Laplacian eigenvectors as positional encoding instead of the learned
  region id, to generalise to the 2 / 4 / 5-player maps.

### 3. Policy head: pointers mapped onto the 133 actions

Like `stottentr`'s `PointerActionNet`, logits are built directly from the output tokens:

* **8–127 (4 region ranges):** `logit[k, i] = MLP_k(region_out_i ⊙ FiLM_k(ctx_out))` —
  one small head per range (conquer / place, redeploy-all, Sorcerer, Dragon),
  phase-conditioned.
* **0–5 combos:** `MLP(combo_out_j ‖ ctx_out)`.
* **6 decline, 7 pass:** `MLP(ctx_out)`.
* **128–132 ally:** `MLP(player_out_k ‖ ctx_out)` (relative offsets, prerequisite 1).
* Last layers initialised near zero → the initial policy is uniform over legal actions.

### 4. Value head

Attention pooling (a learned query over all tokens) ‖ `ctx_out` → MLP 256 → 1, on the
shared trunk. If PPO turns unstable, try `share_features_extractor=False` first.

≈ 0.96 M parameters over 42 tokens; a batch of 256 observations goes through the
policy in ~0.4 s on CPU.

## Why

* **Entity tokens**: the game *is* a set of interacting entities; a flat 579-float
  vector throws that structure away.
* **Pointer heads**: "is this region worth it" is shared across 30 regions × 4 action
  types — every gradient signal is reused 30×.
* **Distance bias**: adjacency is the conquest rule; it is encoded directly.
* **Shared R / P**: 280 possible combos, only 34 embeddings to learn.

## Training recipe

* **Long horizon**: ~150–300 decisions per agent per game (every conquest and every
  redeployed token is a step) → `-g 0.998` (the default 0.99 barely sees turn 10),
  GAE λ 0.95.
* `-os 4096 -ob 1024`, `-ent 0.005`, `-lr 1e-4`, `-t 0.2` (zero-sum reward in
  [-0.75, 1]; a random agent against itself averages 0, same situation as `stotten`).
* Keep `mostly_best` opponents (avoids over-fitting a single opponent with 3 players).
* Next step: `train_mcts.py` — `redeterminize()` exists and this network has a proper
  value head for search.

```sh
python3 train.py -r -e smallw -t 0.2 -g 0.998 -ent 0.005 -lr 1e-4 -os 4096 -ob 1024
```

## Migration

Only an untrained `base.zip` existed in `zoo/smallw/`, so `CustomPolicy` was replaced
in place (regenerate `base` with `-r`); no parallel env name is needed. Once trained
snapshots exist, a breaking change must go into a new class + a new `get_network_arch`
entry (the `stotten` / `stottentr` pattern).
