"""Draw the `toybattle` policy network as `architecture.svg` (next to this file).

Run from `app/`:  python3 -m models.toybattle.draw_architecture          # the SVG
                  python3 -m models.toybattle.draw_architecture --pdf    # + architecture.pdf

Layer sizes and parameter counts are read from `models.py` (the policy is
instantiated on the real observation / action spaces); the board thumbnail and
the relation matrix of the attention-bias panel come from a live observation of
`EXAMPLE`. Re-running it after a change to the network keeps the picture in
sync. The boxes and their captions are hand laid out below: a structural change
(new token type, new head) needs an edit here.

The PDF is the same page printed by a headless Chrome / Chromium (light theme).
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch as th

from environments.toybattle.envs.constants import A_PLACE, A_TARGET, A_TYPE, MAX_BASES, MAX_HQ, N_SLOTS, N_TYPES
from environments.toybattle.envs.toybattle import ToyBattleEnv
from models.smallw.draw_architecture import Svg, fmt, n_params
from models.toybattle import models as M
from utils.register import get_environment, get_network_arch

OUT = Path(__file__).with_name('architecture.svg')
EXAMPLE = 'station'                       # the terrain drawn in the relations column and the bias panel
W, H = 1256, 1500
CW = 186
COLS = [40 + 198 * i for i in range(6)]   # 6 columns, 186 px wide
INNER = COLS[-1] + CW - COLS[0]           # 1176

# token-group colours (context, troop types, bases, regions, H.Q.)
C_CTX, C_TRP, C_BASE, C_REG, C_HQ = '#7c5cbf', '#3a76c4', '#2a9d8f', '#e07a2f', '#c2477a'
TROOPS = ['Kwak', 'Skully', "Cap'n", 'Jumbo', 'Hook', 'XB-42', 'Star', 'Roxy']
GROUPS = [(M.T_CTX, 1, C_CTX, 'ctx'),
          (M.T_TROOPS, N_TYPES, C_TRP, 'troop types (strength 0–7)'),
          (M.T_BASES, MAX_BASES, C_BASE, f'bases (slot 0–{MAX_BASES - 1})'),
          (M.T_REGIONS, M.MAX_REGIONS, C_REG, f'regions 0–{M.MAX_REGIONS - 1}'),
          (M.T_HQS, 2 * MAX_HQ, C_HQ, 'H.Q. (mine, theirs)')]


def token_strip(s: Svg, y, label):
    """The 42-token sequence, one coloured cell per token."""
    cell = 26
    x0 = COLS[0] + (INNER - M.N_TOKENS * cell) // 2
    centers = []
    for start, n, color, name in GROUPS:
        for i in range(start, start + n):
            s.add(f'<rect x="{x0 + i * cell + 1}" y="{y}" width="{cell - 2}" height="28" rx="3" '
                  f'fill="{color}" fill-opacity="0.80"/>')
        gx0, gx1 = x0 + start * cell + 1, x0 + (start + n) * cell - 1
        s.line([(gx0, y + 36), (gx0, y + 40), (gx1, y + 40), (gx1, y + 36)], 'dash', arrow=False)
        s.text((gx0 + gx1) / 2, y + 54, name, 'st', 'middle', color=color, weight=600)
        centers.append((gx0 + gx1) / 2)
    s.text(x0 + M.N_TOKENS * cell / 2, y + 72, label, 't', 'middle')
    return centers


def thumbnail(s: Svg, env, x, y, scale):
    """The example terrain as the network gets it: bases, paths, regions, H.Q."""
    T, me = env.terrain, env.current_player
    at = {b.name: (b.x, b.y) for b in T.bases} | {hq.name: (hq.x, hq.y) for side in T.hqs for hq in side}
    px = lambda p: f'{x + p[0] * scale:.1f},{y + p[1] * scale:.1f}'
    s.add(f'<rect x="{x}" y="{y}" width="{100 * scale:.0f}" height="{166 * scale:.0f}" rx="4" class="box"/>')
    for region in T.regions:
        s.add(f'<polygon points="{" ".join(px(p) for p in region.outline)}" fill="{C_REG}" '
              f'fill-opacity="0.22" stroke="none"/>')
    for a, b in T.edges:
        s.add(f'<polyline points="{px(at[a])} {px(at[b])}" class="ln" fill="none" style="stroke-width:1"/>')
    for base in T.bases:
        cx, cy = px((base.x, base.y)).split(',')
        s.add(f'<circle cx="{cx}" cy="{cy}" r="3.4" fill="{C_BASE}"/>')
    for colour, side in enumerate(T.hqs):
        for hq in side:
            cx, cy = map(float, px((hq.x, hq.y)).split(','))
            s.add(f'<rect x="{cx - 4.5:.1f}" y="{cy - 4.5:.1f}" width="9" height="9" rx="1.5" fill="{C_HQ}" '
                  f'fill-opacity="{1.0 if colour == me else 0.5}"/>')


def relation_matrix(s: Svg, rel, present, mx, my, cell):
    """The (L, L) relation ids of the example observation, one cell per pair of tokens."""
    n = M.N_TOKENS
    for start, count, color, _ in GROUPS:
        s.add(f'<rect x="{mx + start * cell}" y="{my - 6}" width="{count * cell - 1}" height="4" fill="{color}"/>')
        s.add(f'<rect x="{mx - 6}" y="{my + start * cell}" width="4" height="{count * cell - 1}" fill="{color}"/>')
    s.add(f'<rect x="{mx}" y="{my}" width="{n * cell}" height="{n * cell}" class="grid"/>')
    for i in range(n):
        for j in range(n):
            r = int(rel[i, j])
            if r == M.R_NONE:
                continue
            if r <= M.R_DIST + M.MAX_HOPS:
                color, op = C_BASE, 1 - (r - M.R_DIST) / (M.MAX_HOPS + 1)
            elif r in (M.R_IN_REGION, M.R_REGION_OF):
                color, op = C_REG, 0.80
            else:
                color, op = C_HQ, 1.0 if r in (M.R_NEAR_MY_HQ, M.R_MY_HQ_NEAR) else 0.5
            s.add(f'<rect x="{mx + j * cell}" y="{my + i * cell}" width="{cell}" height="{cell}" '
                  f'fill="{color}" fill-opacity="{op:.2f}"/>')
    s.add('<defs><pattern id="hatch" width="4" height="4" patternUnits="userSpaceOnUse" '
          'patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="4" class="hl"/></pattern></defs>')
    for j in range(n):                                   # padded slots are never attended to
        if present[j] < 0.5:
            s.add(f'<rect x="{mx + j * cell}" y="{my}" width="{cell}" height="{n * cell}" class="grid"/>')
            s.add(f'<rect x="{mx + j * cell}" y="{my}" width="{cell}" height="{n * cell}" fill="url(#hatch)"/>')


def draw() -> str:
    env = get_environment('toybattle')()
    policy = get_network_arch('toybattle')(env.observation_space, env.action_space, lambda _: 1e-4)
    trunk, an, lat = policy.features_extractor.trunk, policy.action_net, policy.mlp_extractor
    spaces = env.observation_space.spaces
    d, h, ff = M.D_MODEL, M.HEAD_H, M.FF

    p_block = n_params(trunk.blocks[0])
    p_trunk = n_params(trunk.blocks) + n_params(trunk.final_ln) + n_params(trunk.relation)
    p_tok = n_params(trunk) - p_trunk
    p_pol = n_params(an)
    p_val = n_params(lat) + n_params(policy.value_net)
    p_total = n_params(policy)
    assert p_tok + p_trunk + p_pol + p_val == p_total

    # the example: a live observation of one terrain
    example = ToyBattleEnv(terrain=EXAMPLE)
    example.reset(seed=0)
    T = example.terrain
    obs = {k: th.as_tensor(v).unsqueeze(0) for k, v in example.observation.items()}
    with th.no_grad():
        rel, present = M.relation_ids(obs)[0], trunk._tokens(obs)[1][0]
    n_hq = sum(len(side) for side in T.hqs)
    sizes = f'{len(T.bases)} bases, {len(T.regions)} regions, {n_hq} H.Q.'

    s = Svg(W, H, 'toybattle policy network', bias=C_BASE)
    s.text(40, 40, 'toybattle policy — CustomPolicy (board-agnostic graph transformer)', 'h1')
    s.text(40, 62, f'{fmt(p_total)} parameters · L = {M.N_TOKENS} tokens · d_model = {d} · '
                   f'{M.LAYERS} layers × {M.HEADS} heads · {M.N_ACTIONS} actions · one network for the 8 terrains   '
                   f'(models/toybattle/models.py, design in README.md)', 'st', size=12)

    # ---- 1. observation ------------------------------------------------- #
    s.text(40, 92, 'OBSERVATION  (gymnasium Dict, the mover\'s view of the board)', 'cap')
    shp = lambda k: ' × '.join(map(str, spaces[k].shape))
    obs_boxes = [('glob', f'({shp("glob")})  terrain, phase, tiles…', C_CTX),
                 ('mask', f'({shp("mask")})  legal actions', None),
                 ('bases', f'({shp("bases")})  per base slot', C_BASE),
                 ('regions', f'({shp("regions")})  per medal region', C_REG),
                 ('hqs', f'({shp("hqs")})  2 mine, 2 theirs', C_HQ),
                 ('adj', f'({shp("adj")})  paths between bases', None)]
    for x, (name, sub, color) in zip(COLS, obs_boxes):
        s.box(x, 100, CW, 50, name, [sub], color=color, title_color=color)
    gy, gh = 220, 206
    mid = CW / 2
    for i in (0, 2, 3, 4, 5):
        s.line([(COLS[i] + mid, 150), (COLS[i] + mid, gy - 2)])
    # glob also feeds the troop-type tokens
    s.line([(COLS[0] + 150, 150), (COLS[0] + 150, 166), (COLS[1] + 50, 166), (COLS[1] + 50, gy - 2)])
    # the mask is re-read by the troop, base and H.Q. tokens (legality bits)
    s.line([(COLS[1] + 150, 150), (COLS[1] + 150, gy - 2)], 'dash')
    s.line([(COLS[1] + 150, 196), (COLS[4] + 150, 196)], 'dash', arrow=False)
    for i in (2, 4):
        s.line([(COLS[i] + 150, 196), (COLS[i] + 150, gy - 2)], 'dash')
    s.text(COLS[2] + 158, 191, 'legality bits', 'st')
    # the region membership and the H.Q. adjacency are relations, not token features
    s.line([(COLS[3] + 165, 150), (COLS[3] + 165, 166), (COLS[5] + 56, 166), (COLS[5] + 56, gy - 2)], 'bias')
    s.line([(COLS[4] + 165, 150), (COLS[4] + 165, 166)], 'bias', arrow=False)
    s.text(COLS[3] + 171, 180, 'its bases', 'st')
    s.text(COLS[4] + 171, 180, 'bases next to it', 'st')

    # ---- 2. tokenisation ------------------------------------------------ #
    groups = [
        ('Context token  ×1', C_CTX, [
            f'ctx_in  Linear({trunk.ctx_in.in_features}→{d})',
            '   terrain (8), phase (7),',
            '   colour, medals, objective,',
            '   rack and reserve sizes,',
            '   clock, can draw',
            '+ kind[0]',
        ]),
        (f'Troop-type tokens  ×{N_TYPES}', C_TRP, [
            f'troop_emb  (learned, {N_TYPES} × {d})',
            f'+ troop_in  Linear({trunk.troop_in.in_features}→{d})',
            '   my rack, pinned, discards,',
            '   unseen tiles (mine, theirs),',
            '   mask: placeable, share of',
            '   slots, can be exhumed',
            '+ kind[1]',
        ]),
        (f'Base tokens  ×{MAX_BASES}', C_BASE, [
            f'base_in  Linear({trunk.base_in.in_features}→{d})',
            f'   the {spaces["bases"].shape[1]} columns of bases:',
            '   stack, reach, range, gate,',
            '   medal gain, H.Q. distance',
            f'   mask: legal per troop ({N_TYPES}),',
            '   legal as a target (1)',
            '+ kind[2]',
            'no learned position',
        ]),
        (f'Region tokens  ×{M.MAX_REGIONS}', C_REG, [
            f'region_in  Linear({trunk.region_in.in_features}→{d})',
            '   exists, medals left,',
            '   taken by me / by them,',
            '   share of its bases held',
            '   by me / by them, size',
            '+ kind[3]',
            'no learned position',
        ]),
        (f'H.Q. tokens  ×{2 * MAX_HQ}', C_HQ, [
            f'hq_in  Linear({trunk.hq_in.in_features}→{d})',
            '   exists, can be taken now,',
            f'   strengths allowed ({N_TYPES}),',
            f'   mask: legal per troop ({N_TYPES}),',
            '   mine / theirs',
            '+ kind[4]',
        ]),
    ]
    for x, (title, color, lines) in zip(COLS, groups):
        s.rect(x, gy, CW, gh, color=color)
        s.text(x + 12, gy + 22, title, 'bt', color=color)
        for i, line in enumerate(lines):
            s.text(x + 12, gy + 44 + 18 * i, line, 'st' if line.startswith('   ') else 't', size=11)
        s.text(x + 12, gy + gh - 12, f'Σ  →  token ∈ ℝ^{d}', 'bt', color=color)
    # the board itself: relations between tokens
    x = COLS[5]
    s.rect(x, gy, CW, gh)
    s.text(x + 12, gy + 22, f'Relations  {M.N_TOKENS} × {M.N_TOKENS}', 'bt')
    scale = 0.8
    thumbnail(s, example, x + 12, gy + 34, scale)
    kx, ky = x + 12 + 100 * scale + 12, gy + 48
    s.add(f'<circle cx="{kx + 5}" cy="{ky - 4}" r="3.4" fill="{C_BASE}"/>')
    s.text(kx + 16, ky, 'base', 'st')
    s.add(f'<rect x="{kx + 0.5}" y="{ky + 9.5}" width="9" height="9" rx="1.5" fill="{C_HQ}"/>')
    s.text(kx + 16, ky + 18, 'H.Q.', 'st')
    s.line([(kx, ky + 32), (kx + 10, ky + 32)], arrow=False)
    s.text(kx + 16, ky + 36, 'path', 'st')
    s.add(f'<rect x="{kx}" y="{ky + 45}" width="10" height="10" fill="{C_REG}" fill-opacity="0.35"/>')
    s.text(kx + 16, ky + 54, 'region', 'st')
    for i, line in enumerate(['the board', 'is given', 'only as', 'relations']):
        s.text(kx, ky + 74 + 13 * i, line, 'st')
    s.text(x + 12, gy + gh - 27, f'{T.name}: {len(T.bases)} bases', 'st')
    s.text(x + 12, gy + gh - 12, '→  relation id per pair', 'bt')

    # ---- 3. token sequence ---------------------------------------------- #
    ty = 468
    centers = token_strip(s, ty, f'token sequence  X ∈ ℝ^({M.N_TOKENS} × {d})   '
                                 f'(a terrain with fewer bases, regions or H.Q. leaves padded tokens)')
    for x, cx in zip(COLS, centers):
        s.line([(x + mid, gy + gh), (x + mid, gy + gh + 14), (cx, gy + gh + 14), (cx, ty - 2)])

    # ---- 4. trunk ------------------------------------------------------- #
    by = 568
    s.line([(390, ty + 78), (390, by)])
    s.rect(40, by, 700, 252)
    s.text(56, by + 24, f'EncoderBlock  ×{M.LAYERS}   (pre-LN, no dropout)', 'h2')
    s.text(724, by + 24, f'{fmt(p_block)} params / block', 'st', 'end')
    r1, r2 = by + 70, by + 154            # row centres
    # attention sub-layer
    s.line([(60, by + 36), (60, r1), (88, r1)])
    s.box(90, r1 - 22, 80, 44, 'LayerNorm')
    s.line([(170, r1), (188, r1)])
    s.box(190, r1 - 22, 130, 44, 'qkv  Linear', [f'{d} → 3 × {d}'])
    s.line([(320, r1), (338, r1)])
    s.box(340, r1 - 22, 170, 44, f'attention  {M.HEADS} × {d // M.HEADS}',
          [f'softmax(QKᵀ/√{d // M.HEADS} + B)·V'], color=C_BASE)
    s.line([(510, r1), (528, r1)])
    s.box(530, r1 - 22, 100, 44, 'out  Linear', [f'{d} → {d}'])
    s.line([(630, r1), (657, r1)])
    s.plus(670, r1)
    s.line([(60, r1), (60, r1 - 30), (670, r1 - 30), (670, r1 - 13)])     # residual
    # feed-forward sub-layer
    s.line([(670, r1 + 11), (670, r1 + 38), (60, r1 + 38), (60, r2), (88, r2)])
    s.box(90, r2 - 22, 80, 44, 'LayerNorm')
    s.line([(170, r2), (188, r2)])
    s.box(190, r2 - 22, 130, 44, 'Linear', [f'{d} → {ff}'])
    s.line([(320, r2), (338, r2)])
    s.box(340, r2 - 22, 80, 44, 'GELU')
    s.line([(420, r2), (438, r2)])
    s.box(440, r2 - 22, 130, 44, 'Linear', [f'{ff} → {d}'])
    s.line([(570, r2), (657, r2)])
    s.plus(670, r2)
    s.line([(60, r2), (60, r2 + 38), (670, r2 + 38), (670, r2 + 13)])     # residual
    s.text(682, r2 + 5, '→ next', 'st')
    s.text(56, by + 240, f'{M.LAYERS} blocks with their own weights; the bias B is computed once per '
                         f'observation and reused by every block.', 'st')
    s.line([(390, by + 252), (390, by + 270)])
    s.box(290, by + 272, 200, 36, 'final LayerNorm')

    # attention bias panel
    px, py, pw, ph = 770, by, 446, 322
    s.rect(px, py, pw, ph)
    s.text(px + 16, py + 24, f'Attention bias  B  ({M.HEADS} heads × {M.N_TOKENS} × {M.N_TOKENS})', 'h2')
    s.line([(COLS[5] + 166, gy + gh), (COLS[5] + 166, py - 2)], 'bias')          # from the relations column
    s.line([(px, r1 + 29), (425, r1 + 29), (425, r1 + 23)], 'bias')              # into every attention
    cell, mx, my = 5, px + 26, py + 52
    relation_matrix(s, rel, present, mx, my, cell)
    lx = mx + M.N_TOKENS * cell + 18
    legend = [
        (C_BASE, 1.0, 'base ↔ base', [f'hops 0…{M.MAX_HOPS}+, found in the', 'network from adj']),
        (C_REG, 0.80, 'base ↔ region', ['the regions a base borders']),
        (C_HQ, 1.0, 'base ↔ my H.Q.', ['one path away']),
        (C_HQ, 0.5, 'base ↔ their H.Q.', []),
        ('hatch', 1.0, 'padded slot (as key)', ['−1e9: never attended to']),
        (None, 1.0, 'everything else: 0', []),
    ]
    yy = my + 8
    for color, op, title, lines in legend:
        if color == 'hatch':
            s.add(f'<rect x="{lx}" y="{yy - 10}" width="12" height="12" fill="url(#hatch)"/>')
        elif color:
            s.add(f'<rect x="{lx}" y="{yy - 10}" width="12" height="12" fill="{color}" fill-opacity="{op}"/>')
        else:
            s.add(f'<rect x="{lx}" y="{yy - 10}" width="12" height="12" class="box"/>')
        s.text(lx + 18, yy, title, 't', weight=600)
        for k, line in enumerate(lines):
            s.text(lx + 18, yy + 15 * (k + 1), line, 'st')
        yy += 22 + 15 * len(lines)
    s.text(px + 16, py + 288, f'B = relation[id, head]: {M.N_RELATIONS} relations × {M.HEADS} heads = '
                              f'{n_params(trunk.relation)} learned scalars, zero-initialised.', 'st')
    s.text(px + 16, py + 304, f'Drawn: {T.name} ({sizes}), from a live observation.', 'st')

    # ---- 5. output tokens ------------------------------------------------ #
    oy = by + 340
    s.line([(390, by + 308), (390, oy - 2)])
    token_strip(s, oy, f'output tokens H ∈ ℝ^({M.N_TOKENS} × {d})  ++ {M.N_TOKENS} presence flags  →  '
                       f'features ({M.FEATURES:,})')

    # ---- 6. heads -------------------------------------------------------- #
    hy = oy + 94
    s.line([(300, oy + 78), (300, hy)])
    s.line([(1023, oy + 78), (1023, hy)])
    hh = 334
    s.rect(40, hy, 760, hh)
    s.text(56, hy + 26, f'Policy — PointerActionNet  ({fmt(p_pol)} params)', 'h2')
    s.text(784, hy + 26, 'each logit is read from the tokens it points at', 'st', 'end')
    rows = [
        (C_CTX, ('ctx', [f'{d}']),
         ('plain', [f'MLP {d}→{h}→2, ReLU']), ('logits 0, 1', ['draw · skip'])),
        (C_BASE, ('troop_t ,  slot_s ,  ctx', [f't = 0…{N_TYPES - 1} · s = {MAX_BASES} bases + {MAX_HQ} enemy H.Q.']),
         ('pair MLP + bilinear', [f'relu(W·s + W·t + W·c) → {h} → 1', f'+ ⟨U·t, V·s⟩ / √{h}']),
         (f'logits {A_PLACE}–{A_TARGET - 1}', [f'place: {N_TYPES} troops × {N_SLOTS} slots'])),
        (C_BASE, ('base_s ,  ctx', [f's = 0…{MAX_BASES - 1} · the phase is in ctx']),
         ('target', [f'relu(W·b + W·c) → 1,  {d}→{h}→1']),
         (f'logits {A_TARGET}–{A_TYPE - 1}', ['Jumbo, take back, eruption'])),
        (C_TRP, ('troop_t ,  ctx', [f't = 0…{N_TYPES - 1}']),
         ('type', [f'relu(W·t + W·c) → 1,  {d}→{h}→1']),
         (f'logits {A_TYPE}–{M.N_ACTIONS - 1}', ['troop type (Cursed Cemetery)'])),
    ]
    for i, (color, inp, head, out) in enumerate(rows):
        ry = hy + 44 + 66 * i
        s.box(56, ry, 270, 56, inp[0], inp[1], color=color, title_color=color)
        s.line([(326, ry + 28), (354, ry + 28)])
        s.box(356, ry, 220, 56, head[0], head[1])
        s.line([(576, ry + 28), (604, ry + 28)])
        s.box(606, ry, 178, 56, out[0], out[1], color=color, title_color=color)
    s.text(56, hy + hh - 12, f'concat → {M.N_ACTIONS} logits → illegal actions masked (MaskablePPO) → Categorical.  '
                             'Last Linear of every head zero-init → uniform over legal moves.', 'st')

    vx = 830
    s.rect(vx, hy, 386, hh)
    s.text(vx + 16, hy + 26, f'Value head  ({fmt(p_val)} params)', 'h2')
    chain = [
        (hy + 48, 50, 'attention pooling (learned query q)', [f'softmax(pool_key(H)·q / √{d}), padding masked']),
        (hy + 124, 36, f'pooled ‖ ctx  ({2 * d})', []),
        (hy + 186, 50, 'mlp', [f'{2 * d}→{M.VALUE_H}, ReLU']),
        (hy + 262, 36, f'value_net  Linear {M.VALUE_H}→1   →   V(s)', []),
    ]
    for i, (yy, bh, title, lines) in enumerate(chain):
        s.box(vx + 20, yy, 346, bh, title, lines)
        if i:
            prev = chain[i - 1]
            s.line([(vx + 193, prev[0] + prev[1]), (vx + 193, yy - 2)])

    # ---- 7. action layout ----------------------------------------------- #
    ay = hy + hh + 34
    s.text(40, ay, f'ACTION LAYOUT  (Discrete {M.N_ACTIONS}, environments/toybattle/envs/constants.py)', 'cap')
    cell = 7
    ax0 = COLS[0] + (INNER - M.N_ACTIONS * cell) // 2
    cells = [(C_CTX, 1.0), (C_CTX, 0.55)]
    for t in range(N_TYPES):
        cells += [(C_BASE, 1.0 if t % 2 == 0 else 0.6)] * MAX_BASES + [(C_HQ, 1.0)] * MAX_HQ
    cells += [(C_BASE, 0.35)] * MAX_BASES + [(C_TRP, 1.0)] * N_TYPES
    assert len(cells) == M.N_ACTIONS
    for i, (color, op) in enumerate(cells):
        s.add(f'<rect x="{ax0 + i * cell}" y="{ay + 12}" width="{cell - 1}" height="22" '
              f'fill="{color}" fill-opacity="{op}"/>')
    mid_of = lambda a, b: ax0 + (a + b) / 2 * cell
    for t, name in enumerate(TROOPS):
        a = A_PLACE + t * N_SLOTS
        s.text(mid_of(a, a + N_SLOTS), ay + 52, name, 'st', 'middle', color=C_BASE, weight=600)
    s.text(mid_of(A_TARGET, A_TYPE), ay + 52, f'base pick {A_TARGET}–{A_TYPE - 1}', 'st', 'middle',
           color=C_BASE, weight=600)
    s.text(ax0, ay + 68, 'draw 0 · skip 1', 'st', color=C_CTX, weight=600)
    s.text(mid_of(A_PLACE, A_TARGET), ay + 68,
           f'place {A_PLACE}–{A_TARGET - 1}:  2 + troop × {N_SLOTS} + slot   '
           f'(slots 0–{MAX_BASES - 1} = bases, {MAX_BASES}–{N_SLOTS - 1} = enemy H.Q.)', 'st', 'middle')
    s.text(ax0 + M.N_ACTIONS * cell, ay + 68, f'type pick {A_TYPE}–{M.N_ACTIONS - 1}', 'st', 'end',
           color=C_TRP, weight=600)

    s.text(40, H - 22, f'Parameters: tokenisation {fmt(p_tok)} · trunk {fmt(p_trunk)} · policy heads {fmt(p_pol)} · '
                       f'value {fmt(p_val)} · total {fmt(p_total)}.   Generated by draw_architecture.py.', 'st',
           size=12)
    assert ay + 68 < H - 40
    return s.render(), p_total


def to_pdf(svg: Path, pdf: Path):
    """Print the SVG on one page of its own size with a headless Chrome / Chromium."""
    chrome = next((shutil.which(c) for c in ('google-chrome', 'chromium', 'chromium-browser') if shutil.which(c)), None)
    if chrome is None:
        sys.exit('--pdf needs google-chrome or chromium on the PATH')
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp, 'page.html')
        page.write_text(f'<!doctype html><meta charset="utf-8"><style>@page {{ size: {W}px {H}px; margin: 0 }} '
                        f'html, body {{ margin: 0 }} svg {{ display: block }}</style>'
                        + svg.read_text(encoding='utf-8'), encoding='utf-8')
        subprocess.run([chrome, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                        f'--user-data-dir={tmp}/profile', f'--print-to-pdf={pdf}', page.as_uri()],
                       check=True, capture_output=True)


def main():
    text, p_total = draw()
    OUT.write_text(text, encoding='utf-8')
    print(f'wrote {OUT}  ({fmt(p_total)} params)')
    if '--pdf' in sys.argv[1:]:
        pdf = OUT.with_suffix('.pdf')
        to_pdf(OUT, pdf)
        print(f'wrote {pdf}')


if __name__ == '__main__':
    main()
