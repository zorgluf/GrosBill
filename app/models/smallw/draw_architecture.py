"""Draw the `smallw` policy network as `architecture.svg` (next to this file).

Run from `app/`:  python3 -m models.smallw.draw_architecture

Layer sizes, parameter counts and the region-distance matrix drawn in the
attention-bias panel are read from `models.py` (the policy is instantiated on
the real observation / action spaces), so re-running it after a change to the
network keeps the picture in sync. The boxes and their captions are hand laid
out below: a structural change (new token type, new head) needs an edit here.
"""

from pathlib import Path
from xml.sax.saxutils import escape

from utils.register import get_environment, get_network_arch
from models.smallw import models as M

OUT = Path(__file__).with_name('architecture.svg')
W, H = 1230, 1470
COLS = [40, 274, 508, 742, 976]          # 5 columns, 214 px wide
CW = 214

# token-group colours (context, regions, players, combos)
C_CTX, C_REG, C_PLY, C_CMB = '#7c5cbf', '#2a9d8f', '#e07a2f', '#3a76c4'


# neutral palettes. Plain colours, no CSS variables: librsvg (GNOME image
# viewer, ImageMagick...) ignores var() and falls back to a black fill; it also
# ignores the dark @media block, so non-browser viewers get the light theme.
LIGHT = dict(bg='#ffffff', fg='#1d1f24', muted='#636a76', box='#f5f6f8', stroke='#c3c9d2')
DARK = dict(bg='#15171c', fg='#e6e8ec', muted='#9ba2ae', box='#1f232a', stroke='#3d4450')


def colour_rules(p: dict) -> str:
    return '\n'.join(f'  {rule}' for rule in [
        f'.bg {{ fill:{p["bg"]}; }}',
        f'.box {{ fill:{p["box"]}; stroke:{p["stroke"]}; }}',
        f'.grid {{ fill:{p["box"]}; }}',
        f'.t, .bt, .h1, .h2 {{ fill:{p["fg"]}; }}',
        f'.st, .cap, .ah {{ fill:{p["muted"]}; }}',
        f'.ln, .dash {{ stroke:{p["muted"]}; }}',
        f'.hl {{ stroke:{p["fg"]}; }}',
    ])


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def fmt(n: int) -> str:
    return f'{n:,}'


class Svg:
    def __init__(self):
        self.parts: list[str] = []

    def add(self, s: str):
        self.parts.append(s)

    def text(self, x, y, s, cls='t', anchor='start', color=None, weight=None, size=None):
        style = []
        if color:
            style.append(f'fill:{color}')
        if weight:
            style.append(f'font-weight:{weight}')
        if size:
            style.append(f'font-size:{size}px')
        st = f' style="{";".join(style)}"' if style else ''
        self.add(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}"{st}>{escape(s)}</text>')

    def rect(self, x, y, w, h, cls='box', color=None, rx=8, extra=''):
        st = f' style="stroke:{color}"' if color else ''
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{cls}"{st}{extra}/>')
        if color:  # translucent tint of the group colour
            self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                     f'fill="{color}" fill-opacity="0.10" stroke="none"/>')

    def box(self, x, y, w, h, title, lines=(), color=None, center=True, title_color=None):
        self.rect(x, y, w, h, color=color)
        tx, anchor = (x + w / 2, 'middle') if center else (x + 12, 'start')
        n = 1 + len(lines)
        # vertically centre the block of text (title 13 px + 11 px lines, 15 px pitch)
        y0 = y + h / 2 - (n - 1) * 15 / 2 + 4
        self.text(tx, y0, title, 'bt', anchor, color=title_color or None)
        for i, line in enumerate(lines):
            self.text(tx, y0 + 15 * (i + 1), line, 'st', anchor)

    def line(self, pts, cls='ln', arrow=True):
        d = ' '.join(f'{x},{y}' for x, y in pts)
        mk = ' marker-end="url(#arr)"' if arrow else ''
        self.add(f'<polyline points="{d}" class="{cls}" fill="none"{mk}/>')

    def plus(self, cx, cy):
        self.add(f'<circle cx="{cx}" cy="{cy}" r="11" class="box"/>')
        self.text(cx, cy + 5, '+', 'bt', 'middle', size=16)

    def render(self) -> str:
        head = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Inter, 'Segoe UI', Helvetica, Arial, sans-serif">
<title>smallw policy network</title>
<style>
  .box {{ stroke-width:1.2; }}
  .t  {{ font-size:12px; }}
  .bt {{ font-size:13px; font-weight:600; }}
  .st {{ font-size:11px; }}
  .h1 {{ font-size:22px; font-weight:700; }}
  .h2 {{ font-size:15px; font-weight:700; }}
  .cap {{ font-size:11px; letter-spacing:0.08em; font-weight:600; }}
  .ln {{ stroke-width:1.4; }}
  .dash {{ stroke-width:1.2; stroke-dasharray:4 3; }}
  .bias {{ stroke:{C_REG}; stroke-width:1.4; stroke-dasharray:5 3; }}
  .grid {{ stroke:none; }}
  .hl {{ stroke-width:1.4; }}
{colour_rules(LIGHT)}
  @media (prefers-color-scheme: dark) {{
{colour_rules(DARK)}
  }}
</style>
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" class="ah"/>
  </marker>
</defs>
<rect width="{W}" height="{H}" class="bg"/>
'''
        return head + '\n'.join(self.parts) + '\n</svg>\n'


def token_strip(s: Svg, y, label):
    """The 42-token sequence, one coloured cell per token."""
    cell = 26
    x0 = 40 + (1150 - M.N_TOKENS * cell) // 2
    groups = [(0, 1, C_CTX, 'ctx'),
              (M.TOK_REGIONS, M.MAX_REGIONS, C_REG, f'regions 1–{M.MAX_REGIONS}'),
              (M.TOK_PLAYERS, M.MAX_PLAYERS, C_PLY, 'players (relative seat 0–4)'),
              (M.TOK_COMBOS, M.N_COMBOS, C_CMB, 'combos (slot 0–5)')]
    centers = []
    for start, n, color, name in groups:
        for i in range(start, start + n):
            s.add(f'<rect x="{x0 + i * cell + 1}" y="{y}" width="{cell - 2}" height="28" rx="3" '
                  f'fill="{color}" fill-opacity="0.80"/>')
        gx0, gx1 = x0 + start * cell + 1, x0 + (start + n) * cell - 1
        s.line([(gx0, y + 36), (gx0, y + 40), (gx1, y + 40), (gx1, y + 36)], 'dash', arrow=False)
        s.text((gx0 + gx1) / 2, y + 54, name, 'st', 'middle', color=color, weight=600)
        centers.append((gx0 + gx1) / 2)
    s.text(x0 + M.N_TOKENS * cell / 2, y + 72, label, 't', 'middle')
    return x0, centers


def main():
    env = get_environment('smallw')()
    policy = get_network_arch('smallw')(env.observation_space, env.action_space, lambda _: 1e-4)
    fe, an, lat = policy.features_extractor, policy.action_net, policy.mlp_extractor
    obs = env.observation_space.spaces
    d, h, ff = M.D_MODEL, M.D_MODEL // 2, M.DIM_FEEDFORWARD
    e = M.ID_EMB_DIM

    p_block = n_params(fe.blocks[0])
    p_trunk = n_params(fe.blocks) + n_params(fe.final_ln) + n_params(fe.dist_bias) + fe.own_bias.numel()
    p_tok = n_params(fe) - p_trunk
    p_pol = n_params(an)
    p_val = n_params(lat) + n_params(policy.value_net)
    p_total = n_params(policy)
    assert p_tok + p_trunk + p_pol + p_val == p_total

    s = Svg()
    s.text(40, 40, 'smallw policy — CustomPolicy (entity transformer)', 'h1')
    s.text(40, 62, f'{fmt(p_total)} parameters · L = {M.N_TOKENS} tokens · d_model = {d} · '
                   f'{M.N_LAYERS} layers × {M.N_HEADS} heads · {M.N_ACTIONS} actions   '
                   f'(models/smallw/models.py, design in README.md)', 'st', size=12)

    # ---- 1. observation ------------------------------------------------- #
    s.text(40, 92, 'OBSERVATION  (gymnasium Dict)', 'cap')
    shp = lambda k: ' × '.join(map(str, obs[k].shape))
    obs_boxes = [('global', f'({shp("global")})  turn, phase, die…', C_CTX),
                 ('regions', f'({shp("regions")})  per map region', C_REG),
                 ('players', f'({shp("players")})  relative seat order', C_PLY),
                 ('combos', f'({shp("combos")})  race + power on offer', C_CMB),
                 ('mask', f'({shp("mask")})  legal actions', None)]
    for x, (name, sub, color) in zip(COLS, obs_boxes):
        s.box(x, 100, CW, 50, name, [sub], color=color, title_color=color)
    for x in COLS[:4]:
        s.line([(x + CW / 2, 150), (x + CW / 2, 206)])
    # the mask is re-read by every token type (legality bits)
    s.line([(COLS[4] + CW / 2, 150), (COLS[4] + CW / 2, 172), (COLS[0] + 190, 172)], 'dash', arrow=False)
    for x in COLS[:4]:
        s.line([(x + 190, 172), (x + 190, 206)], 'dash')
    s.text(COLS[4] + CW / 2 + 8, 190, 'legality bits', 'st')

    # ---- 2. tokenisation ------------------------------------------------ #
    gy, gh = 208, 206
    groups = [
        (f'Context token  ×1', C_CTX, [
            'ctx_token  (learned CLS)',
            f'+ phase_emb  ({M.N_PHASES} phases)',
            '+ turn_emb (21) + die_emb (5)',
            f'+ ctx_num  Linear({fe.ctx_num.in_features}→{d})',
            '   7 flags, 4 log1p counts,',
            '   turn fraction, decline/pass legal',
        ]),
        (f'Region tokens  ×{M.MAX_REGIONS}', C_REG, [
            f'terrain_emb + owner_emb ({M.N_OWNER_CODES})',
            '+ conquered_emb + count_emb',
            f'+ cost_emb (attack cost, 0…{M.MAX_ATTACK_COST + 1})',
            f'+ region_race  R → Linear({e}→{d})',
            f'+ region_num  Linear({fe.region_num.in_features}→{d})',
            '   10 flags, 3 log1p, 4 legal bits',
            f'+ region_pos  (learned, {M.MAX_REGIONS})',
        ]),
        (f'Player tokens  ×{M.MAX_PLAYERS}', C_PLY, [
            'player_ids  Linear(128→128) of',
            '   [R active ‖ P ‖ R decl ‖ R decl]',
            f'+ player_num  Linear({fe.player_num.in_features}→{d})',
            '   coins, tokens, regions, income,',
            '   first-conquest, ally legal',
            '+ ally_emb + seat_emb (relative)',
            'absent seats: masked as keys',
        ]),
        (f'Combo tokens  ×{M.N_COMBOS}', C_CMB, [
            'combo_ids  MLP([R ‖ P])',
            f'   {2 * e}→{d}→{d}, GELU',
            '   (race × power interaction)',
            f'+ combo_num  Linear(3→{d})',
            '   coins on it, empty, pickable',
            '+ combo_pos  (slot = cost)',
        ]),
    ]
    for x, (title, color, lines) in zip(COLS, groups):
        s.rect(x, gy, CW, gh, color=color)
        s.text(x + 12, gy + 22, title, 'bt', color=color)
        for i, line in enumerate(lines):
            s.text(x + 12, gy + 44 + 18 * i, line, 'st' if line.startswith('   ') else 't', size=11)
        s.text(x + 12, gy + gh - 12, f'Σ  →  token ∈ ℝ^{d}', 'bt', color=color)
    # shared identity tables
    x = COLS[4]
    s.rect(x, gy, CW, gh)
    s.text(x + 12, gy + 22, 'Shared identity tables', 'bt')
    rows = [('R', f'race_emb   {M.N_RACE_CODES} × {e}'), ('P', f'power_emb  {M.N_POWER_CODES} × {e}')]
    for i, (k, v) in enumerate(rows):
        s.add(f'<rect x="{x + 12}" y="{gy + 38 + 26 * i}" width="20" height="20" rx="4" class="box"/>')
        s.text(x + 22, gy + 53 + 26 * i, k, 'bt', 'middle')
        s.text(x + 40, gy + 53 + 26 * i, v, 't')
    for i, line in enumerate(['one table per id type, read by',
                              'regions, players and combos,',
                              'then re-projected per role:',
                              '"Skeletons in play" and "Skeletons',
                              'on offer" are the same vector.']):
        s.text(x + 12, gy + 112 + 16 * i, line, 'st')

    # ---- 3. token sequence ---------------------------------------------- #
    ty = 448
    x0, centers = token_strip(s, ty, f'token sequence  X ∈ ℝ^({M.N_TOKENS} × {d})')
    for x, cx in zip(COLS, centers):
        s.line([(x + CW / 2, gy + gh), (x + CW / 2, gy + gh + 12), (cx, gy + gh + 12), (cx, ty - 2)])

    # ---- 4. trunk ------------------------------------------------------- #
    by = 548
    s.line([(390, ty + 78), (390, by)])
    s.rect(40, by, 700, 252)
    s.text(56, by + 24, f'EncoderBlock  ×{M.N_LAYERS}   (pre-LN, no dropout)', 'h2')
    s.text(724, by + 24, f'{fmt(p_block)} params / block', 'st', 'end')
    r1, r2 = by + 70, by + 154            # row centres
    # attention sub-layer
    s.line([(60, by + 36), (60, r1), (88, r1)])
    s.box(90, r1 - 22, 80, 44, 'LayerNorm')
    s.line([(170, r1), (188, r1)])
    s.box(190, r1 - 22, 130, 44, 'qkv  Linear', [f'{d} → 3 × {d}'])
    s.line([(320, r1), (338, r1)])
    s.box(340, r1 - 22, 170, 44, f'attention  {M.N_HEADS} × {d // M.N_HEADS}', ['softmax(QKᵀ/√32 + B)·V'],
          color=C_REG)
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
    s.text(56, by + 240, f'{M.N_LAYERS} blocks with their own weights; the bias B is computed once and '
                         f'reused by every block.', 'st')
    s.line([(390, by + 252), (390, 818)])
    s.box(290, 820, 200, 36, 'final LayerNorm')

    # attention bias panel
    px, py = 770, by
    s.rect(px, py, 420, 308)
    s.text(px + 16, py + 24, f'Attention bias  B  ({M.N_HEADS} heads × {M.N_TOKENS} × {M.N_TOKENS})', 'h2')
    s.line([(px, r1 + 29), (425, r1 + 29), (425, r1 + 23)], 'bias')
    cell, mx, my = 5, px + 26, py + 52
    for (start, n, color) in [(0, 1, C_CTX), (M.TOK_REGIONS, M.MAX_REGIONS, C_REG),
                              (M.TOK_PLAYERS, M.MAX_PLAYERS, C_PLY), (M.TOK_COMBOS, M.N_COMBOS, C_CMB)]:
        s.add(f'<rect x="{mx + start * cell}" y="{my - 6}" width="{n * cell - 1}" height="4" fill="{color}"/>')
        s.add(f'<rect x="{mx - 6}" y="{my + start * cell}" width="4" height="{n * cell - 1}" fill="{color}"/>')
    s.add(f'<rect x="{mx}" y="{my}" width="{M.N_TOKENS * cell}" height="{M.N_TOKENS * cell}" class="grid"/>')
    dist = M._distance_matrix()
    r0, p0 = M.TOK_REGIONS, M.TOK_PLAYERS
    for i in range(M.MAX_REGIONS):
        for j in range(M.MAX_REGIONS):
            op = 1 - dist[i, j] / (M.MAX_DIST + 1)
            s.add(f'<rect x="{mx + (r0 + j) * cell}" y="{my + (r0 + i) * cell}" width="{cell}" '
                  f'height="{cell}" fill="{C_REG}" fill-opacity="{op:.2f}"/>')
    # region <-> player owner blocks, then the absent-seat key columns
    for bx, by_, bw, bh in [(p0, r0, M.MAX_PLAYERS, M.MAX_REGIONS), (r0, p0, M.MAX_REGIONS, M.MAX_PLAYERS)]:
        s.add(f'<rect x="{mx + bx * cell}" y="{my + by_ * cell}" width="{bw * cell}" height="{bh * cell}" '
              f'fill="{C_PLY}" fill-opacity="0.45"/>')
    s.add(f'<rect x="{mx + (p0 + 3) * cell}" y="{my}" width="{2 * cell}" height="{M.N_TOKENS * cell}" '
          f'fill="url(#hatch)"/>')
    s.add(f'<defs><pattern id="hatch" width="4" height="4" patternUnits="userSpaceOnUse" '
          f'patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="4" '
          f'class="hl"/></pattern></defs>')
    lx = mx + M.N_TOKENS * cell + 20
    legend = [
        (C_REG, 1.0, 'region ↔ region', [f'dist_bias[d, head], {M.MAX_DIST + 1} × {M.N_HEADS}',
                                         'd = BFS hops on map3p', f'(real map, clipped at {M.MAX_DIST})']),
        (C_PLY, 0.45, 'region ↔ owner seat', [f'own_bias[active|declined,', f'head], 2 × {M.N_HEADS}']),
        ('hatch', 1.0, 'absent seat (as key)', ['−1e9 (e.g. seats 3, 4', 'in a 3-player game)']),
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
    s.text(px + 16, py + 290, f'{n_params(fe.dist_bias) + fe.own_bias.numel()} learned scalars, zero-initialised '
                              f'(a plain transformer at step 0).', 'st')

    # ---- 5. output tokens ------------------------------------------------ #
    oy = 884
    s.line([(390, 856), (390, oy - 2)])
    token_strip(s, oy, f'output tokens H ∈ ℝ^({M.N_TOKENS} × {d})  ++ 5 player-present flags  →  '
                       f'features ({fe.features_dim:,})')

    # ---- 6. heads -------------------------------------------------------- #
    hy = 978
    s.line([(300, oy + 78), (300, hy)])
    s.line([(1010, oy + 78), (1010, hy)])
    s.rect(40, hy, 760, 318)
    s.text(56, hy + 26, f'Policy — PointerActionNet  ({fmt(p_pol)} params)', 'h2')
    s.text(784, hy + 26, 'each logit is read from the token it points at', 'st', 'end')
    rows = [
        (C_CMB, (f'combo_j ‖ ctx', [f'j = 0…{M.N_COMBOS - 1},  {2 * d} each']),
         ('combo_head', [f'MLP {2 * d}→{h}→1, GELU']), ('logits 0–5', ['pick combo j'])),
        (C_CTX, ('ctx', [f'{d}']),
         ('ctx_head', [f'MLP {d}→{h}→2, GELU']), ('logits 6, 7', ['decline · pass'])),
        (C_REG, ('region_i ⊙ (1 + γ_k) + β_k', [f'(γ_k, β_k) = film(ctx), Linear {d}→{M.N_REGION_RANGES * 2 * d}']),
         (f'region_heads[k]  ×{M.N_REGION_RANGES}', [f'MLP {d}→{h}→1, one per range']),
         ('logits 8–127', [f'{M.N_REGION_RANGES} ranges × {M.MAX_REGIONS} regions'])),
        (C_PLY, ('player_k ‖ ctx', [f'k = 0…{M.MAX_PLAYERS - 1},  {2 * d} each']),
         ('ally_head', [f'MLP {2 * d}→{h}→1, GELU']), ('logits 128–132', ['ally, relative offset k'])),
    ]
    for i, (color, inp, head, out) in enumerate(rows):
        ry = hy + 44 + 62 * i
        s.box(56, ry, 270, 50, inp[0], inp[1], color=color, title_color=color)
        s.line([(326, ry + 25), (354, ry + 25)])
        s.box(356, ry, 220, 50, head[0], head[1])
        s.line([(576, ry + 25), (604, ry + 25)])
        s.box(606, ry, 178, 50, out[0], out[1], color=color, title_color=color)
    s.text(56, hy + 306, 'concat → 133 logits → illegal actions masked (MaskablePPO) → Categorical.  '
                         'Last Linear of every head zero-init → uniform over legal moves.', 'st')

    vx = 830
    s.rect(vx, hy, 360, 318)
    s.text(vx + 16, hy + 26, f'Value head  ({fmt(p_val)} params)', 'h2')
    chain = [
        (hy + 44, 50, 'attention pooling (learned query q)', ['softmax(pool_key(H)·q / √128), absent masked']),
        (hy + 116, 36, f'pooled ‖ ctx  ({2 * d})', []),
        (hy + 174, 50, 'value_mlp', [f'{2 * d}→{M.VF_HIDDEN}→{M.VF_HIDDEN}, ReLU']),
        (hy + 246, 36, f'value_net  Linear {M.VF_HIDDEN}→1   →   V(s)', []),
    ]
    for i, (yy, hh, title, lines) in enumerate(chain):
        s.box(vx + 20, yy, 320, hh, title, lines)
        if i:
            prev = chain[i - 1]
            s.line([(vx + 180, prev[0] + prev[1]), (vx + 180, yy - 2)])

    # ---- 7. action layout ----------------------------------------------- #
    ay = 1328
    s.text(40, ay, f'ACTION LAYOUT  (Discrete {M.N_ACTIONS}, smallw.py)', 'cap')
    cell = 8
    ax0 = 40 + (1150 - M.N_ACTIONS * cell) // 2
    ranges = [(0, 6, C_CMB, 1.0, 'combo 0–5', 0), (6, 8, C_CTX, 1.0, 'decline 6 · pass 7', 1),
              (8, 38, C_REG, 1.0, 'conquer / deploy  8–37', 0),
              (38, 68, C_REG, 0.72, 'redeploy all  38–67', 0),
              (68, 98, C_REG, 0.50, 'sorcerer  68–97', 0),
              (98, 128, C_REG, 0.32, 'dragon  98–127', 0),
              (128, 133, C_PLY, 1.0, 'ally 128–132', 0)]
    for a, b, color, op, name, lvl in ranges:
        for i in range(a, b):
            s.add(f'<rect x="{ax0 + i * cell}" y="{ay + 12}" width="{cell - 1}" height="22" '
                  f'fill="{color}" fill-opacity="{op}"/>')
        cx = ax0 + (a + b) / 2 * cell
        s.text(cx, ay + 52 + 16 * lvl, name, 'st', 'middle', color=color, weight=600)

    s.text(40, H - 22, f'Parameters: tokenisation {fmt(p_tok)} · trunk {fmt(p_trunk)} · policy heads {fmt(p_pol)} · '
                       f'value {fmt(p_val)} · total {fmt(p_total)}.   Generated by draw_architecture.py.', 'st',
           size=12)

    OUT.write_text(s.render(), encoding='utf-8')
    print(f'wrote {OUT}  ({fmt(p_total)} params)')


if __name__ == '__main__':
    main()
