"""Architecture diagrams for the OPERA paper (docs/figures/arch_*.{svg,png}).

  conda run -n opera-kagri python docs/figures/build_arch_figures.py

Five figures, drawn to match opera_lm/model.py (forward, _layer_body,
_compose / compose_pair_batch, _hmem_read) and opera_lm/incremental.py:
  arch_overview   one layer: tree + Fenwick readout + memory, cross-MLP, blend
  arch_node       the compose node
  arch_fenwick    prefix states from Fenwick blocks (position is structure)
  arch_memory     the gated holographic memory, one step
  arch_decode     incremental decoding: what one appended byte touches
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'mathtext.fontset': 'dejavusans', 'svg.fonttype': 'none'})

INK = '#1f2430'
MUTED = '#6b7280'
TREE = ('#dbe8fb', '#2f6fd6')      # (fill, edge)
MEM = ('#fde4d0', '#d9622b')
MIX = ('#e3f2e6', '#2e8b57')
HEAD = ('#ece4f7', '#7a4fc2')
NEUTRAL = ('#f3f4f6', '#9ca3af')
HILITE = ('#ffe9a8', '#c98a00')
NEW = ('#d4f5dc', '#1f9d55')


def canvas(w, h, scale=0.78):
    fig, ax = plt.subplots(figsize=(w * scale, h * scale))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax


def box(ax, x, y, w, h, text, colors=NEUTRAL, fs=10, weight='normal',
        ls='-', lw=1.3, color=INK, align='center'):
    fc, ec = colors
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                fc=fc, ec=ec, lw=lw, ls=ls))
    if text:
        tx = x + w / 2 if align == 'center' else x + 0.15
        ax.text(tx, y + h / 2, text, ha=align, va='center', fontsize=fs,
                weight=weight, color=color, linespacing=1.35)


def arrow(ax, p, q, color=INK, lw=1.3, rad=0.0, ls='-', head=True):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>' if head else '-',
                                 mutation_scale=11, color=color, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}',
                                 shrinkA=0, shrinkB=0))


def line(ax, pts, color=INK, lw=1.3, ls='-'):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle='round')


def op(ax, x, y, sym, colors=NEUTRAL, r=0.26, fs=12):
    fc, ec = colors
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=1.3))
    ax.text(x, y, sym, ha='center', va='center', fontsize=fs, color=INK)


def save(fig, name):
    for ext in ('svg', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=220,
                    bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print('wrote', name)


# --------------------------------------------------------------------------
def fig_overview():
    W, H = 11.0, 14.0
    fig, ax = canvas(W, H)
    ax.text(W / 2, H - 0.35, 'OPERA: one layer', ha='center', fontsize=14,
            weight='bold', color=INK)
    ax.text(W / 2, H - 0.85, 'no attention  ·  no tokenizer  ·  no positional encoding',
            ha='center', fontsize=10, color=MUTED)

    # input bytes
    chars = ['T', 'h', 'e', '␣', 'c', 'a', 't']
    x0 = W / 2 - len(chars) * 0.62 / 2
    for i, c in enumerate(chars):
        box(ax, x0 + i * 0.62, 0.35, 0.5, 0.5, c, NEUTRAL, fs=11)
    ax.text(x0 - 0.25, 0.6, 'raw UTF-8 bytes', ha='right', va='center',
            fontsize=9, color=MUTED)
    box(ax, 3.3, 1.35, 4.4, 0.7, 'byte embedding  (259 ids)', NEUTRAL)
    arrow(ax, (5.5, 0.9), (5.5, 1.35))

    # layer frame
    ly0, ly1 = 2.6, 10.55
    box(ax, 0.35, ly0, W - 0.7, ly1 - ly0, '', ('#ffffff', '#9ca3af'), ls='--', lw=1.1)
    ax.text(0.6, ly1 - 0.3, 'layer ℓ   (× L)', fontsize=10, weight='bold', color=MUTED)
    ax.text(5.65, 2.42, 'stream  $s$', ha='left', fontsize=9.5, color=INK)
    arrow(ax, (5.5, 2.05), (5.5, 2.95))

    # split to tree and memory
    line(ax, [(5.5, 2.95), (5.5, 3.25)])
    line(ax, [(3.05, 3.25), (7.95, 3.25)])
    arrow(ax, (3.05, 3.25), (3.05, 3.6))
    arrow(ax, (7.95, 3.25), (7.95, 3.6))

    box(ax, 1.0, 3.6, 4.1, 1.45,
        'binary tree, bottom-up\ncompose node at every pair\n(level $k$ = spans of $2^k$ bytes)',
        TREE, fs=9.5)
    box(ax, 1.0, 5.55, 4.1, 1.25,
        'Fenwick readout\nprefix state per position:\nfold of ≤ ⌈log₂ n⌉+1 tree nodes',
        TREE, fs=9.5)
    arrow(ax, (3.05, 5.05), (3.05, 5.55))

    box(ax, 5.9, 3.6, 4.1, 3.2,
        'gated holographic memory\n\nwrite: bind $k_{t-1}\\otimes v_t$\n'
        'forget: decay $\\lambda_t$ (content-\ndependent, per slot)\n'
        'read: unbind $\\bar{k}_t\\otimes M_t$\n\nstate: nb quaternions (fixed)',
        MEM, fs=9.5)

    op(ax, 5.5, 7.45, '+')
    arrow(ax, (3.05, 6.8), (5.24, 7.45), rad=-0.25)
    arrow(ax, (7.95, 6.8), (5.76, 7.45), rad=0.25)
    ax.text(5.85, 7.75, 'prefix state', fontsize=9, color=MUTED)

    box(ax, 3.6, 8.05, 3.8, 0.65, 'cross-MLP  (position-wise)', MIX, fs=9.5)
    arrow(ax, (5.5, 7.71), (5.5, 8.05))
    box(ax, 3.1, 9.15, 4.8, 0.85,
        'blend gate\n$s \\leftarrow g\\cdot\\mathrm{MLP} + (1-g)\\cdot s$', MIX, fs=9.5)
    arrow(ax, (5.5, 8.7), (5.5, 9.15))
    # skip connection: the stream into the blend
    line(ax, [(5.5, 3.0), (0.75, 3.0), (0.75, 9.57)], color=MIX[1], lw=1.2)
    arrow(ax, (0.75, 9.57), (3.1, 9.57), color=MIX[1], lw=1.2)
    ax.text(0.95, 9.72, 'stream (skip)', fontsize=8.5, color=MIX[1])

    arrow(ax, (5.5, 10.0), (5.5, 11.0))
    ax.text(5.65, 10.3, 'next layer / head', fontsize=8.5, color=MUTED)
    box(ax, 3.1, 11.0, 4.8, 0.7, 'stream norm → byte head', HEAD, fs=9.5)
    arrow(ax, (5.5, 11.7), (5.5, 12.05))
    ax.text(5.5, 12.25, 'next-byte distribution (259)', ha='center', fontsize=9.5,
            color=INK)
    save(fig, 'arch_overview')


# --------------------------------------------------------------------------
def fig_node():
    W, H = 11.0, 12.4
    fig, ax = canvas(W, H)
    ax.text(W / 2, H - 0.35, 'The compose node (per quaternion block)', ha='center',
            fontsize=14, weight='bold', color=INK)

    box(ax, 0.9, 0.4, 3.4, 0.75, 'left child  $h_l=(s_l, v_l)$', TREE, fs=10)
    box(ax, 6.7, 0.4, 3.4, 0.75, 'right child  $h_r=(s_r, v_r)$', TREE, fs=10)

    box(ax, 0.9, 1.85, 3.4, 0.75, '$v_0 = R_L\\, v_l$,   $s_0 = s_l$', NEUTRAL, fs=10)
    box(ax, 6.7, 1.85, 3.4, 0.75, '$v_1 = R_R\\, v_r$,   $s_1 = s_r$', NEUTRAL, fs=10)
    arrow(ax, (2.6, 1.15), (2.6, 1.85))
    arrow(ax, (8.4, 1.15), (8.4, 1.85))

    box(ax, 1.6, 3.4, 7.8, 1.35,
        'geometric (Hamilton) product\n'
        '$(s_0,v_0)\\otimes(s_1,v_1) = (s_0 s_1 - v_0\\cdot v_1,\\;'
        ' s_0 v_1 + s_1 v_0 + v_0\\times v_1)$',
        TREE, fs=10)
    arrow(ax, (2.6, 2.6), (3.2, 3.4))
    arrow(ax, (8.4, 2.6), (7.8, 3.4))

    # gates read both children
    box(ax, 7.3, 5.45, 3.2, 0.95, 'gates\n$g=\\sigma(W[h_l;h_r]+b)$', MIX, fs=9.5)
    line(ax, [(10.1, 0.78), (10.7, 0.78), (10.7, 5.92), (10.5, 5.92)],
         color=MIX[1], lw=1.1, ls='--')
    line(ax, [(0.9, 0.78), (0.4, 0.78), (0.4, 0.15), (10.85, 0.15), (10.85, 5.8)],
         color=MIX[1], lw=1.1, ls='--')
    arrow(ax, (10.85, 5.8), (10.5, 5.8), color=MIX[1], lw=1.1, ls='--')

    box(ax, 0.6, 5.35, 6.2, 1.15,
        'gated fusion\n$f = g_0\\,(s_0,v_0) + g_1\\,(s_1,v_1) + g_2\\,(h_0\\otimes h_1)$',
        MIX, fs=10)
    arrow(ax, (5.5, 4.75), (4.9, 5.35))
    arrow(ax, (7.3, 5.92), (6.8, 5.92), color=MIX[1])
    # identity paths
    arrow(ax, (0.9, 2.22), (1.2, 5.35), rad=-0.35, color=MUTED, lw=1.1)
    line(ax, [(10.1, 2.22), (10.35, 2.22), (10.35, 5.05), (6.45, 5.05)], color=MUTED, lw=1.1)
    arrow(ax, (6.45, 5.05), (6.45, 5.35), color=MUTED, lw=1.1)
    ax.text(0.15, 3.9, 'identity\npaths', fontsize=8.5, color=MUTED, ha='left')

    box(ax, 2.2, 7.1, 6.6, 0.7, 'output map   $f_v \\leftarrow R_O\\, f_v$', NEUTRAL, fs=10)
    arrow(ax, (3.7, 6.5), (4.6, 7.1))
    box(ax, 2.2, 8.35, 6.6, 0.7,
        'LayerNorm, then  $\\phi(x)=\\tanh x + 0.1x$', NEUTRAL, fs=10)
    arrow(ax, (5.5, 7.8), (5.5, 8.35))
    box(ax, 3.3, 9.6, 4.4, 0.75, 'parent  $h_{\\mathrm{parent}}$', TREE, fs=10.5,
        weight='bold')
    arrow(ax, (5.5, 9.05), (5.5, 9.6))

    ax.text(W / 2, 10.85,
            '$R_L, R_R, R_O$: learned 3×3 maps per block, initialised at $I + \\mathcal{N}(0, 0.1^2)$\n'
            'states are quaternions (even subalgebra of Cl(3)); the product is order-sensitive',
            ha='center', fontsize=8.8, color=MUTED, linespacing=1.4)
    save(fig, 'arch_node')


# --------------------------------------------------------------------------
def tree_layout(T=8, dx=1.15, x0=1.55, y0=0.9, dy=1.35):
    """{(level, j): (x, y)} for a balanced binary tree over T leaves."""
    pos = {}
    for j in range(T):
        pos[(0, j)] = (x0 + j * dx, y0)
    k = 1
    while (1 << k) <= T:
        for j in range(T >> k):
            a, b = pos[(k - 1, 2 * j)], pos[(k - 1, 2 * j + 1)]
            pos[(k, j)] = ((a[0] + b[0]) / 2, y0 + k * dy)
        k += 1
    return pos


def span(k, j):
    a, b = j * (1 << k) + 1, (j + 1) * (1 << k)
    return f'{a}' if a == b else f'{a}–{b}'


def draw_tree(ax, pos, style):
    """style(key) -> (colors, edge_color, lw)."""
    for (k, j), (x, y) in pos.items():
        if k > 0:
            for c in (2 * j, 2 * j + 1):
                cx, cy = pos[(k - 1, c)]
                line(ax, [(cx, cy + 0.28), (x, y - 0.28)], color='#c3c8d0', lw=1.1)
    for key, (x, y) in pos.items():
        colors, lw = style(key)
        w = 0.62 if key[0] == 0 else 0.95
        box(ax, x - w / 2, y - 0.27, w, 0.54, span(*key), colors, fs=9, lw=lw)


def fig_fenwick():
    W, H = 15.4, 7.6
    fig, ax = canvas(W, H)
    ax.text(W / 2, H - 0.35, 'Prefix states from Fenwick blocks: position is structure',
            ha='center', fontsize=14, weight='bold', color=INK)
    pos = tree_layout()
    pick = {(2, 0), (1, 2), (0, 6)}                 # prefix 7 = 4 + 2 + 1
    draw_tree(ax, pos, lambda key: (HILITE, 2.0) if key in pick else (TREE, 1.1))
    for i in range(8):
        ax.text(pos[(0, i)][0], 0.3, f'byte {i + 1}', ha='center', fontsize=8.5,
                color=MUTED)
    for k in range(4):
        ax.text(0.05, pos[(0, 0)][1] + k * 1.35, f'level {k}', fontsize=8.5,
                color=MUTED, va='center')
    ax.text(5.0, 6.35,
            'prefix of length 7 = 111₂  →  blocks [1–4], [5–6], [7]\n'
            'prefix state $P_7 = ([1\\text{–}4] \\circ [5\\text{–}6]) \\circ [7]$'
            '   (∘ = the same compose node)',
            ha='center', fontsize=10, color=INK, linespacing=1.5)

    # decomposition table
    tx = 10.6
    ax.text(tx, 5.75, 'n', fontsize=10, weight='bold', color=INK)
    ax.text(tx + 0.75, 5.75, 'binary', fontsize=10, weight='bold', color=INK)
    ax.text(tx + 2.15, 5.75, 'blocks folded', fontsize=10, weight='bold', color=INK)
    rows = [(5, '101', '[1–4] ∘ [5]'), (6, '110', '[1–4] ∘ [5–6]'),
            (7, '111', '[1–4] ∘ [5–6] ∘ [7]'), (8, '1000', '[1–8]  (one node)')]
    for i, (n, b, blk) in enumerate(rows):
        y = 5.1 - i * 0.62
        if n == 7:
            box(ax, tx - 0.15, y - 0.27, 4.75, 0.54, '', HILITE, lw=1.0)
        ax.text(tx, y, str(n), fontsize=10, color=INK, va='center')
        ax.text(tx + 0.75, y, b, fontsize=10, color=INK, va='center', family='DejaVu Sans Mono')
        ax.text(tx + 2.15, y, blk, fontsize=10, color=INK, va='center')
    ax.text(tx - 0.15, 1.7,
            'Every length is computed by a different\n'
            'circuit, so position needs no encoding.\n'
            'All T prefixes: O(T log T) composes per layer.',
            fontsize=9.5, color=MUTED, linespacing=1.45)
    save(fig, 'arch_fenwick')


# --------------------------------------------------------------------------
def fig_memory():
    W, H = 14.2, 7.4
    fig, ax = canvas(W, H)
    ax.text(W / 2, H - 0.35, 'Gated holographic memory (one step, per layer)',
            ha='center', fontsize=14, weight='bold', color=INK)

    box(ax, 0.5, 0.4, 2.0, 0.7, 'stream $x_{t-1}$', NEUTRAL, fs=10)
    box(ax, 3.4, 0.4, 2.0, 0.7, 'stream $x_t$', NEUTRAL, fs=10)

    box(ax, 0.1, 1.75, 2.85, 0.95, 'unit key\n$k_{t-1}=\\mathrm{normalize}(W_k\\, x_{t-1})$', MEM, fs=8.4)
    box(ax, 3.2, 1.75, 2.4, 0.95, 'value\n$v_t = W_v x_t$', MEM, fs=9.5)
    arrow(ax, (1.5, 1.1), (1.5, 1.75))
    arrow(ax, (4.4, 1.1), (4.4, 1.75))

    op(ax, 2.95, 3.55, '⊗', MEM, fs=13)
    ax.text(2.95, 3.95, 'bind', ha='center', fontsize=8.5, color=MUTED)
    arrow(ax, (1.5, 2.7), (2.72, 3.45))
    arrow(ax, (4.4, 2.7), (3.18, 3.45))

    op(ax, 4.75, 3.55, '×', MIX)
    arrow(ax, (3.21, 3.55), (4.49, 3.55))
    box(ax, 5.95, 1.75, 2.3, 0.95, 'write gate\n$g_w=\\sigma(w^\\top x_t+b)$', MIX, fs=8.8)
    line(ax, [(5.4, 0.75), (7.1, 0.75), (7.1, 1.75)], color=INK)
    arrow(ax, (6.35, 2.7), (4.97, 3.4), rad=0.2, color=MIX[1])

    op(ax, 6.9, 4.6, '+', MEM)
    arrow(ax, (5.01, 3.55), (6.65, 4.5), rad=-0.15)

    box(ax, 4.6, 5.55, 2.6, 0.75, '$\\lambda_t \\odot M_{t-1}$', MEM, fs=10.5)
    arrow(ax, (6.3, 5.55), (6.85, 4.86))
    ax.text(3.2, 5.92, 'forget gate\n$\\lambda_t=\\sigma(W_f x_t+b_f)$\nper slot', fontsize=8.6,
            color=MUTED, ha='center', va='center', linespacing=1.3)

    box(ax, 7.6, 4.15, 2.5, 0.9, '$M_t$\nnb quaternion slots', MEM, fs=9.2)
    arrow(ax, (7.16, 4.6), (7.6, 4.6))
    arrow(ax, (8.9, 5.05), (6.6, 6.3), rad=0.35, color=MEM[1], ls='--')
    ax.text(8.95, 6.15, 'carried to $t+1$: fixed size,\nno cache growing with length',
            fontsize=8.6, color=MEM[1], linespacing=1.3)

    op(ax, 10.8, 4.6, '⊗', MEM, fs=13)
    ax.text(10.8, 5.0, 'unbind', ha='center', fontsize=8.5, color=MUTED)
    arrow(ax, (10.1, 4.6), (10.54, 4.6))
    box(ax, 9.6, 1.75, 2.4, 0.95, 'conjugate query\n$\\bar{k}_t$ of current key', MEM, fs=8.8)
    line(ax, [(5.4, 0.6), (10.8, 0.6), (10.8, 1.75)], color=INK)
    arrow(ax, (10.8, 2.7), (10.8, 4.34))

    box(ax, 11.6, 4.15, 2.45, 0.9, 'LN → $W_o$ → $\\times\\gamma$\n($\\gamma$ zero-init)', NEUTRAL, fs=9)
    arrow(ax, (11.06, 4.6), (11.6, 4.6))
    arrow(ax, (12.8, 4.15), (12.8, 3.55))
    ax.text(12.8, 3.25, 'added to the layer\'s\nprefix state', ha='center', va='top',
            fontsize=9, color=INK, linespacing=1.3)

    ax.text(W / 2, -0.15,
            '$M_t=\\lambda_t\\odot M_{t-1}+g_w\\,(k_{t-1}\\otimes v_t)$,   read $r_t=\\gamma\\,W_o\\,'
            '\\mathrm{LN}(\\bar{k}_t\\otimes M_t)$;   unit keys: $\\bar{k}\\otimes(k\\otimes v)=v$',
            ha='center', fontsize=9.5, color=INK)
    save(fig, 'arch_memory')


# --------------------------------------------------------------------------
def fig_decode():
    W, H = 15.4, 7.2
    fig, ax = canvas(W, H)
    ax.text(W / 2, H - 0.35, 'Incremental decoding: what appending byte 8 touches',
            ha='center', fontsize=14, weight='bold', color=INK)
    pos = tree_layout()
    new = {(0, 7), (1, 3), (2, 1), (3, 0)}
    draw_tree(ax, pos, lambda key: (NEW, 2.0) if key in new else (NEUTRAL, 1.0))
    for i in range(8):
        ax.text(pos[(0, i)][0], 0.3, f'byte {i + 1}', ha='center', fontsize=8.5,
                color=NEW[1] if i == 7 else MUTED)
    ax.text(10.6, 5.4, 'new nodes (green): the leaf plus\none ancestor per completed level\n'
            '→ ≤ log₂T + 1 composes per layer', fontsize=10, color=INK, linespacing=1.45)
    ax.text(10.6, 3.65, 'existing nodes never change\n(each covers a fixed span)',
            fontsize=10, color=INK, linespacing=1.45)
    ax.text(10.6, 1.75, 'cross-layer mixing is position-wise,\nso the next layer needs only\n'
            'the new position\'s state', fontsize=10, color=INK, linespacing=1.45)
    save(fig, 'arch_decode')


if __name__ == '__main__':
    fig_overview()
    fig_node()
    fig_fenwick()
    fig_memory()
    fig_decode()
