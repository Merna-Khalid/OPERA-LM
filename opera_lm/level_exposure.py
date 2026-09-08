"""Level-exposure diagnostic: which parts of the circuit does training
actually visit, and which parts does extrapolation ask for?

OPERA's compose function is weight-tied across every tree level and every
fold step, so "more parameters at depth" is not the question. The question
is whether the SHARED function is ever EVALUATED on the inputs that
long-context evaluation will hand it. Three axes are unexplored when you
train at T_train and evaluate at T_eval > T_train:

  1. TREE RECURSION DEPTH. build_tree stops at level floor(log2 T_train).
     Nodes at deeper levels are never built during training, so the
     compose function is never optimized on inputs that are summaries of
     that span size.
  2. FOLD CHAIN LENGTH. A prefix of length L folds popcount(L) blocks,
     i.e. popcount(L) - 1 sequential composes through one accumulator.
     max popcount over L <= T_train bounds the deepest accumulator chain
     training ever produces.
  3. LEVEL-INDEXED INPUTS. `level_sin_enc` (attend fold keys, multistate
     readout), `fold_theta` (fold_scale's angle = level * theta), and the
     per-slot `readout_gate[layer, :S]` are all indexed by level or by
     fold-slot number. Values beyond the trained range are defined but
     never fitted.

This is the OPERA-specific form of the unexplored-states hypothesis
(Buitrago Ruiz & Gu, arXiv:2507.02782): the failure mode is not "the
state drifts out of distribution" but "this circuit was never trained."

Pure combinatorics over the shipped `fenwick_blocks` -- no model, no
data, no GPU. Run:

    python -m opera_lm.level_exposure                    # Path A defaults
    python -m opera_lm.level_exposure --json out.json
    python -m opera_lm.level_exposure --train-len 256 --eval-len 2048
"""
import argparse
import json

from .model import fenwick_blocks

# curriculum_len lives in train.py, which imports the optional `datasets`
# extra via .data. Import it when that works so the two stay provably in
# sync; fall back to a local copy (train.py:81) otherwise.
try:
    from .train import curriculum_len
except ImportError:                                       # pragma: no cover
    def curriculum_len(step, cur0, every, max_len):
        return min(cur0 << (step // every), max_len)


def tree_nodes_per_level(T):
    """Nodes built by build_tree at each level, using its exact on-fly
    halving (model.py:1127): level 0 holds T nodes, each level after
    holds floor(prev / 2), and the loop stops once a level would hold
    fewer than 2. Returns {level: n_nodes}."""
    counts = {0: T}
    n = T
    lvl = 0
    while n >= 2:
        n = n // 2
        lvl += 1
        counts[lvl] = n
    return counts


def fold_exposure(T):
    """Walk every prefix L = 1..T and tally what the fold consumes.

    Returns a dict with:
      entries[k]     -- how many prefixes include a level-k block, i.e.
                        how many times a level-k block enters the fold
      referenced[k]  -- distinct level-k tree nodes any prefix reads
      slot_levels[s] -- {level: count} for fold slot index s (blocks
                        arrive largest-first, so slot 0 is the biggest)
      chain_hist[c]  -- how many prefixes fold c blocks (popcount(L))
      composes       -- total fold composes, sum(popcount(L) - 1)
    """
    table, max_blocks = fenwick_blocks(T)
    entries, referenced, slot_levels, chain_hist = {}, {}, {}, {}
    composes = 0
    for blocks in table:
        chain_hist[len(blocks)] = chain_hist.get(len(blocks), 0) + 1
        composes += len(blocks) - 1
        for s, (level, node) in enumerate(blocks):
            entries[level] = entries.get(level, 0) + 1
            referenced.setdefault(level, set()).add(node)
            slot_levels.setdefault(s, {})
            slot_levels[s][level] = slot_levels[s].get(level, 0) + 1
    return {
        'entries': entries,
        'referenced': {k: len(v) for k, v in referenced.items()},
        'slot_levels': slot_levels,
        'chain_hist': chain_hist,
        'composes': composes,
        'max_blocks': max_blocks,
    }


def curriculum_histogram(steps, cur0, every, max_len):
    """Steps spent at each training length T under arm C's schedule."""
    hist = {}
    for step in range(steps):
        T = curriculum_len(step, cur0, every, max_len)
        hist[T] = hist.get(T, 0) + 1
    return hist


def analyze(train_len, eval_len, steps, cur0, every, batch,
            well_trained_frac=0.01):
    """Per-level exposure over a full training schedule, and the gap
    between what training visits and what evaluation asks for.

    `well_trained_frac` separates "the level was built at all" from "the
    level got a meaningful share of the training signal". The top level
    of a tree over T_train spans the whole sequence, so it is reachable
    by exactly ONE prefix (L = T_train) while every level below it is
    reachable by ~T_train/2 prefixes -- a ~T/2-fold exposure deficit at
    the top. Counting it as "trained" would overstate what training
    covers, so both thresholds are reported.
    """
    cur_hist = curriculum_histogram(steps, cur0, every, train_len)

    # Aggregate fold entries per level over the whole run. One step at
    # length T contributes batch * entries_T[k] level-k fold entries.
    per_T = {T: fold_exposure(T) for T in cur_hist}
    tree_per_T = {T: tree_nodes_per_level(T) for T in cur_hist}

    train_entries, train_nodes = {}, {}
    for T, n_steps in cur_hist.items():
        w = n_steps * batch
        for k, c in per_T[T]['entries'].items():
            train_entries[k] = train_entries.get(k, 0) + w * c
        for k, c in tree_per_T[T].items():
            train_nodes[k] = train_nodes.get(k, 0) + w * c

    ev = fold_exposure(eval_len)
    ev_tree = tree_nodes_per_level(eval_len)

    max_trained_level = max(train_entries)
    max_trained_chain = max(
        max(per_T[T]['chain_hist']) for T in cur_hist)

    # Levels carrying a meaningful share of the signal, not just >0.
    tot_train_e = sum(train_entries.values())
    well = [k for k, c in train_entries.items()
            if c / tot_train_e >= well_trained_frac]
    max_well_trained_level = max(well)

    # Per-position view at the main training length -- far more legible
    # than run totals, and it exposes the top-level deficit directly.
    per_seq = {k: v / train_len
               for k, v in per_T[train_len]['entries'].items()}

    # Headline: how much of evaluation runs through untrained circuitry.
    table, _ = fenwick_blocks(eval_len)
    pos_touching, entries_untrained, deep_chain_pos = 0, 0, 0
    pos_touching_w, entries_untrained_w = 0, 0
    for blocks in table:
        deep = [b for b in blocks if b[0] > max_trained_level]
        deep_w = [b for b in blocks if b[0] > max_well_trained_level]
        if deep:
            pos_touching += 1
        if deep_w:
            pos_touching_w += 1
        entries_untrained += len(deep)
        entries_untrained_w += len(deep_w)
        if len(blocks) > max_trained_chain:
            deep_chain_pos += 1

    return {
        'config': {
            'train_len': train_len, 'eval_len': eval_len, 'steps': steps,
            'cur0': cur0, 'every': every, 'batch': batch,
        },
        'curriculum': cur_hist,
        'train_entries': train_entries,
        'train_nodes': train_nodes,
        'max_trained_level': max_trained_level,
        'max_well_trained_level': max_well_trained_level,
        'well_trained_frac': well_trained_frac,
        'entries_per_sequence': per_seq,
        'max_trained_chain': max_trained_chain,
        'eval': {
            'entries': ev['entries'],
            'referenced': ev['referenced'],
            'chain_hist': ev['chain_hist'],
            'composes': ev['composes'],
            'max_blocks': ev['max_blocks'],
            'tree_nodes': ev_tree,
            'max_level': max(ev_tree),
        },
        'gap': {
            'positions_touching_untrained_level': pos_touching,
            'positions_total': eval_len,
            'fold_entries_untrained_level': entries_untrained,
            'fold_entries_total': sum(ev['entries'].values()),
            'positions_beyond_trained_chain': deep_chain_pos,
            'positions_touching_undertrained_level': pos_touching_w,
            'fold_entries_undertrained_level': entries_untrained_w,
        },
    }


def _bar(frac, width=28):
    return '#' * max(0, min(width, round(frac * width)))


def report(a):
    cfg, gap = a['config'], a['gap']
    mtl, mtc = a['max_trained_level'], a['max_trained_chain']
    print('=' * 72)
    print('OPERA LEVEL-EXPOSURE DIAGNOSTIC')
    print(f"  train T={cfg['train_len']}  eval T={cfg['eval_len']}  "
          f"steps={cfg['steps']}  curriculum=({cfg['cur0']}, "
          f"{cfg['every']})  batch={cfg['batch']}")
    print('=' * 72)

    print('\n[1] CURRICULUM: steps spent at each training length')
    total = cfg['steps']
    for T in sorted(a['curriculum']):
        n = a['curriculum'][T]
        print(f"    T={T:<6} {n:>7} steps  {100*n/total:5.1f}%  "
              f"{_bar(n/total)}")

    print('\n[2] TRAINING EXPOSURE per tree level (whole run)')
    print('    level   span   fold entries    share   entries/position')
    tn, te = a['train_nodes'], a['train_entries']
    ps = a['entries_per_sequence']
    tot_e = sum(te.values())
    for k in sorted(tn):
        e = te.get(k, 0)
        share = e / tot_e if tot_e else 0.0
        print(f"    {k:>5}  {1<<k:>6}  {e:>14,}  {100*share:5.1f}%   "
              f"{ps.get(k, 0.0):8.2f}  {_bar(share, 18)}")
    mwl = a['max_well_trained_level']
    if mwl < mtl:
        deficit = ps.get(mwl, 0) / max(ps.get(mtl, 0), 1e-9)
        print(f"\n    NOTE: level {mtl} (span {1 << mtl} = the whole "
              f"training sequence) is reachable by")
        print(f"    exactly ONE prefix (L={cfg['train_len']}), so it sees "
              f"~{deficit:,.0f}x less signal than every")
        print(f"    level below it. Effective deepest WELL-trained level "
              f"is {mwl} (span {1 << mwl}),")
        print(f"    not {mtl} -- the top of the training tree is barely "
              f"trained either.")

    print(f"\n[3] EVALUATION at T={cfg['eval_len']} — per level")
    print('    level   span   tree nodes    fold entries   status')
    ev = a['eval']
    for k in sorted(ev['tree_nodes']):
        e = ev['entries'].get(k, 0)
        status = 'trained' if k <= mtl else '** NEVER TRAINED **'
        print(f"    {k:>5}  {1<<k:>6}   {ev['tree_nodes'][k]:>10,}  "
              f"{e:>13,}   {status}")

    print('\n[4] FOLD CHAIN LENGTH (sequential composes through one '
          'accumulator)')
    print(f"    deepest chain trained : {mtc - 1} composes "
          f"({mtc} blocks)")
    print(f"    deepest chain at eval : {ev['max_blocks'] - 1} composes "
          f"({ev['max_blocks']} blocks)")
    beyond = gap['positions_beyond_trained_chain']
    print(f"    eval positions folding deeper than ever trained: "
          f"{beyond:,} / {cfg['eval_len']:,} "
          f"({100*beyond/cfg['eval_len']:.1f}%)")

    print('\n[5] HEADLINE — how much of evaluation runs through '
          'untrained circuitry')
    pt, ptot = gap['positions_touching_untrained_level'], gap['positions_total']
    eu, etot = gap['fold_entries_untrained_level'], gap['fold_entries_total']
    print(f"    deepest level trained            : {mtl} "
          f"(span {1 << mtl})")
    print(f"    deepest level required at eval   : {ev['max_level']} "
          f"(span {1 << ev['max_level']})")
    print(f"    levels never built in training   : "
          f"{', '.join(str(k) for k in range(mtl + 1, ev['max_level'] + 1))}")
    print(f"    eval positions reading >=1 such block : {pt:,} / "
          f"{ptot:,}  ({100*pt/ptot:.1f}%)")
    print(f"    eval fold entries at such levels      : {eu:,} / "
          f"{etot:,}  ({100*eu/etot:.1f}%)")
    pw = gap['positions_touching_undertrained_level']
    ew = gap['fold_entries_undertrained_level']
    print(f"\n    counting from the deepest WELL-trained level "
          f"({mwl}, span {1 << mwl}):")
    print(f"    eval positions reading >=1 such block : {pw:,} / "
          f"{ptot:,}  ({100*pw/ptot:.1f}%)")
    print(f"    eval fold entries at such levels      : {ew:,} / "
          f"{etot:,}  ({100*ew/etot:.1f}%)")
    print('\n    Scope: this counts EXPOSURE, not learning. OPERA ties the')
    print('    compose weights across levels, so a level-12 node is a')
    print('    TRAINED function applied at an unseen recursion depth --')
    print('    not an untrained module. Under the default fold there are')
    print('    no level-indexed parameters at all, so nothing here is')
    print('    randomly initialised at eval. What the counts establish is')
    print('    that the INPUT REGIME at eval was never visited; whether')
    print('    the tied function generalises there is the empirical')
    print('    question, and the reason state passing is the instrument.')
    print('=' * 72)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-len', type=int, default=512)
    p.add_argument('--eval-len', type=int, default=8192)
    p.add_argument('--steps', type=int, default=15000)
    p.add_argument('--cur0', type=int, default=64)
    p.add_argument('--every', type=int, default=250)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--json', default=None, help='also write raw counts')
    args = p.parse_args()

    a = analyze(args.train_len, args.eval_len, args.steps, args.cur0,
                args.every, args.batch)
    report(a)
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(a, f, indent=2, default=str)
        print(f"\nwrote {args.json}")


if __name__ == '__main__':
    main()
