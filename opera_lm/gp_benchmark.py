"""Garden-path benchmark: do internal trajectory signals separate
ambiguous from unambiguous sentences the way human reading times do?

THE DESIGN
----------
Stimuli: the SAP benchmark ClassicGP pivot file (caplabnyu/sapbenchmark,
MIT) -- 24 items x 3 constructions (NP/S, NP/Z, MV/RR) x {ambiguous,
unambiguous}, with the 0-indexed disambiguating-word position per
sentence. Human ground truth: the SAP N=2000 self-paced-reading
recollection's posterior mean garden-path effect (ambiguous RT minus
unambiguous RT, ms) per item and per construction, at the critical word
(ROI 0) and two spillover regions (ROI 1, 2). Headline construction
effects at the critical word: NP/Z +121ms > MV/RR +59ms > NP/S +45ms;
at spillover+1 the ordering flips: MV/RR +202ms > NP/Z +150ms > NP/S
+64ms. See assets/gp_stimuli/SOURCES.md for provenance.

For every sentence the model produces, at the disambiguating word:
surprisal, step, turn, extrap (exactly the alignment of
opera_lm.trajectory: arrival of word p -> surp[p-1], step[p-1],
turn[p-2], extrap[p-2]). The per-item EFFECT is metric(ambiguous) -
metric(unambiguous): the model's internal analogue of the human GPE.

THE QUESTIONS (stated before running)
-------------------------------------
1. MAGNITUDE/ORDERING: does each metric, per architecture, reproduce the
   human construction ordering (NPZ > MVRR > NPS at the critical word;
   MVRR > NPZ > NPS at spillover+1)?
2. ITEM-LEVEL TRACKING: correlating per-item model effects against the
   human per-item GPE (n=24 per construction, Fisher-z aggregated) --
   does direction-change (turn/extrap) track the human effect better
   than surprisal? That is the literature's claim (Barenholtz et al.
   2026, arXiv:2606.05346, on larger models) tested here on OPERA, its
   matched RoPE baseline, and Mamba.
3. STANDARDIZED SEPARATION: effect / SD of the metric across all
   stimulus sentences at that position -- which internal signal
   separates AMB from UAMB most cleanly, independent of units?

CAVEATS (honest)
----------------
* OPERA/RoPE are word-level with a 10k Simple-English vocabulary: items
  containing any OOV word are SKIPPED (an <unk> would confound the
  comparison entirely -- garden_path.py's rule), and coverage is
  reported. Word tokenization matches opera_lm.data: lowercase,
  alphabetic tokens only.
* NP/Z is UNMEASURABLE for the word-level arches and is dropped there:
  the NP/Z ambiguity is disambiguated by a COMMA ("...changed, the
  file..."), and punctuation-free tokenization collapses the
  ambiguous/unambiguous pair to an identical token sequence (observed:
  exactly zero effect on every metric). Only Mamba (BPE, punctuation
  intact) can run NP/Z.
* Mamba is BPE: surprisal for a word is SUMMED over its BPE tokens
  (standard practice), trajectory metrics are taken at the word's first
  token. Tokenization differs from the word-level models, so Mamba's
  numbers are a yardstick, not a matched control.
* n=24 items per construction is small (and smaller after OOV
  filtering); item-level correlations are descriptive. The construction
  ordering (3 points) is reported for what it is.
* Nothing trains; frozen checkpoints, inference only.

Usage:
    python -m opera_lm.gp_benchmark --arch opera
    python -m opera_lm.gp_benchmark --arch all --json runs_gp.json
"""
import argparse
import csv
import json
import math
import pickle
import re

import torch
import torch.nn.functional as F

from .garden_path import load_opera, fisher_mean, pearson
from .trajectory import (load_rope, load_mamba, trajectory_metrics,
                         OPERA_CKPT, ROPE_CKPT)

STIMULI = 'assets/gp_stimuli/sapbenchmark_items_ClassicGP.pivot.csv'
HUMAN_ITEM = 'assets/gp_stimuli/sapbenchmark_ClassicGP_GPE_effects_by_item.csv'
HUMAN_CONSTR = ('assets/gp_stimuli/'
                'sapbenchmark_ClassicGP_GPE_effects_by_construction.csv')

CONSTRUCTIONS = {'NPS': 'NP/S', 'NPZ': 'NP/Z', 'MVRR': 'MV/RR'}


def load_stimuli(path=STIMULI):
    """Pivot CSV -> list of dicts, one per sentence (144 rows)."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            constr = r['condition'].split('_')[0]
            rows.append(dict(
                item=int(r['item']), construction=constr,
                ambiguity=r['ambiguity'], sentence=r['Sentence'],
                disamb=int(r['disambPosition_0idx'])))
    return rows


def load_human():
    """Human GPE posterior means: per item (ROI 0,1) and per
    construction (ROI 0,1,2). Keys: (item, GPE_*, roi) and
    (GPE_*, roi)."""
    by_item, by_constr = {}, {}
    with open(HUMAN_ITEM) as f:
        for r in csv.DictReader(f):
            by_item[(int(r['item']), r['coef'], int(r['ROI']))] = \
                float(r['mean'])
    with open(HUMAN_CONSTR) as f:
        for r in csv.DictReader(f):
            by_constr[(r['coef'], int(r['ROI']))] = float(r['mean'])
    return by_item, by_constr


# ----------------------------------------------------------------------------
# Per-sentence scoring
# ----------------------------------------------------------------------------

def word_tokens(sentence):
    """opera_lm.data convention: lowercase, alphabetic tokens."""
    return re.findall(r"[a-z]+", sentence.lower())


@torch.no_grad()
def score_word_level(model, ids, arch):
    """One sentence -> (states [T,D], surprisal [T-1]). OPERA reads the
    prefix states straight off the model; RoPE hooks ln_f (same pattern
    as trajectory.collect_rope)."""
    lens = torch.full((1,), ids.shape[1], dtype=torch.long)
    if arch == 'opera':
        out = model(ids, lens, return_states=True)
        states, logits = out.states[-1][0], out.logits[-1][0]
    else:
        buf = []
        h = model.ln_f.register_forward_hook(
            lambda m, i, o: buf.append(o.detach()))
        try:
            logits = model(ids, lens)[0][0]
        finally:
            h.remove()
        states = buf[0][0]
    lp = F.log_softmax(logits.float(), dim=-1)
    sup = -lp[:-1].gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
    return states.float(), sup


def subword_encode(sentence, repr_mode, tok=None):
    """Sentence -> (ids, char_offsets) for a byte or BPE OPERA.

    Returns CHARACTER offsets so the same word-alignment logic as the
    Mamba path applies. Both modes see the sentence VERBATIM -- original
    case, punctuation intact -- which is the entire point: the word path
    (`word_tokens`) lowercases and drops non-alphabetic tokens, which is
    what deletes NP/Z's disambiguating comma and OOVs 120 of 144 stimuli.

    Byte mode: char offsets are derived per character from its UTF-8
    width, so multi-byte characters map every one of their bytes back to
    the source character. For ASCII stimuli this is the identity, but
    deriving it correctly costs nothing and keeps the function honest for
    non-ASCII text.
    """
    if repr_mode == 'bytes':
        from .reprs import BYTE_OFFSET
        ids, offs = [], []
        for i, ch in enumerate(sentence):
            for b in ch.encode('utf-8'):
                ids.append(b + BYTE_OFFSET)
                offs.append((i, i + 1))
        return ids, offs
    enc = tok.encode(sentence)
    return list(enc.ids), list(enc.offsets)


@torch.no_grad()
def score_opera_subword(model, sentence, repr_mode, tok=None):
    """One sentence -> (states [T,d], surprisal [T-1], offsets, n_tok)
    for a byte/BPE OPERA, matching score_mamba's return contract."""
    ids, offs = subword_encode(sentence, repr_mode, tok)
    t = torch.tensor([ids])
    lengths = torch.tensor([len(ids)])
    out = model(t, lengths, return_states=True)
    states = out.states[-1][0]
    lp = F.log_softmax(out.logits[-1].float(), dim=-1)
    sup = -lp[0, :-1].gather(-1, t[0, 1:].unsqueeze(-1)).squeeze(-1)
    return states.float(), sup, offs, len(ids)


@torch.no_grad()
def score_mamba(model, tok, sentence):
    """One sentence -> (states [T,d], per-token surprisal [T-1], offsets).
    Fast tokenizer gives char offsets for word<->BPE alignment."""
    enc = tok(sentence, return_tensors='pt', return_offsets_mapping=True)
    ids = enc.input_ids
    offsets = enc.offset_mapping[0].tolist()
    out = model(ids, output_hidden_states=True)
    states = model.backbone.norm_f(out.hidden_states[-1])[0]
    lp = F.log_softmax(out.logits.float(), dim=-1)
    sup = -lp[0, :-1].gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
    return states.float(), sup, offsets, ids.shape[1]


def metrics_at(states, p):
    """Trajectory metrics at the arrival of word/token index p.
    Returns (step, turn, extrap) or None if p is too early/late.
    Alignment: surp[p-1], step[p-1], turn[p-2], extrap[p-2]."""
    if p < 2 or p > states.shape[0] - 1:
        return None
    m = trajectory_metrics(states.unsqueeze(0))
    return (m['step'][0, p - 1].item(), m['turn'][0, p - 2].item(),
            m['extrap'][0, p - 2].item())


# ----------------------------------------------------------------------------
# Benchmark driver
# ----------------------------------------------------------------------------

@torch.no_grad()
def run_benchmark(arch, rows, w2i=None, i2w=None, model=None, tok=None,
                  repr_mode=None):
    """Per-sentence metrics at the disambiguating word. Returns a list of
    result rows; OOV items (word-level arches) are skipped and counted.

    `arch='opera-sub'` scores a byte or BPE OPERA (`repr_mode`), which
    sees the sentence verbatim -- so it is NOT subject to the word-level
    path's two coverage losses (OOV filtering and the NP/Z comma) and
    shares the subword offset-alignment path with Mamba."""
    out_rows, skipped = [], 0
    for r in rows:
        if arch in ('opera', 'rope') and r['construction'] == 'NPZ':
            # NP/Z is disambiguated by a comma; punctuation-free
            # tokenization collapses the pair to identical tokens.
            # 'opera-sub' is deliberately NOT in this guard: measuring
            # NP/Z is the reason the subword path exists.
            skipped += 1
            continue
        p = r['disamb']
        if arch in ('opera', 'rope'):
            words = word_tokens(r['sentence'])
            if p >= len(words):
                skipped += 1
                continue
            if any(w not in w2i for w in words):
                skipped += 1
                continue
            ids = torch.tensor([[w2i[w] for w in words]])
            states, sup = score_word_level(model, ids, arch)
            mt = metrics_at(states, p)
            if mt is None or p - 1 >= sup.shape[0]:
                skipped += 1
                continue
            rec = dict(surp=sup[p - 1].item(), step=mt[0], turn=mt[1],
                       extrap=mt[2])
            # spillover +1
            mt1 = metrics_at(states, p + 1)
            if mt1 is not None and p < sup.shape[0]:
                rec.update(surp_sp1=sup[p].item(), step_sp1=mt1[0],
                           turn_sp1=mt1[1], extrap_sp1=mt1[2])
        else:
            # Shared subword path: Mamba (HF tokenizer) and byte/BPE
            # OPERA both return (states, surprisal, char offsets), so the
            # word<->subword alignment below is identical for both.
            if arch == 'mamba':
                states, sup, offsets, ntok = score_mamba(model, tok,
                                                         r['sentence'])
            else:
                states, sup, offsets, ntok = score_opera_subword(
                    model, r['sentence'], repr_mode, tok)
            # char span of the p-th alphabetic word in the sentence
            spans = [(m.start(), m.end()) for m in
                     re.finditer(r"[A-Za-z]+", r['sentence'])]
            if p >= len(spans):
                skipped += 1
                continue
            w0, w1 = spans[p]
            toks_in_word = [j for j, (a, b) in enumerate(offsets)
                            if a < w1 and b > w0 and j >= 1]
            if not toks_in_word:
                skipped += 1
                continue
            j0 = toks_in_word[0]
            # surprisal of the word = sum over its BPE tokens; token j's
            # surprisal is sup[j-1]
            wsup = sum(sup[j - 1].item() for j in toks_in_word
                       if j - 1 < sup.shape[0])
            mt = metrics_at(states, j0)
            if mt is None:
                skipped += 1
                continue
            rec = dict(surp=wsup, step=mt[0], turn=mt[1], extrap=mt[2])
            nxt = max(toks_in_word) + 1
            mt1 = metrics_at(states, nxt)
            if mt1 is not None and nxt - 1 < sup.shape[0]:
                rec.update(surp_sp1=sup[nxt - 1].item(), step_sp1=mt1[0],
                           turn_sp1=mt1[1], extrap_sp1=mt1[2])
        out_rows.append(dict(item=r['item'], construction=r['construction'],
                             ambiguity=r['ambiguity'], **rec))
    return out_rows, skipped


def analyze(rows, by_item, by_constr, arch, skipped):
    """The three questions from the module docstring."""
    metrics = ['surp', 'step', 'turn', 'extrap']
    # effect per (item, construction, metric): amb - unamb
    cells = {}
    for r in rows:
        key = (r['item'], r['construction'])
        cells.setdefault(key, {})[r['ambiguity']] = r
    effects = []        # (item, construction, metric -> effect)
    for (item, constr), d in sorted(cells.items()):
        if 'ambiguous' not in d or 'unambiguous' not in d:
            continue
        e = {'item': item, 'construction': constr}
        for m in metrics:
            e[m] = d['ambiguous'][m] - d['unambiguous'][m]
            if f'{m}_sp1' in d['ambiguous'] and f'{m}_sp1' in d['unambiguous']:
                e[f'{m}_sp1'] = (d['ambiguous'][f'{m}_sp1']
                                 - d['unambiguous'][f'{m}_sp1'])
        effects.append(e)

    print(f"\n  [{arch}] {len(rows)} sentences scored, {skipped} skipped "
          f"(OOV/alignment), {len(effects)} amb/unamb cells")

    # Q3: standardized separation (effect / pooled SD of the metric)
    print(f"\n  Q3. standardized GP effect (amb - unamb, in units of the "
          f"metric's SD across stimulus sentences)")
    print(f"    {'metric':<7} {'crit d':>8} {'spill+1 d':>9}")
    std_sep = {}
    for m in metrics:
        vals = [r[m] for r in rows]
        sd = max(float(torch.tensor(vals).std()), 1e-12)
        ds = [e[m] for e in effects]
        d0 = sum(ds) / len(ds) / sd
        line = f"    {m:<7} {d0:>8.3f}"
        rec = {'critical': d0}
        sp = [e[f'{m}_sp1'] for e in effects if f'{m}_sp1' in e]
        if sp:
            vals1 = [r[f'{m}_sp1'] for r in rows if f'{m}_sp1' in r]
            sd1 = max(float(torch.tensor(vals1).std()), 1e-12)
            rec['spillover1'] = sum(sp) / len(sp) / sd1
            line += f" {rec['spillover1']:>9.3f}"
        std_sep[m] = rec
        print(line)

    # Q1: construction ordering vs human
    print(f"\n  Q1. construction ordering at the critical word "
          f"(human: NPZ +121 > MVRR +59 > NPS +45 ms)")
    ordering = {}
    for m in metrics:
        per_c = {}
        for c in CONSTRUCTIONS:
            es = [e[m] for e in effects if e['construction'] == c]
            if es:
                per_c[c] = sum(es) / len(es)
        ordering[m] = per_c
        human_c = {c: by_constr.get((f'GPE_{c}', 0), float('nan'))
                   for c in CONSTRUCTIONS}
        r_h = pearson([per_c[c] for c in CONSTRUCTIONS if c in per_c],
                      [human_c[c] for c in CONSTRUCTIONS if c in per_c]) \
            if len(per_c) == 3 else float('nan')
        print(f"    {m:<7} " +
              '  '.join(f"{c} {per_c.get(c, float('nan')):+.3f}"
                        for c in CONSTRUCTIONS) +
              f"   | r vs human ordering (3 pts): {r_h:+.2f}")

    # Q2: item-level tracking of the human GPE (Fisher-z over
    # constructions)
    print(f"\n  Q2. per-item effect vs human GPE (n<=24 per construction, "
          f"Fisher-z aggregated)")
    item_r = {}
    for m in metrics:
        rs = []
        for c in CONSTRUCTIONS:
            xs, ys = [], []
            for e in effects:
                if e['construction'] != c:
                    continue
                h = by_item.get((e['item'], f'GPE_{c}', 0))
                if h is not None:
                    xs.append(e[m])
                    ys.append(h)
            if len(xs) >= 4:
                rs.append(pearson(xs, ys))
        agg, k = fisher_mean(rs)
        item_r[m] = dict(agg_r=agg, n_constructions=k)
        print(f"    {m:<7} agg r = {agg:+.3f}  (over {k} constructions)")

    return dict(n_sentences=len(rows), skipped=skipped,
                n_cells=len(effects), standardized=std_sep,
                construction_effects=ordering, item_tracking=item_r)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arch',
                   choices=('opera', 'opera-sub', 'rope', 'mamba', 'all'),
                   default='all')
    # --arch opera-sub: byte/BPE OPERA. Full stimulus coverage including
    # NP/Z, since the sentence is scored verbatim.
    p.add_argument('--repr-mode', choices=('bytes', 'bpe'), default='bytes')
    p.add_argument('--tokenizer', default='opera-chat/tokenizer.json')
    p.add_argument('--vocab-size', type=int, default=259)
    p.add_argument('--json', default=None)
    # arm-checkpoint config for --arch opera (see trajectory.py)
    p.add_argument('--ckpt', default=None)
    p.add_argument('--d', type=int, default=640)
    p.add_argument('--nb', type=int, default=160)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--rot-mode', default='so3')
    p.add_argument('--homeo', default='off')
    p.add_argument('--node-paths', type=int, default=3)
    p.add_argument('--fold-adapt', default='off')
    args = p.parse_args()

    rows = load_stimuli()
    by_item, by_constr = load_human()
    print('=' * 74)
    print('GARDEN-PATH BENCHMARK — SAP ClassicGP, 24 items x 3 constructions')
    print('=' * 74)
    print('  frozen checkpoints, inference only; human GPE from the SAP '
          'N=2000 SPR recollection')

    tr, ts, tl, vocab, w2i, i2w = pickle.load(
        open('assets/vocab_docs.pkl', 'rb'))

    archs = ('opera', 'rope', 'mamba') if args.arch == 'all' \
        else (args.arch,)
    results = {}
    for arch in archs:
        if arch == 'opera-sub':
            # Byte or BPE OPERA: full stimulus coverage, NP/Z included.
            sub_tok = None
            if args.repr_mode == 'bpe':
                from tokenizers import Tokenizer
                sub_tok = Tokenizer.from_file(args.tokenizer)
            model = load_opera(args.ckpt, d=args.d, nb=args.nb,
                               layers=args.layers, rot_mode=args.rot_mode,
                               homeo_mode=args.homeo,
                               node_paths=args.node_paths,
                               fold_adapt=args.fold_adapt,
                               vocab_size=args.vocab_size)
            scored, skipped = run_benchmark(arch, rows, model=model,
                                            tok=sub_tok,
                                            repr_mode=args.repr_mode)
        elif arch == 'opera':
            model = load_opera(args.ckpt or OPERA_CKPT, d=args.d,
                               nb=args.nb, layers=args.layers,
                               rot_mode=args.rot_mode,
                               homeo_mode=args.homeo,
                               node_paths=args.node_paths,
                               fold_adapt=args.fold_adapt)
            scored, skipped = run_benchmark(arch, rows, w2i=w2i,
                                            model=model)
        elif arch == 'rope':
            model = load_rope(ROPE_CKPT)
            scored, skipped = run_benchmark(arch, rows, w2i=w2i,
                                            model=model)
        else:
            model, tok = load_mamba()
            scored, skipped = run_benchmark(arch, rows, model=model,
                                            tok=tok)
        results[arch] = analyze(scored, by_item, by_constr, arch, skipped)
        results[arch]['sentences'] = scored
    print('=' * 74)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        print(f"wrote {args.json}")


if __name__ == '__main__':
    main()
