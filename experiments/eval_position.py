"""Per-position held-out loss past the training length (the length-
generalization figure).

Every arm is scored on the SAME held-out documents: those at least
--eval-len bytes long, each cut to its first --eval-len bytes, so every
position is averaged over an identical document set (no selection drift
between short and long positions). Loss at position p is the next-byte
loss given p bytes of context; positions are binned and reported in bits
per byte, with the training length marked.

OPERA arms are rebuilt from repr_summary.json (experiments/arm_utils.py);
byte-level transformer arms (train_tf_pool.py rows, model='transformer')
are rebuilt from their summary row. Tokenized arms are refused: their
positions are tokens, not bytes.

  python experiments/eval_position.py --arms b2m_hmem128_gated_long b2_long \\
      --test-pkl assets/packed_bytes_T1024_all.test.pkl --eval-len 4096 \\
      --train-len 1024 --device mps

Writes <out>.json (per-position sums, binned BPB, band means) and
<out>.png; --out defaults to $OPERA_RUNS/position_curve.
"""
import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'opera-chat'))
from arm_utils import OUT, load_arm                            # noqa: E402


def load_transformer(name, row, device):
    """A train_tf_pool.py arm from its summary row and final.pt."""
    from opera_transformer_baseline_v2 import TransformerBaseline
    if abs(row.get('units_per_byte', 1.0) - 1.0) > 1e-6:
        sys.exit(f'{name}: tokenized arm (units_per_byte '
                 f'{row["units_per_byte"]:.3f}); per-byte positions are '
                 f'undefined for it')
    m = TransformerBaseline(row['vocab_size'], d=row['d'], nheads=row['nb'],
                            num_layers=row['layers'], pe_mode=row['pe'],
                            max_pe_len=min(row['max_len'] * 4, 2048),
                            qk_norm=row.get('qk_norm', False),
                            attn_logit_cap=row.get('attn_logit_cap', 0.0))
    sd = torch.load(os.path.join(row['ckpt_dir'], 'final.pt'),
                    map_location='cpu', weights_only=True)
    m.load_state_dict(sd, strict=True)
    if row['pe'] == 'learned':
        print(f'  WARNING {name}: learned positions are clamped past '
              f'{m.pos_emb.num_embeddings}', flush=True)
    return m.to(device).eval()


def load_model(name, summ, device):
    row = summ.get(name)
    if row is None:
        sys.exit(f'{name}: not in repr_summary.json (unfinished or misnamed)')
    if row.get('model') == 'transformer':
        return load_transformer(name, row, device), 'transformer'
    m, _ = load_arm(name, device=device)
    return m, 'opera'


def logits_of(model, kind, tok):
    lengths = torch.full((tok.shape[0],), tok.shape[1], device=tok.device)
    if kind == 'opera':
        return model(tok, lengths, head_last_only=True).logits[-1]
    return model(tok, lengths)[-1]


@torch.no_grad()
def position_nll(model, kind, docs, batch, device):
    """Summed next-byte NLL (nats) per position over docs [N, L]:
    entry p is the loss of predicting byte p+1 from bytes 0..p."""
    N, L = docs.shape
    tot = torch.zeros(L - 1, dtype=torch.float64)
    for i in range(0, N, batch):
        tok = torch.from_numpy(docs[i:i + batch]).to(device)
        lg = logits_of(model, kind, tok)[:, :-1].float()
        nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                              tok[:, 1:].reshape(-1), reduction='none')
        # sum on device in fp32 (MPS has no float64), accumulate in fp64
        tot += nll.reshape(tok.shape[0], L - 1).sum(0).cpu().double()
    return tot


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arms', nargs='+', required=True)
    p.add_argument('--test-pkl', required=True,
                   help='held-out set with test_short/test_long lists of '
                        '(len, ids); documents >= --eval-len are used')
    p.add_argument('--eval-len', type=int, default=4096)
    p.add_argument('--train-len', type=int, default=1024)
    p.add_argument('--max-docs', type=int, default=0, help='0 = all')
    p.add_argument('--min-alpha', type=float, default=0.0,
                   help='keep only documents whose first --eval-len bytes are '
                        'at least this fraction ASCII letters/spaces (prose). '
                        'The Simple-Wiki pool\'s 4k-byte held-out chunks are '
                        'mostly wiki tables (minor-planet lists); 0.8 keeps '
                        'the prose. 0 = no filter')
    p.add_argument('--bin', type=int, default=128, help='bytes per plotted bin')
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'mps' if torch.backends.mps.is_available() else 'cpu')
    p.add_argument('--summary', default=os.path.join(OUT, 'repr_summary.json'))
    p.add_argument('--labels', nargs='*', default=None,
                   help='legend labels, one per arm (default: arm names)')
    p.add_argument('--out', default=os.path.join(OUT, 'position_curve'))
    a = p.parse_args()

    tp = pickle.load(open(a.test_pkl, 'rb'))
    docs = [np.asarray(ids[:a.eval_len], dtype=np.int64)
            for _, ids in tp.get('test_long', []) + tp.get('test_short', [])
            if len(ids) >= a.eval_len]
    n_long = len(docs)
    if a.min_alpha > 0:
        prose = set(range(65 + 3, 91 + 3)) | set(range(97 + 3, 123 + 3)) | {32 + 3}
        docs = [d for d in docs
                if np.isin(d, list(prose)).mean() >= a.min_alpha]
    if a.max_docs:
        docs = docs[:a.max_docs]
    if not docs:
        sys.exit(f'no held-out document reaches {a.eval_len} bytes in {a.test_pkl}'
                 + (f' with --min-alpha {a.min_alpha}' if a.min_alpha else ''))
    docs = np.stack(docs)
    print(f'{len(docs)} held-out documents x {a.eval_len} bytes '
          f'({docs.size / 1e6:.2f}M predictions per arm; {n_long} reach '
          f'{a.eval_len} bytes, min-alpha {a.min_alpha}), device {a.device}',
          flush=True)

    summ = json.load(open(a.summary))
    labels = a.labels or a.arms
    assert len(labels) == len(a.arms), '--labels needs one entry per arm'
    LN2 = math.log(2)
    res = {}
    for name, label in zip(a.arms, labels):
        t0 = time.time()
        model, kind = load_model(name, summ, a.device)
        tot = position_nll(model, kind, docs, a.batch, a.device)
        bpb_pos = (tot / len(docs) / LN2).numpy()          # position p -> p+1 bytes of context
        ctx = np.arange(1, a.eval_len)                      # context length per entry
        edges = np.arange(0, a.eval_len, a.bin)
        bins = [(int(lo), float(bpb_pos[(ctx > lo) & (ctx <= lo + a.bin)].mean()))
                for lo in edges]
        bands = {}
        lo = 0
        hi = a.train_len
        while lo < a.eval_len - 1:
            sel = (ctx > lo) & (ctx <= min(hi, a.eval_len - 1))
            bands[f'{lo + 1}-{min(hi, a.eval_len)}'] = float(bpb_pos[sel].mean())
            lo, hi = hi, hi * 2
        res[name] = dict(label=label, kind=kind, bins=bins, bands=bands,
                         bpb_per_position=bpb_pos.tolist())
        inlen = bands[f'1-{a.train_len}']
        print(f'{name} ({kind}): ' + '  '.join(
            f'{k}: {v:.4f} ({v / inlen:.3f}x)' for k, v in bands.items())
            + f'   [{time.time() - t0:.0f}s]', flush=True)
        del model
        if a.device == 'cuda':
            torch.cuda.empty_cache()

    meta = dict(test_pkl=a.test_pkl, n_docs=len(docs), n_reaching=n_long,
                min_alpha=a.min_alpha, eval_len=a.eval_len,
                train_len=a.train_len, bin=a.bin,
                note='same documents at every position; bands are mean BPB '
                     'over context-length ranges; x = ratio to 1-train_len')
    json.dump(dict(meta=meta, arms=res), open(a.out + '.json', 'w'))
    print('wrote', a.out + '.json')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for name in a.arms:
        r = res[name]
        xs = [lo + a.bin / 2 for lo, _ in r['bins']]
        ax.plot(xs, [v for _, v in r['bins']], lw=1.8, label=r['label'])
    ax.axvline(a.train_len, color='0.4', ls='--', lw=1)
    ax.text(a.train_len, ax.get_ylim()[1], ' training length', va='top',
            fontsize=9, color='0.3')
    ax.set_xlabel('bytes of context')
    ax.set_ylabel('held-out loss (bits per byte)')
    ax.set_title(f'Per-position loss beyond the training length '
                 f'({len(docs)} documents, {a.eval_len} bytes each)', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out + '.png', dpi=180)
    print('wrote', a.out + '.png')


if __name__ == '__main__':
    main()
