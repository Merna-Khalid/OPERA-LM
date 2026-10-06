"""BPE-token FineWeb-Edu pool for the transformer-baseline ladder (the
tokenizer-level counterpart of build_fineweb_bytes.py), in the same
opera_lm.packed mmap format, plus a held-out test set in the same
test.pkl shape.

Reuses the project's own tokenizer (opera-chat/tokenizer.json, vocab
16384, id 0 already reserved as <|pad|> -- no extra offset needed,
unlike the byte path's BYTE_OFFSET) instead of introducing a new one.
Same streaming/split/chunking protocol as build_fineweb_bytes.py, just
with `tok.encode(text).ids` in place of raw byte units, so the two
pools are the fairest possible match: same documents in the same
order, same held-out split, same chunk-then-cap convention. Chunks are
T TOKENS long here (not T bytes), which is a real difference in text
coverage per chunk -- expected and recorded via `units_per_byte` in
meta.json, the same BPB-conversion field build_corpus's 'bpe' mode
uses (opera_lm/reprs.py).

  python experiments/build_fineweb_tokens.py --max-bytes 4e9 \
      --out-dir /content/data
"""
import argparse
import json
import os
import pickle
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

HEADER = 128        # fixed .npy header size, patched with the final shape


def npy_header(n, dtype='<u2'):
    d = f"{{'descr': '{dtype}', 'fortran_order': False, 'shape': ({n},), }}"
    body = d.ljust(HEADER - 10 - 1) + '\n'
    return b'\x93NUMPY\x01\x00' + len(body).to_bytes(2, 'little') + body.encode('latin1')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--max-bytes', type=float, required=True,
                   help='raw UTF-8 bytes of SOURCE TEXT to consume (comparable '
                        'to build_fineweb_bytes.py --max-bytes); the resulting '
                        'token count is smaller by ~units_per_byte')
    p.add_argument('--out-dir', default='assets')
    p.add_argument('--config', default='sample-10BT')
    p.add_argument('--tokenizer-path', default='opera-chat/tokenizer.json')
    p.add_argument('--T', type=int, default=1024, help='chunk length in TOKENS')
    p.add_argument('--test-frac', type=float, default=0.005)
    p.add_argument('--test-docs', type=int, default=20000)
    p.add_argument('--n-test', type=int, default=5000)
    args = p.parse_args()
    from datasets import load_dataset
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tok.get_vocab_size()
    T, cap, max_bytes = args.T, 4 * args.T, int(args.max_bytes)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{max_bytes / 1e9:g}GB"
    prefix = os.path.join(args.out_dir, f'fineweb_tok_T{T}_{tag}')

    ds = load_dataset('HuggingFaceFW/fineweb-edu', name=args.config,
                      split='train', streaming=True)
    split_rng = random.Random(42)
    offsets = [0]
    test_short, test_long = [], []
    pos = n_docs = n_test_docs = 0
    raw_bytes_consumed = raw_bytes_train = 0
    t0 = time.time()
    with open(prefix + '.tokens.npy', 'wb') as f:
        f.write(npy_header(0))
        for doc in ds:
            text = doc['text']
            nb = len(text.encode('utf-8'))
            if nb < 5:
                continue
            units = np.asarray(tok.encode(text).ids, dtype=np.uint16)
            if len(units) < 5:
                continue
            raw_bytes_consumed += nb
            if n_test_docs < args.test_docs and split_rng.random() < args.test_frac:
                n_test_docs += 1
                test_short += [(n_docs, units[i:i + T]) for i in range(0, len(units), T)
                               if len(units) - i >= 5]
                if len(units) > T:
                    test_long.append((n_docs, units[:cap]))
                n_docs += 1
                if raw_bytes_consumed >= max_bytes:
                    break
                continue
            n_docs += 1
            raw_bytes_train += nb
            for i in range(0, len(units), T):
                c = units[i:i + T]
                if len(c) >= 5:
                    f.write(c.tobytes())
                    pos += len(c)
                    offsets.append(pos)
            if n_docs % 50000 == 0:
                el = time.time() - t0
                print(f"  {n_docs:,} docs, {pos / 1e6:.1f}M train tokens, "
                      f"{raw_bytes_consumed / 1e9:.2f}G raw bytes seen, "
                      f"{n_test_docs} test docs, "
                      f"{raw_bytes_consumed / el / 1e6:.1f} MB/s", flush=True)
            if raw_bytes_consumed >= max_bytes:
                break
        f.seek(0)
        f.write(npy_header(pos))
    np.save(prefix + '.offsets.npy', np.asarray(offsets, dtype=np.int64))

    arr = np.load(prefix + '.tokens.npy', mmap_mode='r')
    assert arr.shape == (pos,) and arr.dtype == np.uint16, 'bad npy header'
    head = np.asarray(arr[:min(pos, 1 << 20)])
    assert head.min() >= 0 and head.max() < vocab_size

    n_short = len(test_short)
    test_short = random.Random(0).sample(test_short, min(args.n_test, n_short))
    meta = {'prefix': prefix, 'config': args.config, 'n_docs': n_docs,
            'n_test_docs': n_test_docs, 'n_seqs': len(offsets) - 1,
            'total_tokens': pos, 'T': T, 'dtype': 'uint16',
            'vocab_size': vocab_size, 'tokenizer_path': args.tokenizer_path,
            'raw_bytes_train': raw_bytes_train,
            'units_per_byte': pos / raw_bytes_train if raw_bytes_train else None,
            'n_test_short_all': n_short, 'n_test_short': len(test_short),
            'n_test_long': len(test_long), 'minutes': (time.time() - t0) / 60}
    with open(prefix + '.test.pkl', 'wb') as g:
        pickle.dump({'test_short': test_short, 'test_long': test_long,
                     'meta': meta}, g)
    json.dump(meta, open(prefix + '.meta.json', 'w'), indent=2)
    print(json.dumps(meta, indent=2), flush=True)
    # same early-exit rationale as build_fineweb_bytes.py: everything is
    # already written and verified, so skip the streaming client's
    # interpreter-shutdown teardown (it raises on an early break)
    os._exit(0)


if __name__ == '__main__':
    main()
