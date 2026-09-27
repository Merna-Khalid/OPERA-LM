"""Byte-level FineWeb-Edu pool for the large-model runs (Recall research
§8c), in the opera_lm.packed mmap format, plus a held-out test set.

- Stream: HuggingFaceFW/fineweb-edu, config sample-10BT, in dataset
  order (deterministic), until --max-bytes training bytes.
- Split: per document, random.Random(42) draws; a document is held out
  with probability --test-frac until --test-docs are collected, and never
  enters the pool.
- Training chunks: each training document is cut into consecutive
  1,024-byte chunks (trailing chunk kept if >= 5 bytes). Unlike the
  Simple-Wikipedia schedule (doc_chunks, which sends every chunk after
  the eighth to evaluation), no training byte is dropped -- FineWeb
  documents are often longer than 8 KB.
- Test chunks: held-out documents are cut exactly like training ones
  (consecutive T-byte chunks); a seeded sample of --n-test of them is
  the in-length test set. Separately, every held-out document longer
  than T contributes its opening min(len, 4T) bytes to the
  extrapolation set (a different test set, so the overlap is harmless).
- Storage: tokens as uint16 (byte ids 3..258), streamed to disk -- RAM
  stays flat at any size. <prefix>.tokens.npy / .offsets.npy /
  .test.pkl / .meta.json.

  python experiments/build_fineweb_bytes.py --max-bytes 4e9 \
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
from opera_lm.reprs import BYTE_OFFSET                      # noqa: E402

HEADER = 128        # fixed .npy header size, patched with the final shape


def npy_header(n, dtype='<u2'):
    d = f"{{'descr': '{dtype}', 'fortran_order': False, 'shape': ({n},), }}"
    body = d.ljust(HEADER - 10 - 1) + '\n'
    return b'\x93NUMPY\x01\x00' + len(body).to_bytes(2, 'little') + body.encode('latin1')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-bytes', type=float, required=True)
    p.add_argument('--out-dir', default='assets')
    p.add_argument('--config', default='sample-10BT')
    p.add_argument('--T', type=int, default=1024)
    p.add_argument('--test-frac', type=float, default=0.005)
    p.add_argument('--test-docs', type=int, default=20000)
    p.add_argument('--n-test', type=int, default=5000)
    args = p.parse_args()
    from datasets import load_dataset
    T, cap, max_bytes = args.T, 4 * args.T, int(args.max_bytes)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{max_bytes / 1e9:g}GB"
    prefix = os.path.join(args.out_dir, f'fineweb_bytes_T{T}_{tag}')

    ds = load_dataset('HuggingFaceFW/fineweb-edu', name=args.config,
                      split='train', streaming=True)
    split_rng = random.Random(42)
    offsets = [0]
    test_short, test_long = [], []
    pos = n_docs = n_test_docs = 0
    t0 = time.time()
    with open(prefix + '.tokens.npy', 'wb') as f:
        f.write(npy_header(0))
        for doc in ds:
            b = doc['text'].encode('utf-8')
            if len(b) < 5:
                continue
            units = np.frombuffer(b, dtype=np.uint8).astype(np.uint16) + BYTE_OFFSET
            if n_test_docs < args.test_docs and split_rng.random() < args.test_frac:
                n_test_docs += 1
                test_short += [(n_docs, units[i:i + T]) for i in range(0, len(units), T)
                               if len(units) - i >= 5]            # uint16 arrays
                if len(units) > T:
                    test_long.append((n_docs, units[:cap]))
                n_docs += 1
                continue
            n_docs += 1
            for i in range(0, len(units), T):
                c = units[i:i + T]
                if len(c) >= 5:
                    f.write(c.tobytes())
                    pos += len(c)
                    offsets.append(pos)
            if n_docs % 50000 == 0:
                el = time.time() - t0
                print(f"  {n_docs:,} docs, {pos / 1e9:.2f}G train bytes, "
                      f"{n_test_docs} test docs, {pos / el / 1e6:.1f} MB/s", flush=True)
            if pos >= max_bytes:
                break
        f.seek(0)
        f.write(npy_header(pos))
    np.save(prefix + '.offsets.npy', np.asarray(offsets, dtype=np.int64))

    tok = np.load(prefix + '.tokens.npy', mmap_mode='r')
    assert tok.shape == (pos,) and tok.dtype == np.uint16, 'bad npy header'
    head = np.asarray(tok[:min(pos, 1 << 20)])
    assert head.min() >= BYTE_OFFSET and head.max() < BYTE_OFFSET + 256

    n_short = len(test_short)
    test_short = random.Random(0).sample(test_short, min(args.n_test, n_short))
    meta = {'prefix': prefix, 'config': args.config, 'n_docs': n_docs,
            'n_test_docs': n_test_docs, 'n_seqs': len(offsets) - 1,
            'total_tokens': pos, 'T': T, 'dtype': 'uint16',
            'n_test_short_all': n_short, 'n_test_short': len(test_short),
            'n_test_long': len(test_long), 'minutes': (time.time() - t0) / 60}
    with open(prefix + '.test.pkl', 'wb') as g:
        pickle.dump({'test_short': test_short, 'test_long': test_long,
                     'meta': meta}, g)
    json.dump(meta, open(prefix + '.meta.json', 'w'), indent=2)
    print(json.dumps(meta, indent=2), flush=True)
    # the streaming client's prefetch thread raises on interpreter
    # shutdown after an early break ("client has been closed"); all
    # files are written and verified by now, so skip that teardown
    os._exit(0)


if __name__ == '__main__':
    main()
