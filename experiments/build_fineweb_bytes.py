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
- Fresh text after an earlier pool (--skip-docs N): the first N kept
  documents of the stream -- an earlier pool's meta n_docs -- are passed
  over, so the new pool shares no document with it (its training text or
  its held-out set); document ids stay global (the new pool starts at N).
  --expect-test <earlier .test.pkl> proves the skip lines up: while
  skipping, the earlier pool's seeded held-out draws are replayed, and
  every document it held out (> its T bytes) must reappear at the same
  index with identical bytes. Any mismatch (the stream changed) aborts
  before a single training byte is written.

  python experiments/build_fineweb_bytes.py --max-bytes 4e9 \
      --out-dir /content/data
  python experiments/build_fineweb_bytes.py --max-bytes 16e9 --T 2048 \
      --skip-docs <n_docs of the 8 GB pool> \
      --expect-test /content/data/fineweb_bytes_T1024_8GB.test.pkl --out-dir /content/data
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
    p.add_argument('--skip-docs', type=int, default=0,
                   help='pass over the first N kept documents (an earlier '
                        'pool\'s meta n_docs) so this pool is fresh text')
    p.add_argument('--expect-test', default=None,
                   help='the earlier pool\'s .test.pkl: verify the skipped '
                        'range is exactly that pool\'s stream (see docstring)')
    args = p.parse_args()
    from datasets import load_dataset
    T, cap, max_bytes = args.T, 4 * args.T, int(args.max_bytes)
    skip = args.skip_docs
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{max_bytes / 1e9:g}GB" + (f"_skip{skip}" if skip else '')
    prefix = os.path.join(args.out_dir, f'fineweb_bytes_T{T}_{tag}')

    expect, old_T = None, None
    if args.expect_test:
        assert skip, '--expect-test checks a --skip-docs range'
        et = pickle.load(open(args.expect_test, 'rb'))
        if et['meta'].get('config', args.config) != args.config:
            sys.exit(f"--expect-test pool used config {et['meta'].get('config')}, "
                     f"this build {args.config}")
        old_T = et['meta']['T']
        expect = {i: u for i, u in et['test_long'] if i < skip}

    ds = load_dataset('HuggingFaceFW/fineweb-edu', name=args.config,
                      split='train', streaming=True)
    split_rng = random.Random(42)
    replay_rng, replay_test, seen, bad = random.Random(42), 0, set(), 0
    offsets = [0]
    test_short, test_long = [], []
    pos = n_test_docs = 0
    n_docs = skipped = 0
    t0 = time.time()
    with open(prefix + '.tokens.npy', 'wb') as f:
        f.write(npy_header(0))
        for doc in ds:
            b = doc['text'].encode('utf-8')
            if len(b) < 5:
                continue
            if skipped < skip:
                # replay the earlier pool's split draw for this document
                # (same rng, same short-circuit) and check its held-out ones
                if (expect is not None and replay_test < args.test_docs
                        and replay_rng.random() < args.test_frac):
                    replay_test += 1
                    if len(b) > old_T:
                        seen.add(skipped)
                        u = np.frombuffer(b, dtype=np.uint8).astype(np.uint16) + BYTE_OFFSET
                        ref = expect.get(skipped)
                        bad += ref is None or not np.array_equal(u[:256], ref[:256])
                skipped += 1
                n_docs = skipped
                if skipped % 200000 == 0:
                    print(f"  skipped {skipped:,} / {skip:,} documents "
                          f"({time.time() - t0:.0f}s)", flush=True)
                if skipped == skip and expect is not None:
                    missing = set(expect) - seen
                    print(f"  skip check: {len(seen)} held-out documents of the earlier "
                          f"pool replayed, {bad} mismatched, {len(missing)} not found",
                          flush=True)
                    if bad or missing:
                        f.close()
                        os.remove(prefix + '.tokens.npy')
                        sys.exit('the stream no longer matches the earlier pool: the '
                                 'skipped range would not be the same documents. '
                                 'Aborted before writing any training data.')
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

    if skipped < skip:
        sys.exit(f'the stream ended after {skipped:,} documents, before the '
                 f'{skip:,} to skip')
    n_short = len(test_short)
    test_short = random.Random(0).sample(test_short, min(args.n_test, n_short))
    meta = {'prefix': prefix, 'config': args.config, 'n_docs': n_docs,
            'skip_docs': skip, 'skip_verified': bool(expect is not None),
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
