"""Pack the FULL Simple English Wikipedia byte corpus into an mmap pool
(opera_lm.packed format) for scale-up runs.

build_corpus (opera_lm/reprs.py) keeps every chunk as a Python int list,
which caps byte mode at 20,000 articles on 16 GB (~44M training bytes).
Here each article's chunks go straight into an int32 array, so the whole
corpus fits in a few hundred MB.

COMPARABILITY: the article stream, the min_units filter, the seeded
per-article split draw (random.Random(42), one draw per kept article)
and the chunk schedule are IDENTICAL to build_corpus. So for the first
20,000 kept articles every split decision equals the a20000 corpus:
its test articles never enter this training pool, and runs trained on
the pool are evaluated on the unchanged a20000 test sets
(repr_study --packed).

POOL TEST SET (<prefix>.test.pkl): the a20000 test set covers only the
first 20,000 articles, which are longer than the rest of the corpus
(mean train sequence 820 vs 613 bytes). So the test-split chunks of ALL
kept articles are collected and a seeded uniform sample of n_test
in-length chunks is stored (plus every chunk longer than T, for the
extrapolation buckets), each tagged with its article index so the
first-20k articles can be separated out. Evaluate with
experiments/eval_pool.py.

    python experiments/build_packed.py [--max-articles 0 (= all)]
"""
import argparse
import json
import os
import pickle
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from opera_lm.data import doc_chunks                       # noqa: E402
from opera_lm.reprs import _load_articles, _units          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-articles', type=int, default=0, help='0 = all')
    p.add_argument('--T', type=int, default=1024)
    p.add_argument('--eval-mult', type=int, default=4)
    p.add_argument('--min-units', type=int, default=5)
    p.add_argument('--n-test', type=int, default=5000)
    args = p.parse_args()
    T, cap = args.T, args.T * args.eval_mult
    tag = f"a{args.max_articles}" if args.max_articles else "all"
    prefix = os.path.join(ROOT, 'assets', f'packed_bytes_T{T}_{tag}')

    split_rng = random.Random(42)
    buf = np.empty(1 << 28, dtype=np.int32)       # grown on demand
    offsets = [0]
    pos = n_articles = raw_bytes_train = 0
    test_short, test_long = [], []                # (article_idx, chunk)
    for article in _load_articles(100000000, 'docs'):
        if args.max_articles and n_articles >= args.max_articles:
            break
        text = article['text']
        units = _units(text, 'bytes', None)
        if len(units) < args.min_units:
            continue
        n_articles += 1
        if not split_rng.random() < 0.9:           # test article
            for c in doc_chunks(units, T, cap):
                (test_long if len(c) > T else test_short).append((n_articles - 1, c))
            continue
        raw_bytes_train += len(text.encode('utf-8'))
        for c in doc_chunks(units, T, cap):
            if len(c) > T:                         # build_corpus: long -> eval only
                continue
            if pos + len(c) > buf.size:
                buf = np.resize(buf, buf.size * 2)
            buf[pos:pos + len(c)] = c
            pos += len(c)
            offsets.append(pos)
        if n_articles % 20000 == 0:
            print(f"  {n_articles:,} articles, {pos / 1e6:.1f}M train bytes", flush=True)
    np.save(prefix + '.tokens.npy', buf[:pos])
    np.save(prefix + '.offsets.npy', np.asarray(offsets, dtype=np.int64))
    meta = {'prefix': prefix, 'n_articles': n_articles, 'n_seqs': len(offsets) - 1,
            'total_tokens': int(pos), 'raw_bytes_train': raw_bytes_train,
            'units_per_byte': pos / raw_bytes_train, 'T': T}
    json.dump(meta, open(prefix + '.meta.json', 'w'), indent=2)
    print(json.dumps(meta, indent=2))

    n_short = len(test_short)
    test_short = random.Random(0).sample(test_short, min(args.n_test, n_short))
    tmeta = {'n_test_articles': len({a for a, _ in test_short + test_long}),
             'n_short_all': n_short, 'n_short': len(test_short),
             'n_long': len(test_long), 'sample_seed': 0, 'T': T, 'cap': cap,
             'n_articles': n_articles}
    with open(prefix + '.test.pkl', 'wb') as f:
        pickle.dump({'test_short': test_short, 'test_long': test_long,
                     'meta': tmeta}, f)
    print(json.dumps(tmeta, indent=2))


if __name__ == '__main__':
    main()
