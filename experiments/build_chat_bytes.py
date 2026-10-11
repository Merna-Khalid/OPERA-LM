"""Byte-level chat SFT pool from HuggingFaceTB/smol-smoltalk (the SmolTalk
subset built for small models), in the opera_lm.packed format.

Chat format (bytes; the only non-byte id is EOS = 2, which ends every
assistant turn and is the generation stop token):

    <|system|>\\n{text}\\n            (if present)
    <|user|>\\n{text}\\n
    <|assistant|>\\n{text}<EOS>
    ... repeated per turn

A generation prompt is the formatted history followed by "<|assistant|>\\n"
(opera_lm.chat.format_prompt). Conversations longer than T bytes are cut
at a turn boundary, keeping the longest prefix that fits and ends with an
assistant turn; conversations whose first exchange does not fit are
dropped. The loss covers whole conversations (as in opera-chat).

  python experiments/build_chat_bytes.py --out-dir /content/data [--T 2048]
"""
import argparse
import json
import os
import pickle
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from opera_lm.chat import encode_conversation                # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', default='assets')
    p.add_argument('--T', type=int, default=2048)
    p.add_argument('--dataset', default='HuggingFaceTB/smol-smoltalk')
    p.add_argument('--extra-dataset', action='append', default=[],
                   help='additional dataset(s) with the same messages '
                        'schema, concatenated into this pool; exact '
                        'duplicate conversations are dropped by content '
                        'hash (the pools share upstream sources), e.g. '
                        '--extra-dataset HuggingFaceTB/smoltalk')
    p.add_argument('--name', default='',
                   help='pool-name suffix: chat_bytes_T{T}_{name} '
                        '(e.g. --name xl for the combined pool)')
    p.add_argument('--n-test', type=int, default=2000)
    p.add_argument('--limit', type=int, default=0, help='rows per split (testing)')
    args = p.parse_args()
    from datasets import load_dataset
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(
        args.out_dir, f'chat_bytes_T{args.T}' +
        (f'_{args.name}' if args.name else ''))
    datasets = [args.dataset] + args.extra_dataset

    import hashlib
    stats = {}
    for split in ('train', 'test'):
        seqs, n_cut, n_drop, n_dup = [], 0, 0, 0
        seen = set()
        per = []
        for dname in datasets:
            # 'name::config' pins a config explicitly (e.g.
            # HuggingFaceTB/smoltalk::all); without it a multi-config
            # dataset resolves to the hub default, which is not always
            # what you get on every datasets version
            dname, _, cfg = dname.partition('::')
            try:
                ds = (load_dataset(dname, cfg, split=split) if cfg
                      else load_dataset(dname, split=split))
            except Exception as e:
                if split == 'test':
                    # train-only sources (e.g. OpenHermes-2.5,
                    # tulu-3-sft-mixture ship no test split): the pool's
                    # held-out set comes from the sources that have one
                    print(f'{dname}: no test split -- skipped for test '
                          f'set (train split unaffected)', flush=True)
                    per.append(dict(dataset=dname, rows=0, kept=0,
                                    note='no test split'))
                    continue
                where = f'{dname} (config {cfg})' if cfg else dname
                print(f'FAILED loading {where} split={split}: {e!r}', flush=True)
                raise
            if args.limit:
                ds = ds.select(range(min(args.limit, len(ds))))
            kept0 = len(seqs)
            for row in ds:
                ids, cut = encode_conversation(row['messages'], args.T)
                if ids is None:
                    n_drop += 1
                    continue
                arr = np.asarray(ids, dtype=np.uint16)
                h = hashlib.blake2b(arr.tobytes(), digest_size=16).digest()
                if h in seen:
                    n_dup += 1
                    continue
                seen.add(h)
                n_cut += cut
                seqs.append(arr)
            per.append(dict(dataset=dname, rows=len(ds),
                            kept=len(seqs) - kept0))
        stats[split] = dict(kept=len(seqs), cut=n_cut, dropped=n_drop,
                            duplicates=n_dup,
                            bytes=int(sum(len(s) for s in seqs)),
                            per_dataset=per)
        print(split, stats[split], flush=True)
        if split == 'train':
            random.Random(0).shuffle(seqs)
            offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
            offsets[1:] = np.cumsum([len(s) for s in seqs])
            # stream-write the token array instead of np.concatenate: the
            # xl pool is ~4 GB and concatenating doubles the peak RSS --
            # an OOM kill on a standard (12.7 GB) Colab CPU runtime, mid-
            # build, leaving the pool in a partial state. open_memmap
            # writes the .npy to disk progressively; free the per-conv
            # list right after.
            tok_mm = np.lib.format.open_memmap(
                prefix + '.tokens.npy', mode='w+', dtype=np.uint16,
                shape=(int(offsets[-1]),))
            pos = 0
            for s in seqs:
                tok_mm[pos:pos + len(s)] = s
                pos += len(s)
            tok_mm.flush()
            del tok_mm
            np.save(prefix + '.offsets.npy', offsets)
        else:
            test = random.Random(0).sample(seqs, min(args.n_test, len(seqs)))
            with open(prefix + '.test.pkl', 'wb') as f:
                pickle.dump({'test_short': [(0, s) for s in test],
                             'test_long': [], 'meta': stats}, f)
    meta = {'prefix': prefix, 'datasets': [d.partition('::')[0] for d in datasets],
            'load_specs': datasets, 'T': args.T,
            'n_seqs': stats['train']['kept'],
            'total_tokens': stats['train']['bytes'], 'splits': stats}
    json.dump(meta, open(prefix + '.meta.json', 'w'), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
