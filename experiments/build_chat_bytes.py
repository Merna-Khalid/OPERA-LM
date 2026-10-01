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
    p.add_argument('--n-test', type=int, default=2000)
    p.add_argument('--limit', type=int, default=0, help='rows per split (testing)')
    args = p.parse_args()
    from datasets import load_dataset
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, f'chat_bytes_T{args.T}')

    stats = {}
    for split in ('train', 'test'):
        ds = load_dataset(args.dataset, split=split)
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        seqs, n_cut, n_drop = [], 0, 0
        for row in ds:
            ids, cut = encode_conversation(row['messages'], args.T)
            if ids is None:
                n_drop += 1
                continue
            n_cut += cut
            seqs.append(np.asarray(ids, dtype=np.uint16))
        stats[split] = dict(rows=len(ds), kept=len(seqs), cut=n_cut,
                            dropped=n_drop,
                            bytes=int(sum(len(s) for s in seqs)))
        print(split, stats[split], flush=True)
        if split == 'train':
            random.Random(0).shuffle(seqs)
            offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
            offsets[1:] = np.cumsum([len(s) for s in seqs])
            np.save(prefix + '.tokens.npy', np.concatenate(seqs))
            np.save(prefix + '.offsets.npy', offsets)
        else:
            test = random.Random(0).sample(seqs, min(args.n_test, len(seqs)))
            with open(prefix + '.test.pkl', 'wb') as f:
                pickle.dump({'test_short': [(0, s) for s in test],
                             'test_long': [], 'meta': stats}, f)
    meta = {'prefix': prefix, 'dataset': args.dataset, 'T': args.T,
            'n_seqs': stats['train']['kept'],
            'total_tokens': stats['train']['bytes'], 'splits': stats}
    json.dump(meta, open(prefix + '.meta.json', 'w'), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
