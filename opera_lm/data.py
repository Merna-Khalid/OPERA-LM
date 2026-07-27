"""Data loading for OPERA language-model training.

`datasets` (HuggingFace) is imported lazily inside `load_data`, so the
package imports fine without it installed.
"""
import random
from collections import Counter

# ============================================================================
# DATA
# ============================================================================

def doc_chunk_sizes(train_max_len, cap, long_first=False):
    """Deterministic chunk-length schedule. Default ('docs', historical,
    UNCHANGED for comparability): 8 train-length chunks, then one chunk
    per bucket length. long_first ('docs-en', opt6): bucket-length chunks
    FIRST, descending. The historical order requires an article longer
    than 8*train_max_len before it produces its FIRST long chunk -- at
    train 1024 that is >8192 words, which almost no article has (observed
    test_long n=29). Long-first means EVERY article of length X > L
    contributes its opening chunk (length min(X, cap)) directly to the
    bucket containing it: bucket population follows the article-length
    distribution instead of requiring 8k-word articles."""
    if long_first:
        ks = list(range(cap // train_max_len, 1, -1))     # e.g. [4, 3, 2]
        return [k * train_max_len for k in ks] + [train_max_len] * 8
    sizes = [train_max_len] * 8
    k = 2
    while k * train_max_len <= cap:
        sizes.append(k * train_max_len)
        k += 1
    return sizes


def doc_chunks(words, train_max_len, cap, long_first=False):
    """Cut one article's word stream into consecutive chunks following the
    schedule. Deterministic (no RNG). Trailing chunk kept if >= 5 words."""
    sizes = doc_chunk_sizes(train_max_len, cap, long_first=long_first)
    out = []
    i = 0
    si = 0
    n = len(words)
    while i < n:
        L = sizes[si % len(sizes)]
        si += 1
        chunk = words[i:i + L]
        i += L
        if len(chunk) >= 5:
            out.append(chunk)
    return out


def load_data(vocab_size=10000, train_max_len=40, eval_max_len=80,
              data_mode='sentences', docs_limit=100000):
    print(f"Loading {'FULL English' if data_mode == 'docs-en' else 'Simple English'}"
          f" Wikipedia ({data_mode})...", flush=True)
    from datasets import load_dataset
    if data_mode == 'docs-en':
        # v8.4 SCALE DATA: full English Wikipedia, STREAMED, first
        # docs_limit articles (deterministic order -> deterministic
        # split). Median enwiki article is far longer than Simple's
        # ~1024-word ceiling, so extrapolation buckets up to eval_max_len
        # are powered BY CONSTRUCTION at train_max_len=1024 -- the regime
        # Simple could not populate (observed n=9/4/1/0 above 1024).
        from itertools import islice
        ds = islice(load_dataset("wikimedia/wikipedia", "20231101.en",
                                 split="train", streaming=True), docs_limit)
    else:
        ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
    sentences = []
    cap = max(train_max_len, eval_max_len)
    if data_mode in ('docs', 'docs-en'):
        # v8.0: whole-article word streams, chunked on the deterministic
        # schedule. Per-token filter (keep alphabetic words) instead of the
        # sentence mode's per-sentence filter, so streams stay long.
        # v8.5 SPLIT HYGIENE: article-level 90/10 split BEFORE chunking.
        # Previously chunks from all articles were pooled and split at the
        # chunk level, so consecutive chunks of the same article landed in
        # both train and test -- inflating in-length PPL and especially the
        # extrapolation buckets (a long test chunk could be the direct
        # continuation of train chunks from the same article). The Bernoulli
        # draw per article keeps memory flat for streamed docs-en.
        split_rng = random.Random(42)
        train_chunks, test_chunks = [], []
        for article in ds:
            words = [w for w in article['text'].lower().split() if w.isalpha()]
            if len(words) < 5:
                continue
            tgt = train_chunks if split_rng.random() < 0.9 else test_chunks
            tgt.extend(doc_chunks(words, train_max_len, cap,
                                  long_first=(data_mode == 'docs-en')))
        sentences = train_chunks + test_chunks
    else:
        for article in ds:
            text = article['text'].lower()
            for sent in text.split('.'):
                sent = sent.strip()
                if not sent:
                    continue
                words = sent.split()
                if 5 <= len(words) <= cap:
                    if all(w.isalpha() or w in '.,;:!?-"' for w in words):
                        sentences.append(words)
    print(f"  {len(sentences)} sequences (5-{cap} words)", flush=True)

    word_counts = Counter()
    for s in sentences:
        word_counts.update(s)
    vocab = ['<pad>', '<unk>', '<bos>', '<eos>'] + [w for w, _ in word_counts.most_common(vocab_size - 4)]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    data = [[word2idx.get(w, 1) for w in s] for s in sentences]
    if data_mode in ('docs', 'docs-en'):
        # Split already happened at the article level above; just bucket.
        n_train = len(train_chunks)
        train_all, test_all = data[:n_train], data[n_train:]
        random.Random(42).shuffle(train_all)
    else:
        random.seed(42)
        random.shuffle(data)
        split = int(0.9 * len(data))
        train_all, test_all = data[:split], data[split:]

    train_data = [s for s in train_all if len(s) <= train_max_len]
    test_short = [s for s in test_all if len(s) <= train_max_len]
    test_long = [s for s in test_all if len(s) > train_max_len]
    print(f"  train (<= {train_max_len}): {len(train_data)}", flush=True)
    print(f"  test  (<= {train_max_len}): {len(test_short)}", flush=True)
    print(f"  test  ({train_max_len+1}-{eval_max_len}): {len(test_long)}", flush=True)
    return train_data, test_short, test_long, vocab, word2idx, idx2word
