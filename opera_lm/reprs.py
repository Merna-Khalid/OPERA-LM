"""Matched input representations: word-level, byte-level BPE, raw bytes.

WHY THIS EXISTS
---------------
Two separate problems, one cause.

1. **Measurement.** The garden-path benchmark (`opera_lm.gp_benchmark`)
   skips 120 of 144 SAP ClassicGP stimuli as OOV, leaving 12 cells, and
   NP/Z is structurally unmeasurable: `load_data`'s word path runs
   `[w for w in text.lower().split() if w.isalpha()]`, which DELETES the
   comma that disambiguates NP/Z. At n=12 no arm can be distinguished
   from noise, so the input representation gates every downstream
   result.

2. **Research.** OPERA removes positional encoding; tokenization is the
   other imposed discretization. Byte-level models (BLT; H-Net;
   ByteFlow, arXiv:2603.03583) all *downsample* -- they must choose K
   chunk boundaries so an expensive backbone runs on a shorter sequence,
   and ByteFlow's ablation shows the choice matters a lot (coding-rate
   50.89% > word boundaries 49.38% > random 41.34%). OPERA does not
   downsample: it composes every dyadic span at every level and reads
   every prefix exactly, so it never chooses boundaries at all. Whether
   that survives contact with morphology -- OPERA's tree covers only
   intervals [i*2^k, (i+1)*2^k), so a word at bytes 5..11 is no single
   node -- is the open question this module exists to test.

WHAT IS HELD FIXED
------------------
All three modes stream the SAME articles in the same order, apply the
SAME article-level Bernoulli(0.9) split under `random.Random(42)`, and
cut with the SAME `doc_chunks` schedule. Only the article -> unit-stream
step differs. Reusing `data.doc_chunks` rather than re-deriving it keeps
the v8.5 split hygiene (article-level split BEFORE chunking) intact.

WHAT IS NOT COMPARABLE
----------------------
**BPB is meaningful only between `bpe` and `bytes`.** They model the
same raw UTF-8 string, so bits-per-byte converts between them exactly.
The `word` mode lowercases and drops every non-alphabetic token -- it
models a *different, easier string* -- so its loss cannot be converted
to BPB against raw text, and any such comparison would flatter it.
`word` is here as the historical incumbent for the GP benchmark, not as
a BPB arm. `corpus_stats()` reports raw-byte totals so the conversion is
explicit rather than assumed.

CONTENT MATCHING
----------------
Bytes are ~4-5x more units per word than word-level, so equal `max_len`
is NOT equal content. `suggested_max_len()` converts a word-level budget
into the byte/BPE budget covering the same text, measured from the
corpus rather than guessed.

    from opera_lm.reprs import build_corpus
    data, meta = build_corpus('bytes', max_len=1024, eval_max_len=4096)
    train(..., data=data, vocab_size=meta['vocab_size'])
"""
import pickle
import random
from collections import Counter

from .data import doc_chunks

# Byte mode: 0/1/2 reserved to match the word path's <pad>/<bos>/<eos>
# convention (pad MUST stay 0 -- make_batch_full zero-fills, and every
# loss mask assumes it). 256 byte values then occupy 3..258.
BYTE_OFFSET = 3
BYTE_VOCAB = 256 + BYTE_OFFSET
PAD, BOS, EOS = 0, 1, 2


def encode_bytes(text):
    """UTF-8 bytes -> ids in [3, 258]. Lossless for any str."""
    return [b + BYTE_OFFSET for b in text.encode('utf-8')]


def decode_bytes(ids):
    """Inverse of encode_bytes. Reserved ids are dropped; invalid UTF-8
    is replaced rather than raising, so partial/sampled sequences decode."""
    raw = bytes(i - BYTE_OFFSET for i in ids
                if BYTE_OFFSET <= i < BYTE_VOCAB)
    return raw.decode('utf-8', errors='replace')


def _load_articles(docs_limit, data_mode):
    from datasets import load_dataset
    if data_mode == 'docs-en':
        from itertools import islice
        return islice(load_dataset("wikimedia/wikipedia", "20231101.en",
                                   split="train", streaming=True),
                      docs_limit)
    return load_dataset("wikimedia/wikipedia", "20231101.simple",
                        split="train")


def _units(text, repr_mode, tok):
    """Article text -> unit stream. The ONLY step that differs by mode."""
    if repr_mode == 'word':
        # Verbatim from data.load_data so the word arm stays bitwise the
        # historical incumbent -- including the lossy lowercase+isalpha
        # filter that makes NP/Z unmeasurable.
        return [w for w in text.lower().split() if w.isalpha()]
    if repr_mode == 'bytes':
        return encode_bytes(text)
    if repr_mode == 'bpe':
        return tok.encode(text).ids
    raise ValueError(repr_mode)


def build_corpus(repr_mode, max_len, eval_max_len, docs_limit=100000,
                 data_mode='docs', word_vocab_size=10000,
                 tokenizer_path='opera-chat/tokenizer.json',
                 min_units=5, cache=None, verbose=True,
                 max_articles=None):
    """Returns ((train, test_short, test_long, vocab_size), meta).

    The 4-tuple is exactly what `opera_lm.train.train(data=...)` expects
    (train.py:531). `meta` carries vocab_size, raw-byte totals for BPB,
    and the decoder needed by the GP benchmark.

    `max_articles` caps the article stream. This is a MEMORY control,
    not a sampling choice: byte mode emits ~8.4x more units per article
    than word mode (measured units/byte 0.119 vs ~1.0), and the full
    Simple English corpus at byte granularity is ~250M Python ints,
    which does not fit in 16 GB. The cap is applied to the article
    stream BEFORE the split draw, so train/test proportions and the
    per-article split decisions are unchanged for a given cap -- and
    every representation must be built with the SAME cap to stay
    comparable.
    """
    if cache:
        try:
            with open(cache, 'rb') as f:
                data, meta = pickle.load(f)
            if verbose:
                print(f"  [reprs] loaded cache {cache}", flush=True)
            return data, meta
        except FileNotFoundError:
            pass

    tok = None
    if repr_mode == 'bpe':
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tokenizer_path)

    if verbose:
        print(f"  [reprs] repr={repr_mode} max_len={max_len} "
              f"eval_max_len={eval_max_len} mode={data_mode}", flush=True)

    cap = max(max_len, eval_max_len)
    # IDENTICAL to load_data: seeded article-level split BEFORE chunking.
    split_rng = random.Random(42)
    train_chunks, test_chunks = [], []
    raw_bytes_train = raw_bytes_test = 0
    n_articles = 0

    for article in _load_articles(docs_limit, data_mode):
        if max_articles is not None and n_articles >= max_articles:
            break
        text = article['text']
        units = _units(text, repr_mode, tok)
        if len(units) < min_units:
            continue
        n_articles += 1
        is_train = split_rng.random() < 0.9
        chunks = doc_chunks(units, max_len, cap,
                            long_first=(data_mode == 'docs-en'))
        if is_train:
            train_chunks.extend(chunks)
        else:
            test_chunks.extend(chunks)
        # Raw-byte accounting for BPB. Measured on the SAME text the
        # units came from, so the conversion is exact for bpe/bytes.
        nb = len(text.encode('utf-8'))
        if is_train:
            raw_bytes_train += nb
        else:
            raw_bytes_test += nb

    if repr_mode == 'word':
        counts = Counter()
        for c in train_chunks + test_chunks:
            counts.update(c)
        vocab = ['<pad>', '<unk>', '<bos>', '<eos>'] + [
            w for w, _ in counts.most_common(word_vocab_size - 4)]
        w2i = {w: i for i, w in enumerate(vocab)}
        train_chunks = [[w2i.get(w, 1) for w in c] for c in train_chunks]
        test_chunks = [[w2i.get(w, 1) for w in c] for c in test_chunks]
        vocab_size = len(vocab)
        idx2word = {i: w for w, i in w2i.items()}
    else:
        vocab_size = (BYTE_VOCAB if repr_mode == 'bytes'
                      else tok.get_vocab_size())
        idx2word = None

    random.Random(42).shuffle(train_chunks)
    train_data = [c for c in train_chunks if len(c) <= max_len]
    test_short = [c for c in test_chunks if len(c) <= max_len]
    test_long = [c for c in test_chunks if len(c) > max_len]

    # Units per raw byte -- the exact BPB conversion factor for this
    # representation. bytes ~= 1.0 (minus reserved-id overhead); BPE is
    # the compression ratio actually achieved on THIS corpus.
    tot_units = sum(len(c) for c in train_chunks)
    meta = {
        'repr_mode': repr_mode, 'vocab_size': vocab_size,
        'idx2word': idx2word, 'n_articles': n_articles,
        'raw_bytes_train': raw_bytes_train,
        'raw_bytes_test': raw_bytes_test,
        'units_train': tot_units,
        'units_per_byte': (tot_units / raw_bytes_train
                           if raw_bytes_train else None),
        'max_len': max_len, 'eval_max_len': eval_max_len,
        'bpb_comparable': repr_mode in ('bpe', 'bytes'),
    }
    if verbose:
        print(f"  [reprs] {n_articles} articles -> train {len(train_data)}, "
              f"test_short {len(test_short)}, test_long {len(test_long)}",
              flush=True)
        print(f"  [reprs] vocab {vocab_size}, units/byte "
              f"{meta['units_per_byte']}", flush=True)
        if not meta['bpb_comparable']:
            print("  [reprs] NOTE: word mode is lossy (lowercase+isalpha); "
                  "its loss is NOT BPB-comparable to bpe/bytes", flush=True)

    data = (train_data, test_short, test_long, vocab_size)
    if cache:
        with open(cache, 'wb') as f:
            pickle.dump((data, meta), f)
        if verbose:
            print(f"  [reprs] wrote cache {cache}", flush=True)
    return data, meta


def nats_to_bpb(mean_nats_per_unit, units_per_byte):
    """Convert a model's mean next-unit CE (nats/unit) to bits per byte.

    bpb = (nats/unit) * (units/byte) / ln 2

    Valid only where units and bytes describe the same string -- i.e.
    `bpe` and `bytes`, not `word` (see module docstring).
    """
    import math
    return mean_nats_per_unit * units_per_byte / math.log(2)


def suggested_max_len(word_max_len, units_per_byte_target,
                      units_per_byte_word):
    """Content-matched sequence budget. Equal max_len across
    representations is equal LENGTH, not equal CONTENT: bytes are ~4-5x
    more units per word. Scale from measured corpus ratios, not guesses,
    then round up to a power of two (the Fenwick index cache is keyed by
    T, and power-of-two stages keep shapes static within a stage --
    train.py:79)."""
    raw = word_max_len * units_per_byte_target / units_per_byte_word
    p = 1
    while p < raw:
        p *= 2
    return p
