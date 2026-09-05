"""prepare_kagri_data.py -- tokenize generate_selfplay.py's raw (obs,
action) logs into windowed, model-ready chunks (Phase 3 of the plan).

Each chunk is turn-aligned (never splits a turn's observation+action
across a window boundary -- OPERA has no positional encoding, so a
window is meaningful as long as the LOCAL structure within it is
intact; splitting a turn would leave a dangling action segment with no
observation context). Split is by EPISODE, not by chunk, so consecutive
chunks of one episode never straddle train/test (same hygiene principle
as opera-chat/prepare_fineweb.py's per-document split).

test_long uses the SAME held-out episodes as test_short but windowed at
--eval-max-len instead of --max-len, mirroring opera-chat/
prepare_data.py's train/test_short/test_long contract (an extrapolation
check downstream, if one is ever run here).

Usage:
  python prepare_kagri_data.py --logs selfplay_logs.pkl --out kagri_data.pkl
"""
import argparse
import pickle
import random

import numpy as np

import kagri_common as kc


def _episode_to_turn_tokens(turns, use_geom):
    """[(obs, action), ...] -> list of (tokens, geoms, mask) per turn,
    one call to encode_turn per turn (kept separate, not yet
    concatenated, so windowing can cut cleanly between turns)."""
    out = []
    for obs, action in turns:
        tokens, geoms, mask = kc.encode_turn(obs, action, use_geom=use_geom)
        out.append((tokens, geoms, mask))
    return out


def _window_episode(turn_tokens, max_len):
    """Turn-aligned, non-overlapping windows of at most max_len tokens.
    A single turn longer than max_len (shouldn't happen at this vocab's
    scale, but not impossible with many hired hands) is hard-truncated
    on its own, logged, and kept as its own window."""
    chunks = []
    cur_tok, cur_geom, cur_mask = [], [], []

    def _flush():
        if cur_tok:
            chunks.append((list(cur_tok), list(cur_geom), list(cur_mask)))

    for tokens, geoms, mask in turn_tokens:
        if len(tokens) > max_len:
            print(f"WARNING: single turn has {len(tokens)} tokens > "
                  f"max_len={max_len}; truncating that turn alone", flush=True)
            tokens, geoms, mask = (tokens[:max_len], geoms[:max_len],
                                   mask[:max_len])
        if len(cur_tok) + len(tokens) > max_len:
            _flush()
            cur_tok, cur_geom, cur_mask = [], [], []
        cur_tok.extend(tokens)
        cur_geom.extend(geoms)
        cur_mask.extend(mask)
    _flush()
    return chunks


def _pack_chunk(tokens, geoms, mask):
    """(tokens, geoms(list[np.ndarray|None]), mask) -> the dense arrays
    opera_lm.model.forward's geom/geom_mask kwargs expect directly."""
    T = len(tokens)
    geom_arr = np.zeros((T, 3), dtype=np.float32)
    geom_mask = np.zeros((T,), dtype=bool)
    for i, g in enumerate(geoms):
        if g is not None:
            geom_arr[i] = g
            geom_mask[i] = True
    return {
        "tokens": np.array(tokens, dtype=np.int64),
        "geom": geom_arr,
        "geom_mask": geom_mask,
        "action_mask": np.array(mask, dtype=bool),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="selfplay_logs.pkl")
    p.add_argument("--out", default="kagri_data.pkl")
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--eval-max-len", type=int, default=512)
    p.add_argument("--test-frac", type=float, default=0.1,
                   help="fraction of EPISODES held out for test_short/"
                        "test_long combined")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-geom", action="store_true",
                   help="Phase 4b ablation: encode positions as plain "
                        "discrete POS_X_<n>/POS_Y_<n> tokens instead of "
                        "injected geometry (see kagri_common's use_geom). "
                        "Train a model on each of the two resulting pkls "
                        "and compare -- that comparison IS the "
                        "geometric-vs-token-blind hypothesis test.")
    a = p.parse_args()
    use_geom = not a.no_geom

    with open(a.logs, "rb") as f:
        episodes = pickle.load(f)
    print(f"loaded {len(episodes)} episodes from {a.logs}", flush=True)

    rng = random.Random(a.seed)
    idx = list(range(len(episodes)))
    rng.shuffle(idx)
    n_test = max(1, int(round(len(idx) * a.test_frac)))
    test_idx = set(idx[:n_test])
    train_idx = [i for i in idx if i not in test_idx]

    def _build(indices, max_len):
        out = []
        for i in indices:
            turn_tokens = _episode_to_turn_tokens(episodes[i]["turns"], use_geom)
            for tokens, geoms, mask in _window_episode(turn_tokens, max_len):
                out.append(_pack_chunk(tokens, geoms, mask))
        return out

    train = _build(train_idx, a.max_len)
    test_short = _build(list(test_idx), a.max_len)
    test_long = [c for c in _build(list(test_idx), a.eval_max_len)
                if len(c["tokens"]) > a.max_len]

    assert train and test_short and test_long, \
        "train/test_short/test_long must all be non-empty -- generate " \
        "more episodes or lower --test-frac/--eval-max-len"

    print(f"train chunks: {len(train)}  test_short: {len(test_short)}  "
          f"test_long: {len(test_long)}", flush=True)
    print(f"train tokens: {sum(len(c['tokens']) for c in train)}", flush=True)

    bundle = {
        "vocab_size": kc.VOCAB_SIZE,
        "geom_block": kc.GEOM_BLOCK,
        "use_geom": use_geom,
        "max_len": a.max_len,
        "eval_max_len": a.eval_max_len,
        "train": train,
        "test_short": test_short,
        "test_long": test_long,
    }
    with open(a.out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
