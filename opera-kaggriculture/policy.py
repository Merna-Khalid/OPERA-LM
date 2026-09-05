"""policy.py -- shared action-generation logic for the OPERA Kaggriculture
agent, used by both main.py (inference: argmax, force_harvest configurable
from model_config.json, default True) and train_rl.py (rollout collection:
sampling, force_harvest=False so the preference can be learned by reward
instead of hand-coded).

Grammar-constrained generation: forced structural tokens (slot markers,
MOVE_DELTA, NONE_ARG where the verb has no argument) interleaved with
selected content tokens (which verb, which item/qty, how many market
orders). kagri_common.decode_action's fuzz-safe fallback is a backstop,
not the primary correctness mechanism.

force_harvest is the ONE preference override left in here (force HARVEST
over WATER when both are legal on the same tile -- a live trace showed
the BC model systematically preferring WATER, tanking money). It is
architecturally different from every other restriction in this file:
tile-occupancy legality, seed-availability gating, and the fixed
MOVE_DELTA/NONE_ARG facts are all constraints on actions that were never
real choices (a guaranteed no-op is not a strategy). HARVEST-vs-WATER
was a real choice the model was making badly, so overriding it is
scripting strategy, not masking illegality -- hence it is the one thing
this module makes optional and the one thing RL fine-tuning targets.
"""
import numpy as np
import torch
import torch.nn.functional as F

import kagri_common as kc

_MOVE_VERBS = set(kc.FARMER_MOVES)
_ITEM_ARG_VERBS = {"PLANT", "PICKUP", "PLACE"}
_QTY_ARG_VERBS = {"PICKUP", "PLACE"}
_NO_ARG_MARKET_VERBS = {"HIRE", "BUY_LAND"}
_VERB_NAME = {tok: name for name, tok in kc.VERB.items()}
_MVERB_NAME = {tok: name for name, tok in kc.MVERB.items()}
_ARG1_ITEM_IDS = sorted(set(kc.ITEM.values()) | {kc.NONE_ARG})
_QTY_IDS = sorted(set(kc.QTY.values()) | {kc.NONE_ARG})
_MARKET_SLOT_IDS = sorted(set(kc.MVERB.values()) | {kc.ORDER_END})
_ITEM_IDS = sorted(kc.ITEM.values())

# Worst-case action-segment length (grammar upper bound, computed from
# the constants themselves so it can't drift out of sync): slot+verb+
# optional MOVE_DELTA+arg1+arg2 (<=5) per unit (farmer + MAX_HANDS
# hands), + MARKET slot + up to 10 orders * 3 tokens + ORDER_END, + EOT.
MAX_UNIT_TOKENS = 5
MAX_MARKET_TOKENS = 1 + 10 * 3 + 1
MAX_ACTION_TOKENS = (MAX_UNIT_TOKENS * (1 + kc.MAX_HANDS)
                    + MAX_MARKET_TOKENS + 1)
DEFAULT_CONTEXT_WINDOW = 256   # matches the training data's max_len


def select_token(logits, allowed_ids, mode):
    """(token_id, log_prob) restricted to allowed_ids. mode="argmax":
    deterministic (today's inference path), log_prob is None. mode=
    "sample": categorical sample over softmax(logits[allowed_ids]) --
    needed so RL rollouts actually explore instead of being as
    deterministic as inference -- with the log-prob under that same
    restricted distribution (what the policy-gradient update needs)."""
    allowed = torch.tensor(allowed_ids, dtype=torch.long)
    restricted = logits[allowed]
    if mode == "argmax":
        idx = int(torch.argmax(restricted).item())
        return int(allowed[idx].item()), None
    if mode == "sample":
        log_probs = F.log_softmax(restricted, dim=-1)
        idx = int(torch.multinomial(log_probs.exp(), 1).item())
        return int(allowed[idx].item()), float(log_probs[idx].item())
    raise ValueError(f"unknown mode {mode!r}")


def generate_action(dec, farm, day, seeds, use_geom, geom_block,
                    mode="argmax", force_harvest=True, record=None):
    """Generate one turn's action dict via dec.append (an OperaDecoder).

    record: optional list; if given, one entry
    {"local_pos": int, "allowed_ids": [...], "token": int,
    "log_prob": float|None} is appended per generated token, in
    generation order -- everything train_rl.py needs to later recompute
    log-probs with gradients under the exact same restricted support
    that was actually sampled from. "local_pos" is the index within
    THIS call's own generated tokens (0-based) -- generate_action only
    sees the action segment it itself produces, not the observation
    tokens or prior turns already fed to `dec` before this call, so the
    caller must add its own running window offset to get an absolute
    position before indexing into a forward() pass over the full window.

    Returns (action_dict, tokens, geoms): the full token/geom sequence
    generated this call (structural tokens included, not just the
    recorded choices) -- train_rl.py's rollout collector needs this to
    extend its own running window buffer to match exactly what `dec`
    was actually fed."""
    tiles = farm["tiles"]
    board_size = len(tiles)
    hands = farm.get("hands", [])
    num_hands = min(kc.MAX_HANDS, len(hands))
    positions = [tuple(farm["farmer"])] + [tuple(h) for h in hands[:num_hands]]
    tokens = []
    geoms = []

    def step(tok, geom=None):
        tokens.append(tok)
        geoms.append(geom)
        return dec.append(tok, geom=geom, geom_block=geom_block)

    def choose(logits, allowed_ids):
        tok, log_prob = select_token(logits, allowed_ids, mode)
        if record is not None:
            record.append({"local_pos": len(tokens), "allowed_ids": list(allowed_ids),
                          "token": tok, "log_prob": log_prob})
        return tok

    def gen_unit(slot_tok, pos):
        x, y = pos
        tile = (tiles[y][x] if 0 <= x < board_size and 0 <= y < board_size
               else "LOCKED")
        legal_names = kc.legal_unit_verbs(tile, day, pos, board_size, seeds=seeds)
        if force_harvest and "HARVEST" in legal_names:
            legal_names = {"HARVEST"}
        legal_ids = sorted(kc.VERB[n] for n in legal_names)
        logits = step(slot_tok)
        verb_tok = choose(logits, legal_ids)
        logits = step(verb_tok)
        verb = _VERB_NAME[verb_tok]
        if verb in _MOVE_VERBS and use_geom:
            dx, dy = kc.FARMER_MOVES[verb]
            geom = np.array([float(dx), float(dy), 0.0], dtype=np.float32)
            logits = step(kc.MOVE_DELTA, geom=geom)
        if verb == "PLANT":
            plantable = kc.legal_plant_crops(seeds)
            a1_ids = sorted(kc.ITEM[c] for c in plantable)
            a1 = choose(logits, a1_ids)
        elif verb in _ITEM_ARG_VERBS:
            a1 = choose(logits, _ARG1_ITEM_IDS)
        else:
            a1 = kc.NONE_ARG
        logits = step(a1)
        if verb in _QTY_ARG_VERBS:
            a2 = choose(logits, _QTY_IDS)
        else:
            a2 = kc.NONE_ARG
        return step(a2)

    logits = gen_unit(kc.FARMER, positions[0])
    for i in range(num_hands):
        logits = gen_unit(kc.HAND[i], positions[1 + i])

    logits = step(kc.MARKET)
    for _ in range(10):
        mv = choose(logits, _MARKET_SLOT_IDS)
        if mv == kc.ORDER_END:
            break
        logits = step(mv)
        verb = _MVERB_NAME[mv]
        if verb in _NO_ARG_MARKET_VERBS:
            item_tok, qty_tok = kc.NONE_ARG, kc.NONE_ARG
            logits = step(item_tok)
            logits = step(qty_tok)
        else:
            item_tok = choose(logits, _ITEM_IDS)
            logits = step(item_tok)
            qty_tok = choose(logits, _QTY_IDS)
            logits = step(qty_tok)
    if tokens[-1] != kc.ORDER_END:
        step(kc.ORDER_END)
    step(kc.EOT)

    return kc.decode_action(tokens, num_hands), tokens, geoms
