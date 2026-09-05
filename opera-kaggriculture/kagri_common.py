"""kagri_common.py -- vocab + turn<->token encoding for the OPERA
Kaggriculture agent. Mirrors chat_common.py's role for the chat pipeline:
this is the one place the token format is defined, imported by every
other script in opera-kaggriculture/.

Design (see /Users/mernahafez/.claude/plans/polished-foraging-canyon.md):
OBSERVATION/ACTION content is tokenized (coarse log-magnitude buckets for
context numerics like money/price/inventory; exact small integers where
decode must reconstruct precisely, e.g. order quantities). GRID POSITIONS
(farmer/hand coordinates, movement deltas) are the one deliberate
exception: they are injected as literal 3-vectors into a leaf's
paravector block via opera_lm.model.inject_geometry, not tokenized --
that's the whole point of this experiment (does OPERA's rotor
composition reduce the "spatial blindness" ordinary token-id positions
cause).

Game constants (CROPS/ANIMALS/PRODUCTS/SHOPS/FARMER_MOVES/...) are
copied from kaggle_environments/envs/kaggriculture/kaggriculture.py
rather than imported, so training-time code here has no runtime
dependency on kaggle_environments being installed -- only self-play
generation (generate_selfplay.py) and live testing (main.py's local
checklist) need that package.
"""
import math
import os
import sys

import numpy as np


def ensure_opera_lm():
    """Make `import opera_lm` work without installing the package: put
    the repo root (parent of this file's opera-kaggriculture/ dir, which
    directly contains the opera_lm/ package) on sys.path."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


# ============================================================================
# Game constants (copied from kaggriculture.py -- keep in sync with the
# installed kaggle_environments version if the rules ever change).
# ============================================================================

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
ANIMAL_STRUCTURE = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}
# first_yield_day per crop (kaggriculture.py's CROPS table): a one-time
# crop's yield_units is set to 1 the INSTANT it's planted (_new_plant),
# but HARVEST is still blocked by the engine until this many days pass.
CROPS_FIRST_YIELD_DAY = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8,
                        "STRAWBERRY": 10, "MELON": 10}
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
           "EGG", "MILK", "WOOL", "FERTILIZER"]
SHOPS = ["BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
        "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET"]
QUADRANTS = ["NW", "NE", "SW", "SE"]
# (dx, dy); y grows downward -- matches kaggriculture.py's FARMER_MOVES.
FARMER_MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
MAX_YIELD_UNITS = 6            # max_held/max_yield across crops+animals; caps YIELD_<n>

UNIT_VERBS = ["PASS", "NORTH", "SOUTH", "EAST", "WEST",
             "PICKUP", "DROP", "PLACE", "PLANT", "WATER", "HARVEST",
             "FERTILIZE", "FEED", "CARE", "COLLECT_FERTILIZER",
             "BUILD_COOP", "BUILD_PASTURE", "DIG"]
MARKET_VERBS = ["BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "SELL",
               "HIRE", "BUY_LAND"]
ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
        "GOOSE", "COW", "SHEEP", "EGG", "MILK", "WOOL", "FERTILIZER"]
QTY_VALUES = list(range(1, 21)) + [50, 100]     # exact, invertible

MAX_HANDS = 4                  # supported hired hands per turn (v1 scope)
MAX_DAY = 40                   # clamp ceiling for the DAY_<n> token family
N_MAG_BUCKETS = 20             # generic log2-magnitude bucket family
MAX_HIRES = 8                  # clamp ceiling for HIRES_<n>
GEOM_BLOCK = 0                 # paravector block index geometry is injected into
MAX_BOARD = 16                  # ceiling for the POS_X_<n>/POS_Y_<n> fallback

# ============================================================================
# Vocab construction
# ============================================================================

_names = ["<pad>"]


def _add(name):
    assert name not in _names, f"duplicate token {name!r}"
    _names.append(name)
    return len(_names) - 1


PAD = 0
TURN_START = _add("TURN_START")
FARMER = _add("FARMER")
HAND = [_add(f"HAND{i}") for i in range(MAX_HANDS)]
MARKET = _add("MARKET")
ORDER_END = _add("ORDER_END")
EOT = _add("EOT")
NONE_ARG = _add("NONE_ARG")
FARMER_POS = _add("FARMER_POS")
HAND_POS = _add("HAND_POS")
MOVE_DELTA = _add("MOVE_DELTA")
NEIGHBORS = _add("NEIGHBORS")
# Phase 4b ablation fallback (use_geom=False): a coordinate becomes two
# EXACT discrete tokens instead of an injected vector -- NOT the same
# as just dropping FARMER_POS/HAND_POS, which would leave an identical
# marker token regardless of position (equivalent to "no position
# information" rather than "position as an arbitrary token id", biasing
# the ablation). This is the fair, "ordinary LLM" baseline the plan's
# hypothesis is measured against.
POS_X = {n: _add(f"POS_X_{n}") for n in range(MAX_BOARD)}
POS_Y = {n: _add(f"POS_Y_{n}") for n in range(MAX_BOARD)}

VERB = {v: _add(f"V_{v}") for v in UNIT_VERBS}
MVERB = {v: _add(f"M_{v}") for v in MARKET_VERBS}
ITEM = {it: _add(f"I_{it}") for it in ITEMS}
QTY = {n: _add(f"QTY_{n}") for n in QTY_VALUES}

TILE_EMPTY = _add("TILE_EMPTY")
TILE_LOCKED = _add("TILE_LOCKED")
TILE_WEED = _add("TILE_WEED")
TILE_OOB = _add("TILE_OOB")
TILE_COOP_EMPTY = _add("TILE_COOP_EMPTY")
TILE_PASTURE_EMPTY = _add("TILE_PASTURE_EMPTY")
TILE_PLANT = {c: _add(f"TILE_PLANT_{c}") for c in CROPS}
TILE_ANIMAL = {a: _add(f"TILE_ANIMAL_{a}") for a in ANIMALS}

WATERED_YES, WATERED_NO = _add("WATERED_YES"), _add("WATERED_NO")
FERT_ACTIVE_YES, FERT_ACTIVE_NO = _add("FERT_ACTIVE_YES"), _add("FERT_ACTIVE_NO")
FED_YES, FED_NO = _add("FED_YES"), _add("FED_NO")
CARED_YES, CARED_NO = _add("CARED_YES"), _add("CARED_NO")
FERTAVAIL_YES, FERTAVAIL_NO = _add("FERTAVAIL_YES"), _add("FERTAVAIL_NO")
YIELD = {n: _add(f"YIELD_{n}") for n in range(MAX_YIELD_UNITS + 1)}

DAY = {n: _add(f"DAY_{n}") for n in range(MAX_DAY)}
HOUR = {n: _add(f"HOUR_{n}") for n in range(24)}
MAG = {n: _add(f"MAG_{n}") for n in range(N_MAG_BUCKETS)}
HIRES = {n: _add(f"HIRES_{n}") for n in range(MAX_HIRES + 1)}

QUAD_OWNED = {q: _add(f"QUAD_{q}_OWNED") for q in QUADRANTS}
QUAD_LOCKED = {q: _add(f"QUAD_{q}_LOCKED") for q in QUADRANTS}
SHOP = {s: _add(f"SHOP_{s}") for s in SHOPS}

ID2TOK = list(_names)
TOK2ID = {name: i for i, name in enumerate(_names)}
VOCAB_SIZE = len(_names)


# ============================================================================
# Numeric helpers
# ============================================================================

def norm_coord(c, n):
    """Grid coordinate c in [0, n-1] -> [-1, 1], the scale OPERA's tree
    states actually live in (tanh(x)+0.1x bounded regime)."""
    if n <= 1:
        return 0.0
    return 2.0 * c / (n - 1) - 1.0


def mag_bucket(x):
    """Generic log2-magnitude bucket for a nonnegative context numeric
    (money, price, inventory/shed level): lossy on purpose -- it's
    context, not something decode must reconstruct exactly."""
    x = max(0.0, float(x))
    b = int(math.log2(x + 1.0))
    return max(0, min(N_MAG_BUCKETS - 1, b))


def clamp_qty(n):
    """Nearest QTY_* token value <= n (or the smallest if n < 1). The
    game engine clamps every SELL/BUY/PICKUP/PLACE order to what's
    actually available/affordable per-unit (see kaggriculture.py's
    _commit_unit/_apply_unit_action), so a generous round-number token
    like QTY_50 already behaves as "as much as possible, capped at 50"
    -- no separate 'ALL' token is needed."""
    n = int(n)
    best = QTY_VALUES[0]
    for v in QTY_VALUES:
        if v <= n:
            best = v
        else:
            break
    return best


# ============================================================================
# Observation encoding
# ============================================================================

def _tile_type_token(tile):
    if tile is None:
        return TILE_EMPTY
    if tile == "LOCKED":
        return TILE_LOCKED
    kind = tile.get("kind")
    if kind == "WEED":
        return TILE_WEED
    if kind == "PLANT":
        return TILE_PLANT[tile["crop"]]
    if kind == "COOP":
        return TILE_ANIMAL[tile["animal"]] if "animal" in tile else TILE_COOP_EMPTY
    if kind == "PASTURE":
        return TILE_ANIMAL[tile["animal"]] if "animal" in tile else TILE_PASTURE_EMPTY
    return TILE_EMPTY


def _own_tile_detail_tokens(tile, day):
    """Extra tokens for the tile a unit is STANDING ON (not emitted for
    neighbor tiles, to keep the sequence compact -- see the plan's
    'local neighborhood, not full grid' scoping)."""
    if not isinstance(tile, dict):
        return []
    kind = tile.get("kind")
    if kind == "PLANT":
        watered = WATERED_YES if tile.get("watered_today") else WATERED_NO
        fert = (FERT_ACTIVE_YES if tile.get("fertilized_until_day", -1) >= day
                else FERT_ACTIVE_NO)
        yld = YIELD[min(MAX_YIELD_UNITS, tile.get("yield_units", 0))]
        return [watered, fert, yld]
    if kind in ("COOP", "PASTURE") and "animal" in tile:
        fed = FED_YES if tile.get("fed_today") else FED_NO
        cared = CARED_YES if tile.get("cared_today") else CARED_NO
        fertavail = (FERTAVAIL_YES if tile.get("fertilizer_available")
                    else FERTAVAIL_NO)
        yld = YIELD[min(MAX_YIELD_UNITS, tile.get("yield_units", 0))]
        return [fed, cared, fertavail, yld]
    return []


def is_shed_adjacent(pos, board_size):
    """Mirrors kaggriculture.py's _shed_access_tiles/_is_shed_adjacent:
    the four center tiles, one per quadrant."""
    half = board_size // 2
    return tuple(pos) in {(half - 1, half - 1), (half, half - 1),
                          (half - 1, half), (half, half)}


def legal_plant_crops(seeds):
    """Crops the player actually has a seed for right now (private.seeds
    is a shared per-PLAYER pool, not per-unit -- any unit can plant with
    it). Used to restrict PLANT's item argument, not just the verb: a
    live trace caught the model choosing PLANT with a crop it had zero
    seeds of (wasting the turn) while holding seeds for a different
    crop it never picked instead."""
    return {c for c in CROPS if seeds.get(c, 0) > 0}


def legal_unit_verbs(tile, day, pos, board_size, seeds=None):
    """The subset of UNIT_VERBS that are NOT guaranteed no-ops for this
    unit right now, derived directly from kaggriculture.py's
    _apply_unit_action preconditions (ground truth, not a guess) --
    e.g. PLANT on an already-occupied tile, or HARVEST before
    first_yield_day, always silently does nothing. Used to keep a
    decoder from ever choosing a verb that's certain to waste the turn;
    PASS is always included as a safe fallback.

    seeds (optional, private.seeds): PLANT is additionally gated on
    having a seed for AT LEAST ONE crop -- omit (None) to skip this
    check (e.g. for tests that only care about tile-state legality).
    Every OTHER item/inventory dependency (PICKUP's item, PLACE's
    animal) is deliberately still excluded here: those depend on which
    item gets chosen at the argument slot, not the verb slot, and are
    lower-stakes (a wasted no-op there costs nothing) than PLANT's
    "wrong crop, whole turn wasted" failure mode this was added for."""
    x, y = int(pos[0]), int(pos[1])
    legal = {"PASS"}
    for verb, (dx, dy) in FARMER_MOVES.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < board_size and 0 <= ny < board_size:
            legal.add(verb)
    if is_shed_adjacent(pos, board_size):
        legal.add("PICKUP")
        legal.add("DROP")
        legal.add("PLACE")   # shed-drop branch; animal-placement branch
                             # needs a matching structure, checked below
    if tile is None:
        if seeds is None or legal_plant_crops(seeds):
            legal.add("PLANT")
        legal.add("BUILD_COOP")
        legal.add("BUILD_PASTURE")
    elif tile != "LOCKED" and isinstance(tile, dict):
        kind = tile.get("kind")
        if kind == "PLANT":
            if not tile.get("watered_today"):
                legal.add("WATER")
            first_yield = CROPS_FIRST_YIELD_DAY.get(tile.get("crop"), 0)
            mature = (day - tile.get("planted_day", day)) >= first_yield
            if mature and tile.get("yield_units", 0) > 0:
                legal.add("HARVEST")
            legal.add("FERTILIZE")   # inventory-gated, see docstring
        elif kind in ("COOP", "PASTURE"):
            if "animal" in tile:
                # note: no DIG here -- an occupied structure can't be dug.
                if tile.get("yield_units", 0) > 0:
                    legal.add("HARVEST")
                if not tile.get("fed_today"):
                    legal.add("FEED")
                if not tile.get("cared_today"):
                    legal.add("CARE")
                if tile.get("fertilizer_available"):
                    legal.add("COLLECT_FERTILIZER")
            else:
                legal.add("PLACE")
        if kind == "WEED" or (kind in ("COOP", "PASTURE") and "animal" not in tile) \
                or kind == "PLANT":
            legal.add("DIG")
    return legal


def _unit_block(tokens, geoms, unit_tok, pos_tok, pos, tiles, day, board_size,
                include_neighbors=False, use_geom=True):
    x, y = int(pos[0]), int(pos[1])
    tokens.append(unit_tok); geoms.append(None)
    if use_geom:
        tokens.append(pos_tok)
        geoms.append(np.array([norm_coord(x, board_size),
                              norm_coord(y, board_size), 0.0], dtype=np.float32))
    else:
        # Phase 4b fallback: two EXACT discrete tokens, not injected
        # geometry -- see POS_X/POS_Y's module-level comment.
        tokens.append(POS_X[min(MAX_BOARD - 1, x)]); geoms.append(None)
        tokens.append(POS_Y[min(MAX_BOARD - 1, y)]); geoms.append(None)
    tile = tiles[y][x]
    tokens.append(_tile_type_token(tile)); geoms.append(None)
    for tok in _own_tile_detail_tokens(tile, day):
        tokens.append(tok); geoms.append(None)
    if include_neighbors:
        tokens.append(NEIGHBORS); geoms.append(None)
        for dx, dy in FARMER_MOVES.values():
            nx, ny = x + dx, y + dy
            if 0 <= nx < board_size and 0 <= ny < board_size:
                tokens.append(_tile_type_token(tiles[ny][nx]))
            else:
                tokens.append(TILE_OOB)
            geoms.append(None)


def encode_observation(obs, use_geom=True):
    """obs -> (tokens: list[int], geoms: list[np.ndarray([3]) | None]),
    parallel lists (geoms[i] is not None iff tokens[i]'s leaf should get
    inject_geometry'd). Deterministic field order throughout, so the
    model always finds a given fact at a stable relative position.

    use_geom=False is the Phase 4b ablation condition: positions become
    plain discrete POS_X_<n>/POS_Y_<n> tokens instead of injected
    vectors (see POS_X/POS_Y's comment) -- everything else about the
    encoding is identical, so the ablation isolates the geometry
    question rather than confounding it with an unrelated schema
    change."""
    tokens, geoms = [], []
    farms = obs["farms"]
    player = obs["player"]
    farm = farms[player]
    private = obs.get("private") or {}
    tiles = farm["tiles"]
    board_size = len(tiles)
    day = obs.get("day", 0)

    tokens.append(TURN_START); geoms.append(None)
    tokens.append(DAY[min(MAX_DAY - 1, int(day))]); geoms.append(None)
    tokens.append(HOUR[int(obs.get("hour", 0)) % 24]); geoms.append(None)
    tokens.append(MAG[mag_bucket(farm.get("money", 0))]); geoms.append(None)

    _unit_block(tokens, geoms, FARMER, FARMER_POS, farm["farmer"], tiles,
               day, board_size, include_neighbors=True, use_geom=use_geom)

    hands = farm.get("hands", [])
    for i in range(min(MAX_HANDS, len(hands))):
        _unit_block(tokens, geoms, HAND[i], HAND_POS, hands[i], tiles,
                   day, board_size, include_neighbors=False, use_geom=use_geom)

    # Farmer's own held inventory (private.inventories[0]) -- what's in
    # hand right now, distinct from the shed.
    inventories = private.get("inventories") or []
    farmer_inv = inventories[0] if inventories else {}
    for item, n in farmer_inv.items():
        if n > 0 and item in ITEM:
            tokens.append(ITEM[item]); geoms.append(None)
            tokens.append(MAG[mag_bucket(n)]); geoms.append(None)

    market = obs.get("market") or {}
    m_inv = market.get("inventory") or {}
    m_price = market.get("prices") or {}
    for p in PRODUCTS:
        tokens.append(ITEM[p]); geoms.append(None)
        tokens.append(MAG[mag_bucket(m_price.get(p, 0))]); geoms.append(None)
        tokens.append(MAG[mag_bucket(m_inv.get(p, 0))]); geoms.append(None)

    town = obs.get("town") or {}
    unlocked = town.get("unlocked_shops") or []
    counts = {s: unlocked.count(s) for s in SHOPS}
    for s in SHOPS:
        tokens.append(SHOP[s]); geoms.append(None)
        tokens.append(MAG[mag_bucket(counts[s])]); geoms.append(None)

    shed = private.get("shed") or {}
    for p in PRODUCTS:
        n = shed.get(p, 0)
        if n > 0:
            tokens.append(ITEM[p]); geoms.append(None)
            tokens.append(MAG[mag_bucket(n)]); geoms.append(None)

    seeds = private.get("seeds") or {}
    for c in CROPS:
        n = seeds.get(c, 0)
        if n > 0:
            tokens.append(ITEM[c]); geoms.append(None)
            tokens.append(MAG[mag_bucket(n)]); geoms.append(None)

    owned = set(farm.get("unlocked_quadrants", ["NW"]))
    for q in QUADRANTS:
        tokens.append(QUAD_OWNED[q] if q in owned else QUAD_LOCKED[q])
        geoms.append(None)

    tokens.append(HIRES[min(MAX_HIRES, int(farm.get("hires_today", 0)))])
    geoms.append(None)

    return tokens, geoms


# ============================================================================
# Action encoding / decoding
# ============================================================================

def _encode_unit_action(action):
    """[verb, arg1, arg2] -> (verb_tok, arg1_tok, arg2_tok), fixed
    arity, NONE_ARG-padded. `action` is the raw ["VERB", ...] list the
    game engine consumes (see kaggriculture.py's _apply_unit_action)."""
    if not isinstance(action, list) or not action or action[0] not in VERB:
        return VERB["PASS"], NONE_ARG, NONE_ARG
    verb = action[0]
    v_tok = VERB[verb]
    a1, a2 = NONE_ARG, NONE_ARG
    if verb == "PLANT" and len(action) >= 2 and action[1] in ITEM:
        a1 = ITEM[action[1]]
    elif verb in ("PICKUP", "PLACE") and len(action) >= 2 and action[1] in ITEM:
        a1 = ITEM[action[1]]
        n = int(action[2]) if len(action) >= 3 else 1
        a2 = QTY[clamp_qty(n)]
    return v_tok, a1, a2


def _decode_unit_action(v_tok, a1_tok, a2_tok):
    verb = None
    for name, tok in VERB.items():
        if tok == v_tok:
            verb = name
            break
    if verb is None or verb == "PASS":
        return ["PASS"]
    if verb in FARMER_MOVES or verb in ("WATER", "HARVEST", "FERTILIZE",
                                        "FEED", "CARE", "COLLECT_FERTILIZER",
                                        "BUILD_COOP", "BUILD_PASTURE", "DIG",
                                        "DROP"):
        return [verb]
    if verb == "PLANT":
        item = next((it for it, tok in ITEM.items() if tok == a1_tok), None)
        return [verb, item] if item in CROPS else ["PASS"]
    if verb in ("PICKUP", "PLACE"):
        item = next((it for it, tok in ITEM.items() if tok == a1_tok), None)
        n = next((v for v, tok in QTY.items() if tok == a2_tok), 1)
        return [verb, item, n] if item is not None else ["PASS"]
    return ["PASS"]


def encode_action(action, use_geom=True):
    """action dict (the {"farmer":..., "hands":[...], "market":[...]}
    the game engine expects) -> token ids. Fixed-arity template per
    actor slot; movement verbs additionally get a MOVE_DELTA leaf
    carrying the literal (dx,dy,0) as a second, redundant geometric
    injection alongside the verb token (verb = which action, geometry =
    how much/which direction -- what makes the Phase 4b ablation mean
    anything). Returns (tokens, geoms) parallel lists, matching
    encode_observation's contract.

    use_geom=False (Phase 4b ablation): MOVE_DELTA is skipped entirely
    rather than replaced by a discrete token -- the verb (NORTH/SOUTH/
    EAST/WEST) already names the direction exactly, so there's nothing
    lossy about dropping the redundant leaf when there's no geometry to
    inject into it."""
    tokens, geoms = [], []

    def _emit_unit(slot_tok, unit_action):
        v_tok, a1, a2 = _encode_unit_action(unit_action)
        tokens.append(slot_tok); geoms.append(None)
        tokens.append(v_tok); geoms.append(None)
        verb = unit_action[0] if isinstance(unit_action, list) and unit_action else None
        if verb in FARMER_MOVES and use_geom:
            dx, dy = FARMER_MOVES[verb]
            tokens.append(MOVE_DELTA)
            geoms.append(np.array([float(dx), float(dy), 0.0], dtype=np.float32))
        tokens.append(a1); geoms.append(None)
        tokens.append(a2); geoms.append(None)

    _emit_unit(FARMER, action.get("farmer", ["PASS"]))
    hands = action.get("hands", []) or []
    for i in range(min(MAX_HANDS, len(hands))):
        _emit_unit(HAND[i], hands[i])

    tokens.append(MARKET); geoms.append(None)
    orders = (action.get("market", []) or [])[:10]
    for order in orders:
        if not isinstance(order, list) or not order or order[0] not in MVERB:
            continue
        verb = order[0]
        tokens.append(MVERB[verb]); geoms.append(None)
        if verb in ("HIRE", "BUY_LAND"):
            tokens.append(NONE_ARG); geoms.append(None)
            tokens.append(NONE_ARG); geoms.append(None)
        else:
            item = order[1] if len(order) >= 2 else None
            n = order[2] if len(order) >= 3 else 1
            tokens.append(ITEM.get(item, NONE_ARG)); geoms.append(None)
            tokens.append(QTY[clamp_qty(n)] if item in ITEM else NONE_ARG)
            geoms.append(None)
    tokens.append(ORDER_END); geoms.append(None)
    tokens.append(EOT); geoms.append(None)
    return tokens, geoms


def decode_action(token_ids, num_hands):
    """token ids -> action dict, in the same fixed-stride layout
    encode_action wrote. num_hands (from len(obs['farms'][player]
    ['hands']) at inference time, or len(action['hands']) when
    round-tripping known data) says how many hand-action slots to
    actually read -- the game only accepts as many hand actions as
    there are currently hired hands. Never raises: any malformed/
    truncated/out-of-vocab stream degrades to an all-PASS/no-orders
    action instead."""
    try:
        return _decode_action_inner(list(int(t) for t in token_ids), num_hands)
    except Exception:
        return {"farmer": ["PASS"], "hands": [["PASS"]] * max(0, num_hands),
               "market": []}


def _decode_action_inner(toks, num_hands):
    i = 0
    n = len(toks)

    def _read_unit():
        nonlocal i
        # slot token (ignored -- position tells us which slot); verb;
        # optional MOVE_DELTA (skip, geometry-only, redundant with verb);
        # arg1; arg2.
        i += 1  # slot token
        if i >= n:
            return ["PASS"]
        v_tok = toks[i]; i += 1
        if i < n and toks[i] == MOVE_DELTA:
            i += 1
        a1 = toks[i] if i < n else NONE_ARG; i += 1
        a2 = toks[i] if i < n else NONE_ARG; i += 1
        return _decode_unit_action(v_tok, a1, a2)

    farmer = ["PASS"]
    if i < n and toks[i] == FARMER:
        farmer = _read_unit()

    hands_out = []
    hand_slot_ids = set(HAND)
    while i < n and toks[i] in hand_slot_ids:
        hands_out.append(_read_unit())
    hands_out = hands_out[:max(0, num_hands)]
    while len(hands_out) < max(0, num_hands):
        hands_out.append(["PASS"])

    orders = []
    if i < n and toks[i] == MARKET:
        i += 1
        mverb_ids = set(MVERB.values())
        while i < n and toks[i] in mverb_ids and len(orders) < 10:
            v_tok = toks[i]; i += 1
            verb = next(name for name, tok in MVERB.items() if tok == v_tok)
            a1 = toks[i] if i < n else NONE_ARG; i += 1
            a2 = toks[i] if i < n else NONE_ARG; i += 1
            if verb in ("HIRE", "BUY_LAND"):
                orders.append([verb])
            else:
                item = next((it for it, tok in ITEM.items() if tok == a1), None)
                qty = next((v for v, tok in QTY.items() if tok == a2), None)
                if item is not None and qty is not None:
                    orders.append([verb, item, qty])

    return {"farmer": farmer, "hands": hands_out, "market": orders}


def encode_turn(obs, action=None, use_geom=True):
    """One turn = encode_observation(obs) ++ (encode_action(action) if
    action is not None). Returns (tokens, geoms, action_mask): action_mask
    marks exactly the action-segment positions, for Phase 4's masked
    behavior-cloning loss (observation tokens are exogenous JSON, not
    model-generated content, and must not be predicted-and-scored).

    use_geom=False selects the Phase 4b ablation encoding throughout
    (see encode_observation/encode_action) -- pick one setting per
    dataset, never mix within a run."""
    obs_tok, obs_geom = encode_observation(obs, use_geom=use_geom)
    mask = [False] * len(obs_tok)
    tokens, geoms = list(obs_tok), list(obs_geom)
    if action is not None:
        act_tok, act_geom = encode_action(action, use_geom=use_geom)
        tokens += act_tok
        geoms += act_geom
        mask += [True] * len(act_tok)
    return tokens, geoms, mask
