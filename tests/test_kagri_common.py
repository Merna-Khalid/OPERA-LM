"""Round-trip and fuzz tests for opera-kaggriculture/kagri_common.py's
turn<->token encoding (Phase 1 of the OPERA-Kaggriculture plan). Mirrors
tests/test_selftest.py's role: pytest-collected, one assertion group per
test function so a failure in one doesn't hide the others.
"""
import os
import random
import sys

import numpy as np
import pytest

_KAGRI_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "opera-kaggriculture")
if _KAGRI_DIR not in sys.path:
    sys.path.insert(0, _KAGRI_DIR)

import kagri_common as kc  # noqa: E402


def _empty_farm(board_size=10, hands=0):
    tiles = [[None if x < board_size // 2 and y < board_size // 2 else "LOCKED"
             for x in range(board_size)] for y in range(board_size)]
    return {
        "money": 3000, "tiles": tiles, "farmer": [2, 2],
        "hands": [[3, 3 + i] for i in range(hands)],
        "unlocked_quadrants": ["NW"], "hires_today": 0,
    }


def _base_obs(board_size=10, hands=0):
    return {
        "player": 0, "day": 5, "hour": 10,
        "farms": [_empty_farm(board_size, hands), _empty_farm(board_size)],
        "market": {
            "inventory": {p: 10000 for p in kc.PRODUCTS},
            "prices": {p: 50 for p in kc.PRODUCTS},
        },
        "town": {"unlocked_shops": ["BAKERY", "BAKERY", "YARN_STORE"]},
        "private": {
            "shed": {"WHEAT": 5, "CARROT": 0},
            "seeds": {"WHEAT": 2},
            "inventories": [{"WHEAT": 1}] + [{} for _ in range(hands)],
        },
    }


# ============================================================================
# Vocab sanity
# ============================================================================

def test_vocab_no_collisions():
    assert len(kc.ID2TOK) == kc.VOCAB_SIZE
    assert len(set(kc.ID2TOK)) == kc.VOCAB_SIZE, "duplicate token names"
    assert kc.ID2TOK[kc.PAD] == "<pad>"
    for name, idx in kc.TOK2ID.items():
        assert kc.ID2TOK[idx] == name


# ============================================================================
# Unit-action round trips
# ============================================================================

_NO_ARG_VERBS = ["PASS", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST",
                 "FERTILIZE", "FEED", "CARE", "COLLECT_FERTILIZER",
                 "BUILD_COOP", "BUILD_PASTURE", "DIG", "DROP"]


@pytest.mark.parametrize("verb", _NO_ARG_VERBS)
def test_unit_action_roundtrip_no_arg(verb):
    action = {"farmer": [verb], "hands": [], "market": []}
    tokens, _ = kc.encode_action(action)
    decoded = kc.decode_action(tokens, num_hands=0)
    assert decoded["farmer"] == [verb]
    assert decoded["hands"] == []
    assert decoded["market"] == []


def test_unit_action_roundtrip_plant():
    for crop in kc.CROPS:
        action = {"farmer": ["PLANT", crop], "hands": [], "market": []}
        tokens, _ = kc.encode_action(action)
        decoded = kc.decode_action(tokens, num_hands=0)
        assert decoded["farmer"] == ["PLANT", crop]


@pytest.mark.parametrize("verb", ["PICKUP", "PLACE"])
@pytest.mark.parametrize("item,n,expected_n", [
    # QTY_VALUES = 1..20, 50, 100: below 20 is exact; 25 clamps DOWN to
    # 20 (the engine clamps every order to what's available/affordable
    # anyway, so under-asking is harmless -- see clamp_qty's docstring).
    ("WHEAT", 1, 1), ("FERTILIZER", 7, 7), ("EGG", 25, 20), ("MILK", 1000, 100),
])
def test_unit_action_roundtrip_item_qty(verb, item, n, expected_n):
    action = {"farmer": [verb, item, n], "hands": [], "market": []}
    tokens, _ = kc.encode_action(action)
    decoded = kc.decode_action(tokens, num_hands=0)
    assert decoded["farmer"] == [verb, item, expected_n]


def test_unit_action_unknown_verb_falls_back_to_pass():
    action = {"farmer": ["TELEPORT", "X"], "hands": [], "market": []}
    tokens, _ = kc.encode_action(action)
    decoded = kc.decode_action(tokens, num_hands=0)
    assert decoded["farmer"] == ["PASS"]


def test_movement_gets_geometric_move_delta():
    for verb, (dx, dy) in kc.FARMER_MOVES.items():
        action = {"farmer": [verb], "hands": [], "market": []}
        tokens, geoms = kc.encode_action(action)
        idx = tokens.index(kc.MOVE_DELTA)
        np.testing.assert_allclose(geoms[idx], [dx, dy, 0.0])


# ============================================================================
# Hands round trips
# ============================================================================

@pytest.mark.parametrize("n_hands", [0, 1, kc.MAX_HANDS, kc.MAX_HANDS + 2])
def test_hands_roundtrip(n_hands):
    verbs = ["NORTH", "WATER", "HARVEST", "PASS", "SOUTH", "EAST"]
    hand_actions = [[verbs[i % len(verbs)]] for i in range(n_hands)]
    action = {"farmer": ["PASS"], "hands": hand_actions, "market": []}
    tokens, _ = kc.encode_action(action)
    # encode_action caps at MAX_HANDS; decode_action must be told the
    # TRUE current hand count (from obs), which may legitimately be
    # less than what got encoded (e.g. a hand was let go since).
    effective = min(n_hands, kc.MAX_HANDS)
    decoded = kc.decode_action(tokens, num_hands=effective)
    assert len(decoded["hands"]) == effective
    for i in range(effective):
        assert decoded["hands"][i] == hand_actions[i]


# ============================================================================
# Market order round trips
# ============================================================================

def test_market_orders_hire_and_buy_land():
    action = {"farmer": ["PASS"], "hands": [],
             "market": [["HIRE"], ["BUY_LAND"]]}
    tokens, _ = kc.encode_action(action)
    decoded = kc.decode_action(tokens, num_hands=0)
    assert decoded["market"] == [["HIRE"], ["BUY_LAND"]]


@pytest.mark.parametrize("verb", ["BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "SELL"])
def test_market_orders_item_qty(verb):
    action = {"farmer": ["PASS"], "hands": [],
             "market": [[verb, "WHEAT", 3]]}
    tokens, _ = kc.encode_action(action)
    decoded = kc.decode_action(tokens, num_hands=0)
    assert decoded["market"] == [[verb, "WHEAT", 3]]


def test_market_orders_zero_and_ten():
    action0 = {"farmer": ["PASS"], "hands": [], "market": []}
    tokens0, _ = kc.encode_action(action0)
    assert kc.decode_action(tokens0, num_hands=0)["market"] == []

    orders10 = [["SELL", "WHEAT", i + 1] for i in range(10)]
    action10 = {"farmer": ["PASS"], "hands": [], "market": orders10}
    tokens10, _ = kc.encode_action(action10)
    decoded10 = kc.decode_action(tokens10, num_hands=0)
    assert decoded10["market"] == orders10

    # >10 orders: encode_action itself caps at 10 (matches the game's
    # own maxMarketOrdersPerTurn default -- extra orders are dropped
    # the same way the engine would silently drop them).
    orders15 = [["SELL", "WHEAT", i + 1] for i in range(15)]
    action15 = {"farmer": ["PASS"], "hands": [], "market": orders15}
    tokens15, _ = kc.encode_action(action15)
    decoded15 = kc.decode_action(tokens15, num_hands=0)
    assert decoded15["market"] == orders15[:10]


# ============================================================================
# Fuzz: decode_action must never raise, always shape-correct
# ============================================================================

def test_decode_action_fuzz_never_raises():
    rng = random.Random(0)
    for _ in range(500):
        length = rng.randint(0, 40)
        toks = [rng.randint(0, kc.VOCAB_SIZE - 1) for _ in range(length)]
        num_hands = rng.randint(0, kc.MAX_HANDS)
        decoded = kc.decode_action(toks, num_hands)
        assert set(decoded.keys()) == {"farmer", "hands", "market"}
        assert isinstance(decoded["farmer"], list) and decoded["farmer"]
        assert len(decoded["hands"]) == num_hands
        assert isinstance(decoded["market"], list)


def test_decode_action_handles_empty_and_truncated():
    assert kc.decode_action([], 0)["farmer"] == ["PASS"]
    assert kc.decode_action([kc.FARMER], 0)["farmer"] == ["PASS"]
    assert kc.decode_action([kc.FARMER, kc.VERB["PLANT"]], 0)["farmer"] == ["PASS"]


# ============================================================================
# Observation encoding
# ============================================================================

@pytest.mark.parametrize("n_hands", [0, 1, kc.MAX_HANDS])
def test_encode_observation_shapes(n_hands):
    obs = _base_obs(hands=n_hands)
    tokens, geoms = kc.encode_observation(obs)
    assert len(tokens) == len(geoms)
    assert all(isinstance(t, int) for t in tokens)
    geo_positions = [i for i, t in enumerate(tokens)
                     if t in (kc.FARMER_POS, kc.HAND_POS)]
    assert len(geo_positions) == 1 + n_hands
    for i in geo_positions:
        assert geoms[i] is not None and geoms[i].shape == (3,)
    for i, t in enumerate(tokens):
        if t not in (kc.FARMER_POS, kc.HAND_POS):
            assert geoms[i] is None


def test_encode_observation_covers_tile_variants():
    # Detail tokens (watered/fed/cared/fertavail/yield) are only emitted
    # for the tile a unit is STANDING ON (see _own_tile_detail_tokens);
    # neighbor tiles only get the coarse type token. So to see detail
    # tokens for the animal tile, put the farmer ON it.
    obs = _base_obs()
    tiles = obs["farms"][0]["tiles"]
    obs["farms"][0]["farmer"] = [1, 2]
    tiles[2][1] = {"kind": "COOP", "animal": "GOOSE", "placed_day": 0,
                  "yield_units": 1, "fed_today": False,
                  "consecutive_unfed": 1, "cared_today": True,
                  "fertilizer_available": True, "pending_care_bonus": 0}
    tiles[2][2] = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
                  "watered_today": True, "yield_units": 2,
                  "fertilized_until_day": -1}
    tiles[1][1] = {"kind": "WEED"}
    tokens, geoms = kc.encode_observation(obs)
    assert kc.TILE_PLANT["WHEAT"] in tokens        # neighbor, coarse only
    assert kc.TILE_ANIMAL["GOOSE"] in tokens        # own tile
    assert kc.TILE_WEED in tokens                   # neighbor, coarse only
    assert kc.FED_NO in tokens
    assert kc.CARED_YES in tokens
    assert kc.FERTAVAIL_YES in tokens


def test_encode_turn_action_mask():
    obs = _base_obs()
    action = {"farmer": ["NORTH"], "hands": [], "market": [["SELL", "WHEAT", 1]]}
    tokens, geoms, mask = kc.encode_turn(obs, action)
    obs_tok, _ = kc.encode_observation(obs)
    act_tok, _ = kc.encode_action(action)
    assert len(tokens) == len(obs_tok) + len(act_tok)
    assert len(mask) == len(tokens)
    assert not any(mask[:len(obs_tok)])
    assert all(mask[len(obs_tok):])
    assert tokens == obs_tok + act_tok


# ============================================================================
# Phase 4b ablation encoding (use_geom=False)
# ============================================================================

def test_use_geom_false_has_no_injected_vectors():
    obs = _base_obs(hands=1)
    action = {"farmer": ["NORTH"], "hands": [["WATER"]], "market": []}
    tokens, geoms, mask = kc.encode_turn(obs, action, use_geom=False)
    assert all(g is None for g in geoms)
    assert kc.FARMER_POS not in tokens and kc.HAND_POS not in tokens
    assert kc.MOVE_DELTA not in tokens
    fx, fy = obs["farms"][0]["farmer"]
    assert kc.POS_X[fx] in tokens and kc.POS_Y[fy] in tokens
    decoded = kc.decode_action(kc.encode_action(action, use_geom=False)[0],
                              num_hands=1)
    assert decoded["farmer"] == ["NORTH"]
    assert decoded["hands"] == [["WATER"]]


def test_encode_turn_no_action():
    obs = _base_obs()
    tokens, geoms, mask = kc.encode_turn(obs)
    obs_tok, _ = kc.encode_observation(obs)
    assert tokens == obs_tok
    assert not any(mask)


# ============================================================================
# legal_unit_verbs -- the root cause of the live PLANT-on-occupied-tile
# loop this was added to prevent; each case here is taken directly from
# kaggriculture.py's _apply_unit_action preconditions.
# ============================================================================

def test_legal_verbs_empty_tile():
    legal = kc.legal_unit_verbs(None, day=0, pos=(2, 2), board_size=10)
    assert "PLANT" in legal and "BUILD_COOP" in legal and "BUILD_PASTURE" in legal
    assert "WATER" not in legal and "HARVEST" not in legal and "DIG" not in legal
    assert "PASS" in legal


def test_legal_verbs_locked_tile_blocks_everything_but_move_and_pass():
    legal = kc.legal_unit_verbs("LOCKED", day=0, pos=(2, 2), board_size=10)
    for v in ("PLANT", "WATER", "HARVEST", "DIG", "BUILD_COOP", "BUILD_PASTURE"):
        assert v not in legal
    assert "PASS" in legal
    assert {"NORTH", "SOUTH", "EAST", "WEST"} <= legal   # in-bounds at (2,2)


def test_legal_verbs_immature_plant_no_harvest_but_can_water():
    # the exact bug: yield_units=1 immediately at planting, but HARVEST
    # is blocked until first_yield_day passes (WHEAT: 2).
    tile = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
           "watered_today": False, "yield_units": 1,
           "fertilized_until_day": -1}
    legal = kc.legal_unit_verbs(tile, day=0, pos=(2, 2), board_size=10)
    assert "PLANT" not in legal, "tile is occupied -- replanting is a no-op"
    assert "HARVEST" not in legal, "day 0 < first_yield_day (2)"
    assert "WATER" in legal
    assert "DIG" in legal
    assert "FERTILIZE" in legal


def test_legal_verbs_mature_plant_can_harvest():
    tile = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
           "watered_today": True, "yield_units": 4,
           "fertilized_until_day": -1}
    legal = kc.legal_unit_verbs(tile, day=2, pos=(2, 2), board_size=10)
    assert "HARVEST" in legal
    assert "WATER" not in legal, "already watered_today"


def test_legal_verbs_already_watered_plant():
    tile = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
           "watered_today": True, "yield_units": 0,
           "fertilized_until_day": -1}
    legal = kc.legal_unit_verbs(tile, day=0, pos=(2, 2), board_size=10)
    assert "WATER" not in legal and "HARVEST" not in legal and "PLANT" not in legal


def test_legal_verbs_weed():
    legal = kc.legal_unit_verbs({"kind": "WEED"}, day=0, pos=(2, 2), board_size=10)
    assert "DIG" in legal
    assert "PLANT" not in legal and "WATER" not in legal


def test_legal_verbs_animal_structure_states():
    empty_coop = {"kind": "COOP"}
    legal = kc.legal_unit_verbs(empty_coop, day=0, pos=(2, 2), board_size=10)
    assert "PLACE" in legal and "DIG" in legal
    assert "FEED" not in legal and "CARE" not in legal

    occupied = {"kind": "COOP", "animal": "GOOSE", "placed_day": 0,
               "yield_units": 0, "fed_today": False, "cared_today": True,
               "fertilizer_available": True}
    legal2 = kc.legal_unit_verbs(occupied, day=0, pos=(2, 2), board_size=10)
    assert "DIG" not in legal2, "can't dig an occupied structure"
    assert "FEED" in legal2 and "COLLECT_FERTILIZER" in legal2
    assert "CARE" not in legal2, "already cared_today"
    assert "HARVEST" not in legal2, "yield_units is 0"


def test_legal_verbs_shed_adjacency_gates_pickup_drop():
    # board_size=10 -> shed-access tiles are (4,4),(5,4),(4,5),(5,5)
    on_shed = kc.legal_unit_verbs(None, day=0, pos=(4, 4), board_size=10)
    assert "PICKUP" in on_shed and "DROP" in on_shed
    off_shed = kc.legal_unit_verbs(None, day=0, pos=(2, 2), board_size=10)
    assert "PICKUP" not in off_shed and "DROP" not in off_shed


def test_legal_verbs_movement_blocked_at_board_edge():
    legal = kc.legal_unit_verbs(None, day=0, pos=(0, 0), board_size=10)
    assert "NORTH" not in legal and "WEST" not in legal
    assert "SOUTH" in legal and "EAST" in legal


def test_legal_plant_crops():
    assert kc.legal_plant_crops({"WHEAT": 0, "CARROT": 3}) == {"CARROT"}
    assert kc.legal_plant_crops({}) == set()
    assert kc.legal_plant_crops({"WHEAT": 1, "MELON": 2}) == {"WHEAT", "MELON"}


def test_legal_verbs_plant_gated_by_seed_availability():
    # The exact live-trace bug: an empty tile with zero seeds of ANY
    # crop must not offer PLANT at all (it would be a guaranteed no-op
    # regardless of which crop token gets chosen next).
    no_seeds = kc.legal_unit_verbs(None, day=0, pos=(2, 2), board_size=10,
                                   seeds={"WHEAT": 0, "CARROT": 0})
    assert "PLANT" not in no_seeds
    assert "PASS" in no_seeds

    has_seeds = kc.legal_unit_verbs(None, day=0, pos=(2, 2), board_size=10,
                                    seeds={"WHEAT": 0, "CARROT": 3})
    assert "PLANT" in has_seeds

    # seeds=None (default) skips the check entirely -- existing
    # tile-only callers/tests are unaffected.
    no_check = kc.legal_unit_verbs(None, day=0, pos=(2, 2), board_size=10)
    assert "PLANT" in no_check


def test_legal_verbs_always_includes_pass():
    for tile in (None, "LOCKED", {"kind": "WEED"},
                {"kind": "PLANT", "crop": "MELON", "planted_day": 0,
                 "watered_today": True, "yield_units": 0,
                 "fertilized_until_day": -1}):
        assert "PASS" in kc.legal_unit_verbs(tile, day=5, pos=(1, 1),
                                             board_size=10)
