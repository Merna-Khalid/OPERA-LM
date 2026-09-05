"""Regression gate for policy.py's extraction out of main.py (RL
fine-tuning phase): a fixed-seed, deterministic scenario must reproduce
EXACTLY the action dict the original inline main.py code produced,
byte-for-byte, so the refactor itself can never silently change deployed
(argmax, force_harvest=True) behavior. Also covers the new sampling mode
(train_rl.py's rollout path) and the record hook with light structural
assertions, since there's no "before" behavior to pin those against.
"""
import os
import random
import sys

import torch
import pytest

_KAGRI_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "opera-kaggriculture")
if _KAGRI_DIR not in sys.path:
    sys.path.insert(0, _KAGRI_DIR)

import kagri_common as kc  # noqa: E402
import policy  # noqa: E402


class FakeDecoder:
    """Same fake used throughout this session's ad hoc fuzz checks:
    ignores the token/geom actually appended and returns fresh random
    logits, driven entirely by the ambient torch RNG state -- so a fixed
    torch.manual_seed + a fixed sequence of calls is fully reproducible."""
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def append(self, tok, geom=None, geom_block=0):
        return torch.randn(self.vocab_size)


def _fixed_scenario():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[4][4] = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
                  "watered_today": False, "yield_units": 1,
                  "fertilized_until_day": -1}
    farm = {"farmer": [4, 4], "hands": [[3, 3]], "tiles": tiles}
    seeds = {"WHEAT": 2, "CARROT": 3, "TOMATO": 0, "STRAWBERRY": 0, "MELON": 0}
    return farm, seeds


def test_argmax_force_harvest_matches_pinned_baseline():
    # Pinned by running this exact scenario against the original inline
    # main.py generation code (pre-refactor) with torch.manual_seed(42).
    # If this ever changes, the refactor changed live agent behavior.
    torch.manual_seed(42)
    dec = FakeDecoder(kc.VOCAB_SIZE)
    farm, seeds = _fixed_scenario()
    action, _tok, _geo = policy.generate_action(dec, farm, 3, seeds, True, kc.GEOM_BLOCK,
                                    mode="argmax", force_harvest=True)
    assert action == {"farmer": ["HARVEST"], "hands": [["EAST"]],
                      "market": [["SELL", "WOOL", 2]]}


def test_force_harvest_false_can_choose_water():
    # Same scenario, force_harvest off: HARVEST is still LEGAL (offered),
    # but no longer the only option, so argmax over the unrestricted
    # legal set can land on something else (WATER is also legal here:
    # a mature, unwatered wheat tile). This is exactly the toggle
    # train_rl.py trains through.
    torch.manual_seed(42)
    dec = FakeDecoder(kc.VOCAB_SIZE)
    farm, seeds = _fixed_scenario()
    action, _tok, _geo = policy.generate_action(dec, farm, 3, seeds, True, kc.GEOM_BLOCK,
                                    mode="argmax", force_harvest=False)
    assert action["farmer"][0] in kc.legal_unit_verbs(
        farm["tiles"][4][4], 3, (4, 4), 10, seeds=seeds)


@pytest.mark.parametrize("trial_seed", range(20))
def test_argmax_fuzz_always_well_formed(trial_seed):
    rng = random.Random(trial_seed)

    def rand_tile():
        return rng.choice([None, "LOCKED",
            {"kind": "PLANT", "crop": rng.choice(kc.CROPS), "planted_day": 0,
             "watered_today": rng.choice([True, False]),
             "yield_units": rng.randint(0, 4), "fertilized_until_day": -1},
            {"kind": "WEED"}, {"kind": "COOP"}])

    torch.manual_seed(trial_seed)
    dec = FakeDecoder(kc.VOCAB_SIZE)
    board_size = 10
    tiles = [[rand_tile() for _ in range(board_size)] for _ in range(board_size)]
    n_hands = rng.randint(0, kc.MAX_HANDS + 1)
    farm = {"farmer": [rng.randint(0, 9), rng.randint(0, 9)],
           "hands": [[rng.randint(0, 9), rng.randint(0, 9)] for _ in range(n_hands)],
           "tiles": tiles}
    seeds = {c: rng.choice([0, 0, 0, 3]) for c in kc.CROPS}
    action, _tok, _geo = policy.generate_action(dec, farm, rng.randint(0, 10), seeds,
                                    rng.choice([True, False]), kc.GEOM_BLOCK,
                                    mode="argmax", force_harvest=True)
    assert set(action.keys()) == {"farmer", "hands", "market"}
    assert len(action["hands"]) == min(kc.MAX_HANDS, n_hands)
    if action["farmer"][0] == "PLANT":
        assert action["farmer"][1] in kc.legal_plant_crops(seeds)


def test_sample_mode_returns_log_probs_and_records_positions():
    torch.manual_seed(0)
    dec = FakeDecoder(kc.VOCAB_SIZE)
    tiles = [[None] * 10 for _ in range(10)]
    farm = {"farmer": [4, 4], "hands": [], "tiles": tiles}
    seeds = {"WHEAT": 3, "CARROT": 0, "TOMATO": 0, "STRAWBERRY": 0, "MELON": 0}
    record = []
    action, _tok, _geo = policy.generate_action(dec, farm, 0, seeds, True, kc.GEOM_BLOCK,
                                    mode="sample", force_harvest=False,
                                    record=record)
    assert record, "sample mode with a record list must record every choice"
    assert all(r["log_prob"] is not None for r in record)
    assert all(r["log_prob"] <= 0 for r in record)   # log-prob, never positive
    positions = [r["local_pos"] for r in record]
    assert positions == sorted(positions), "local_pos must be non-decreasing"
    assert len(set(positions)) == len(positions), "no duplicate positions"
    assert set(action.keys()) == {"farmer", "hands", "market"}


def test_argmax_mode_records_none_log_prob():
    torch.manual_seed(0)
    dec = FakeDecoder(kc.VOCAB_SIZE)
    tiles = [[None] * 10 for _ in range(10)]
    farm = {"farmer": [4, 4], "hands": [], "tiles": tiles}
    seeds = {c: 0 for c in kc.CROPS}
    record = []
    _act, _tok, _geo = policy.generate_action(dec, farm, 0, seeds, True, kc.GEOM_BLOCK,
                                                mode="argmax", force_harvest=True, record=record)
    assert record and all(r["log_prob"] is None for r in record)
