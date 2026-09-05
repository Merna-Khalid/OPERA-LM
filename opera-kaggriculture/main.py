"""main.py -- OPERA Kaggriculture submission (Phase 5 of the plan).

CPU-only, turn-by-turn incremental inference via opera_lm.incremental.
OperaDecoder, driven by policy.generate_action in deterministic (argmax)
mode. See policy.py for the generation logic itself (shared with
train_rl.py's RL-fine-tuning rollout collector, which drives the same
function in sampling mode).

Packaging for actual Kaggle submission (not needed for local testing,
where opera_lm is already importable from the repo checkout): bundle
this file together with kagri_common.py, policy.py, the opera_lm/
package directory, kagri_model.pt, and model_config.json, all at the
tar.gz root -- kaggriculture's rules page notes files land in
/kaggle_simulations/agent/ and imports must be set up accordingly.

Local test (per the plan's Phase 5 checklist):
  python -c "
  from kaggle_environments import make
  env = make('kaggriculture')
  env.run(['main.py', 'starter'])
  print([s.reward for s in env.steps[-1]])
  "
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch

import kagri_common as kc
# Local dev/testing: opera_lm lives in the repo root (this file's parent
# dir), not bundled alongside main.py the way a real submission would
# have it -- ensure_opera_lm's sys.path trick covers that case; a no-op
# once opera_lm/ is actually bundled next to this file (already
# importable via _HERE).
kc.ensure_opera_lm()
from opera_lm.model import OperaSpinorFenwickTree
from opera_lm.incremental import OperaDecoder

import policy

_CONFIG_PATH = os.path.join(_HERE, "model_config.json")
_state = {"model": None, "dec": None, "cfg": None}


def _load():
    with open(_CONFIG_PATH) as f:
        cfg = json.load(f)
    model = OperaSpinorFenwickTree(
        vocab_size=cfg["vocab_size"], d=cfg["d"], nb=cfg["nb"],
        num_layers=cfg["num_layers"], tie=cfg.get("tie", True),
        pe_mode=cfg.get("pe_mode", "none"), fold_mode=cfg.get("fold_mode", "left"),
        rot_mode=cfg.get("rot_mode", "free"))
    ckpt_path = cfg["ckpt"]
    if not os.path.isabs(ckpt_path) or not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(_HERE, os.path.basename(cfg["ckpt"]))
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    dec = OperaDecoder(model)
    _state["model"] = model
    _state["dec"] = dec
    _state["cfg"] = cfg


def agent(obs):
    try:
        if _state["dec"] is None:
            _load()
        dec = _state["dec"]
        cfg = _state["cfg"]

        if obs.get("day", 0) == 0 and obs.get("hour", 0) == 0:
            dec.reset()

        use_geom = cfg.get("use_geom", True)
        geom_block = cfg.get("geom_block", kc.GEOM_BLOCK)
        obs_tok, obs_geom = kc.encode_observation(obs, use_geom=use_geom)

        # Bounded-context reset: kaggriculture's actTimeout is 1s/turn,
        # and OperaDecoder's O(log T)-per-token cost still means total
        # per-turn latency grows (slowly) with T -- measured empirically
        # to climb from ~0.04s to ~0.45s over a 720-turn episode if the
        # tree is allowed to grow across the whole game. Separately (and
        # more fundamentally), training data was windowed to max_len
        # tokens (see prepare_kagri_data.py) -- the model has never seen
        # a context anywhere near a full episode's length, so letting it
        # grow unboundedly is also out-of-distribution, not just slow.
        # The observation is FULLY OBSERVED each turn (current tile/
        # market/inventory state, not partial), so a fresh window loses
        # no information the model actually needs to act correctly --
        # resetting here trades away only whatever incidental cross-turn
        # pattern the model might have picked up within one training
        # window, in exchange for a hard, flat latency ceiling.
        context_window = cfg.get("context_window", cfg.get(
            "max_len", policy.DEFAULT_CONTEXT_WINDOW))
        if dec.t > 0 and dec.t + len(obs_tok) + policy.MAX_ACTION_TOKENS > context_window:
            dec.reset()

        for tok, geom in zip(obs_tok, obs_geom):
            dec.append(tok, geom=geom, geom_block=geom_block)

        player = obs.get("player", 0)
        farm = obs["farms"][player]
        seeds = (obs.get("private") or {}).get("seeds", {})
        force_harvest = cfg.get("force_harvest", True)
        action, _tokens, _geoms = policy.generate_action(
            dec, farm, obs.get("day", 0), seeds, use_geom, geom_block,
            mode="argmax", force_harvest=force_harvest)
        return action
    except Exception as e:
        print(f"main.agent: falling back to PASS after exception: {e!r}",
              flush=True)
        num_hands = 0
        try:
            num_hands = len(obs["farms"][obs.get("player", 0)].get("hands", []))
        except Exception:
            pass
        return {"farmer": ["PASS"], "hands": [["PASS"]] * num_hands, "market": []}
