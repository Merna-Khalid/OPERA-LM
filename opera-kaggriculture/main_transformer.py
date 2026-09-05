"""main_transformer.py -- the matched-baseline counterpart to main.py:
same grammar-constrained generation (policy.py, unchanged), same
kagri_common encoding, but driving TransformerBaseline instead of
OperaSpinorFenwickTree.

NaiveTransformerDecoder below is a drop-in replacement for OperaDecoder's
minimal interface (.append(tok, geom, geom_block) -> logits, .reset(),
.t) -- but unlike OperaDecoder it has NO incremental/KV-cache support
(TransformerBaseline doesn't have one; see opera_transformer_baseline_v2.py).
It simply re-runs a full forward pass over the whole window on every
appended token. This is only viable because the context window is
capped the same way main.py's is (same context_window bound) and the
model is tiny: measured ~0.76ms/forward at T=256, so even a worst-case
~40-token turn costs ~30ms total, comfortably inside the 1s/turn
actTimeout. policy.generate_action works with this object completely
unchanged -- it only ever calls .append/.reset/.t, never OperaDecoder
internals.
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_CHAT_DIR = os.path.join(os.path.dirname(_HERE), "opera-chat")
if _CHAT_DIR not in sys.path:
    sys.path.insert(0, _CHAT_DIR)

import torch

import kagri_common as kc
kc.ensure_opera_lm()

import policy
from opera_transformer_baseline_v2 import TransformerBaseline

_CONFIG_PATH = os.path.join(_HERE, "model_config_transformer.json")
_state = {"model": None, "dec": None, "cfg": None}


class NaiveTransformerDecoder:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.reset()

    def reset(self):
        self.tokens = []
        self.t = 0

    @torch.no_grad()
    def append(self, token_id, geom=None, geom_block=0):
        # geom is always None here (this model trains with use_geom=False
        # and has no injection mechanism), kept only so this class matches
        # OperaDecoder's call signature and policy.generate_action can
        # drive either one unmodified.
        self.tokens.append(int(token_id))
        self.t += 1
        tok = torch.tensor([self.tokens], dtype=torch.long)
        lengths = torch.tensor([self.t])
        logits = self.model(tok, lengths)[0]
        return logits[0, -1, :]


def _load():
    with open(_CONFIG_PATH) as f:
        cfg = json.load(f)
    model = TransformerBaseline(
        vocab_size=cfg["vocab_size"], d=cfg["d"], nheads=cfg["nheads"],
        num_layers=cfg["num_layers"], pe_mode=cfg.get("pe_mode", "nope"),
        ffn_mult=cfg.get("ffn_mult", 3.3), tie=cfg.get("tie", True))
    ckpt_path = cfg["ckpt"]
    if not os.path.isabs(ckpt_path) or not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(_HERE, os.path.basename(cfg["ckpt"]))
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    dec = NaiveTransformerDecoder(model)
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

        use_geom = False   # matched baseline: no injection mechanism
        obs_tok, obs_geom = kc.encode_observation(obs, use_geom=use_geom)

        context_window = cfg.get("context_window", cfg.get(
            "max_len", policy.DEFAULT_CONTEXT_WINDOW))
        if dec.t > 0 and dec.t + len(obs_tok) + policy.MAX_ACTION_TOKENS > context_window:
            dec.reset()

        for tok, geom in zip(obs_tok, obs_geom):
            dec.append(tok, geom=geom, geom_block=0)

        player = obs.get("player", 0)
        farm = obs["farms"][player]
        seeds = (obs.get("private") or {}).get("seeds", {})
        force_harvest = cfg.get("force_harvest", True)
        action, _tokens, _geoms = policy.generate_action(
            dec, farm, obs.get("day", 0), seeds, use_geom, 0,
            mode="argmax", force_harvest=force_harvest)
        return action
    except Exception as e:
        print(f"main_transformer.agent: falling back to PASS after "
              f"exception: {e!r}", flush=True)
        num_hands = 0
        try:
            num_hands = len(obs["farms"][obs.get("player", 0)].get("hands", []))
        except Exception:
            pass
        return {"farmer": ["PASS"], "hands": [["PASS"]] * num_hands, "market": []}
