"""make_artifacts.py -- final Path A stage: compute the pre-registered
gates (docs/OPERA_PathA_prereg.md §3) from the three position-curve
JSONs and the SFT results.jsonl rows, and generate the long-context
continuation samples for each arm. Writes artifacts/gates.json and
artifacts/summary.md.
"""
import argparse
import glob
import json
import os
import pickle
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "opera-chat"))

import torch                                             # noqa: E402
import torch.nn.functional as F                          # noqa: E402
from chat_common import ensure_opera_lm, load_tokenizer  # noqa: E402

ensure_opera_lm()
from generate_chat import load_model, _sample            # noqa: E402
from opera_lm import OperaDecoder                        # noqa: E402
from opera_transformer_baseline_v2 import TransformerBaseline  # noqa: E402

HONESTY = """## Honesty box (prereg §6, verbatim)

A 155M model pretrained on ~250M tokens (0.01% of SmolLM2-135M's 2T) and
SFT'd on smoltalk. **Will:** correct chat format (turn-taking, stopping),
fluent grammatical multi-turn replies that reference context, simple
instruction-following behaviors, recall of very common facts, and -- the
study's claim -- measurably better length behavior at 4-8k context than
the matched RoPE baseline if H1/H2 pass. **Will not:** reliable factual
knowledge, arithmetic, complex multi-constraint instructions, or
assistant-level helpfulness."""


def weighted_mean_ce(pc, min_lo):
    """Token-weighted mean CE over bands whose start >= min_lo."""
    s = sum(b["ce"] * b["tokens"] for b in pc["bands"] if b["lo"] >= min_lo)
    c = sum(b["tokens"] for b in pc["bands"] if b["lo"] >= min_lo)
    return (s / c) if c else None


def last_result_row(run_dir):
    rows = []
    path = os.path.join(run_dir, "results.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            rows = [json.loads(x) for x in f if x.strip()]
    return rows[-1] if rows else None


def compute_gates(pcs):
    o, r, n = pcs["opera"], pcs["rope"], pcs["nope"]
    deg = {k: pcs[k]["degradation_best_to_final_horizon"] for k in pcs}
    gates = {}
    gates["H1_flatness"] = {
        "rule": "deg_opera <= 0.5 * deg_rope AND opera boundary_delta <= +0.10",
        "deg_opera": deg["opera"], "deg_rope": deg["rope"],
        "opera_boundary_delta": o["boundary_delta"],
        "pass": bool(deg["opera"] <= 0.5 * deg["rope"]
                     and o["boundary_delta"] is not None
                     and o["boundary_delta"] <= 0.10),
    }
    m_o = weighted_mean_ce(o, 4096)
    m_r = weighted_mean_ce(r, 4096)
    gates["H2_absolute_beyond_4k"] = {
        "rule": "token-weighted mean CE over bands >= 4096: opera < rope",
        "opera_ce": m_o, "rope_ce": m_r,
        "pass": bool(m_o is not None and m_r is not None and m_o < m_r),
    }
    gates["H4_rope_flat_null"] = {
        "rule": "if deg_rope < 0.15 nats, flatness does not discriminate",
        "deg_rope": deg["rope"],
        "fires": bool(deg["rope"] < 0.15),
    }
    return gates


@torch.no_grad()
def sample_opera(ckpt, config, ids, n_tokens, device):
    model = load_model(config, ckpt, device)
    dec = OperaDecoder(model)
    logits = dec.prefill(list(ids))
    out = []
    for _ in range(n_tokens):
        nxt = _sample(logits, temp=0.8, top_p=0.9)
        if nxt == 3:                      # <|end|>
            out.append(nxt)
            break
        out.append(nxt)
        logits = dec.append(nxt)
    return out


@torch.no_grad()
def sample_tf(ckpt, pe, ids, n_tokens, device, vocab):
    model = TransformerBaseline(vocab, d=1264, nheads=8, num_layers=7,
                                pe_mode=pe, max_pe_len=8192, tie=True)
    st = torch.load(ckpt, map_location=device, weights_only=True)
    if "model" in st:
        st = st["model"]
    model.load_state_dict(st)
    model = model.to(device).eval()
    ctx = list(ids)
    out = []
    for _ in range(n_tokens):
        t = torch.tensor([ctx], dtype=torch.long, device=device)
        lens = torch.tensor([len(ctx)], device=device)
        logits = model(t, lens)[0][0, -1]
        nxt = _sample(logits, temp=0.8, top_p=0.9)
        if nxt == 3:
            out.append(nxt)
            break
        out.append(nxt)
        ctx.append(nxt)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True)
    p.add_argument("--data", required=True, help="smoltalk eval pkl")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--n-tokens", type=int, default=80)
    a = p.parse_args()
    art = os.path.join(a.work, "artifacts")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pcs = {arm: json.load(open(os.path.join(art, f"pc_{arm}.json")))
           for arm in ("opera", "rope", "nope")}
    gates = compute_gates(pcs)

    rows = {arm: last_result_row(os.path.join(a.work, "runs", f"sft_{arm}"))
            for arm in ("opera", "rope", "nope")}
    if all(rows.values()):
        ppl = {arm: rows[arm]["test_perplexity_in_length"] for arm in rows}
        ranking = sorted(ppl, key=ppl.get)
        gates["H3_two_stage_flip"] = {
            "rule": "report in-length PPL ranking; flip persists iff "
                    "rope < opera",
            "in_length_ppl": ppl, "ranking_best_to_worst": ranking,
            "flip_persists": bool(ppl["rope"] < ppl["opera"]),
        }
    with open(os.path.join(art, "gates.json"), "w") as f:
        json.dump(gates, f, indent=2)

    # ---------------- long-context continuation samples
    with open(a.data, "rb") as f:
        d = pickle.load(f)
    tok = load_tokenizer(a.tokenizer)
    long_pool = sorted((s for s in d["test_long"] if len(s) >= 2048),
                       key=len, reverse=True)[:3]
    ctxs_opera = [1024, 2048, 4096, 8192]
    ctxs_tf = [1024, 4096]
    lines = ["# Path A long-context samples", ""]
    o_final = max((f for f in glob.glob(os.path.join(a.work, "runs",
                                                     "sft_opera", "*.pt"))
                   if "_train_ckpt" not in f), key=os.path.getmtime)
    o_cfg = os.path.join(a.work, "runs", "sft_opera", "model_config.json")
    for i, seq in enumerate(long_pool):
        for ctx in ctxs_opera:
            ids = seq[-ctx:]
            gen = sample_opera(o_final, o_cfg, ids, a.n_tokens, device)
            lines += [f"## sample {i + 1} @ ctx {ctx} (prompt tail + "
                      f"OPERA continuation)", "```",
                      tok.decode(ids[-96:]).replace("```", "'''") + " ...",
                      ">>> " + tok.decode(gen), "```", ""]
        for pe in ("rope", "nope"):
            ck = os.path.join(a.work, "runs", f"sft_{pe}",
                              f"tf_{pe}_opt-muon.pt")
            for ctx in ctxs_tf:
                ids = seq[-ctx:]
                gen = sample_tf(ck, pe, ids, min(a.n_tokens, 60), device,
                                d["vocab_size"])
                lines += [f"## sample {i + 1} @ ctx {ctx} (TF {pe.upper()} "
                          f"continuation)", "```",
                          tok.decode(ids[-96:]).replace("```", "'''") + " ...",
                          ">>> " + tok.decode(gen), "```", ""]

    # ---------------- summary
    def curve_table(arm):
        rows_ = [f"| {b['lo']}-{b['hi'] - 1} | {b['ce']:.4f} | {b['tokens']} |"
                 for b in pcs[arm]["bands"]]
        return (f"### {arm}\n\n| band | CE | tokens |\n|---|---|---|\n"
                + "\n".join(rows_))

    summary = ["# OPERA Path A -- results summary (auto-assembled)", "",
               "## Gates (prereg §3, computed verbatim)", "",
               "```json", json.dumps(gates, indent=2), "```", "",
               "## In-length PPL and extrapolation (SFT runs)", ""]
    for arm in ("opera", "rope", "nope"):
        if rows[arm]:
            r_ = rows[arm]
            summary.append(f"- **{arm}**: in-length PPL "
                           f"{r_['test_perplexity_in_length']:.2f}; "
                           f"extrapolation "
                           f"{json.dumps(r_.get('extrapolation', {}))}")
    summary += ["", "## Position curves (train 512, horizon 8192)", "",
                curve_table("opera"), "", curve_table("rope"), "",
                curve_table("nope"), "", HONESTY, ""]
    with open(os.path.join(art, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n\n---\n\n" + "\n".join(summary))
    print(f"gates + summary + samples -> {art}")


if __name__ == "__main__":
    main()
