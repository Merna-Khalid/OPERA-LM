"""train_rl.py -- REINFORCE + KL-to-reference RL fine-tune on top of the
behavior-cloned OPERA Kaggriculture checkpoint (see the plan doc,
polished-foraging-canyon.md, for the full design rationale).

Why this exists: BC plateaued around 55-60% win rate vs `starter`, and
closing the gap further required a hand-scripted decode-time override
(force HARVEST over WATER) rather than a genuine legality mask -- the
user correctly flagged that as scripting the strategy, not testing
OPERA's own judgment. This trains that specific preference (and any
other systematic BC mistakes reward can see) through actual reward
signal instead: rollouts sample with force_harvest=False (see
policy.py), and the policy gradient has to learn on its own whether
harvesting now beats watering now.

Why rollout collection and the gradient step are separate passes:
OperaDecoder.append is @torch.no_grad() (opera_lm/incremental.py) --
correctly fast for rollout, but nothing sampled through it can be
backpropped through directly. Instead, each turn's rollout records the
exact conditioning window (tokens/geom since the last context-window
reset, extended by that turn's own generated tokens) and which token
was sampled from which restricted candidate set; the policy-gradient
update re-derives log-probs for those exact (window, position,
candidate-set) triples via the batched, gradient-enabled forward() --
the only path in this codebase that can produce gradients.

Usage (start with the small smoke-scale defaults on the CLI before a
real run -- see the plan's staged-verification section):
  python train_rl.py --iterations 3 --episodes-per-iter 3 --eval-every 3 --eval-episodes 2
  python train_rl.py --iterations 30 --episodes-per-iter 8
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from kagri_common import ensure_opera_lm

ensure_opera_lm()

from opera_lm.model import OperaSpinorFenwickTree      # noqa: E402
from opera_lm.incremental import OperaDecoder           # noqa: E402

import kagri_common as kc                                # noqa: E402
import policy                                             # noqa: E402
from heuristic_bot import agent as heuristic_agent        # noqa: E402

from kaggle_environments import make                       # noqa: E402

OPPONENTS = [heuristic_agent, "starter", "random"]
OPPONENT_WEIGHTS = [0.4, 0.4, 0.2]


def _load_model(cfg, ckpt_path):
    model = OperaSpinorFenwickTree(
        vocab_size=cfg["vocab_size"], d=cfg["d"], nb=cfg["nb"],
        num_layers=cfg["num_layers"], tie=cfg.get("tie", True),
        pe_mode=cfg.get("pe_mode", "none"), fold_mode=cfg.get("fold_mode", "left"),
        rot_mode=cfg.get("rot_mode", "free"))
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model


def _resolve_ckpt_path(cfg_path, cfg):
    ckpt_path = cfg["ckpt"]
    if not os.path.isabs(ckpt_path) or not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(cfg_path)),
                                 os.path.basename(cfg["ckpt"]))
    return ckpt_path


def collect_episode(policy_model, cfg, rng):
    """One self-play episode: policy_model (sampling, force_harvest=False)
    vs a randomly-chosen fixed opponent. Returns
    (turn_records, reward, own_money, opp_money). turn_records is a list
    of {"tokens", "geoms", "choices": [{"abs_pos","allowed_ids","token"}]}
    -- reward gets attached to every record after the episode ends,
    since it's a terminal, whole-episode signal (no per-turn shaping)."""
    dec = OperaDecoder(policy_model)
    use_geom = cfg.get("use_geom", True)
    geom_block = cfg.get("geom_block", kc.GEOM_BLOCK)
    context_window = cfg.get("context_window", cfg.get(
        "max_len", policy.DEFAULT_CONTEXT_WINDOW))

    window_tokens, window_geoms = [], []
    turn_records = []
    seat = rng.randint(0, 1)

    def tracing_agent(obs):
        nonlocal window_tokens, window_geoms
        if obs.get("day", 0) == 0 and obs.get("hour", 0) == 0:
            dec.reset()
            window_tokens, window_geoms = [], []

        farm = obs["farms"][obs["player"]]
        seeds = (obs.get("private") or {}).get("seeds", {})
        obs_tok, obs_geom = kc.encode_observation(obs, use_geom=use_geom)

        if dec.t > 0 and dec.t + len(obs_tok) + policy.MAX_ACTION_TOKENS > context_window:
            dec.reset()
            window_tokens, window_geoms = [], []

        for tok, geom in zip(obs_tok, obs_geom):
            dec.append(tok, geom=geom, geom_block=geom_block)
        window_tokens.extend(obs_tok)
        window_geoms.extend(obs_geom)

        base_offset = len(window_tokens)
        record = []
        action, act_tok, act_geom = policy.generate_action(
            dec, farm, obs.get("day", 0), seeds, use_geom, geom_block,
            mode="sample", force_harvest=False, record=record)
        window_tokens.extend(act_tok)
        window_geoms.extend(act_geom)

        if record:
            turn_records.append({
                "tokens": list(window_tokens),
                "geoms": list(window_geoms),
                "choices": [{"abs_pos": base_offset + r["local_pos"],
                            "allowed_ids": r["allowed_ids"],
                            "token": r["token"]} for r in record],
            })
        return action

    opponent = rng.choices(OPPONENTS, weights=OPPONENT_WEIGHTS, k=1)[0]
    agents = [tracing_agent, opponent] if seat == 0 else [opponent, tracing_agent]
    env = make("kaggriculture", configuration={"episodeSteps": 720})
    env.run(agents)
    final = env.steps[-1]
    own_money = final[seat].reward
    opp_money = final[1 - seat].reward
    reward = math.tanh((own_money - opp_money) / 2000.0)
    for tr in turn_records:
        tr["reward"] = reward
    return turn_records, reward, own_money, opp_money


def _pack_window(tokens, geoms):
    T = len(tokens)
    geom_arr = np.zeros((T, 3), dtype=np.float32)
    geom_mask = np.zeros((T,), dtype=bool)
    for i, g in enumerate(geoms):
        if g is not None:
            geom_arr[i] = g
            geom_mask[i] = True
    return np.array(tokens, dtype=np.int64), geom_arr, geom_mask


def _batch_windows(records, device):
    B = len(records)
    T = max(len(r["tokens"]) for r in records)
    token_ids = torch.zeros(B, T, dtype=torch.long)
    geom = torch.zeros(B, T, 3, dtype=torch.float32)
    geom_mask = torch.zeros(B, T, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, r in enumerate(records):
        tok, ga, gm = _pack_window(r["tokens"], r["geoms"])
        t = len(tok)
        token_ids[i, :t] = torch.from_numpy(tok)
        geom[i, :t] = torch.from_numpy(ga)
        geom_mask[i, :t] = torch.from_numpy(gm)
        lengths[i] = t
    return (token_ids.to(device), geom.to(device), geom_mask.to(device),
           lengths.to(device))


def _score_records(model, records, device):
    """Full-vocab logits [B,T,V] for the batch of recorded windows;
    caller indexes per-choice (position, allowed_ids)."""
    token_ids, geom, geom_mask, lengths = _batch_windows(records, device)
    out = model(token_ids, lengths, geom=geom, geom_mask=geom_mask,
               head_last_only=True)
    return out.logits[-1]


def compute_losses(policy_model, reference_model, records, baseline, device):
    """REINFORCE-with-baseline policy loss + mean KL(policy || reference)
    over the exact restricted candidate sets sampling actually used."""
    logits = _score_records(policy_model, records, device)
    with torch.no_grad():
        ref_logits = _score_records(reference_model, records, device)

    policy_loss = torch.zeros((), device=device)
    kl_total = torch.zeros((), device=device)
    n_choices = 0
    for i, r in enumerate(records):
        advantage = r["reward"] - baseline
        turn_logprob = torch.zeros((), device=device)
        for c in r["choices"]:
            pos = c["abs_pos"] - 1     # logits[pos] predicts the token AT abs_pos
            allowed = torch.tensor(c["allowed_ids"], dtype=torch.long, device=device)
            row = logits[i, pos, :][allowed]
            ref_row = ref_logits[i, pos, :][allowed]
            log_p = F.log_softmax(row, dim=-1)
            log_q = F.log_softmax(ref_row, dim=-1)
            token_idx = c["allowed_ids"].index(c["token"])
            turn_logprob = turn_logprob + log_p[token_idx]
            p = log_p.exp()
            kl_total = kl_total + (p * (log_p - log_q)).sum()
            n_choices += 1
        policy_loss = policy_loss - advantage * turn_logprob
    policy_loss = policy_loss / max(1, len(records))
    kl_loss = kl_total / max(1, n_choices)
    return policy_loss, kl_loss


def build_eval_agent(model, cfg, force_harvest):
    """Argmax inference agent, mirrors main.py's agent() but with
    force_harvest as an explicit parameter (main.py reads it from
    config; here we sweep it to test whether RL superseded the hack)."""
    dec = OperaDecoder(model)
    use_geom = cfg.get("use_geom", True)
    geom_block = cfg.get("geom_block", kc.GEOM_BLOCK)
    context_window = cfg.get("context_window", cfg.get(
        "max_len", policy.DEFAULT_CONTEXT_WINDOW))

    def agent(obs):
        if obs.get("day", 0) == 0 and obs.get("hour", 0) == 0:
            dec.reset()
        farm = obs["farms"][obs["player"]]
        seeds = (obs.get("private") or {}).get("seeds", {})
        obs_tok, obs_geom = kc.encode_observation(obs, use_geom=use_geom)
        if dec.t > 0 and dec.t + len(obs_tok) + policy.MAX_ACTION_TOKENS > context_window:
            dec.reset()
        for tok, geom in zip(obs_tok, obs_geom):
            dec.append(tok, geom=geom, geom_block=geom_block)
        action, _tok, _geom = policy.generate_action(
            dec, farm, obs.get("day", 0), seeds, use_geom, geom_block,
            mode="argmax", force_harvest=force_harvest)
        return action

    return agent


def run_matches(agent_fn, opponent, n):
    wins = losses = ties = 0
    for i in range(n):
        seat = i % 2
        agents = [agent_fn, opponent] if seat == 0 else [opponent, agent_fn]
        env = make("kaggriculture", configuration={"episodeSteps": 720})
        env.run(agents)
        final = env.steps[-1]
        own, opp = final[seat].reward, final[1 - seat].reward
        if own > opp:
            wins += 1
        elif opp > own:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def evaluate_and_save(policy_model, bc_cfg, a, it):
    for force_harvest in (True, False):
        agent_fn = build_eval_agent(policy_model, bc_cfg, force_harvest)
        wins, losses, ties = run_matches(agent_fn, "starter", a.eval_episodes)
        print(f"    eval vs starter, force_harvest={force_harvest}: "
              f"{wins}/{a.eval_episodes} wins, {losses} losses, {ties} ties",
              flush=True)

    ckpt_path = os.path.join(a.out_dir, f"kagri_rl_iter{it + 1}.pt")
    torch.save(policy_model.state_dict(), ckpt_path)
    cfg = dict(bc_cfg)
    cfg["ckpt"] = ckpt_path
    # Default the RL-tuned checkpoint to NOT need the hand-scripted
    # override -- that's the whole point of this phase. If the
    # force_harvest=False eval above is still much worse, this is the
    # honest signal to flip it back for an actual deployment while
    # continuing to train.
    cfg["force_harvest"] = False
    with open(os.path.join(a.out_dir, "model_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"    checkpoint -> {ckpt_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bc-config", default="model_config.json")
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--episodes-per-iter", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl-coef", type=float, default=1.0)
    p.add_argument("--minibatch-size", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-episodes", type=int, default=6)
    p.add_argument("--out-dir", default="runs_kagri_rl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    with open(a.bc_config) as f:
        bc_cfg = json.load(f)
    ckpt_path = _resolve_ckpt_path(a.bc_config, bc_cfg)

    policy_model = _load_model(bc_cfg, ckpt_path).to(a.device)
    reference_model = _load_model(bc_cfg, ckpt_path).to(a.device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad_(False)

    opt = torch.optim.AdamW(policy_model.parameters(), lr=a.lr)
    rng = random.Random(a.seed)

    print(f"RL fine-tune: {a.iterations} iterations x {a.episodes_per_iter} "
          f"episodes, lr={a.lr}, kl_coef={a.kl_coef}", flush=True)

    for it in range(a.iterations):
        t0 = time.time()
        all_records, rewards = [], []
        for _ in range(a.episodes_per_iter):
            recs, reward, own_m, opp_m = collect_episode(policy_model, bc_cfg, rng)
            all_records.extend(recs)
            rewards.append(reward)
        collect_time = time.time() - t0

        mean_reward = sum(rewards) / len(rewards)
        # Per-batch mean, not a cross-iteration EMA: an EMA starting at 0
        # badly lags the true mean for the first several iterations
        # (momentum 0.9 -> only 10%/iteration catch-up), so early on
        # nearly every episode's advantage (reward - baseline) is large
        # and the SAME sign -- there's no "this one was less bad than
        # that one" signal, just "push down everything, hard, uniformly"
        # -- exactly what caused KL to blow up (0.07 -> 4.8 over 5
        # iterations) and the policy to collapse below the BC baseline
        # in the first real run. A plain per-batch mean zero-centers
        # advantages within THIS batch from iteration 1, which is the
        # standard choice for small-batch REINFORCE.
        baseline = mean_reward

        policy_model.train()
        rng.shuffle(all_records)
        total_policy_loss = total_kl_loss = 0.0
        n_batches = 0
        for i in range(0, len(all_records), a.minibatch_size):
            batch = all_records[i:i + a.minibatch_size]
            if not batch:
                continue
            policy_loss, kl_loss = compute_losses(
                policy_model, reference_model, batch, baseline, a.device)
            loss = policy_loss + a.kl_coef * kl_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            opt.step()
            total_policy_loss += policy_loss.item()
            total_kl_loss += kl_loss.item()
            n_batches += 1
        policy_model.eval()

        train_time = time.time() - t0 - collect_time
        print(f"iter {it:3d}  mean_reward {mean_reward:+.3f}  "
              f"baseline {baseline:+.3f}  "
              f"policy_loss {total_policy_loss / max(1, n_batches):.4f}  "
              f"kl {total_kl_loss / max(1, n_batches):.4f}  "
              f"turns {len(all_records)}  "
              f"({collect_time:.1f}s collect, {train_time:.1f}s train)",
              flush=True)

        if (it + 1) % a.eval_every == 0 or it == a.iterations - 1:
            evaluate_and_save(policy_model, bc_cfg, a, it)


if __name__ == "__main__":
    main()
