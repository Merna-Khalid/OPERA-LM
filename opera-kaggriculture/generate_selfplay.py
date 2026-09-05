"""generate_selfplay.py -- run local Kaggriculture episodes and log raw
(obs, action) pairs per turn per player to disk (Phase 2 of the plan).
Tokenization is deliberately deferred to prepare_kagri_data.py (Phase 3)
so the encoding scheme in kagri_common.py can iterate without
re-simulating games -- simulation is the expensive step here, not
tokenization.

Usage:
  python generate_selfplay.py --episodes 300 --out selfplay_logs.pkl
"""
import argparse
import pickle
import time

from kaggle_environments import make
from heuristic_bot import agent as heuristic_agent

# heuristic-vs-heuristic dominates (the actual policy Phase 4 imitates,
# both seats), with a slice of heuristic-vs-starter/-random for
# behavioral diversity (different opponent styles produce different
# game trajectories -- e.g. contested land/market timing against
# another heuristic instance vs. a static single-tile opponent).
MATCHUPS = [
    (heuristic_agent, heuristic_agent),
    (heuristic_agent, "starter"),
    ("starter", heuristic_agent),
    (heuristic_agent, "random"),
]
MATCHUP_WEIGHTS = [0.7, 0.15, 0.05, 0.10]


def run_one_episode(agents, steps=720):
    env = make("kaggriculture", configuration={"episodeSteps": steps})
    env.run(list(agents))
    ep_steps = env.steps
    # steps[t][i].action is the action player i's agent computed WHEN
    # SHOWN steps[t-1]'s observation -- it's the action that PRODUCED
    # steps[t], not the response TO steps[t] (verified directly: calling
    # the agent again on steps[t].observation reproduces steps[t+1].action
    # bit-for-bit, not steps[t].action). So the correct behavior-cloning
    # pair is (this step's observation, NEXT step's action); steps[0]'s
    # action is the environment's pre-agent reset default and is
    # discarded entirely, and the last step has no next action to pair
    # with, so both ends are excluded from the range below.
    episode = {0: [], 1: []}
    for t in range(len(ep_steps) - 1):
        for i in (0, 1):
            obs = ep_steps[t][i].observation
            action = ep_steps[t + 1][i].action
            episode[i].append((obs, action))
    final = ep_steps[-1]
    rewards = (final[0].reward, final[1].reward)
    return episode, rewards


def _verify_step_pairing(steps=30):
    """Regression guard for the exact bug this file's run_one_episode
    comment documents: assert steps[t+1][i].action really does equal
    heuristic_agent(steps[t][i].observation) for a few turns, using the
    heuristic (deterministic, memoryless -- see heuristic_bot.py) as an
    oracle. If a future kaggle_environments version changes this
    step/action indexing convention, this fails loudly at the start of
    generation instead of silently mislabeling the whole dataset again."""
    env = make("kaggriculture", configuration={"episodeSteps": steps})
    env.run([heuristic_agent, "random"])
    ep_steps = env.steps
    for t in range(min(10, len(ep_steps) - 1)):
        expected = heuristic_agent(ep_steps[t][0].observation)
        actual = ep_steps[t + 1][0].action
        assert actual == expected, (
            f"step-pairing assumption broke at t={t}: "
            f"steps[{t+1}][0].action={actual} != "
            f"heuristic_agent(steps[{t}][0].observation)={expected}. "
            f"run_one_episode's (obs, action) pairing needs re-deriving "
            f"against the installed kaggle_environments version.")


def main():
    _verify_step_pairing()
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--steps", type=int, default=720)
    p.add_argument("--out", default="selfplay_logs.pkl")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import random
    rng = random.Random(a.seed)

    logs = []          # list of (player_perspective_episode, rewards, matchup_tag)
    n_crashed = 0
    t0 = time.time()
    for ep in range(a.episodes):
        agents = rng.choices(MATCHUPS, weights=MATCHUP_WEIGHTS, k=1)[0]
        tag = "+".join(x if isinstance(x, str) else "heuristic" for x in agents)
        try:
            episode, rewards = run_one_episode(agents, steps=a.steps)
        except Exception as e:
            n_crashed += 1
            print(f"  episode {ep} ({tag}) CRASHED: {e!r}", flush=True)
            continue
        # Only keep the heuristic's own perspective(s) -- imitation
        # learning should learn the teacher's policy, not random's or
        # starter's.
        for i, agent in enumerate(agents):
            if agent is heuristic_agent:
                logs.append({"turns": episode[i], "reward": rewards[i],
                            "matchup": tag, "seat": i})
        if (ep + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {ep + 1}/{a.episodes} episodes "
                  f"({elapsed:.0f}s, {elapsed / (ep + 1):.1f}s/ep), "
                  f"{n_crashed} crashed so far", flush=True)

    print(f"done: {len(logs)} heuristic-perspective episodes logged, "
          f"{n_crashed} crashed", flush=True)
    with open(a.out, "wb") as f:
        pickle.dump(logs, f)
    print(f"saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
