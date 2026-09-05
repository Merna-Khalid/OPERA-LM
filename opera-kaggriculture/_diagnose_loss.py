"""Throwaway diagnostic: trace one OPERA-vs-starter episode turn by turn,
comparing the BC model's chosen action against what the heuristic
teacher would have done in the exact same spot, to localize where/why
the live agent's behavior diverges from the policy it was cloned from.
"""
import sys
from collections import Counter
from kaggle_environments import make
import main as m
from heuristic_bot import policy as heuristic_policy

log = []


def tracing_agent(obs):
    farm = obs["farms"][obs["player"]]
    act = m.agent(obs)
    teacher_act = heuristic_policy(obs)
    log.append({
        "day": obs["day"], "hour": obs["hour"],
        "farmer": tuple(farm["farmer"]), "money": farm["money"],
        "act": act, "teacher_act": teacher_act,
        "match": act == teacher_act,
    })
    return act


env = make("kaggriculture", configuration={"episodeSteps": 720})
env.run([tracing_agent, "starter"])
final = env.steps[-1]
print("final rewards:", final[0].reward, final[1].reward)

n = len(log)
matches = sum(1 for r in log if r["match"])
print(f"live turn-by-turn match vs teacher: {matches}/{n} = {matches/n:.1%}")

# Farmer-verb-only match (ignore hands/market, just: did it pick the
# same verb the teacher would have, regardless of exact args)
verb_matches = sum(1 for r in log
                   if r["act"]["farmer"][0] == r["teacher_act"]["farmer"][0])
print(f"farmer VERB-only match: {verb_matches}/{n} = {verb_matches/n:.1%}")

# Money trajectory
print("\nmoney trajectory:")
for idx in (0, 24, 100, 240, 360, 480, 600, 719):
    if idx < n:
        r = log[idx]
        print(f"  turn {idx}: day={r['day']} money={r['money']} "
              f"farmer={r['farmer']} act={r['act']['farmer']} "
              f"(teacher would: {r['teacher_act']['farmer']})")

# Repetition check: how often does the SAME farmer action repeat back to
# back, and how often does farmer position repeat (stuck in place)?
same_action_streak = 0
max_streak = 0
same_pos_streak = 0
max_pos_streak = 0
for i in range(1, n):
    if log[i]["act"]["farmer"] == log[i - 1]["act"]["farmer"]:
        same_action_streak += 1
        max_streak = max(max_streak, same_action_streak)
    else:
        same_action_streak = 0
    if log[i]["farmer"] == log[i - 1]["farmer"]:
        same_pos_streak += 1
        max_pos_streak = max(max_pos_streak, same_pos_streak)
    else:
        same_pos_streak = 0
print(f"\nlongest identical-consecutive-farmer-action streak: {max_streak}")
print(f"longest farmer-stuck-in-same-position streak: {max_pos_streak}")

# Most common farmer verbs chosen overall (live) vs teacher's
live_verbs = Counter(r["act"]["farmer"][0] for r in log)
teacher_verbs = Counter(r["teacher_act"]["farmer"][0] for r in log)
print("\nlive farmer verb distribution:", live_verbs.most_common())
print("teacher farmer verb distribution (same states):", teacher_verbs.most_common())

# First 40 turns in detail (mirrors the earlier heuristic-bug trace style)
print("\nfirst 40 turns:")
for r in log[:40]:
    tag = "OK" if r["match"] else "DIFF"
    print(f"  [{tag}] day{r['day']} h{r['hour']} pos={r['farmer']} "
          f"money={r['money']} live={r['act']['farmer']} "
          f"teacher={r['teacher_act']['farmer']}")
