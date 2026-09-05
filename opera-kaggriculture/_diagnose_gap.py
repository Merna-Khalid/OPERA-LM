"""Throwaway diagnostic: run several OPERA-vs-starter episodes, and for
each track economic indicators (hands hired, quadrants owned, harvest/
plant/market-order counts, movement-direction distribution) to find
what actually differs between episodes OPERA wins vs loses.
"""
import sys
from collections import Counter
from kaggle_environments import make
import main as m

N_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def run_one(seat0_is_opera=True):
    stats = {"verbs": Counter(), "market_verbs": Counter(),
            "max_hands": 0, "max_quads": 1, "money_hist": []}

    def tracing_agent(obs):
        farm = obs["farms"][obs["player"]]
        act = m.agent(obs)
        stats["verbs"][act["farmer"][0]] += 1
        for h in act.get("hands", []):
            stats["verbs"][h[0]] += 1
        for order in act.get("market", []):
            stats["market_verbs"][order[0]] += 1
        stats["max_hands"] = max(stats["max_hands"], len(farm.get("hands", [])))
        stats["max_quads"] = max(stats["max_quads"],
                                 len(farm.get("unlocked_quadrants", ["NW"])))
        stats["money_hist"].append(farm["money"])
        return act

    agents = [tracing_agent, "starter"] if seat0_is_opera else ["starter", tracing_agent]
    env = make("kaggriculture", configuration={"episodeSteps": 720})
    env.run(agents)
    final = env.steps[-1]
    opera_reward = final[0].reward if seat0_is_opera else final[1].reward
    starter_reward = final[1].reward if seat0_is_opera else final[0].reward
    stats["opera_reward"] = opera_reward
    stats["starter_reward"] = starter_reward
    stats["won"] = opera_reward > starter_reward
    return stats


results = []
for i in range(N_EPISODES):
    seat0 = (i % 2 == 0)
    r = run_one(seat0)
    results.append(r)
    print(f"ep{i} (opera P{0 if seat0 else 1}): "
          f"opera={r['opera_reward']:.0f} starter={r['starter_reward']:.0f} "
          f"won={r['won']} max_hands={r['max_hands']} max_quads={r['max_quads']} "
          f"market={dict(r['market_verbs'])}", flush=True)

wins = [r for r in results if r["won"]]
losses = [r for r in results if not r["won"]]
print(f"\n{len(wins)}/{len(results)} wins")


def avg(key, rows):
    return sum(r[key] for r in rows) / max(1, len(rows))


for key in ("max_hands", "max_quads"):
    print(f"avg {key}: wins={avg(key, wins):.2f} losses={avg(key, losses):.2f}")

for verb in ("HIRE", "BUY_LAND", "BUY_SEED", "SELL"):
    w = sum(r["market_verbs"].get(verb, 0) for r in wins) / max(1, len(wins))
    l = sum(r["market_verbs"].get(verb, 0) for r in losses) / max(1, len(losses))
    print(f"avg {verb} count/episode: wins={w:.1f} losses={l:.1f}")

print("\naggregate farmer+hands verb distribution across ALL episodes:")
total_verbs = Counter()
for r in results:
    total_verbs.update(r["verbs"])
print(total_verbs.most_common())
