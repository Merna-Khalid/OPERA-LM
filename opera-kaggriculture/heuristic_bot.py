"""heuristic_bot.py -- rule-based Kaggriculture agent (Phase 2 of the
plan). Two jobs: (1) a valid, dependency-free submission from day one,
(2) the self-play teacher generate_selfplay.py records for Phase 4's
behavior cloning.

Scope (v1, deliberate): multi-tile, multi-crop farming (WHEAT/CARROT/
MELON) across every currently-owned tile, opportunistic land purchases
and hand hiring, sells the shed's entire contents every turn. Skips
animals (GOOSE/COW/SHEEP) and FERTILIZE -- those need a PICKUP-from-shed
carry step and a longer buy -> build structure -> place chain, and the
crop loop alone already (a) clears the "beat starter" gate and (b)
generates genuinely varied multi-unit, multi-tile self-play data for
Phase 4. Revisit animals as a v2 improvement if behavior cloning on this
data plateaus below the teacher's own ceiling.

Movement is memoryless (recomputed fresh from `obs` every call, no
internal state): each unit heads for the nearest unclaimed task
(harvest > water > dig weed > plant), one Manhattan step per turn,
preferring the axis with the larger remaining distance.
"""
CROPS_SEED_COST = {"WHEAT": 10, "CARROT": 20, "MELON": 80}
# first_yield_day per crop (kaggriculture.py's CROPS table): one-time
# crops get yield_units=1 the INSTANT they're planted (_new_plant), but
# HARVEST is still blocked by the engine until this many days have
# passed -- checking yield_units alone (without this gate) makes the
# bot spam a no-op HARVEST instead of WATERing, and an unwatered plant
# turns into a weed after 2 consecutive missed days.
CROPS_FIRST_YIELD_DAY = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8,
                        "STRAWBERRY": 10, "MELON": 10}
CROP_PRIORITY = ["WHEAT", "CARROT", "MELON"]   # cheap+fast first
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
           "EGG", "MILK", "WOOL", "FERTILIZER"]
LAND_PRICES = [1000, 2000, 4000]
MAX_HANDS = 3


def _manhattan_step(pos, target):
    fx, fy = pos
    tx, ty = target
    dx, dy = tx - fx, ty - fy
    if dx == 0 and dy == 0:
        return None
    if abs(dx) >= abs(dy) and dx != 0:
        return "EAST" if dx > 0 else "WEST"
    if dy != 0:
        return "SOUTH" if dy > 0 else "NORTH"
    return "EAST" if dx > 0 else "WEST"


def _tile_task(tile, day):
    """(priority, verb) for a tile that needs unit attention, else None.
    Lower priority = more urgent."""
    if not isinstance(tile, dict):
        return None
    kind = tile.get("kind")
    if kind == "PLANT":
        crop = tile.get("crop")
        first_yield = CROPS_FIRST_YIELD_DAY.get(crop, 0)
        mature = (day - tile.get("planted_day", day)) >= first_yield
        if mature and tile.get("yield_units", 0) > 0:
            return (0, "HARVEST")
        if not tile.get("watered_today"):
            return (1, "WATER")
        return None
    if kind == "WEED":
        return (2, "DIG")
    return None


def _find_tasks(tiles, board_size, day):
    tasks = []
    for y in range(board_size):
        for x in range(board_size):
            t = _tile_task(tiles[y][x], day)
            if t is not None:
                tasks.append((t[0], (x, y), t[1]))
    tasks.sort(key=lambda t: t[0])
    return tasks


def _plantable_tiles(tiles, board_size):
    return [(x, y) for y in range(board_size) for x in range(board_size)
            if tiles[y][x] is None]


def _choose_crop(seeds):
    for c in CROP_PRIORITY:
        if seeds.get(c, 0) > 0:
            return c
    return None


def _next_land_cost(farm):
    n_owned = len(farm.get("unlocked_quadrants", ["NW"]))
    idx = n_owned - 1              # NW is free/default, not a purchase
    return LAND_PRICES[idx] if 0 <= idx < len(LAND_PRICES) else None


def _decide_market(farm, private):
    money = farm.get("money", 0)
    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    orders = []

    for p in PRODUCTS:
        n = shed.get(p, 0)
        if n > 0:
            orders.append(["SELL", p, n])

    for crop in CROP_PRIORITY:
        cost = CROPS_SEED_COST[crop]
        if seeds.get(crop, 0) == 0 and money >= cost * 3:
            orders.append(["BUY_SEED", crop, 3])
            break                   # one seed purchase per turn is plenty

    if farm.get("hires_today", 0) == 0 and money > 2500 \
            and len(farm.get("hands", [])) < MAX_HANDS:
        orders.append(["HIRE"])

    if len(farm.get("unlocked_quadrants", ["NW"])) < 4:
        cost = _next_land_cost(farm)
        if cost is not None and money > cost * 3:
            orders.append(["BUY_LAND"])

    return orders[:10]


def _decide_units(farm, private, board_size, day):
    tiles = farm["tiles"]
    seeds = private.get("seeds", {}) or {}
    units = [tuple(farm["farmer"])] + [tuple(h) for h in farm.get("hands", [])]

    tasks = _find_tasks(tiles, board_size, day)              # [(prio, pos, verb)]
    task_map = {pos: verb for _, pos, verb in tasks}
    # PLANT-on-empty-tile is priority 3: strictly worse than any real
    # task (harvest=0/water=1/dig=2), so units always finish tending
    # existing plants before wandering off to start new ones -- planting
    # is what caused the original bug (units abandoned freshly-planted,
    # not-yet-watered tiles to go plant somewhere else, and the
    # abandoned plants died to the 2-day-unwatered rule).
    plantable = (_plantable_tiles(tiles, board_size)
                if _choose_crop(seeds) is not None else [])
    all_targets = [(p, pos) for p, pos, _ in tasks] + [(3, pos) for pos in plantable]

    claimed = set()
    actions = []
    for pos in units:
        x, y = pos
        in_bounds = 0 <= x < board_size and 0 <= y < board_size
        tile = tiles[y][x] if in_bounds else None

        if in_bounds and pos in task_map and pos not in claimed:
            claimed.add(pos)
            actions.append([task_map[pos]])
            continue
        if in_bounds and tile is None and pos in plantable and pos not in claimed:
            crop = _choose_crop(seeds)
            claimed.add(pos)
            actions.append(["PLANT", crop])
            continue

        candidates = [(p, t) for p, t in all_targets if t not in claimed]
        if candidates:
            best_prio = min(p for p, _ in candidates)
            target = min((t for p, t in candidates if p == best_prio),
                        key=lambda t: abs(t[0] - x) + abs(t[1] - y))
            claimed.add(target)
            step = _manhattan_step(pos, target)
            actions.append([step] if step else ["PASS"])
        else:
            actions.append(["PASS"])
    return actions[0], actions[1:]


def policy(obs):
    """obs -> action dict, the game engine's exact expected shape."""
    farms = obs.get("farms", [])
    player = obs.get("player", 0)
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    farm = farms[player]
    private = obs.get("private") or {}
    board_size = len(farm["tiles"])

    market = _decide_market(farm, private)
    day = obs.get("day", 0)
    farmer_action, hands_actions = _decide_units(farm, private, board_size, day)
    return {"farmer": farmer_action, "hands": hands_actions, "market": market}


def agent(obs):
    return policy(obs)


agents = {"heuristic": agent}
