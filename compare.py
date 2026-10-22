import numpy as np
from world import GRID, SENSE_RADIUS, make_scenario, predict, drift_core
from expert import START, can_see, cross_off, run_expert

MAX_STEPS = 800

def lawnmower_path():
    path = []
    r, c = START
    while c > 0:                      # fly to the bottom-left corner first
        c -= 1; path.append((r, c))
    while r > SENSE_RADIUS:
        r -= 1; path.append((r, c))
    spacing = 2 * SENSE_RADIUS + 1
    right = True
    while r < GRID:
        cols = range(GRID) if right else range(GRID - 1, -1, -1)
        for cc in cols:
            path.append((r, cc))
        end = path[-1][1]
        for rr in range(r + 1, min(r + spacing, GRID)):
            path.append((rr, end))
        r += spacing
        right = not right
    return path

LAWN = lawnmower_path()

def run_lawnmower(sc):
    core = sc["true_core"].copy()
    pos = START
    for step in range(1, MAX_STEPS + 1):
        if can_see(pos, core):
            return step
        pos = LAWN[min(step - 1, len(LAWN) - 1)]
        core = drift_core(core, sc["current"])
    return MAX_STEPS

def run_greedy(sc):
    prob = sc["prior"].copy()
    core = sc["true_core"].copy()
    pos = START
    for step in range(1, MAX_STEPS + 1):
        if can_see(pos, core):
            return step
        cross_off(prob, pos)
        tr, tc = np.unravel_index(np.argmax(prob), prob.shape)
        dr, dc = tr - pos[0], tc - pos[1]
        if abs(dr) >= abs(dc) and dr != 0:
            pos = (pos[0] + int(np.sign(dr)), pos[1])
        elif dc != 0:
            pos = (pos[0], pos[1] + int(np.sign(dc)))
        prob = predict(prob, sc["current_est"])
        core = drift_core(core, sc["current"])
    return MAX_STEPS

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    N = 200
    lawn, greedy, expert = [], [], []
    for i in range(N):
        sc = make_scenario(rng)
        lawn.append(run_lawnmower(sc))
        greedy.append(run_greedy(sc))
        expert.append(run_expert(sc, max_steps=MAX_STEPS)[0])

    for name, res in [("Lawnmower", lawn), ("Greedy", greedy), ("Ergodic expert", expert)]:
        res = np.array(res)
        print(f"{name:15s} avg {res.mean():6.1f} steps   "
              f"median {np.median(res):6.1f}   failed {np.sum(res >= MAX_STEPS)}/{N}")