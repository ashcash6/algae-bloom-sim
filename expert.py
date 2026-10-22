import numpy as np
import matplotlib.pyplot as plt
from world import GRID, SENSE_RADIUS, make_scenario, predict, drift_core

K = 10
k1, k2 = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
rows, cols = np.meshgrid(np.arange(GRID) + 0.5, np.arange(GRID) + 0.5, indexing="ij")
BASIS = (np.cos(k1[:, :, None, None] * np.pi * rows / GRID) *
         np.cos(k2[:, :, None, None] * np.pi * cols / GRID))
LAMBDA = (1 + k1**2 + k2**2) ** -1.5

MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1)]
START = (3, GRID // 2)
GAMMA = 0.95  # how fast the drone forgets old coverage

def can_see(pos, core):
    r, c = int(round(core[0])), int(round(core[1]))
    return abs(pos[0] - r) <= SENSE_RADIUS and abs(pos[1] - c) <= SENSE_RADIUS

def cross_off(prob, pos):
    """Bayesian update: the area the drone sees has no bloom core."""
    R = SENSE_RADIUS
    r0, r1 = max(0, pos[0] - R), min(GRID, pos[0] + R + 1)
    c0, c1 = max(0, pos[1] - R), min(GRID, pos[1] + R + 1)
    prob[r0:r1, c0:c1] = 0
    s = prob.sum()
    if s > 0:
        prob /= s

def legal_moves(pos):
    out = []
    for i, (dr, dc) in enumerate(MOVES):
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < GRID and 0 <= nc < GRID:
            out.append((i, (nr, nc)))
    return out

def expert_action(pos, prob, visit, count):
    phi = (BASIS * prob).sum(axis=(2, 3))  # recomputed every step since the map moves
    best_i, best_score = None, np.inf
    for i, (nr, nc) in legal_moves(pos):
        c = (GAMMA * visit + BASIS[:, :, nr, nc]) / (GAMMA * count + 1)
        score = np.sum(LAMBDA * (c - phi) ** 2)
        if score < best_score:
            best_i, best_score = i, score
    return best_i

def run_expert(sc, max_steps=500):
    prob = sc["prior"].copy()
    core = sc["true_core"].copy()
    pos = START
    visit, count = np.zeros((K, K)), 0.0
    path, cores = [pos], [core.copy()]

    for step in range(1, max_steps + 1):
        if can_see(pos, core):
            return step, path, cores
        cross_off(prob, pos)                       # update: we didn't see it here
        a = expert_action(pos, prob, visit, count)
        dr, dc = MOVES[a]
        pos = (pos[0] + dr, pos[1] + dc)
        visit = GAMMA * visit + BASIS[:, :, pos[0], pos[1]]
        count = GAMMA * count + 1
        prob = predict(prob, sc["current_est"])    # predict: the bloom drifts
        core = drift_core(core, sc["current"])     # the real bloom moves
        path.append(pos); cores.append(core.copy())
    return max_steps, path, cores

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sc = make_scenario(rng)
    steps, path, cores = run_expert(sc)

    plt.imshow(sc["prior"], origin="lower", cmap="viridis")
    plt.plot([p[1] for p in path], [p[0] for p in path], c="white", linewidth=1, label="drone")
    plt.plot([c[1] for c in cores], [c[0] for c in cores], c="red", linewidth=2, label="bloom drift")
    plt.scatter(cores[-1][1], cores[-1][0], c="red", s=80)
    plt.title(f"Expert found the bloom in {steps} steps")
    plt.legend()
    plt.show()