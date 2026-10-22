import numpy as np
from world import GRID, make_scenario, predict, drift_core
from expert import K, BASIS, START, MOVES, GAMMA, can_see, cross_off, expert_action

VIEW = 5  # local window radius, so an 11x11 patch

def make_obs(pos, prob, trail):
    pad_p = np.pad(prob, VIEW)
    pad_t = np.pad(trail, VIEW)
    r, c = pos[0] + VIEW, pos[1] + VIEW
    local_p = pad_p[r - VIEW:r + VIEW + 1, c - VIEW:c + VIEW + 1] / (prob.max() + 1e-12)
    local_t = pad_t[r - VIEW:r + VIEW + 1, c - VIEW:c + VIEW + 1]
    coarse = prob.reshape(10, 5, 10, 5).sum(axis=(1, 3))   # 50x50 -> 10x10 overview
    coarse = coarse / (coarse.max() + 1e-12)
    position = np.array([pos[0] / GRID, pos[1] / GRID])
    return np.concatenate([local_p.ravel(), local_t.ravel(),
                           coarse.ravel(), position]).astype(np.float32)

def collect(n_scenarios, seed, max_steps=400):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    kept = 0
    for i in range(n_scenarios):
        sc = make_scenario(rng)
        prob = sc["prior"].copy()
        core = sc["true_core"].copy()
        pos = START
        visit, count = np.zeros((K, K)), 0.0
        trail = np.zeros((GRID, GRID))
        ep_X, ep_Y, found = [], [], False

        for step in range(max_steps):
            if can_see(pos, core):
                found = True
                break
            cross_off(prob, pos)
            a = expert_action(pos, prob, visit, count)
            ep_X.append(make_obs(pos, prob, trail))
            ep_Y.append(a)
            dr, dc = MOVES[a]
            pos = (pos[0] + dr, pos[1] + dc)
            visit = GAMMA * visit + BASIS[:, :, pos[0], pos[1]]
            count = GAMMA * count + 1
            trail *= GAMMA
            trail[pos] = 1.0
            prob = predict(prob, sc["current_est"])
            core = drift_core(core, sc["current"])

        if found:            # only learn from runs where the expert succeeded
            X += ep_X; Y += ep_Y; kept += 1
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{n_scenarios} scenarios, kept {kept}, samples so far {len(X)}")

    return np.array(X), np.array(Y)

if __name__ == "__main__":
    X, Y = collect(500, seed=123)
    np.savez("demos.npz", X=X, Y=Y)
    print("Saved", X.shape, Y.shape)
    print("Action counts (up, down, right, left):", np.bincount(Y, minlength=4))