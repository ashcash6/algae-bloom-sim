import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from world import GRID, make_scenario, predict, drift_core
from expert import START, MOVES, GAMMA, can_see, cross_off, legal_moves, run_expert
from dataset import make_obs
from evaluate import model
from compare import MAX_STEPS

def greedy_path(sc):
    prob, core, pos = sc["prior"].copy(), sc["true_core"].copy(), START
    path, cores = [pos], [core.copy()]
    for step in range(1, MAX_STEPS + 1):
        if can_see(pos, core):
            return step, path, cores
        cross_off(prob, pos)
        tr, tc = np.unravel_index(np.argmax(prob), prob.shape)
        dr, dc = tr - pos[0], tc - pos[1]
        if abs(dr) >= abs(dc) and dr != 0:
            pos = (pos[0] + int(np.sign(dr)), pos[1])
        elif dc != 0:
            pos = (pos[0], pos[1] + int(np.sign(dc)))
        prob = predict(prob, sc["current_est"])
        core = drift_core(core, sc["current"])
        path.append(pos); cores.append(core.copy())
    return MAX_STEPS, path, cores

def learned_path(sc):
    prob, core, pos = sc["prior"].copy(), sc["true_core"].copy(), START
    trail = np.zeros((GRID, GRID))
    path, cores = [pos], [core.copy()]
    for step in range(1, MAX_STEPS + 1):
        if can_see(pos, core):
            return step, path, cores
        cross_off(prob, pos)
        obs = torch.tensor(make_obs(pos, prob, trail)).unsqueeze(0)
        with torch.no_grad():
            logits = model(obs)[0]
        legal = [i for i, _ in legal_moves(pos)]
        mask = torch.full((4,), float("-inf"))
        mask[legal] = 0
        a = int((logits + mask).argmax())
        dr, dc = MOVES[a]
        pos = (pos[0] + dr, pos[1] + dc)
        trail *= GAMMA
        trail[pos] = 1.0
        prob = predict(prob, sc["current_est"])
        core = drift_core(core, sc["current"])
        path.append(pos); cores.append(core.copy())
    return MAX_STEPS, path, cores

if __name__ == "__main__":
    os.makedirs("paths", exist_ok=True)
    rng = np.random.default_rng(1)
    for k in range(6):
        sc = make_scenario(rng)
        runs = [
            ("Greedy", greedy_path(sc)),
            ("Ergodic expert", run_expert(sc, max_steps=MAX_STEPS)),
            ("Learned policy", learned_path(sc)),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (name, (steps, path, cores)) in zip(axes, runs):
            ax.imshow(sc["prior"], origin="lower", cmap="viridis")
            ax.plot([p[1] for p in path], [p[0] for p in path], c="white", linewidth=1)
            ax.plot([c[1] for c in cores], [c[0] for c in cores], c="red", linewidth=2)
            ax.scatter(cores[-1][1], cores[-1][0], c="red", s=60)
            ax.scatter(sc["reported"][1], sc["reported"][0], c="white", marker="x", s=80)
            ax.set_title(f"{name}: {steps} steps")
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Scenario {k + 1} (report {sc['delay']} steps old)  "
                     f"white = drone, red = bloom drift, x = reported spot")
        plt.tight_layout()
        plt.savefig(f"paths/scenario_{k + 1:02d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("Saved 6 images in paths/")