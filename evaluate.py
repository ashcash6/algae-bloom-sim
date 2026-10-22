import time
import numpy as np
import torch
from world import GRID, make_scenario, predict, drift_core
from expert import START, MOVES, GAMMA, can_see, cross_off, legal_moves, run_expert
from dataset import make_obs
from train import Policy
from compare import run_lawnmower, run_greedy, MAX_STEPS

model = Policy()
model.load_state_dict(torch.load("policy.pt"))
model.eval()

def run_learned(sc):
    prob = sc["prior"].copy()
    core = sc["true_core"].copy()
    pos = START
    trail = np.zeros((GRID, GRID))
    for step in range(1, MAX_STEPS + 1):
        if can_see(pos, core):
            return step
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
    return MAX_STEPS

if __name__ == "__main__":
    rng = np.random.default_rng(1)   # same test scenarios as compare.py, never trained on
    N = 200
    scenarios = [make_scenario(rng) for _ in range(N)]

    strategies = [
        ("Lawnmower", run_lawnmower),
        ("Greedy", run_greedy),
        ("Ergodic expert", lambda sc: run_expert(sc, max_steps=MAX_STEPS)[0]),
        ("Learned policy", run_learned),
    ]

    for name, fn in strategies:
        t0 = time.time()
        res = np.array([fn(sc) for sc in scenarios])
        elapsed = time.time() - t0
        print(f"{name:16s} avg {res.mean():6.1f}   median {np.median(res):6.1f}   "
              f"failed {np.sum(res >= MAX_STEPS):3d}/{N}   time {elapsed:5.1f}s")