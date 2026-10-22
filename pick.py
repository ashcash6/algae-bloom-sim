import numpy as np
from world import make_scenario
from expert import run_expert
from compare import MAX_STEPS
from plot_paths import greedy_path, learned_path

rng = np.random.default_rng(1)
for k in range(20):
    sc = make_scenario(rng)
    g = greedy_path(sc)[0]
    e = run_expert(sc, max_steps=MAX_STEPS)[0]
    l = learned_path(sc)[0]
    print(f"Scenario {k + 1:2d}:  greedy {g:4d}   expert {e:4d}   learned {l:4d}")