import os
import numpy as np
import matplotlib.pyplot as plt
from world import make_scenario
from expert import START, run_expert
from compare import MAX_STEPS
from plot_paths import greedy_path, learned_path

STYLES = [("Greedy", "orange", -0.2),
          ("Ergodic expert", "cyan", 0.0),
          ("Learned policy", "magenta", 0.2)]

if __name__ == "__main__":
    os.makedirs("overlays", exist_ok=True)
    rng = np.random.default_rng(1)
    for k in range(20):
        sc = make_scenario(rng)
        results = [greedy_path(sc), run_expert(sc, max_steps=MAX_STEPS), learned_path(sc)]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(sc["prior"], origin="lower", cmap="Greys", alpha=0.6)

        for (name, color, off), (steps, path, cores) in zip(STYLES, results):
            txt = f"{steps} steps" if steps < MAX_STEPS else "not found"
            ax.plot([p[1] + off for p in path], [p[0] + off for p in path],
                    c=color, lw=2, label=f"{name}: {txt}")

        cores = max(results, key=lambda r: len(r[2]))[2]
        ax.plot([c[1] for c in cores], [c[0] for c in cores], c="red", lw=2.5, label="bloom drift")
        ax.scatter(START[1], START[0], c="black", marker="s", s=80, label="start", zorder=5)
        ax.scatter(sc["reported"][1], sc["reported"][0], c="red", marker="x", s=100,
                   label="reported spot", zorder=5)

        ax.legend(loc="upper left", fontsize=9)
        ax.set_title(f"Scenario {k + 1}")
        ax.set_xticks([]); ax.set_yticks([])
        plt.savefig(f"overlays/scenario_{k + 1:02d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("Saved 20 overlays in overlays/")