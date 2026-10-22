import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift, gaussian_filter

GRID = 50
SENSE_RADIUS = 2

def gaussian_map(center, sigma, size=GRID):
    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    d2 = (rows - center[0])**2 + (cols - center[1])**2
    m = np.exp(-d2 / (2 * sigma**2))
    return m / m.sum()

def make_scenario(rng):
    reported = rng.uniform(15, 35, size=2)              # where someone saw it (row, col)
    delay = rng.integers(10, 40)                        # steps between report and drone arrival
    current = rng.normal(0, 0.3, size=2)                # true drift per step
    current_est = current + rng.normal(0, 0.1, size=2)  # our imperfect estimate of it

    true_core = np.clip(reported + current * delay, 0, GRID - 1)
    prior_center = np.clip(reported + current_est * delay, 0, GRID - 1)
    prior_sigma = 2 + 0.15 * delay                      # older report = more uncertainty
    prior = gaussian_map(prior_center, prior_sigma)

    return dict(reported=reported, delay=delay, current=current,
                current_est=current_est, true_core=true_core, prior=prior)

def predict(prob, current_est, blur=0.5):
    """Predict step: move the map with the estimated current, spread it a bit."""
    moved = shift(prob, current_est, order=1, mode="constant")
    moved = gaussian_filter(moved, blur)
    s = moved.sum()
    return moved / s if s > 0 else prob

def drift_core(core, current):
    """The real bloom keeps drifting while the drone searches."""
    return np.clip(core + current, 0, GRID - 1)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sc = make_scenario(rng)

    plt.imshow(sc["prior"], origin="lower", cmap="viridis")
    plt.scatter(sc["reported"][1], sc["reported"][0], c="white", marker="x", s=100, label="reported")
    plt.scatter(sc["true_core"][1], sc["true_core"][0], c="red", marker="o", s=80, label="actual bloom now")
    plt.title(f"Where the bloom probably is (report was {sc['delay']} steps ago)")
    plt.legend()
    plt.colorbar(label="probability")
    plt.show()