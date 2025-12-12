import sys
from pathlib import Path

# --------------------------------------------------
# Fix import path so "src" is discoverable
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt

from src.peg_models import simulate_ou_peg_paths

# --------------------------------------------------
# Output directory
# --------------------------------------------------
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Simulation parameters (match paper defaults)
# --------------------------------------------------
prices = simulate_ou_peg_paths(
    n_paths=300,
    n_steps=252,
    T=1.0,
    kappa=5.0,
    sigma=0.02,
    p0=1.0,
    peg=1.0,
    random_seed=42,
)

# --------------------------------------------------
# Plot
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(prices.iloc[:, :40], alpha=0.4, linewidth=1)
ax.axhline(1.0, color="black", linestyle="--", label="Target peg = 1.0")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.set_title("Example OU Peg Paths")
ax.legend()
ax.grid(True)

# --------------------------------------------------
# Save figure
# --------------------------------------------------
out_path = FIG_DIR / "fig_peg_paths.png"
fig.tight_layout()
fig.savefig(out_path, dpi=300)
plt.close(fig)

print(f"Saved {out_path}")
