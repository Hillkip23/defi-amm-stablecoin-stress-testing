# experiments/export_depeg_severity.py

# experiments/export_depeg_severity.py

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- path setup ---
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
sys.path.append(str(ROOT))

from defi_risk.peg_models import simulate_peg_paths
from defi_risk.peg_stress import depeg_probabilities


# -------------------------------------------------
# Configuration (paper baseline)
# -------------------------------------------------
N_PATHS = 5000
N_STEPS = 252
T = 1.0
KAPPA = 5.0
SIGMA = 0.03
P0 = 1.0
PEG = 1.0
SEED = 42

# -------------------------------------------------
# Simulate peg paths (standalone)
# -------------------------------------------------
prices_peg = simulate_peg_paths(
    model="basic_ou",
    n_paths=N_PATHS,
    n_steps=N_STEPS,
    T=T,
    kappa=KAPPA,
    sigma=SIGMA,
    p0=P0,
    peg=PEG,
    random_seed=SEED,
)

# -------------------------------------------------
# Compute depeg severity
# -------------------------------------------------
p_T = prices_peg.iloc[-1].values
severity = np.maximum(0.0, PEG - p_T)

# -------------------------------------------------
# Plot severity distribution
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(severity, bins=50, color="steelblue", edgecolor="black")
ax.set_xlabel("Depeg severity $(1 - p_T)^+$")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Terminal Depeg Severity")

fig.tight_layout()

# -------------------------------------------------
# Save figure
# -------------------------------------------------
out_path = OUT / "fig_depeg_severity.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)

print(f"Saved: {out_path}")
