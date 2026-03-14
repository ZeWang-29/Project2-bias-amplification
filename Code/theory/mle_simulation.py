"""
Figure 11: Standard Maximum Likelihood Estimation (MLE) simulation (control).

Shows that without pretrained bias weighting, estimated distributions remain
stable across generations — no bias amplification occurs.

Paper reference: Appendix L (Theoretical Intuition), Figure 11
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ============================================================
# Configuration
# ============================================================
OUTPUT_FILE = "MLE.png"
N_SAMPLES = 10000
N_GENERATIONS = 11
SELECTED_GENERATIONS = [1, 5, 7, 11]

generation_colors = {1: "green", 5: "blue", 7: "purple", 11: "orange"}
generation_labels = {1: "Generation 0", 5: "Generation 4", 7: "Generation 6", 11: "Generation 10"}

# ============================================================
# MLE Simulation (no weighting)
# ============================================================
def neg_log_likelihood_beta(params, data):
    a, b = params
    if a <= 0 or b <= 0:
        return np.inf
    return -np.sum(np.log(beta.pdf(data, a, b)))

bounds = [(1e-6, None), (1e-6, None)]

# Fine-tuning data from true Beta(2,2)
alpha_true, beta_true = 2, 2
data_finetune = np.random.beta(alpha_true, beta_true, size=N_SAMPLES)

# Iterative standard MLE
data_synthetic = data_finetune.copy()
generations_data = []
initial_params = [1.0, 1.0]

for gen in range(1, N_GENERATIONS + 1):
    result = minimize(neg_log_likelihood_beta, initial_params, args=(data_synthetic,), bounds=bounds)
    alpha_est, beta_est = result.x
    data_synthetic = np.random.beta(alpha_est, beta_est, size=N_SAMPLES)

    if gen in SELECTED_GENERATIONS:
        generations_data.append((gen, data_synthetic.copy(), alpha_est, beta_est))

    initial_params = [alpha_est, beta_est]

# Plot
x = np.linspace(0, 1, 100)
plt.figure(figsize=(12, 8))
plt.plot(x, beta.pdf(x, alpha_true, beta_true), color="red", linestyle="--", lw=2, label="Beta(2,2)")

for gen, data_gen, _, _ in generations_data:
    sns.kdeplot(data_gen, bw_adjust=0.5, label=generation_labels[gen],
                color=generation_colors[gen], linewidth=2)

plt.title("Beta Distributions Estimation Using MLE", fontsize=16)
plt.xlabel("Value", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
plt.show()
