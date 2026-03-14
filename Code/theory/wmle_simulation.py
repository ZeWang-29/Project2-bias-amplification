"""
Figure 10: Weighted Maximum Likelihood Estimation (WMLE) simulation.

Demonstrates bias amplification through iterative estimation when the
fine-tuning process is influenced by a biased pretrained model (Beta(3,2)).

Paper reference: Appendix L (Theoretical Intuition), Figure 10
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
OUTPUT_FILE = "WMLE.png"
N_SAMPLES = 10000
N_GENERATIONS = 11
SELECTED_GENERATIONS = [1, 5, 7, 11]   # Generations to plot

generation_colors = {1: "green", 5: "blue", 7: "purple", 11: "orange"}
generation_labels = {1: "Generation 0", 5: "Generation 4", 7: "Generation 6", 11: "Generation 10"}

# ============================================================
# WMLE Simulation
# ============================================================
def neg_log_likelihood_beta(params, data):
    a, b = params
    if a <= 0 or b <= 0:
        return np.inf
    return -np.sum(np.log(beta.pdf(data, a, b)))

def neg_log_likelihood_beta_weighted(params, data, weights):
    a, b = params
    if a <= 0 or b <= 0:
        return np.inf
    return -np.sum(weights * np.log(beta.pdf(data, a, b)))

bounds = [(1e-6, None), (1e-6, None)]

# Step 1: Pretraining — estimate Beta(3,2) (biased pretrained distribution)
alpha_pretrain, beta_pretrain = 3, 2
data_pretrain = np.random.beta(alpha_pretrain, beta_pretrain, size=N_SAMPLES)
result = minimize(neg_log_likelihood_beta, [1.0, 1.0], args=(data_pretrain,), bounds=bounds)
alpha_est_pre, beta_est_pre = result.x

# Step 2: Fine-tuning data from true Beta(2,2) (unbiased)
alpha_true, beta_true = 2, 2
data_finetune = np.random.beta(alpha_true, beta_true, size=N_SAMPLES)

# Step 3: Iterative WMLE
data_synthetic = data_finetune.copy()
generations_data = []
initial_params = [2.0, 2.0]

for gen in range(1, N_GENERATIONS + 1):
    weights = beta.pdf(data_synthetic, alpha_est_pre, beta_est_pre)
    result = minimize(neg_log_likelihood_beta_weighted, initial_params,
                      args=(data_synthetic, weights), bounds=bounds)
    alpha_est, beta_est = result.x
    data_synthetic = np.random.beta(alpha_est, beta_est, size=N_SAMPLES)

    if gen in SELECTED_GENERATIONS:
        generations_data.append((gen, data_synthetic.copy(), alpha_est, beta_est))

    alpha_est_pre, beta_est_pre = alpha_est, beta_est
    initial_params = [alpha_est, beta_est]

# Step 4: Plot
x = np.linspace(0, 1, 100)
plt.figure(figsize=(12, 8))
plt.plot(x, beta.pdf(x, 3, 2), color="black", linestyle="-", lw=2, label="Beta(3,2)")
plt.plot(x, beta.pdf(x, 2, 2), color="red", linestyle="--", lw=2, label="Beta(2,2)")

for gen, data_gen, _, _ in generations_data:
    sns.kdeplot(data_gen, bw_adjust=0.5, label=generation_labels[gen],
                color=generation_colors[gen], linewidth=2)

plt.title("Beta Distributions Estimation Using WMLE", fontsize=16)
plt.xlabel("Value", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
plt.show()
