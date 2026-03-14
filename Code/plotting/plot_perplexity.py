"""
Figure 9: Average perplexity across generations with 95% confidence intervals.

Paper reference: Appendix E, Figure 9
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 13})

# ============================================================
# Configuration
# ============================================================
INPUT_CSV = "../Data/Bias_Performance_and_Generation_Quality/Synthetic_Perplexity.csv"
OUTPUT_FILE = "average_perplexity.png"

# ============================================================
# Plot
# ============================================================
df = pd.read_csv(INPUT_CSV)
df["Generation"] = df["Generation"].astype(int)

avg = df.groupby("Generation")["Perplexity"].mean()
std = df.groupby("Generation")["Perplexity"].std()
count = df.groupby("Generation").size()
ci = 1.96 * (std / np.sqrt(count))

plt.figure(figsize=(14, 8))
plt.errorbar(avg.index, avg.values, yerr=ci.values,
             marker="o", capsize=5, label="Synthetic", color="blue", linestyle="-")
plt.title("Average Perplexity Across Generations")
plt.xlabel("Generation")
plt.ylabel("Perplexity")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
plt.show()
