"""
Figure 8: Distribution of Text Quality Index across generations (histogram).

Paper reference: Appendix D, Figure 8
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 14})

# ============================================================
# Configuration
# ============================================================
INPUT_CSV = "../Data/Bias_Performance_and_Generation_Quality/Synthetic_Generation_Quality.csv"
SELECTED_GENERATIONS = [1, 3, 5, 7, 9, 11]     # Generation 0, 2, 4, 6, 8, 10 in paper numbering
OUTPUT_FILE = "distribution_text_quality.png"

# ============================================================
# Plot
# ============================================================
results = pd.read_csv(INPUT_CSV)
grouped = results.groupby("Generation")["GibberishLevel"].apply(list)

plt.figure(figsize=(12, 6))
colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(SELECTED_GENERATIONS)))

for idx, generation in enumerate(SELECTED_GENERATIONS):
    if generation in grouped:
        plt.hist(grouped[generation], bins=100, alpha=0.75, color=colors[idx],
                 label=f"Generation {generation - 1}")     # CSV gen 1 = paper gen 0

plt.xlabel("Text Quality Index")
plt.ylabel("Frequency")
plt.title("Distribution of Text Quality Index Across Generations")
plt.legend()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.show()
