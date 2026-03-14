"""
Figure 3b: Text Quality Index across generations with 95% confidence intervals.

Paper reference: Section 4.2, Figure 3b
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 13})

# ============================================================
# Configuration
# ============================================================
INPUT_FILES = {
    "Synthetic":                    "../Data/Bias_Performance_and_Generation_Quality/Synthetic_Generation_Quality.csv",
    "Synthetic with Preservation":  "../Data/Bias_Performance_and_Generation_Quality/Preservation_Generation_Quality.csv",
    "Synthetic with Accumulation":  "../Data/Bias_Performance_and_Generation_Quality/Accumulation_Generation_Quality.csv",
    "Synthetic with Overfitting":   "../Data/Bias_Performance_and_Generation_Quality/Overfitting_Generation_Quality.csv",
}
OUTPUT_FILE = "text_quality_index.png"

# ============================================================
# Plot
# ============================================================
plt.figure(figsize=(12, 8))

for label, file_path in INPUT_FILES.items():
    df = pd.read_csv(file_path)
    df["Generation"] = df["Generation"].astype(int)
    avg = df.groupby("Generation")["GibberishLevel"].mean()
    std = df.groupby("Generation")["GibberishLevel"].std()
    count = df.groupby("Generation")["GibberishLevel"].size()
    ci = 1.96 * (std / np.sqrt(count))

    plt.errorbar(avg.index, avg.values, yerr=ci.values,
                 marker="o", capsize=5, label=label, linestyle="-")

plt.title("Text Quality Index Across Generations")
plt.xlabel("Generation")
plt.ylabel("Text Quality Index")
plt.legend(title="Dataset")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
plt.show()
