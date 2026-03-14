"""
Figure 4 (variant): Neuron-bias Pearson correlation scatter plot with layer coloring.

Paper reference: Section 4.4, Figure 4
"""

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15})

# ============================================================
# Configuration
# ============================================================
INPUT_CSV = "../Data/Mechanistic_Interpretation/Pearson Correlation Between Neuron Weight and Bias Performance.csv"
OUTPUT_FILE = "neuron_correlation_scatter.png"

# ============================================================
# Plot
# ============================================================
correlation_df = pd.read_csv(INPUT_CSV)

significant = correlation_df[
    (correlation_df["pearson_correlation"] >= 0.80) | (correlation_df["pearson_correlation"] <= -0.80)
]
print(f"Neurons with |correlation| >= 0.80: {len(significant)}")

plt.figure(figsize=(15, 10))

for layer in sorted(correlation_df["layer"].unique()):
    layer_data = correlation_df[correlation_df["layer"] == layer]
    plt.scatter(layer_data["neuron_id"], layer_data["pearson_correlation"],
                label=f"Layer {layer}", alpha=0.5)

plt.xlabel("Neuron ID")
plt.ylabel("Correlation Level")
plt.title("Pearson Correlation Between Neuron Weight and Bias Performance")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, bbox_inches="tight", dpi=300)
plt.show()
