"""
Scatter plot of Newey-West adjusted p-values for neuron-bias correlations.

Paper reference: Section 4.4
"""

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15})

# ============================================================
# Configuration
# ============================================================
INPUT_CSV = "../Data/Mechanistic_Interpretation/Regression and Statistical Tests for Relationship Between Neuron Weight and Bias Performance.csv"
OUTPUT_FILE = "pvalue_scatter.png"
BONFERRONI_THRESHOLD = 0.05 / 9216

# ============================================================
# Plot
# ============================================================
correlation_df = pd.read_csv(INPUT_CSV)

plt.figure(figsize=(15, 10))

for layer in sorted(correlation_df["layer"].unique()):
    layer_data = correlation_df[correlation_df["layer"] == layer]
    plt.scatter(layer_data["neuron_id"], layer_data["significance_level_newey_west"],
                label=f"Layer {layer}", alpha=0.5)

plt.axhline(y=0.05, color="red", linestyle="--", label="p = 0.05")
plt.axhline(y=BONFERRONI_THRESHOLD, color="darkred", linestyle=":", label=f"Bonferroni ({BONFERRONI_THRESHOLD:.2e})")
plt.xlabel("Neuron ID")
plt.ylabel("P-value (Newey-West)")
plt.title("Newey-West Adjusted P-values for Neuron Weight vs Bias Performance")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, bbox_inches="tight", dpi=300)
plt.show()
