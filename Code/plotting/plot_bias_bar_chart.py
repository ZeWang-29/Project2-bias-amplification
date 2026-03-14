"""
Bar charts of Left/Center/Right article distribution per generation (subplot grid).

Paper reference: Section 4.1
"""

import pandas as pd
import matplotlib.pyplot as plt
import math

# ============================================================
# Configuration
# ============================================================
INPUT_CSV = "../Data/Bias_Performance_and_Generation_Quality/Synthetic_Bias_Performance.csv"
OUTPUT_FILE = "bias_bar_chart.png"

# ============================================================
# Plot
# ============================================================
df = pd.read_csv(INPUT_CSV)

def classify_bias(row):
    scores = {"Center": row["Center_Score"], "Right": row["Right_Score"], "Left": row["Left_Score"]}
    return max(scores, key=scores.get)

df["Bias_Label"] = df.apply(classify_bias, axis=1)
df["Generation"] = df["Generation"].astype(str)

bias_counts = df.groupby(["Generation", "Bias_Label"]).size().unstack(fill_value=0)
generations = sorted(df["Generation"].unique(), key=lambda x: int(x) if x.isdigit() else -1)
bias_labels = ["Left", "Center", "Right"]

cols = 3
rows = math.ceil(len(generations) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()

for idx, generation in enumerate(generations):
    ax = axes[idx]
    counts = bias_counts.loc[generation][bias_labels] if generation in bias_counts.index else pd.Series([0, 0, 0], index=bias_labels)
    counts.plot(kind="bar", ax=ax, color=["blue", "grey", "red"])
    ax.set_title(f"Generation {generation}")
    ax.set_xlabel("Bias Label")
    ax.set_ylabel("Number of Articles")
    ax.set_ylim(0, bias_counts.max().max() + 10)
    ax.set_xticklabels(bias_labels, rotation=0)

for idx in range(len(generations), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
plt.show()
