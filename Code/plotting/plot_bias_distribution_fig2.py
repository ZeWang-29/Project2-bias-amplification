"""
Figure 2: Distribution of political bias labels for initial GPT-2 synthetic outputs.

Paper reference: Section 4.1, Figure 2
"""

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
INPUT_CSV = "../Data/Bias_Performance_and_Generation_Quality/GPT2_Bias_Performance.csv"
OUTPUT_FILE = "bias_distribution_fig2.png"

# ============================================================
# Plot
# ============================================================
df = pd.read_csv(INPUT_CSV)

def classify_bias(row):
    scores = {"Center": row["Center_Score"], "Right": row["Right_Score"], "Left": row["Left_Score"]}
    return max(scores, key=scores.get)

df["Bias_Label"] = df.apply(classify_bias, axis=1)
df["Generation"] = df["Generation"].astype(str)

df_gen0 = df[df["Generation"] == "0"]
bias_counts = df_gen0["Bias_Label"].value_counts().reindex(["Left", "Center", "Right"], fill_value=0)

plt.figure(figsize=(10, 8))
bias_counts.plot(kind="bar", color=["blue", "grey", "red"], width=0.6)
plt.title("Distribution of Articles in GPT-2 Outputs by Political Leaning", fontsize=15)
plt.xlabel("", fontsize=15)
plt.ylabel("Number of Articles", fontsize=15)
plt.xticks(rotation=0, fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(0, bias_counts.max() + 50)
plt.grid(axis="y", linestyle="-", alpha=0.7)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
plt.show()
