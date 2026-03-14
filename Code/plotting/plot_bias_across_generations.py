"""
Figure 3a: Percentage of right-leaning articles across generations.

Also used for Figures 6 (center-leaning) and 7 (left-leaning) by changing
the bias label in the filtering step.

Paper reference: Section 4.1, Figure 3a, Appendix B (Figures 6, 7)
"""

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 13})

# ============================================================
# Configuration
# ============================================================
INPUT_FILES = {
    "Synthetic":                    "../Data/Bias_Performance_and_Generation_Quality/Synthetic_Bias_Performance.csv",
    "Synthetic with Preservation":  "../Data/Bias_Performance_and_Generation_Quality/Preservation_Bias_Performance.csv",
    "Synthetic with Accumulation":  "../Data/Bias_Performance_and_Generation_Quality/Accumulation_Bias_Performance.csv",
    "Synthetic with Overfitting":   "../Data/Bias_Performance_and_Generation_Quality/Overfitting_Bias_Performance.csv",
}
BIAS_LABEL = "Right"                # Change to "Center" for Fig 6, "Left" for Fig 7
OUTPUT_FILE = "percentage_right_biased.png"

# ============================================================
# Plot
# ============================================================
plt.figure(figsize=(12, 8))

for label, file_path in INPUT_FILES.items():
    df = pd.read_csv(file_path)
    df["Generation"] = df["Generation"].astype(int)
    df["Bias_Label"] = df[["Left_Score", "Center_Score", "Right_Score"]].idxmax(axis=1).str.replace("_Score", "")
    pct = df[df["Bias_Label"] == BIAS_LABEL].groupby("Generation").size() / df.groupby("Generation").size() * 100
    pct = pct.sort_index()
    plt.plot(pct.index, pct, marker="o", linestyle="-", label=label)

plt.title(f"Percentage of {BIAS_LABEL} Biased Articles Across Generations")
plt.xlabel("Generation")
plt.ylabel(f"Percentage of {BIAS_LABEL} Biased Articles (%)")
plt.legend(title="Dataset")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
plt.show()
