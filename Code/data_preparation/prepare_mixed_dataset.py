"""
Prepare the mixed (unbiased) dataset for the main experiment.

Samples 506 articles each from Left, Center, and Right categories (1,518 total)
from the Webis-Bias-Flipper-18 dataset and saves as D_mixed.txt.

Paper reference: Section 3.1 (Dataset Preparation)
"""

import pandas as pd

# ============================================================
# Configuration
# ============================================================
INPUT_CSV = "data_public.csv"       # Webis-Bias-Flipper-18 dataset (https://zenodo.org/records/3271061)
OUTPUT_FILE = "D_mixed.txt"         # Output: 1,518 mixed articles
N_SAMPLES_PER_CATEGORY = 506        # 506 x 3 = 1,518 total
RANDOM_SEED = 42

# ============================================================
# Sample and format articles
# ============================================================
df = pd.read_csv(INPUT_CSV, on_bad_lines="skip")

biases = ["From the Left", "From the Center", "From the Right"]
sampled_dfs = []

for bias in biases:
    df_bias = df[df["bias"] == bias].copy()
    if len(df_bias) < N_SAMPLES_PER_CATEGORY:
        raise ValueError(f"Not enough articles for bias '{bias}' to sample {N_SAMPLES_PER_CATEGORY}.")
    df_sampled = df_bias.sample(n=N_SAMPLES_PER_CATEGORY, random_state=RANDOM_SEED)
    sampled_dfs.append(df_sampled)

df_combined = pd.concat(sampled_dfs, ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

def format_article(row):
    return f"title: {row['original_title']}\nbody: {row['original_body']}"

df_combined["formatted"] = df_combined.apply(format_article, axis=1)
df_combined["formatted"].to_csv(OUTPUT_FILE, index=False, header=False)

print(f"Saved {len(df_combined)} articles to {OUTPUT_FILE}")
