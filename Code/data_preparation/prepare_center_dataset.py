"""
Prepare the center-leaning dataset for the alternative experiment.

Filters all center-leaning articles from the Webis-Bias-Flipper-18 dataset
and saves as D0.txt for the alternative experimental setup.

Paper reference: Section 3.1, Appendix I (Alternative Experimental Setup)
"""

import pandas as pd

# ============================================================
# Configuration
# ============================================================
INPUT_CSV = "data_public.csv"       # Webis-Bias-Flipper-18 dataset (https://zenodo.org/records/3271061)
OUTPUT_FILE = "D0.txt"              # Output: center-leaning articles only

# ============================================================
# Filter and format articles
# ============================================================
df = pd.read_csv(INPUT_CSV, on_bad_lines="skip")
df_center = df[df["bias"] == "From the Center"].copy()

def format_article(row):
    return f"title: {row['original_title']}\nbody: {row['original_body']}"

df_center["formatted"] = df_center.apply(format_article, axis=1)
df_center["formatted"].to_csv(OUTPUT_FILE, index=False, header=False)

print(f"Saved {len(df_center)} center-leaning articles to {OUTPUT_FILE}")
