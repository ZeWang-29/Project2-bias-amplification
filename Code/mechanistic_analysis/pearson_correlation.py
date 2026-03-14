"""
Compute Pearson correlation between neuron activations and model bias
across all fine-tuned GPT-2 versions.

Paper reference: Section 3.6, Section 4.4 (Mechanistic Interpretation)
"""

import pandas as pd
from scipy.stats import pearsonr

# ============================================================
# Configuration
# ============================================================
ACTIVATION_CSV_FILES = [
    # Activation CSVs for each experimental setup (from extract_activations.py)
    "MM_activations.csv",
    "MMA_activations.csv",
    "MMP_activations.csv",
    "MMO_activations.csv",
    # "MMB_activations.csv",    # Optional: Beam Search (not in paper)
    # "MMN_activations.csv",    # Optional: Nucleus Sampling (not in paper)
]
BIAS_CSV = "ModelBias.csv"                  # Model bias data (model_name, bias)
OUTPUT_CSV = "neuron_bias_correlation.csv"

# ============================================================
# Compute correlations
# ============================================================
activation_data = pd.concat([pd.read_csv(f) for f in ACTIVATION_CSV_FILES], ignore_index=True)
bias_data = pd.read_csv(BIAS_CSV)
merged_data = pd.merge(activation_data, bias_data, on="model_name")

results = []
grouped = merged_data.groupby(["neuron_id", "layer"])

for (neuron_id, layer), group in grouped:
    correlation, p_value = pearsonr(group["activation"], group["bias"])
    results.append({
        "neuron_id": neuron_id,
        "layer": layer,
        "pearson_correlation": correlation,
        "significance_level": p_value,
    })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Pearson correlations saved to {OUTPUT_CSV}")
