"""
Perform linear regression with Newey-West HAC-adjusted standard errors
to test statistical significance of neuron-metric correlations.

Paper reference: Section 3.6, Appendix G (Mathematical Details for Statistical Tests)
"""

import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr

# ============================================================
# Configuration
# ============================================================
ACTIVATION_CSV_FILES = [
    "MM_activations.csv",
    "MMA_activations.csv",
    "MMP_activations.csv",
    "MMO_activations.csv",
]
BIAS_CSV = "ModelBias.csv"
OUTPUT_CSV = "neuron_bias_correlation_newey_west.csv"

# ============================================================
# Compute correlations with Newey-West adjusted p-values
# ============================================================
activation_data = pd.concat([pd.read_csv(f) for f in ACTIVATION_CSV_FILES], ignore_index=True)
bias_data = pd.read_csv(BIAS_CSV)
merged_data = pd.merge(activation_data, bias_data, on="model_name")

results = []
grouped = merged_data.groupby(["neuron_id", "layer"])

for (neuron_id, layer), group in grouped:
    correlation, _ = pearsonr(group["activation"], group["bias"])

    # OLS regression with Newey-West HAC standard errors
    X = sm.add_constant(group["activation"])
    model = sm.OLS(group["bias"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    p_value_newey_west = model.pvalues.iloc[1]

    results.append({
        "neuron_id": neuron_id,
        "layer": layer,
        "pearson_correlation": correlation,
        "significance_level_newey_west": p_value_newey_west,
    })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Newey-West adjusted correlations saved to {OUTPUT_CSV}")
