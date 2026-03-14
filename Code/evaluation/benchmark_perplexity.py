"""
Compute GPT-2 perplexity on generated articles across generations.

Paper reference: Appendix E (Average Perplexity Across Generations)
"""

import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
START_GEN = 0
END_GEN = 11
GENERATION_PATHS = {
    # Map generation index to file path. Adjust paths to your environment.
    # 0: "synthetic_data/DD0.txt",
    # 1: "synthetic_data/DD1.txt",
    # ...
}
OUTPUT_CSV = "perplexity.csv"

# ============================================================
# Perplexity computation
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)
model.eval()


def compute_perplexities(texts):
    """Compute perplexity for a list of article texts."""
    perplexities = []
    for text in tqdm(texts, desc="Computing perplexity"):
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=model.config.n_positions)
        input_ids = encodings.input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
            perplexity = torch.exp(torch.tensor(loss)).item()
            perplexities.append(perplexity)
    return perplexities


# ============================================================
# Main loop
# ============================================================
results = pd.DataFrame()

for i in range(START_GEN, END_GEN + 1):
    path = GENERATION_PATHS.get(i, f"synthetic_data/DD{i}.txt")
    with open(path, "r", encoding="utf-8") as file:
        content = file.read()
    articles = [a.strip() for a in content.split("\n\n") if a.strip()]

    perplexities = compute_perplexities(articles)
    gen_results = pd.DataFrame({"Generation": [i] * len(perplexities), "Perplexity": perplexities})
    results = pd.concat([results, gen_results], ignore_index=True)

results.to_csv(OUTPUT_CSV, index=False)
print(f"Saved perplexity scores to {OUTPUT_CSV}")
