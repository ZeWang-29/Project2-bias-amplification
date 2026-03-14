"""
Extract average activation values for all 9,216 neurons (768 per layer x 12 layers)
across multiple fine-tuned GPT-2 models.

Registers forward hooks on each transformer block to capture activations,
then averages over all articles and tokens.

Paper reference: Section 3.6 (Mechanistic Analysis)
"""

import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, GPT2Tokenizer

# ============================================================
# Configuration
# ============================================================
MODEL_NAMES = [
    # List all model names/paths to extract activations from.
    # Example for the main experiment (MM series):
    *[f"refipsai/MM{i}" for i in range(1, 12)],
    # Add other series as needed:
    # *[f"refipsai/MMP{i}" for i in range(1, 12)],
    # *[f"refipsai/MMA{i}" for i in range(1, 12)],
    # *[f"refipsai/MMO{i}" for i in range(1, 12)],
]
INPUT_ARTICLES_PATH = "D_mixed.txt"             # Common input file for all models
OUTPUT_CSV = "activations.csv"

# ============================================================
# Helper functions
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token


def get_activation(name, activations_dict):
    """Create a forward hook to capture activations."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        activations = output.detach().cpu().numpy()
        if name not in activations_dict:
            activations_dict[name] = activations
        else:
            if activations.shape[1] != activations_dict[name].shape[1]:
                min_tokens = min(activations.shape[1], activations_dict[name].shape[1])
                activations = activations[:, :min_tokens, :]
                activations_dict[name] = activations_dict[name][:, :min_tokens, :]
            activations_dict[name] = np.concatenate((activations_dict[name], activations), axis=0)
    return hook


def read_articles_from_text(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    raw_articles = content.split('"title: ')
    articles = []
    for raw_article in raw_articles[1:]:
        try:
            title, body = raw_article.split("body: ", 1)
            formatted_text = f"title: {title.strip()}\nbody: {body.strip()}"
            articles.append({"formatted": formatted_text})
        except ValueError:
            continue
    return pd.DataFrame(articles)


# ============================================================
# Extract activations for each model
# ============================================================
articles_df = read_articles_from_text(INPUT_ARTICLES_PATH)
all_activations = []
num_tokens = 512

for model_name in MODEL_NAMES:
    print(f"Processing model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Register hooks
    activations_dict = {}
    hooks = []
    for i, layer in enumerate(model.transformer.h):
        hook = layer.register_forward_hook(get_activation(f"layer_{i}", activations_dict))
        hooks.append(hook)

    # Process articles
    for _, article in articles_df.iterrows():
        inputs = gpt2_tokenizer(
            article["formatted"], return_tensors="pt",
            padding=True, truncation=True, max_length=num_tokens,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask, use_cache=False)

    for hook in hooks:
        hook.remove()

    # Average activations over articles and tokens
    num_articles = len(articles_df)
    for layer_name, activation in activations_dict.items():
        if activation.shape[1] < num_tokens:
            activation = np.pad(activation, ((0, 0), (0, num_tokens - activation.shape[1]), (0, 0)), mode="constant")
        else:
            activation = activation[:, :num_tokens, :]
        avg_activations = activation.sum(axis=(0, 1)) / (num_articles * num_tokens)
        for neuron_id, neuron_activation in enumerate(avg_activations):
            all_activations.append({
                "model_name": model_name,
                "layer": layer_name,
                "neuron_id": neuron_id,
                "activation": neuron_activation,
            })

    print(f"  Collected activations for {model_name}")
    del model
    torch.cuda.empty_cache()

activations_df = pd.DataFrame(all_activations)
activations_df.to_csv(OUTPUT_CSV, index=False)
print(f"All activations saved to {OUTPUT_CSV}")
