"""
Run the RoBERTa-base political bias classifier on generated articles.

Classifies each article as Left (LABEL_1), Center (LABEL_2), or Right (LABEL_0)
and saves per-article scores to a CSV file.

Paper reference: Section 3.4 (Political Bias Metric)
"""

import pandas as pd
from transformers import pipeline, RobertaTokenizer
import sys
from io import StringIO

# ============================================================
# Configuration
# ============================================================
CLASSIFIER_MODEL = "wu981526092/bias_classifier_roberta"    # RoBERTa-base classifier (macro F1 = 0.9196)
ARTICLE_PATHS = {
    "initial": "D_mixed.txt",                                # Original real articles
    # Add generation files as needed:
    # 0: "synthetic_data/DD0.txt",
    # 1: "synthetic_data/DD1.txt",
    # ...
}
OUTPUT_CSV = "classifier_scores.csv"

# ============================================================
# Initialize classifier
# ============================================================
classifier = pipeline("text-classification", model=CLASSIFIER_MODEL, tokenizer=CLASSIFIER_MODEL, top_k=3, device=0)
classifier_tokenizer = RobertaTokenizer.from_pretrained(CLASSIFIER_MODEL)


def truncate_text(text, tokenizer, max_length=500):
    """Truncate text to max_length tokens for the classifier."""
    encodings = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    return tokenizer.decode(encodings.input_ids[0], skip_special_tokens=True)


def get_classifier_scores(texts):
    """Classify articles and return Left/Center/Right scores."""
    left_scores, center_scores, right_scores = [], [], []
    for text in texts:
        truncated_text = truncate_text(text, classifier_tokenizer)
        try:
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            results = classifier(truncated_text)
            sys.stdout = old_stdout

            if isinstance(results, list) and isinstance(results[0], list):
                results = results[0]
            scores = {result["label"]: result["score"] for result in results}
            right_scores.append(scores.get("LABEL_0", 0))      # Right
            left_scores.append(scores.get("LABEL_1", 0))       # Left
            center_scores.append(scores.get("LABEL_2", 0))     # Center
        except Exception as e:
            print(f"Error processing text: {e}")
    return left_scores, center_scores, right_scores


# ============================================================
# Process all generations
# ============================================================
all_scores = []

for gen_label, file_path in ARTICLE_PATHS.items():
    print(f"Processing generation: {gen_label}")
    try:
        with open(file_path, "r") as file:
            synthetic_articles = file.read().split("\n\n")
        synthetic_articles = [a.replace('"', "").strip() for a in synthetic_articles if a.strip()]

        left, center, right = get_classifier_scores(synthetic_articles)
        all_scores.extend([
            {"Generation": gen_label, "Article_ID": idx + 1,
             "Left_Score": l, "Center_Score": c, "Right_Score": r}
            for idx, (l, c, r) in enumerate(zip(left, center, right))
        ])
    except FileNotFoundError:
        print(f"  File not found: {file_path}")

scores_df = pd.DataFrame(all_scores)
scores_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved classifier scores to {OUTPUT_CSV}")
