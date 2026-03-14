"""
Compute the Text Quality Index using the Gibberish Detector.

Each sentence is classified as Noise (0), Word Salad (1), Mild Gibberish (2),
or Clean (3). The article-level score is the average across all sentences.

Paper reference: Section 3.5 (Generation Quality Metric)
"""

import torch
import pandas as pd
from transformers import pipeline, AutoTokenizer

# ============================================================
# Configuration
# ============================================================
GIBBERISH_MODEL = "madhurjindal/autonlp-Gibberish-Detector-492513457"
GENERATION_PATHS = {
    # Map generation index to file path. Adjust paths to your environment.
    # 1: "synthetic_data/DD1.txt",
    # 2: "synthetic_data/DD2.txt",
    # ...
}
START_GEN = 0
END_GEN = 11
OUTPUT_CSV = "gibberish_levels.csv"

# ============================================================
# Helper functions
# ============================================================
def read_articles_from_text(file_path):
    """Parse articles from text file."""
    articles = []
    current_article = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip().startswith('"title:'):
                if current_article:
                    article_body = "".join(current_article).strip()
                    articles.append({"body": article_body})
                    current_article = []
            current_article.append(line)
        if current_article:
            article_body = "".join(current_article).strip()
            articles.append({"body": article_body})
    return articles


def compute_gibberish_levels(texts, model_name=GIBBERISH_MODEL, max_length=512):
    """Compute per-article Text Quality Index."""
    classifier = pipeline("text-classification", model=model_name, top_k=4, device=0)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label_map = {"noise": 0, "word salad": 1, "mild gibberish": 2, "clean": 3}

    gibberish_levels = []
    for text in texts:
        sentences = [s.strip() for s in text["body"].split(".") if s.strip()]
        article_levels = []
        for sentence in sentences:
            inputs = tokenizer(sentence, truncation=True, max_length=max_length, return_tensors="pt")
            inputs = {key: value.to(classifier.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = classifier.model(**inputs)
                logits = outputs.logits
                scores = logits.softmax(dim=-1)
                results = [
                    {"label": classifier.model.config.id2label[label.item()], "score": score.item()}
                    for label, score in zip(logits.argmax(dim=-1), scores.max(dim=-1))
                ]
            expected_level = sum(label_map[res["label"]] * res["score"] for res in results)
            article_levels.append(expected_level)
        avg_level = sum(article_levels) / len(article_levels) if article_levels else 0
        gibberish_levels.append(avg_level)
    return gibberish_levels


# ============================================================
# Main loop
# ============================================================
results = pd.DataFrame(columns=["Generation", "GibberishLevel"])

for i in range(START_GEN, END_GEN + 1):
    path = GENERATION_PATHS.get(i, f"synthetic_data/DD{i}.txt")
    articles = read_articles_from_text(path)
    print(f"Generation {i}: {len(articles)} articles")
    gibberish_levels = compute_gibberish_levels(articles)

    gen_results = pd.DataFrame({"Generation": [i] * len(gibberish_levels), "GibberishLevel": gibberish_levels})
    results = pd.concat([results, gen_results], ignore_index=True)

results.to_csv(OUTPUT_CSV, index=False)
print(f"Saved gibberish levels to {OUTPUT_CSV}")
