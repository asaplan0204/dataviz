
import json
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "bluesky_text_corpus.csv"
OUTPUT_FILE = BASE_DIR / "midterm_bertopic_results.json"

df = pd.read_csv(DATA_FILE)
texts = df["text"].dropna().astype(str).tolist()
texts = [text for text in texts if text.strip()]


def compute_topics(documents, exclude_stopwords):
    vectorizer = CountVectorizer(
        stop_words="english" if exclude_stopwords else None
    )

    model = BERTopic(
        vectorizer_model=vectorizer,
        min_topic_size=20,
        verbose=False
    )

    topics, probabilities = model.fit_transform(documents)

    info = model.get_topic_info()
    info = info[info["Topic"] != -1].copy()
    info = info.sort_values("Count", ascending=False).head(10)

    results = []

    for _, row in info.iterrows():
        words = model.get_topic(int(row["Topic"])) or []

        results.append({
            "Topic": int(row["Topic"]),
            "Count": int(row["Count"]),
            "Topic Words": ", ".join(
                word for word, score in words[:5]
            )
        })

    return results


results = {
    "exclude_stopwords": compute_topics(texts, True),
    "include_stopwords": compute_topics(texts, False)
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved results to {OUTPUT_FILE}")
