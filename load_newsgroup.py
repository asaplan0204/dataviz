import os
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

DATA_FILE = "20newsgroups.csv"

def load_news_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)["text"].tolist()

    data = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes")
    )

    texts = data.data

    pd.DataFrame({"text": texts}).to_csv(
        DATA_FILE,
        index=False
    )

    return texts

texts = load_news_data()