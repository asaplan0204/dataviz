import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import PorterStemmer
from bertopic import BERTopic
from pathlib import Path
import json

st.set_page_config(
    page_title="20 Newsgroups Text Visualization",
    layout="wide"
)

st.title("20 Newsgroups Text Visualization")
st.write("""
**Dataset Description:**
The dataset is a text corpus of 18000 news groups posts, including discussion on sports, religion, politics, and more.  Each document 
is composed of text from an online conversation.  
""")

df = pd.read_csv("20newsgroups.csv")
texts = df["text"].dropna().tolist()

st.sidebar.header("Honest Frequency Options")
stopword_option = st.sidebar.radio(
    "Stop words",
    ["Exclude stop words", "Include stop words"]
)
stemming_option = st.sidebar.radio(
    "Stemming",
    ["Exclude stemming", "Include stemming"]
)
ngram_option = st.sidebar.radio(
    "N-gram",
    ["Unigrams", "Bigrams"]
)
representation_option = st.sidebar.radio(
    "Representation",
    ["Frequency", "TF-IDF"]
)


if stopword_option == "Exclude stop words":
    stop_words = "english"
else:
    stop_words = None

if ngram_option == "Unigrams":
    ngram_range = (1, 1)
else:
    ngram_range = (2, 2)

if stemming_option == "Include stemming":

    stemmer = PorterStemmer()

    base_vectorizer = CountVectorizer(
        stop_words=stop_words
    )

    tokenizer = base_vectorizer.build_tokenizer()

    def stem_tokenizer(text):
        words = tokenizer(text)
        return [stemmer.stem(word) for word in words]

    if representation_option == "Frequency":

        vectorizer = CountVectorizer(
            stop_words=stop_words,
            tokenizer=stem_tokenizer,
            token_pattern=None,
            ngram_range=ngram_range
        )

    else:

        vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            tokenizer=stem_tokenizer,
            token_pattern=None,
            ngram_range=ngram_range
        )

else:

    if representation_option == "Frequency":

        vectorizer = CountVectorizer(
            stop_words=stop_words,
            ngram_range=ngram_range
        )

    else:

        vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            ngram_range=ngram_range
        )
texts = pd.Series(texts).dropna().astype(str).tolist()
X = vectorizer.fit_transform(texts)

terms = vectorizer.get_feature_names_out()
scores = X.sum(axis=0).A1
results = pd.DataFrame({
    "Term": terms,
    "Score": scores
})

top_terms = (
    results
    .sort_values("Score", ascending=False)
    .head(20)
)

st.header("1. Honest Frequency Visualization")

if representation_option == "Frequency":

    chart_title = f"Top 20 Most Frequent {ngram_option}"
    x_label = "Number of Occurrences"

else:

    chart_title = f"Top 20 {ngram_option} by TF-IDF"
    x_label = "Total TF-IDF Score"


fig_frequency = px.bar(
    top_terms.sort_values("Score"),
    x="Score",
    y="Term",
    orientation="h",
    title=chart_title,
    labels={
        "Score": x_label,
        "Term": "Word / Phrase"
    }
)

fig_frequency.update_layout(
    height=700
)

st.plotly_chart(
    fig_frequency,
    use_container_width=True
)

st.write("""
**Honest Frequency Visualization Insight and Limitation:**
The honest frequency visualization allows the users to view the 20 most frequently used words or phrases within the text corpus. It can reveal 
the most often used vocabulary within the text corpus.  One way the honest frequency visualization can mislead is it does not represent important 
topics.  A word or phrase that appears many times within the text corpus does not necessarily make the word or phrase meaningful it can just mean 
that it is used a lot and that is it.
""")


st.header("2. BERTopic Visualization(Precomputed)")

# Load precomputed BERTopic results
RESULTS_FILE = Path(__file__).parent / "bertopic_results.json"

st.sidebar.header("BERTopic Options")

bertopic_stopwords = st.sidebar.radio(
    "BERTopic stop words",
    ["Exclude stop words", "Include stop words"]
)

exclude_bertopic_stopwords = (
    bertopic_stopwords == "Exclude stop words"
)

if not RESULTS_FILE.exists():
    st.error(
        "Precomputed BERTopic results were not found. "
        "Please add bertopic_results.json to the app repository."
    )
    st.stop()

with open(RESULTS_FILE, "r", encoding="utf-8") as file:
    bertopic_results = json.load(file)

# Select the precomputed configuration
if exclude_bertopic_stopwords:
    results_key = "exclude_stopwords"
else:
    results_key = "include_stopwords"

top_topics = pd.DataFrame(
    bertopic_results[results_key]
)

if top_topics.empty:
    st.warning("No topics are available for this configuration.")
else:
    fig_topics = px.bar(
        top_topics.sort_values("Count"),
        x="Count",
        y="Topic Words",
        orientation="h",
        title="Top 10 BERTopic Topics",
        labels={
            "Count": "Number of Documents",
            "Topic Words": "Representative Topic Words"
        },
        hover_data={
            "Topic": True,
            "Count": True
        }
    )

    fig_topics.update_layout(height=600)

    st.plotly_chart(
        fig_topics,
        use_container_width=True
    )

    st.subheader("Topic Details")

    display_topics = top_topics[
        ["Topic", "Count", "Topic Words"]
    ].copy()

    st.dataframe(
        display_topics,
        use_container_width=True,
        hide_index=True
    )

st.write("""
**BERTopic Visualization Insight and Limitation:**

The BERTopic visualization displays the 10 most represented topics,
each described by its five representative words. It reveals potential
themes within the text corpus. However, the representative words may
be interpreted differently by users than intended by the model,
potentially leading to misunderstandings about a topic.
""")
