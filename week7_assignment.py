import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import PorterStemmer
from bertopic import BERTopic

st.set_page_config(
    page_title="20 Newsgroups Text Visualization",
    layout="wide"
)

st.title("20 Newsgroups Text Visualization")
st.write("""
**Dataset Description:**
This dataset is a text corpus of 18000 news groups posts, including discussion on sports, religion, politics, and more.  Each document 
is composed of text from an online conversation.
""")

@st.cache_data
def load_news_data():
    data = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
        download_if_missing=True,
        data_home="/tmp/20newsgroups"
    )
    return data.data

texts = load_news_data()

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

st.header("2. BERTopic Visualization")

@st.cache_resource
def create_topic_model(documents, exclude_stopwords):

    if exclude_stopwords:
        vectorizer_model = CountVectorizer(
            stop_words="english"
        )
    else:
        vectorizer_model = CountVectorizer()

    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        min_topic_size=20,
        verbose=False
    )

    topics, probabilities = topic_model.fit_transform(documents)

    return topic_model, topics, probabilities

st.sidebar.header("BERTopic Options")

bertopic_stopwords = st.sidebar.radio(
    "BERTopic stop words",
    ["Exclude stop words", "Include stop words"]
)

exclude_bertopic_stopwords = (
    bertopic_stopwords == "Exclude stop words"
)


with st.spinner("Running BERTopic... This may take a moment."):

    topic_model, topics, probabilities = create_topic_model(
    texts,
    exclude_bertopic_stopwords
)

topic_info = topic_model.get_topic_info()
topic_info = topic_info[
    topic_info["Topic"] != -1
].copy()

top_topics = (
    topic_info
    .sort_values("Count", ascending=False)
    .head(10)
)

def get_topic_words(topic):

    words = topic_model.get_topic(topic)

    if words is None:
        return "Unknown"

    return ", ".join(
        [word for word, score in words[:5]]
    )


top_topics["Topic Words"] = top_topics["Topic"].apply(
    get_topic_words
)

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

fig_topics.update_layout(
    height=600
)

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
The BERTopic visualization allows the users to view a bar chart and table with the 10 most represented topics with each topic represented by its 5 
most representative words. The BERTopic visualization reveals themes or topics within the text corpus.  One way BERTopic visualization could mislead 
the user is the representative words may be interpreted differently than the model, causing the user to misunderstand the topic.
""")