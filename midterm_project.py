import pandas as pd
import plotly.express as px
import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import math
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import PorterStemmer
from pathlib import Path
import json
import re
import plotly.graph_objects as go


@st.cache_data(max_entries=1)
def load_and_process_data():
    usecols = [
        "event_type",
        "created_at",
        "user_did",
        "target_did",
        "text",
        "image_urls"
    ]

    df = pd.read_csv(
        "bluesky_all_events.csv",
        usecols=usecols
    )

    df_network = df.loc[
        df["event_type"] == "follow",
        ["created_at", "user_did", "target_did"]
    ].copy()

    text_mask = (
        df["event_type"].isin(["post", "reply", "repost"])
        & df["text"].notna()
        & df["text"].str.strip().ne("")
    )

    df_text = df.loc[text_mask, ["event_type", "text"]].copy()
    df_corpus = df_text[["text"]].copy()

    image_mask = (
        df["event_type"].isin(["post", "reply", "repost"])
        & df["image_urls"].notna()
        & df["image_urls"].str.strip().ne("")
        & df["image_urls"].str.strip().ne("[]")
    )

    df_image = df.loc[
        image_mask,
        ["event_type", "created_at", "image_urls"]
    ].copy()

    return df_network, df_text, df_corpus, df_image


df_network, df_text, df_corpus, df_image = load_and_process_data()

# @st.cache_data
# def load_and_process_data():
#     df = pd.read_csv("bluesky_all_events.csv")

#     df_network = df[df["event_type"] == "follow"]

#     df_text = df[
#         df["event_type"].isin(["post", "reply", "repost"])
#     ]
#     df_text = df_text[
#         df_text["text"].notna()
#         & (df_text["text"].str.strip() != "")
#     ]

#     df_corpus = df_text[["text"]].copy()

#     df_image = df[
#         df["event_type"].isin(["post", "reply", "repost"])
#     ]
#     df_image = df_image[
#         df_image["image_urls"].notna()
#         & (df_image["image_urls"].str.strip() != "")
#         & (df_image["image_urls"] != "[]")
#     ]

#     return df, df_network, df_text, df_corpus, df_image


# df, df_network, df_text, df_corpus, df_image = load_and_process_data()

@st.cache_data
def load_rgb_data():
    return pd.read_csv("bluesky_reply_rgb_intensity.csv")


@st.cache_data
def load_bertopic_results(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

st.set_page_config(
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center;'>Bluesky Social Media Explorer</h1>",
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs(["About the Data", "Network Analysis","Text Analysis","Image Analysis"])

with tab1:
    with st.container(border=True):
        st.title("Explore the Data")

        st.markdown("""
        Explore public activity collected from Bluesky's real-time event stream that was created on 19 October 2026.

        This dataset includes social media posts, user relationships, timestamps, and image content from the activity on the Bluesky app.
        """)


    preview, summary = st.tabs(["Data Preview","Summary Statistics"])
    with preview:
        st.subheader("First 5 Records")
        st.dataframe(df.head())

        st.subheader("Last 5 Records")
        st.dataframe(df.tail())

    with summary:
        st.subheader("Bluesky Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Events", len(df))
        col2.metric("Text Records", len(df_text))
        col3.metric("Follow Relationships", len(df_network))
        col4.metric("Records with Images", len(df_image))

with tab2:
    st.subheader("Bluesky Relationship Network")
    st.markdown("""
    **Research Question:** What do relationships between users of Bluesky look like based on time of follow request?
    """)
    df_network["created_at"] = pd.to_datetime(
        df_network["created_at"],
        format="ISO8601",
        utc=True
    )
    df_network["minute"] = df_network["created_at"].dt.floor("min")

    minutes = sorted(df_network["minute"].unique())

    selected_minute = st.selectbox(
        "Select a minute",
        options=minutes,
        format_func=lambda x: x.strftime("%H:%M UTC")
    )

    df_minute = df_network[df_network["minute"] == selected_minute]

    G = nx.from_pandas_edgelist(
        df_minute,
        source="user_did",
        target="target_did",
        create_using=nx.DiGraph()
    )

    G.remove_edges_from(nx.selfloop_edges(G))

    if G.number_of_edges() > 0:
        G_undirected = G.to_undirected()

        communities = nx.community.louvain_communities(
            G_undirected,
            seed=42
        )

        partition = {
            node: community_id
            for community_id, community in enumerate(communities)
            for node in community
        }

        nx.set_node_attributes(G, partition, "community")

        community_sizes = {
            community_id: len(community)
            for community_id, community in enumerate(communities)
        }

        selected_communities = st.multiselect(
            "Select communities",
            options=list(community_sizes.keys()),
            default=list(community_sizes.keys()),
            format_func=lambda c: (
                f"Community {c} ({community_sizes[c]} users)"
            )
        )

        largest_community = max(community_sizes.values())

        min_community_size = st.slider(
            "Minimum users per community",
            min_value=1,
            max_value=min(100, largest_community),
            value=min(5, largest_community)
        )

        keep_nodes = {
            node
            for node, community_id in partition.items()
            if community_id in selected_communities
            and community_sizes[community_id] >= min_community_size
        }

        G_filtered = G.subgraph(keep_nodes).copy()

        layout_choice = st.radio(
            "Network Layout",
            ["Force-Directed", "Random", "Circle"],
            horizontal=True
        )

        col1, col2 = st.columns(2)

        col1.metric("Users Displayed", G_filtered.number_of_nodes())
        col2.metric("Connections Displayed", G_filtered.number_of_edges())

        if G_filtered.number_of_edges() > 0:

            net = Network(
                height="650px",
                width="100%",
                directed=False,
                bgcolor="#ffffff",
                font_color="#222222"
            )

            for node, data in G_filtered.nodes(data=True):
                community_id = data["community"]

                net.add_node(
                    node,
                    label=str(node)[-8:],
                    title=(
                        f"User ID: {node}<br>"
                        f"Community: {community_id}<br>"
                        f"Community size: {community_sizes[community_id]}"
                    ),
                    group=community_id,
                    size=10
                )
            for source, target in G_filtered.edges():
                net.add_edge(source, target)

            node_ids = list(G_filtered.nodes())
            total_nodes = len(node_ids)
            radius = max(150, total_nodes * 3)

            if layout_choice == "Circle":

                for i, node in enumerate(node_ids):
                    angle = 2 * math.pi * i / total_nodes

                    net.get_node(node)["x"] = radius * math.cos(angle)
                    net.get_node(node)["y"] = radius * math.sin(angle)
                    net.get_node(node)["physics"] = False

                net.set_options("""
                {
                "nodes": {
                    "shape": "dot",
                    "font": {"size": 10}
                },
                "edges": {
                    "smooth": false
                },
                "physics": {
                    "enabled": false
                },
                "interaction": {
                    "hover": true,
                    "navigationButtons": true,
                    "zoomView": true
                }
                }
                """)

            elif layout_choice == "Random":

                random.seed(42)

                for node in node_ids:
                    net.get_node(node)["x"] = random.uniform(-radius, radius)
                    net.get_node(node)["y"] = random.uniform(-radius, radius)
                    net.get_node(node)["physics"] = False

                net.set_options("""
                {
                "nodes": {
                    "shape": "dot",
                    "font": {"size": 10}
                },
                "edges": {
                    "smooth": false
                },
                "physics": {
                    "enabled": false
                },
                "interaction": {
                    "hover": true,
                    "navigationButtons": true,
                    "zoomView": true
                }
                }
                """)

            else:

                net.set_options("""
                {
                "nodes": {
                    "shape": "dot",
                    "font": {"size": 10}
                },
                "edges": {
                    "smooth": false
                },
                "physics": {
                    "enabled": true,
                    "solver": "forceAtlas2Based",
                    "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08,
                    "avoidOverlap": 1
                    },
                    "stabilization": {
                    "enabled": true,
                    "iterations": 200
                    }
                },
                "interaction": {
                    "hover": true,
                    "navigationButtons": true,
                    "zoomView": true
                }
                }
                """)
            components.html(
                net.generate_html(),
                height=670,
                scrolling=True
            )

        else:
            st.info(
                "No connections match the selected communities. "
                "Select more communities or lower the minimum size."
            )

    else:
        st.info("No follow connections are available for this minute.")

    if "show_network_answer" not in st.session_state:
        st.session_state.show_network_answer = False

    if st.button("Show Research Findings", key="network_answer_button"):
        st.session_state.show_network_answer = not st.session_state.show_network_answer

    if st.session_state.show_network_answer:
        st.markdown("""
        **Research Answer:** Looking at the time 17:39 UTC due to it having multiple users instead of just a few, the graph reveals multiple connections with multiple 
        communities. Many of the connections are within individual communities that are not connected to other communities.  The shape of the connections and lack of 
        community connectivity suggest that Bluesky has small groups of interconnected users.  Overall, the pattern of Bluesky users suggest that they 
        form relationships within specific social circles. However, this is only during the data collection period and does not represent the entire app's relationships.
        """)

with tab3:
    st.markdown("""
    **Research Question:** What words and topics are most common in Bluesky posts during the data collection period?
    """)
    frequency, bertopic = st.tabs(["Frequency Analysis","BERTopic Analysis(Precomputed)"])
    with frequency:
        df = pd.read_csv("bluesky_text_corpus.csv")
        texts = df["text"].dropna().tolist()

        st.subheader("Honest Frequency Options")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            stopword_option = st.radio(
                "Stop words",
                ["Exclude stop words", "Include stop words"]
            )
        with col2:
            stemming_option = st.radio(
                "Stemming",
                ["Exclude stemming", "Include stemming"]
            )
        with col3:
            ngram_option = st.radio(
                "N-gram",
                ["Unigrams", "Bigrams"]
            )
        with col4:
            representation_option = st.radio(
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
    with bertopic:
        st.header("2. BERTopic Visualization(Precomputed)")

        RESULTS_FILE = Path(__file__).parent / "midterm_bertopic_results.json"

        st.subheader("BERTopic Options")

        col1 = st.columns(1)
        with col1[0]:
            bertopic_stopwords = st.radio(
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

        bertopic_results = load_bertopic_results(str(RESULTS_FILE))

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

    if "show_text_answer" not in st.session_state:
        st.session_state.show_text_answer = False

    if st.button("Show Research Findings",  key="text_answer_button"):
        st.session_state.show_text_answer = not st.session_state.show_text_answer

    if st.session_state.show_text_answer:
        st.markdown("""
        **Research Answer:** When it comes to which words are most common in the Bluesky posts during the data collection period, it truly depends on the options 
        selected.  If the options selected are exclude stops words, unigrams, and frequency.  Then when the option of exclude or include stemming changes the top 
        most common words.  Sometimes changing the options change the most common words completely and sometimes changing the options only change the rank of the 
        words.  Thus, the most common words or frequency of words depends on what the user is looking for and what options they choose.  Looking at the precomputed 
        BERTopic analysis, it can be seen what the top 10 representative topic words are for the Bluesky social media app.  These words give insight into what themes 
        or topics are popular within the Bluesky app.  However, these interpretations of themes and topics are only within the data collection period and does not 
        represent the entire app's themes or topics.
        """)


with tab4:
    st.markdown("""
        **Research Question:** What visual characteristics can be observed in reply images of the Bluesky app.
    """)
    
    montage, intensity = st.tabs(["Montage Analysis(Precomputed)", "Intensity Analysis(Precomputed)"])
    with montage:
        st.subheader("Bluesky Image Montage(Precomputed)")
        st.markdown("""
            **Side Note:** The montage only shows the first 40 reply images in the dataset no more and no less.  I precomputed the montage and only used the reply images
            due to the first 40 posts and reply images had images that were extremely explicit and not appropriate.  That is why the montage will only show the first 40 
                reply images.
        """)
        st.image(
            "bluesky_montage.png",
            caption="First 40 reply images from the Bluesky dataset",
            use_container_width=True
        )
    with intensity:
        st.subheader("RGB Intensity Distribution by Reply Image")
        st.markdown("""
            **Side Note:** Just like the montage this RGB Intensitiy Distribution will be limited to only the first 10 reply images, again this is due to the other
            images in the dataset being very explicit and not appropriate. Also, when using all 40 images for the RGB Intensity Distribution the app was failing, due to
            the precomputed file being too large for streamlit and github to handle. 
        """)
        rgb_df = load_rgb_data()
        image_options = sorted(rgb_df["image_number"].unique().tolist())

        selected_image = st.selectbox(
            "Select an image",
            options=image_options,
            format_func=lambda x: f"Image {x}"
        )

        plot_df = rgb_df[rgb_df["image_number"] == selected_image]

        fig = go.Figure()

        for channel, color in [
            ("Red", "red"),
            ("Green", "green"),
            ("Blue", "blue")
        ]:
            fig.add_trace(
                go.Histogram(
                    x=plot_df[channel],
                    name=channel,
                    marker_color=color,
                    opacity=0.55,
                    xbins=dict(
                        start=0,
                        end=256,
                        size=1
                    )
                )
            )

        fig.update_layout(
            title=f"RGB Intensity Distribution: Image {selected_image}",
            xaxis_title="Pixel Intensity (0–255)",
            yaxis_title="Pixel Count",
            barmode="overlay",
            bargap=0.05,
            legend_title="Color Channel",
            height=500,
            template="plotly_white"
        )

        fig.update_xaxes(range=[0, 255])

        st.plotly_chart(fig, use_container_width=True)

        df_image = df_image[
            (df_image["event_type"] == "reply")
            & df_image["image_urls"].notna()
            & (df_image["image_urls"].str.strip() != "")
            & (df_image["image_urls"] != "[]")
        ]

        images = []

        for _, row in df_image.iterrows():
            try:
                urls = row["image_urls"]

                if isinstance(urls, str):
                    urls = json.loads(urls)

                if not isinstance(urls, list):
                    continue

                for value in urls:
                    if not isinstance(value, str):
                        continue

                    match = re.search(r"\]\((https?://[^)]+)\)", value)
                    url = match.group(1) if match else value.strip()

                    if url.startswith("https://"):
                        images.append(url)

                    if len(images) >= 40:
                        break

                if len(images) >= 40:
                    break

            except (json.JSONDecodeError, TypeError):
                continue

        if 1 <= selected_image <= len(images):
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                st.image(
                    images[selected_image - 1],
                    caption=f"Reply Image {selected_image}",
                    width=500
                )
        else:
            st.warning("Could not find the URL for this image.")

    if "show_image_answer" not in st.session_state:
        st.session_state.show_image_answer = False

    if st.button("Show Research Findings",  key="text_image_button"):
        st.session_state.show_image_answer = not st.session_state.show_image_answer

    if st.session_state.show_image_answer:
        st.markdown("""
        **Research Answer:** The 40 reply images that were used in the montage and the 10 reply images used in the RGB intensity distribution reveal a diverse range of visual characteristics.
        The montage shows that some images have similarities such as being images of food, but they are different foods in the image. While most of the other images
        have nothing to do with each other.  The RGB intensity distribution shows that at the pixel level each image has it own variation of RGB distributions and intensity
        values. Suggesting that the images vary in color composition and overall visual apperance.  Overall, the image analysis suggests that the reply images in the
        Bluesky social media app have a very wide range of visual characteristics.
        """)
