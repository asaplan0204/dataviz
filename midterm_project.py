import pandas as pd
import plotly.express as px
import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import math
import random

df = pd.read_csv("bluesky_all_events.csv")

df_network = df[df["event_type"] == "follow"]

df_text = df[(df["event_type"] == "post") | (df["event_type"] == "reply") | (df["event_type"] == "repost")]
df_text = df_text[df_text["text"].notna() &(df_text["text"].str.strip() != "")]

df_image = df[(df["event_type"] == "post") | (df["event_type"] == "reply") | (df["event_type"] == "repost")]
df_image = df_image[df_image["image_urls"].notna() & (df_image["image_urls"].str.strip() != "") & (df_image["image_urls"] != "[]")]

st.set_page_config(
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center;'>Bluesky Social Media Explorer</h1>",
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs(["About the Data", "Network","Text","Image"])

with tab1:
    with st.container(border=True):
        st.title("Explore the Data")

        st.markdown("""
        Explore public activity collected from Bluesky's real-time event stream that was created on 19 October 2026 between 5:34:58 PM and 5:40:18 PM.

        This dataset includes social media posts, user relationships, timestamps, and image content from 5 minutes of activity on the Bluesky app.
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
    st.subheader("Bluesky Follow Network by Community")
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