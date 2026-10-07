import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import math

st.set_page_config(page_title="Network Graph", layout="wide")
st.title("Twitch User Network Graph(Brazil)")

st.subheader("Description of Network")
st.info("""
This network was collected from the Stanford SNAP Datasets website. This network represents twitch users that speak portugues in Brazil.  Each node is a user on 
the twitch platform and each edge is a mutual friendship between each user.
""")

df = pd.read_csv("musae_PTBR_edges.csv")

# graph
G = nx.from_pandas_edgelist(
    df,
    source="from",
    target="to"
)

# communities
communities = nx.community.louvain_communities(
    G,
    seed=42
)

node_community = {}
for community_id, community in enumerate(communities):
    for node in community:
        node_community[node] = community_id

# attibutes
nx.set_node_attributes(
    G,
    node_community,
    "community"
)

degrees = dict(G.degree())
nx.set_node_attributes(
    G,
    degrees,
    "degree"
)

# sidebar
st.sidebar.title("Graph Filters")

community_ids = sorted(set(node_community.values()))

selected_communities = st.sidebar.multiselect(
    "Select communities",
    community_ids,
    default=community_ids[:1]
)

layout_type = st.sidebar.selectbox(
    "Graph layout",
    [
        "Random",
        "Circular",
        "Force Directed"
    ]
)

random_seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    value=42,
    step=1
)


col1, col2 = st.columns(2)

with col1:
    min_degree = st.slider(
        "Minimum node degree",
        min_value=1,
        max_value=max(degrees.values()),
        value=1
    )


selected_nodes = [
    node
    for node, community in node_community.items()
    if community in selected_communities
    and degrees[node] >= min_degree
]

G_filtered = G.subgraph(selected_nodes).copy()

with col2:
    st.metric(
        "Nodes shown",
        G_filtered.number_of_nodes()
    )

st.write(
    f"Showing {G_filtered.number_of_nodes()} nodes "
    f"and {G_filtered.number_of_edges()} edges"
)

community_colors = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#ffff33",  # yellow
    "#a65628",  # brown
    "#f781bf",  # pink
    "#999999",  # gray
]

if layout_type == "Random":

    pos = nx.random_layout(
        G_filtered,
        seed=random_seed
    )

elif layout_type == "Circular":

    pos = nx.circular_layout(
        G_filtered
    )

elif layout_type == "Force Directed":

    pos = nx.spring_layout(
        G_filtered,
        seed=random_seed
    )

# pyvis graph
net = Network(
    height="750px",
    width="100%",
    bgcolor="#ffffff",
    font_color="black"
)

# adding nodes
for node in G_filtered.nodes():

    community = G_filtered.nodes[node]["community"]
    degree = G_filtered.nodes[node]["degree"]

    color = community_colors[
        community % len(community_colors)
    ]

    size = float(5 + math.log1p(degree) * 5)

    x = float(pos[node][0]) * 1000
    y = float(pos[node][1]) * 1000

    physics = layout_type == "Force Directed"

    net.add_node(
        node,
        label=str(node),
        color=color,
        size=size,
        x=x,
        y=y,
        physics=physics,
        title=(
            f"Node: {node}<br>"
            f"Community: {community}<br>"
            f"Degree: {degree}"
        )
    )

for source, target in G_filtered.edges():

    net.add_edge(
        source,
        target,
        color="#999999"
    )

# pyvis settings
net.set_options("""
{
  "nodes": {
    "shape": "dot"
  },
  "edges": {
    "smooth": false
  },
  "physics": {
    "enabled": false,
    "solver": "repulsion",
    "repulsion": {
      "centralGravity": 0.1,
      "springLength": 250,
      "springConstant": 0.05,
      "nodeDistance": 300,
      "damping": 0.09
    }
  },
  "interaction": {
    "hover": true,
    "navigationButtons": true,
    "zoomView": true
  }
}
""")

net.save_graph("network.html")

with open("network.html", "r", encoding="utf-8") as f:
    html = f.read()

components.html(
    html,
    height=800,
    scrolling=False
)

st.subheader("Encodings and What They Reveal")
st.info("""
I chose my encodings and widgets to help break down the hairball problem.  The color encoding allows users to visualize the different communities when looking 
at multiple communities.  The size of nodes based on degree allows users to visualize which nodes have larger connectivity and which nodes have smaller 
connectivity.  In order to solve the hairball problem I added a sidebar and some filters for the users so that they could actually visualize a clean looking 
graph.  The sidebar includes a filter for which community the user wants to look at whether it's one community or multiple at once.  I also added a graph layout 
filter so that users can choose between forced directed, circular, or random graphs depending on what the user wants to visualize.  There is also a random seed 
filter so that the random graph can be reproducible if the users need it to be.  The last filter I added to pair with the other filters to solve the hairball 
problem is a slider at the top that only displays a user-provided minimum node degree of nodes.  Overall, the graph with the filters reveals the relationships 
each node has with each other which otherwise could not be visualized due to the hairball problem.
""")

st.subheader("Limitation")
st.info("""
One limitation is that the filters cannot show nodes with a high degree of connections connected to a low degree of connections.  I could not figure out a way 
to handle the hairball problem and display a node with a low degree like degree 3 and a node with a high degree like degree 100.  This misleads the user by 
allowing the possibility of thinking that a low degree node doesn’t have a connection with a high degree node due to after filtering not having all the nodes 
on the graph.  Overall, that limitation can either not be a huge problem or it can be a huge problem depending on what the user wants or needs to look at based 
on the problem they are trying to solve.
""")