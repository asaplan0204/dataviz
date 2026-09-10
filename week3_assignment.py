import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, leaves_list
import numpy as np

# title
st.markdown(
    "<h1 style='text-align: center;'>Pizza</h1>",
    unsafe_allow_html=True
)

# loading the data
df = pd.read_csv("pizza.csv")

# preview of the data
st.subheader("Pizza Dataset Preview")
st.dataframe(df.head(20))

# summary statistics
summary = df.drop(columns=["id"]).describe()

st.subheader("Summary Statistics")
st.dataframe(summary)


# correlation heatmap unordered
st.subheader("Correlation among pizza ingredients unordered")
num = df.select_dtypes(include="number").drop(columns=["id"])
corr = num.corr()
fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", title="Correlation among pizza ingredients unordered")
st.plotly_chart(fig, use_container_width=True)

# ordered heatmap using clustermapping
st.subheader("Correlation among pizza ingredients ordered")
num = df.select_dtypes(include="number").drop(columns=["id"])
corr = num.corr()
linkage_matrix = linkage(corr, method="average")
order = leaves_list(linkage_matrix)
corr_ordered = corr.iloc[order, order]
fig = px.imshow(corr_ordered, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", title="Correlation among pizza ingredients ordered")
st.plotly_chart(fig, use_container_width=True)

# scatterplot to show ingredients have a linear correlation
st.subheader("Relationship Between Sodium and Ash")
fig = px.scatter(df, x="sodium", y="ash", title="Sodium vs. Ash", labels={"sodium": "Sodium","ash": "Ash"})
st.plotly_chart(fig)

# stadardizing the data
X = StandardScaler().fit_transform(df.select_dtypes(include="number").drop(columns=["id"]))
st.subheader("PCA of Pizza Ingredients")
pca = PCA(n_components=2)
pcs = pca.fit_transform(X)
st.write("Explained variance:")
st.write(f"PC1: {pca.explained_variance_ratio_[0]:.3f}")
st.write(f"PC2: {pca.explained_variance_ratio_[1]:.3f}")

# PCA scatterplot
st.subheader("PCA Scatter Plot of Pizza Ingredients")
brands = df["brand"].map({"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9})
fig = px.scatter(
    df,
    x=pcs[:, 0],
    y=pcs[:, 1],
    color="brand",
    title="Pizza Data PCA",
    labels={
        "x": f"PC1 ({pca.explained_variance_ratio_[0]:.0%} variance)",
        "y": f"PC2 ({pca.explained_variance_ratio_[1]:.0%} variance)",
        "brand": "Brand"
    }
)
st.plotly_chart(fig, use_container_width=True)

# pc1 bar graph
st.subheader("PC1 Loadings")
num = df.select_dtypes(include="number").drop(columns=["id"])
loadings = pd.Series(
    pca.components_[0],
    index=num.columns
).sort_values()

loading_df = pd.DataFrame({
    "Variable": loadings.index,
    "Loading": loadings.values,
    "Direction": np.where(loadings.values > 0, "Positive", "Negative")
})

fig = px.bar(
    loading_df,
    x="Loading",
    y="Variable",
    orientation="h",
    color="Direction",
    title="PC1 Loadings for Pizza Data",
    labels={
        "Loading": "PC1 Loading",
        "Variable": "Pizza Variable"
    },
    color_discrete_map={
        "Positive": "#2E6E8E",
        "Negative": "#d9534f"
    }
)
fig.add_vline(
    x=0,
    line_width=1,
    line_color="#999"
)
st.plotly_chart(fig)

# pc2 bar graph
st.subheader("PC2 Loadings")
num = df.select_dtypes(include="number").drop(columns=["id"])
loadings = pd.Series(
    pca.components_[1],
    index=num.columns
).sort_values()

loading_df = pd.DataFrame({
    "Variable": loadings.index,
    "Loading": loadings.values,
    "Direction": np.where(loadings.values > 0, "Positive", "Negative")
})

fig = px.bar(
    loading_df,
    x="Loading",
    y="Variable",
    orientation="h",
    color="Direction",
    title="PC2 Loadings for Pizza Data",
    labels={
        "Loading": "PC2 Loading",
        "Variable": "Pizza Variable"
    },
    color_discrete_map={
        "Positive": "#2E6E8E",
        "Negative": "#d9534f"
    }
)
fig.add_vline(
    x=0,
    line_width=1,
    line_color="#999"
)
st.plotly_chart(fig)