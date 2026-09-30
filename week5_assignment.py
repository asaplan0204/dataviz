import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, leaves_list
import numpy as np
import matplotlib.pyplot as plt

# make the width as wide as the screen
st.set_page_config(layout="wide")

# title
st.markdown(
    "<h1 style='text-align: center;'>Pizza</h1>",
    unsafe_allow_html=True
)

# loading the data
df = pd.read_csv("Pizza.csv")

# variable legend
st.subheader("Variable Legend")

variable_legend = {
    "Variable": ["brand", "id", "mois", "prot", "fat", "ash", "sodium", "carb", "cal"],
    "Definition": [
        "Pizza brand",
        "ID",
        "Amount of water per 100 grams in the sample",
        "Amount of protein per 100 grams in the sample",
        "Amount of fat per 100 grams in the sample",
        "Amount of ash per 100 grams in the sample",
        "Amount of sodium per 100 grams in the sample",
        "Amount of carbohydrates per 100 grams in the sample",
        "Amount of calories per 100 grams in the sample"
    ]
}

legend = pd.DataFrame(variable_legend)

st.dataframe(legend, hide_index=True, use_container_width=True)

# preview of the data
st.subheader("Pizza Dataset Preview")
st.dataframe(df.head(20))

# summary statistics
summary = df.drop(columns=["id"]).describe()

st.subheader("Summary Statistics")
st.dataframe(summary)


# correlation heatmap unordered
st.subheader("Correlation among pizza ingredients unordered")
col1, col2 = st.columns(2)

num = df.select_dtypes(include="number").drop(columns=["id"])
corr = num.corr()
# unordered heatmap 
with col1:
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", title="Correlation among pizza ingredients unordered")
    st.plotly_chart(fig, use_container_width=True)

# ordered heatmap using clustermapping
st.subheader("Correlation among pizza ingredients ordered")
num = df.select_dtypes(include="number").drop(columns=["id"])
corr = num.corr()
with col2:
    linkage_matrix = linkage(corr, method="average")
    order = leaves_list(linkage_matrix)
    corr_ordered = corr.iloc[order, order]
    fig = px.imshow(corr_ordered, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", title="Correlation among pizza ingredients ordered")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("What is this heatmap showing?")
st.info("""
**Structure Interpretation**

The structure of the ordered heatmap indicates that there are some groupings between variables with similar correlation patterns, 
while some variables have little to no relationship with each other.  The carb variable on the ordered heatmap shows a group of 
variables with negative correlations.  For example, the carb variable and several of the other variables such as ash, prot, sodium, and fat.  Looking at the mois variable 
it includes positive correlations, negative correlations, and little to no correlations with the other variables.  In contrast, cal, 
fat, prot, sodium, and ash all share either positive correlations or little to no correlations with each other.  Overall, the ordered 
heatmap shows that there are some individual relationships or correlations between variables.
""")

# scatterplot to show ingredients have a linear correlation
st.subheader("Relationship Between Ingredients")
ingredients = [
    "mois",
    "prot",
    "fat",
    "ash",
    "sodium",
    "carb",
    "cal"
]

# selectbox widget within sidebar
st.sidebar.title("Relationship Between Ingredients Scatterplot Options")
x_ingredient = st.sidebar.selectbox(
    "Select X-axis ingredient",
    ingredients,
    index=4
)

y_ingredient = st.sidebar.selectbox(
    "Select Y-axis ingredient",
    ingredients,
    index=3
)

# radio widget within sidebar
chart_type = st.sidebar.radio(
    "Select chart type",
    ["Plotly", "Matplotlib", "Streamlit"]
)

st.subheader(f"{x_ingredient.capitalize()} vs. {y_ingredient.capitalize()}")

# plotly
if chart_type == "Plotly":
    fig = px.scatter(df, x=x_ingredient, y=y_ingredient, title="Relationship Between Ingredients", labels={"sodium": "Sodium","ash": "Ash"})
    st.plotly_chart(fig)

# matplot
elif chart_type == "Matplotlib":
    fig, ax = plt.subplots()
    ax.scatter(df[x_ingredient],df[y_ingredient])
    ax.set_xlabel(x_ingredient.capitalize())
    ax.set_ylabel(y_ingredient.capitalize())
    ax.set_title("Relationship Between Ingredients")
    st.pyplot(fig)

# streamlit
else:
    chart_data = df[[x_ingredient, y_ingredient]]
    st.scatter_chart(chart_data, x=x_ingredient, y=y_ingredient)

st.subheader("What is this scatterplot showing?")
st.info("""

This scatterplot shows the relationships between the ingredients in order to check for linearity.
""")

# caching and running pca
@st.cache_data
def pca_calc(df):
    X = StandardScaler().fit_transform(df.select_dtypes(include="number").drop(columns=["id"])) 
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    return pca, pcs
pca, pcs = pca_calc(df)

st.subheader("PCA of Pizza Ingredients")
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

st.subheader("What is PC1?")
st.info("""
**PC1 Representation**

PC1 captures 60 percent of pizza ingredients total variation.  The loadings or weights of PC1 indicate that ash, fat, sodium, prot, cal, 
and mois are set against carbs.  Indicating that ash, fat, prot, cal, and mois increase the PC1 while carb decreases the PC1.
 Thus, PC1 represents the direction that captures the largest amount of variation within the pizza ingredient dataset.

""")

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

st.subheader("What is PC2?")
st.info("""
**PC2 Representation**

PC2 captures 33 percent of pizza ingredients datas total variation.  The loadings or weights of PC2 indicate that mois, 
prot, and ash are set against sodium, fat, carb, and cal.  Indicating that mois, prot, and ash increase the PC2, while sodium, fat, carb, and cal decrease the PC2.
Thus, PC2 represents the direction that captures the second largest amount of variation within the pizza ingredient dataset.
""")

st.subheader("Conclusion")
st.info("""
The heatmap shows the individual relationships of the variables within the pizza ingredient dataset.  While PCA reduces the dimensionality of the pizza ingredient dataset 
by condensing the variables into principal components.  Which means that 7 variables can be reduced down to 2 principal components while retaining 93 percent of variance
within this pizza ingredient dataset.
""")

st.subheader("Week 5 Streamlit Additions")
st.info("""
I added a st.sidebar to hold the controls for the ingredient relationship scatterplot.  The two widgets I added was a st.selectbox so that users can choose what ingredients
to use for the scatterplot, and radio so that users can choose between plotly, matplotlib, and streamlit for which type of chart they want to use.  I used st.columns on my unordered
and ordered correlation heatmaps, so that users can see them easier and at the same time.  Finally, I @st.cache_data my pca calculation so that whenever the user clicks a different
chart in the st.radio or changes the ingredient using the st.selectbox the calculation of the pca does not have to rerun for no reason.
""")
