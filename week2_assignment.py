import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Diabetes")

st.write("""
### Question:

How does hba1c levels vary across different age groups?
""")

st.write("""
### Data:
""")

df = pd.read_csv("diabetes_risk.csv")

st.dataframe(df.head(20))

# age groups
df["age_group"] = pd.cut(
    df["age"],
    bins = [18, 30, 42, 54, 66, 80],
    labels = ["18-30", "31-42", "43-54", "55-66", "67-80"]
)

# average hba1c level based on age group
avg_hba1c = df.groupby("age_group", observed=False)["hba1c_level"].mean().reset_index()

# color mapping using color brewer
age_colors = {
    "18-30": "#a6cee3", 
    "31-42": "#1f78b4", 
    "43-54": "#b2df8a", 
    "55-66": "#33a02c", 
    "67-80": "#fb9a99"
}

# scatter plot
scatt = px.scatter(
    df, 
    x="age", 
    y="hba1c_level", 
    color = "age_group",
    color_discrete_map=age_colors,
    title="HbA1c Levels Over Age Groups ",
    labels = {"age": "Age", "hba1c_level": "HbA1c Level"}
)
st.plotly_chart(scatt)
st.write("""
Channel: Position and color

Perceptual-accuracy: Scatterplots coordinate based position channel is the most accurate
visualization for viewers to see how hba1c levels vary across each individual age. 

Gestalt: I used similarity principle by making the different age groups different colors, so that the
points with the same colors can be perceived as being in the same group.
""")

# bar chart
bar = px.bar(
    avg_hba1c,
    x="age_group",
    y="hba1c_level",
    color = "age_group",
    color_discrete_map=age_colors,
    title="Average HbA1c Levels by Age Groups",
    labels = {"age_group": "Age Group", "hba1c_level": "Avgerage HbA1c Level"}
)
st.plotly_chart(bar)
st.write("""
Channel: Position/length and color

Perceptual-accuracy: Bar charts having a common baseline makes it easy to compare the
average hba1c levels between the different age groups in order to visualize which average
hba1c level is higher or lower. Bar charts are not as accurate as scatter plots however they
are more accurate then angle or area channels. 

Gestalt: I used similarity principle for this bar chart, but instead of showing the grouping it is
distinguishing the groups.  Each color represents a different group which allows the viewer
to make clearer perceptions of which group has larger, smaller, or the same average hba1c levels.
""")

# line chart
line = px.line(
    avg_hba1c,
    x="age_group",
    y="hba1c_level",
    markers= True,
    title="Average HbA1c Levels by Age Groups",
    labels = {"age_group": "Age Group", "hba1c_level": "Avgerage HbA1c Level"}
)

line.update_traces(
    line=dict(color="#444444"),
    marker=dict(color=[age_colors[group] for group in avg_hba1c["age_group"]])
)
st.plotly_chart(line)
st.write("""
Channel: Position and color

Perceptual-accuracy: Line charts are coordinate based positions meaning they have the highest accuracy.
It also does a good job as showing the average hba1c levels trend amoungst the age groups.  Viewers can
easily see if the average hba1c level increases, decreases, or stays the same throughout the age groups.

Gestalt: I used continuity principle, the connected lines allow viewers to follow a smooth path of the average
hba1c levels through the different age groups.
""")
