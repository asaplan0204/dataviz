import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Diabetes")

df = pd.read_csv("diabetes_risk.csv")

st.dataframe(df.head(20))

# st.write(df["hba1c_level"].min())
# st.write(df["hba1c_level"].max())

hist = px.histogram(df, x="hba1c_level", nbins=18, title="Distribution of HbA1c Levels", labels = {"hba1c_level": "HbA1c Levels"})
hist.update_traces(marker_line_width=1, marker_line_color="black")

st.plotly_chart(hist)

st.write("""
### About the Dataset:

This diabetes dataset was found on kaggle.com.  It includes health, demographic, and lifestyle
factors such as gender, age, bmi, physical activity level, smoking status, and hba1c levels just 
to name a few.  This dataset can be used to explore risk of diabetes and analyze how different
factors relate to gaining diabetes.
""")

st.write("""
### About the Histogram:

The histogram above indicates that the hba1c levels for this dataset has a right skewed distribution.  The 
highest bar being 6.5-6.9 tells us that the majority of the people in this dataset would be 
considered high or diabetic.  There is a small amount of people that would be considered moderately elevated or pre diabetic(5.7-6.4), and
an even smaller amount of people that would be considered low or healthy(below 5.7).  Also the histogram having a right 
skewed distribution shows that there are some extreme cases for hba1c levels such as three people having hba1c
levels between 11 and 11.4.  Anything above 10 is considered very high hba1c levels, which is present in this dataset.  Overall, majority of
people in this dataset have healthy, moderately elevated, and high hba1c levels, but there are some extreme values or very high levels as well.
""")