# demo_1_basic.py — Week 5, Part 1: a first real app
# Run with:  streamlit run demo_1_basic.py
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

st.title("Tips Explorer")
st.write("Every interaction re-runs this script top to bottom. Watch.")

df = px.data.tips()

day = st.selectbox("Day", sorted(df["day"].unique()))
sub = df[df["day"] == day]

st.write(f"{len(sub)} parties on {day}")

# ---------------------------------------------------------------------------
# Three ways to put a chart on the page. The first is Streamlit's own helper;
# the other two are the figures you already know how to build.
# ---------------------------------------------------------------------------
how = st.radio("Draw it with", ["st.scatter_chart", "matplotlib", "plotly"],
               horizontal=True)

if how == "st.scatter_chart":
    # Quick: hand it a dataframe and Streamlit picks the encodings for you.
    st.scatter_chart(sub, x="total_bill", y="tip")

elif how == "matplotlib":
    # Your own figure object. This is how a Weeks 2-4 chart moves in unchanged.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(sub["total_bill"], sub["tip"], c="#2E6E8E", alpha=0.7)
    ax.set_xlabel("total bill ($)")
    ax.set_ylabel("tip ($)")
    ax.set_title(f"Tips on {day}")
    st.pyplot(fig)
    plt.close(fig)        # without this, figures pile up across re-runs

else:
    # Plotly figures stay interactive in the browser: hover, zoom, pan.
    fig = px.scatter(sub, x="total_bill", y="tip", color="time",
                     labels={"total_bill": "total bill ($)", "tip": "tip ($)"})
    st.plotly_chart(fig, use_container_width=True)

st.caption(
    "st.scatter_chart is convenient but chooses the encodings for you. "
    "st.pyplot and st.plotly_chart take a figure you built, so every encoding "
    "decision from Weeks 2-4 survives the move into an app."
)
