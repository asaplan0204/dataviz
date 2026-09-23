#%%
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from statsmodels.tsa.seasonal import seasonal_decompose

# make the width as wide as the screen
# st.set_page_config(layout="wide")

# title
st.markdown(
    "<h1 style='text-align: center;'>Philippines Weather</h1>",
    unsafe_allow_html=True
)

# loading the data
df = pd.read_csv("Philippines_weather_data.csv")

# converting date to datetime format
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
df = df.set_index("Date")

# preview of the data
st.subheader("Philippines Weather Dataset Preview")
st.dataframe(df.head(20))

# summary statistics
summary = df.describe()


st.subheader("Summary Statistics")
st.dataframe(summary)

st.subheader("Philippines Sunshine Trend With Uncertainty Band")
# Resolution widget
resolution = st.selectbox(
    "Select Resolution",
    ["Monthly", "Quarterly", "Yearly"]
)

if resolution == "Monthly":
    sun = df["Sunshine_Duration"].resample("MS").mean()

elif resolution == "Quarterly":
    sun = df["Sunshine_Duration"].resample("QS").mean()

else:
    sun = df["Sunshine_Duration"].resample("YS").mean()

# Rolling mean
rolling_mean = sun.rolling(12 if resolution == "Monthly" else 4 if resolution == "Quarterly" else 1).mean()

# Rolling standard deviation
rolling_std = sun.rolling(12 if resolution == "Monthly" else 4 if resolution == "Quarterly" else 1).std()

# bounds
lower = rolling_mean - rolling_std
upper = rolling_mean + rolling_std

# chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=upper.index, y=upper, mode="lines", line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=lower.index, y=lower, mode="lines", fill="tonexty", fillcolor="rgba(31, 119, 180, 0.2)", line=dict(width=0), name="±1 Standard Deviation"))
fig.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean, mode="lines", name="12-Month Rolling Mean"))
fig.update_layout(title="Sunshine Duration Between 2000 and 2023 In The Philippines", xaxis_title="Date",yaxis_title="Sunshine Duration",hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)


# rolling menas for seasonaility chart
monthly_sunshine = df["Sunshine_Duration"].resample("MS").mean()
rolling_3 = monthly_sunshine.rolling(window=3).mean()
rolling_6 = monthly_sunshine.rolling(window=6).mean()
rolling_12 = monthly_sunshine.rolling(window=12).mean()
rolling_60 = monthly_sunshine.rolling(window=60).mean()

st.subheader("Philippines Sunshine Rolling Means")
fig = go.Figure()
fig.add_trace(go.Scatter(x=rolling_3.index, y=rolling_3, mode="lines", name="3-Month Rolling Mean", visible=True))
fig.add_trace(go.Scatter(x=rolling_6.index, y=rolling_6, mode="lines", name="6-Month Rolling Mean", visible=False))
fig.add_trace(go.Scatter(x=rolling_12.index, y=rolling_12, mode="lines", name="12-Month Rolling Mean", visible=False))
fig.add_trace(go.Scatter(x=rolling_60.index, y=rolling_60, mode="lines", name="60-Month Rolling Mean",visible=False))

# dropdown code
fig.update_layout(
    updatemenus=[
        dict(
            buttons=[
                dict(
                    label="3 Months",
                    method="update",
                    args=[
                        {"visible": [True, False, False, False]},
                        {"title": "3-Month Rolling Mean"}
                    ]
                ),
                dict(
                    label="6 Months",
                    method="update",
                    args=[
                        {"visible": [False, True, False, False]},
                        {"title": "6-Month Rolling Mean"}
                    ]
                ),
                dict(
                    label="12 Months",
                    method="update",
                    args=[
                        {"visible": [False, False, True, False]},
                        {"title": "12-Month Rolling Mean"}
                    ]
                ),
                dict(
                    label="60 Months",
                    method="update",
                    args=[
                        {"visible": [False, False, False, True]},
                        {"title": "60-Month Rolling Mean"}
                    ]
                )
            ],
            direction="down",
            showactive=True
        )
    ],
    title="Sunshine Duration Rolling Mean",
    xaxis_title="Date",
    yaxis_title="Sunshine Duration",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)



# stats model decompose seasonality
monthly_sun = df["Sunshine_Duration"].resample("MS").mean()
st.subheader("Seasonality Of Philippines Sunshine Decomposed")
result = seasonal_decompose(monthly_sun, period=12)

# create charts
def make_plot(series, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name=title))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Sunshine Duration",hovermode="x unified")
    return fig

# the four decomposed charts
st.plotly_chart(
    make_plot(result.observed, "Observed"),
    use_container_width=True
)

st.plotly_chart(
    make_plot(result.trend, "Trend"),
    use_container_width=True
)

st.plotly_chart(
    make_plot(result.seasonal, "Seasonal"),
    use_container_width=True
)

st.plotly_chart(
    make_plot(result.resid, "Residual"),
    use_container_width=True
)


# animation experiement
st.subheader("Animation of Trend")
if "running" not in st.session_state:
    st.session_state.running = False

if "position" not in st.session_state:
    st.session_state.position = 12

# buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Start"):
        st.session_state.position = 12
        st.session_state.running = True

with col2:
    if st.button("Pause"):
        st.session_state.running = False

with col3:
    if st.button("Resume"):
        st.session_state.running = True

# chart
frame = st.empty()
def draw_chart(upto):

    x = rolling_mean.index[:upto]
    y = rolling_mean.iloc[:upto]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="12-Month Rolling Mean"))
    fig.update_layout(title="Sunshine Duration Trend", xaxis_title="Date", yaxis_title="Sunshine Duration", hovermode="x unified")
    return fig

fig = draw_chart(st.session_state.position)
frame.plotly_chart(fig, use_container_width=True)

# animation
if st.session_state.running:
    while (
        st.session_state.running
        and st.session_state.position < len(rolling_mean)
    ):
        st.session_state.position += 1
        fig = draw_chart(st.session_state.position)
        frame.plotly_chart(fig, use_container_width=True)
        time.sleep(0.1)


st.subheader("Temporal Honesty Choice")
st.info("""

For this week's assignment compared to last week's assignment I choose to keep the charts center screened instead of wide screened.  This decision was 
made based on the temporal honesty choice of aspect ratio.  When the seasonality charts are wide screened instead of 
centered it makes the slopes look less steep, indicating a flatter seasonality even though it is the same data being used.  Thus, by not 
using the wide screened option for the charts I prevented the slopes from being exaggerated.  
""")