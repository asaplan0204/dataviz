import pandas as pd
import geopandas as gpd
import numpy as np
import folium
import streamlit as st

from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

st.set_page_config(
    page_title="USGS Volcano Dashboard",
    page_icon="🌋",
    layout="wide"
)

st.title("Volcano Distribution and Activity in the United States")

st.markdown("""
Explore the geographic distribution of U.S. volcanoes
and compare volcano activity across states.
""")

# load the volcanoes_us.csv to see without spatial join unmatched volcanoes
# volcanoes = pd.read_csv("volcanoes_us.csv")
volcanoes = pd.read_csv("volcanoes.csv")
acs = pd.read_csv("acs.csv")
states = gpd.read_file("state_boundaries.geojson")

def normalize_fips(series):
    return (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64")
        .astype("string")
        .str.zfill(2)
    )


acs["fips"] = normalize_fips(acs["fips"])
states["fips"] = normalize_fips(states["fips"])

volcanoes["latitude"] = pd.to_numeric(
    volcanoes["latitude"], errors="coerce"
)

volcanoes["longitude"] = pd.to_numeric(
    volcanoes["longitude"], errors="coerce"
)

volcanoes["alert_level"] = (
    volcanoes["alert_level"]
    .fillna("Unknown")
    .astype(str)
)

volcanoes["alert_level"] = (
    volcanoes["alert_level"]
    .replace("", "Unknown")
)

volcanoes["alert_date"] = pd.to_datetime(
    volcanoes["alert_date"],
    errors="coerce"
)

volcano_gdf = gpd.GeoDataFrame(
    volcanoes,
    geometry=gpd.points_from_xy(
        volcanoes["longitude"],
        volcanoes["latitude"]
    ),
    crs="EPSG:4326"
)

states = states.to_crs("EPSG:4326")

volcano_gdf = gpd.sjoin(
    volcano_gdf,
    states[["fips", "geometry"]],
    how="left",
    predicate="within"
)

total_volcanoes = len(volcano_gdf)

matched_volcanoes = (
    volcano_gdf["fips"].notna().sum()
)

unmatched_volcanoes = (
    volcano_gdf["fips"].isna().sum()
)

state_counts = (
    volcano_gdf
    .dropna(subset=["fips"])
    .groupby("fips")
    .agg(
        Volcano_Count=("volcano_name", "count")
    )
    .reset_index()
)

state_join_check = states[["fips"]].drop_duplicates().merge(
    acs[["fips", "pop_total"]],
    on="fips",
    how="left",
    indicator=True
)

matched_states = (
    state_join_check["_merge"] == "both"
).sum()

unmatched_states = (
    state_join_check["_merge"] == "left_only"
).sum()

state_stats = acs.merge(
    state_counts,
    on="fips",
    how="left"
)

state_stats["Volcano_Count"] = (
    state_stats["Volcano_Count"].fillna(0)
)

state_stats["Volcanoes_per_100k"] = (
    state_stats["Volcano_Count"]
    / state_stats["pop_total"].replace(0, np.nan)
    * 100000
)

map_data = states[
    ["fips", "state", "geometry"]
].merge(
    state_stats.drop(columns=["state"]),
    on="fips",
    how="left"
)

st.sidebar.header("Volcano Alert Level Filters")

all_alert_levels = sorted(
    volcano_gdf["alert_level"].dropna().unique()
)

selected_alerts = st.sidebar.multiselect(
    "Select Alert Levels",
    options=all_alert_levels,
    default=all_alert_levels
)

filtered_volcanoes = volcano_gdf[
    volcano_gdf["alert_level"].isin(selected_alerts)
]

filtered_counts = (
    filtered_volcanoes.groupby("fips")
    .size()
    .reset_index(name="Volcano_Count")
)

filtered_stats = acs.merge(
    filtered_counts,
    on="fips",
    how="left"
)

filtered_stats["Volcano_Count"] = (
    filtered_stats["Volcano_Count"].fillna(0)
)

filtered_stats["Volcanoes_per_100k"] = (
    filtered_stats["Volcano_Count"]
    / filtered_stats["pop_total"].replace(0, np.nan)
    * 100000
)

filtered_map = states.merge(
    filtered_stats[
        ["fips", "Volcano_Count", "Volcanoes_per_100k"]
    ],
    on="fips",
    how="left"
)

alert_colors = {
    "NORMAL": "green",
    "ADVISORY": "orange",
    "WATCH": "red",
    "WARNING": "darkred",
    "UNASSIGNED": "gray"
}

tab1, tab2, tab3 = st.tabs([
    "Volcano Distribution Map",
    "Volcano Activity Analysis",
    "Join Results"
])

with tab1:

    st.header("Geographic Distribution of Volcanoes")

    if filtered_map.empty:

        st.warning("Select at least one alert level in the sidebar.")

    else:
        values = map_data[
            "Volcanoes_per_100k"
        ].dropna()

        if values.max() > values.min():

            thresholds = np.linspace(
                values.min(),
                values.max(),
                6
            ).tolist()

            m = folium.Map(
                location=[39.0, -110.0],
                zoom_start=3,
                tiles="OpenStreetMap"
            )

            folium.Choropleth(
                geo_data=filtered_map.to_json(),
                data=filtered_map,
                columns=[
                    "fips",
                    "Volcanoes_per_100k"
                ],
                key_on="feature.properties.fips",
                fill_color="YlOrRd",
                fill_opacity=0.75,
                line_opacity=0.3,
                threshold_scale=thresholds,
                legend_name="Volcanoes per 100,000 Residents"
            ).add_to(m)

        else:

            m = folium.Map(
                location=[39.5, -98.35],
                zoom_start=4,
                tiles="OpenStreetMap"
            )

        volcano_layer = folium.FeatureGroup(
            name="Individual Volcano Locations",
            show=True
        )

        marker_cluster = MarkerCluster().add_to(
            volcano_layer
        )

        for _, row in filtered_volcanoes.iterrows():

            if (
                pd.notna(row["latitude"])
                and pd.notna(row["longitude"])
            ):

                alert = row["alert_level"]

                marker_color = alert_colors.get(
                    alert,
                    "gray"
                )

                folium.CircleMarker(
                    location=[
                        row["latitude"],
                        row["longitude"]
                    ],
                    radius=6,
                    color=marker_color,
                    fill=True,
                    fill_color=marker_color,
                    fill_opacity=0.8,
                    popup=f"""
                        Volcano: {row['volcano_name']}<br>
                        Region: {row['region']}<br>
                        Alert Level: {alert}<br>
                        Aviation Color: {row['color_code']}<br>
                        Threat Level: {row['threat_level']}
                    """
                ).add_to(marker_cluster)

        volcano_layer.add_to(m)

        folium.LayerControl().add_to(m)

        st_folium(
            m,
            width=1200,
            height=700
        )

        st.caption("""
        **Map Parameters**

        - **Layer 1 – State choropleth:** Encodes the number of volcanoes per 100,000 residents. .
        - **Layer 2 – Volcano locations:** Individual volcano locations displayed as clustered circle markers. Marker colors represent USGS alert levels: green (Normal), orange (Advisory), red (Watch), dark red (Warning), and gray (Unknown).
        - **Classification:** Equal-interval classification with 5 classes. 
        - **Projection:** EPSG:4326, used for geographic coordinates and web mapping.
        """)

with tab2:

    st.header("Volcano Activity Analysis")

    if filtered_volcanoes.empty:

        st.warning("Select at least one alert level in the sidebar.")

    else:
        st.subheader("Volcano Counts by State")

        volcano_chart = filtered_stats.sort_values(
            "Volcano_Count",
            ascending=False
        )

        st.bar_chart(
            volcano_chart,
            x="state",
            y="Volcano_Count"
        )

        st.subheader("Volcanoes by Alert Level")

        alert_counts = (
            filtered_volcanoes["alert_level"]
            .value_counts()
            .rename_axis("Alert Level")
            .reset_index(name="Volcano Count")
        )

        st.bar_chart(
            alert_counts,
            x="Alert Level",
            y="Volcano Count"
        )

        st.subheader("Volcanoes by Geographic Region")

        region_counts = (
            filtered_volcanoes["region"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Region")
            .reset_index(name="Volcano Count")
        )

        st.bar_chart(
            region_counts,
            x="Region",
            y="Volcano Count"
        )

        st.subheader("Volcanoes by Threat Classification")

        threat_counts = (
            filtered_volcanoes["threat_level"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Threat Level")
            .reset_index(name="Volcano Count")
        )

        st.bar_chart(
            threat_counts,
            x="Threat Level",
            y="Volcano Count"
        )

    st.divider()

    st.subheader("Volcano Information")

    st.dataframe(
        filtered_volcanoes[
            [
                "volcano_name",
                "region",
                "alert_level",
                "color_code",
                "threat_level",
                "observatory",
                "alert_date"
            ]
        ],
        use_container_width=True
    )

    st.caption("""
    Volcano counts are based on the spatial join between volcano locations and state boundaries. Alert levels, aviation color codes, and threat classifications are obtained from the USGS volcano monitoring dataset.
    """)

with tab3:

    st.header("Join Results")

    st.subheader("Spatial Join: Volcano Locations")

    st.markdown("""
    The spatial join assigns each volcano point to the state polygon containing its geographic coordinates.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Volcanoes",
        f"{total_volcanoes:,}"
    )

    col2.metric(
        "Matched Volcanoes",
        f"{matched_volcanoes:,}"
    )

    col3.metric(
        "Unmatched Volcanoes",
        f"{unmatched_volcanoes:,}"
    )

    unmatched = volcano_gdf[
        volcano_gdf["fips"].isna()
    ]

    st.caption("""
    The reason for unmatched volcanoes is that there are coordinates outside of the state polygon or points are exactly on the boundaries of the polygon.  There is an option at the beginning to load in 
    volcanoes_us.csv file in order to prevent unmatched volcanoes, however I chose to leave in the unmatched volcanoes for demonstration purposes.
    """)

    st.divider()

    st.subheader("Attribute Join: ACS Population Data")

    st.markdown("""
    The attribute join connects state boundary records to American Community Survey population data using the shared two-digit state FIPS code.
    """)

    col1, col2 = st.columns(2)

    col1.metric(
        "Matched States",
        f"{matched_states:,}"
    )

    col2.metric(
        "Unmatched States",
        f"{unmatched_states:,}"
    )

    st.caption("""
    The population data is used to normalize volcano counts, which results in a rate representing volcanoes per 100,000 residents.  This allows for comparisons between states with different population sizes.
    Population data is used to normalize volcano counts.
    """)

    st.divider()
