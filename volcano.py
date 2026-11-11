#%%
import requests
import pandas as pd
import geopandas as gpd

url = "https://volcanoes.usgs.gov/vsc/api/volcanoApi/geojson"

r = requests.get(url, timeout=30)

print("Status:", r.status_code)
r.raise_for_status()

payload = r.json()

print("Number of volcanoes:", len(payload["features"]))

rows = []

for f in payload["features"]:

    lon, lat = f["geometry"]["coordinates"][:2]

    p = f["properties"]

    rows.append({
        "volcano_name": p["volcanoName"],
        "vnum": p["vnum"],
        "volcano_code": p["volcanoCd"],
        "region": p["region"],
        "observatory": p["obs"],
        "alert_level": p["alertLevel"],
        "color_code": p["colorCode"],
        "threat_level": p["nvewsThreat"],
        "alert_date": p["alertDate"],
        "notice": p["noticeSynopsis"],
        "latitude": lat,
        "longitude": lon
    })

volcanoes = pd.DataFrame(rows)

volcanoes["alert_date"] = pd.to_datetime(
    volcanoes["alert_date"],
    errors="coerce"
)

volcanoes = volcanoes.sort_values(
    ["region", "volcano_name"]
).reset_index(drop=True)

print("Dataset shape:", volcanoes.shape)

volcanoes.head()
# %%
volcanoes.to_csv("volcanoes.csv", index=False)


# %%
volcanoes = pd.read_csv("volcanoes.csv")
states = gpd.read_file("state_boundaries.geojson")

volcanoes["latitude"] = pd.to_numeric(
    volcanoes["latitude"], errors="coerce"
)

volcanoes["longitude"] = pd.to_numeric(
    volcanoes["longitude"], errors="coerce"
)

volcanoes = volcanoes.dropna(
    subset=["latitude", "longitude"]
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

usa_boundary = states.geometry.union_all()

volcano_gdf = volcano_gdf[
    volcano_gdf.geometry.within(usa_boundary)
].copy()

volcanoes_us = pd.DataFrame(
    volcano_gdf.drop(columns="geometry")
)

volcanoes_us.to_csv(
    "volcanoes_us.csv",
    index=False
)

print("U.S. volcanoes:", len(volcanoes_us))
# %%
