"""Export the forecasts in formats other tools read directly.

Run after forecast.py:  python code/export.py
- outputs/forecasts_wide.csv: one row per city and year, a column per metric, with city name,
  latitude and longitude (spreadsheets, pandas, R, Tableau).
- outputs/forecasts.geojson: one point per city with its 2019-2025 forecasts and the change
  versus 1989-2018 (QGIS, ArcGIS, geojson.io, kepler.gl, Leaflet, the Hub viewer).
"""

import json

import pandas as pd

import climate_lib as cl

panel, meta = cl.load_panel()
sub = pd.read_csv(cl.ROOT / "submission.csv")
hist = panel[panel.index.get_level_values("year") >= 1989][cl.METRICS].groupby("city_id").mean()

wide = sub.pivot_table(index=["city_id", "year"], columns="metric", values="predicted_value").reset_index()
wide = meta.rename(columns={"Name": "name", "Lat": "lat", "Lon": "lon"}).reset_index().merge(wide, on="city_id")
(cl.ROOT / "outputs").mkdir(exist_ok=True)
wide[["city_id", "name", "lat", "lon", "year"] + cl.METRICS].to_csv(cl.ROOT / "outputs" / "forecasts_wide.csv", index=False)

features = []
for city, g in wide.groupby("city_id"):
    props = {"city_id": city, "name": g["name"].iloc[0]}
    for m in cl.METRICS:
        props[f"{m}_1989_2018"] = round(float(hist.loc[city, m]), 2) if pd.notna(hist.loc[city, m]) else None
        props[f"{m}_2019_2025_mean"] = round(float(g[m].mean()), 2)
        for _, r in g.iterrows():
            props[f"{m}_{int(r.year)}"] = float(r[m])
    features.append({"type": "Feature", "properties": props,
                     "geometry": {"type": "Point", "coordinates": [float(g["lon"].iloc[0]), float(g["lat"].iloc[0])]}})
(cl.ROOT / "outputs" / "forecasts.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))
print(f"wrote outputs/forecasts_wide.csv ({len(wide)} rows) and outputs/forecasts.geojson ({len(features)} cities)")
