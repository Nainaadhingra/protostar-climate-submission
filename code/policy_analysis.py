"""Numbers behind the README's Policy Idea: forecast change in cooling demand per city.

Run after forecast.py:  python code/policy_analysis.py
Writes outputs/cdd_change_by_city.csv (one row per city): 1989-2018 mean CDD and annual
temperature (valid years only), the mean 2019-2025 forecast, and the change.
"""

import pandas as pd

import climate_lib as cl

panel, meta = cl.load_panel()
sub = pd.read_csv(cl.ROOT / "submission.csv")
fc = sub.groupby(["city_id", "metric"])["predicted_value"].mean().unstack()
hist = panel[panel.index.get_level_values("year") >= 1989][cl.METRICS].groupby("city_id").mean()

out = pd.DataFrame({
    "name": meta["Name"], "lat": meta["Lat"], "lon": meta["Lon"],
    "cdd_1989_2018": hist["cdd"], "cdd_2019_2025_forecast": fc["cdd"],
    "temp_1989_2018": hist["annual_avg_temp"], "temp_2019_2025_forecast": fc["annual_avg_temp"],
})
out["cdd_change"] = out["cdd_2019_2025_forecast"] - out["cdd_1989_2018"]
out["cdd_change_pct"] = 100 * out["cdd_change"] / out["cdd_1989_2018"]
out["temp_change"] = out["temp_2019_2025_forecast"] - out["temp_1989_2018"]
# "Cooling-gap" cities: low historical cooling demand (< 1,000 °F-days) but >= 20% and
# >= 50 °F-days growth (the absolute floor drops near-zero cities like Crescent City)
out["cooling_gap_city"] = ((out["cdd_1989_2018"] < 1000) & (out["cdd_change_pct"] >= 20)
                           & (out["cdd_change"] >= 50))

(cl.ROOT / "outputs").mkdir(exist_ok=True)
out.round(2).sort_values("cdd_change_pct", ascending=False).to_csv(cl.ROOT / "outputs" / "cdd_change_by_city.csv")
print(f"cities with rising CDD: {(out.cdd_change > 0).sum()} of {len(out)}; "
      f"median change {out.cdd_change_pct.median():.1f}%")
print("cooling-gap cities:")
print(out[out.cooling_gap_city].sort_values("cdd_change_pct", ascending=False)
      [["name", "cdd_1989_2018", "cdd_2019_2025_forecast", "cdd_change_pct"]].round(1).to_string())
