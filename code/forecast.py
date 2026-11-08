"""Write submission.csv: forecasts for 2019-2025 made from data up to 2018.

Run from the repository root:  python code/forecast.py
Also re-runs the backtests of the final model and writes code/backtest_results.csv.
"""

import pandas as pd

import climate_lib as cl

# Per-metric settings of the level + pooled-trend model, chosen with code/backtest.py.
PARAMS = {
    "annual_avg_temp": dict(level_k=15, trend_k=50, radius_km=1000, damp=1.0, adjust_stations=True),
    "cdd": dict(level_k=15, trend_k=50, radius_km=1000, damp=1.0, adjust_stations=True),
    "hottest_day_temp": dict(level_k=30, trend_k=50, radius_km=1000, damp=0.5, adjust_stations=True),
    "total_precip_in": dict(level_k=30, damp=0.0, adjust_stations=False),  # = 30-year mean
}
YEARS = list(range(2019, 2026))
CDD_DELTA_YEARS = 30  # daily weather of the last 30 valid years is replayed, warmed


def model(panel, meta, origin, years):
    """Final model.
    annual_avg_temp, hottest_day_temp, total_precip_in: level + pooled trend (PARAMS).
    cdd: delta method, the city's last 30 years of daily weather warmed to the temperature
    forecast, CDD recomputed and averaged."""
    base = cl.f_level_trend(panel, meta, origin, years, params=PARAMS,
                            metrics=["annual_avg_temp", "hottest_day_temp", "total_precip_in"])
    cdd = cl.f_cdd_delta(panel, meta, origin, years, temp_params=PARAMS["annual_avg_temp"], k=CDD_DELTA_YEARS)
    return pd.concat([base, cdd], ignore_index=True)


if __name__ == "__main__":
    from backtest import CHECK, TUNE

    panel, meta = cl.load_panel()

    bt = cl.backtest(model, panel, meta, TUNE + CHECK)
    bt["split"] = ["tune" if o in TUNE else "check" for o in bt.origin]
    bt.round(4).to_csv(cl.ROOT / "code" / "backtest_results.csv", index=False)
    print(bt.groupby("split")[["score"] + cl.METRICS].mean().round(3))

    pred = model(panel, meta, cl.CUTOFF, YEARS)
    template = pd.read_csv(cl.ROOT / "submission.csv", usecols=["city_id", "year", "metric"])
    out = template.merge(pred, on=["city_id", "year", "metric"], how="left")
    if out["predicted_value"].isna().any():
        # a city-metric with no usable recent years: fall back to its all-years mean
        allmean = panel[cl.METRICS].groupby("city_id").mean().stack().rename("fb").reset_index()
        allmean.columns = ["city_id", "metric", "fb"]
        out = out.merge(allmean, on=["city_id", "metric"], how="left")
        print(f"fallback used for {out.predicted_value.isna().sum()} rows")
        out["predicted_value"] = out["predicted_value"].fillna(out.pop("fb"))
    out["predicted_value"] = out["predicted_value"].round(2)
    out.to_csv(cl.ROOT / "submission.csv", index=False)
    print(f"wrote submission.csv ({len(out)} rows)")
