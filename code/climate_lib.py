"""Reusable pieces for the Protostar climate forecasts.

Everything here works on a "panel": a DataFrame indexed by (city_id, year) with one
column per metric, NaN where the challenge would not count the value. Forecasters are
plain functions `f(panel, meta, origin, horizon_years) -> DataFrame(city_id, year, metric,
predicted_value)` that only look at rows with year <= origin, so the same code runs the
backtests (origin 2004, 2011, ...) and the real forecast (origin 2018).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CUTOFF = 2018  # data rule: no weather after 2018-12-31

METRICS = ["annual_avg_temp", "total_precip_in", "hottest_day_temp", "cdd"]
SOURCE_COLUMN = {
    "annual_avg_temp": "Avg Temp",
    "total_precip_in": "ToT Prcp",
    "hottest_day_temp": "Max Tmax",
    "cdd": "cdd",
}
MAX_MISSING = 10  # challenge rule: a year counts if <= 10 days are missing


# --------------------------------------------------------------------------- data

def load_panel(data_dir: Path = DATA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (panel, meta). panel: (city_id, year) x METRICS with invalid years as NaN,
    plus `segment` (station segment). meta: one row per city (Name, Lat, Lon)."""
    annual = pd.read_parquet(data_dir / "annual_indices.parquet")
    cover = pd.read_parquet(data_dir / "coverage_and_stations.parquet")
    df = annual.merge(cover, on=["ID", "Name", "Year"], how="left")
    if df["Year"].max() > CUTOFF:
        raise RuntimeError(f"Data rule: found weather dated after {CUTOFF}; refusing to continue.")

    days = np.where(pd.to_datetime(df["Year"].astype(str) + "-12-31").dt.is_leap_year, 366, 365)
    temp_ok = (df["Miss Tmax"] <= MAX_MISSING) & (df["Miss Tmin"] <= MAX_MISSING)
    valid = {
        "annual_avg_temp": temp_ok,
        "total_precip_in": df["Miss Prcp"] <= MAX_MISSING,
        "hottest_day_temp": temp_ok,
        "cdd": temp_ok & (days - df["Days Both"] <= MAX_MISSING),
    }
    panel = pd.DataFrame({"city_id": df["ID"], "year": df["Year"], "segment": df["Station Segment"]})
    for m in METRICS:
        v = df[SOURCE_COLUMN[m]].replace([np.inf, -np.inf], np.nan)
        panel[m] = v.where(valid[m])
    # extra series (not scored): annual_avg_temp = (tmax + tmin) / 2
    panel["tmax"] = df["Avg Tmax"].where(temp_ok)
    panel["tmin"] = df["Avg Tmin"].where(temp_ok)
    panel = panel.set_index(["city_id", "year"]).sort_index()
    meta = annual.groupby("ID")[["Name", "Lat", "Lon"]].last()
    meta.index.name = "city_id"
    return panel, meta


def wide(panel: pd.DataFrame, metric: str, origin: int) -> pd.DataFrame:
    """cities x years matrix of one metric, years <= origin only."""
    p = panel[panel.index.get_level_values("year") <= origin]
    return p[metric].unstack("year")


# --------------------------------------------------------------------------- helpers

def _window(w: pd.DataFrame, origin: int, k: int) -> pd.DataFrame:
    cols = [y for y in w.columns if origin - k < y <= origin]
    return w[cols]


def _long(pred: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = []
    for m, frame in pred.items():
        s = frame.stack()
        s.index.names = ["city_id", "year"]
        out.append(s.rename("predicted_value").reset_index().assign(metric=m))
    return pd.concat(out, ignore_index=True)[["city_id", "year", "metric", "predicted_value"]]


def _broadcast(level: pd.Series, years) -> pd.DataFrame:
    return pd.DataFrame({y: level for y in years})


def segment_offsets(panel: pd.DataFrame, metric: str, origin: int, min_years: int = 5) -> pd.Series:
    """Additive correction putting each (city, year) on the level of the station in use at
    `origin`. For each station change, offset = mean(after) - mean(before) using up to 10
    years each side, with the regional mean of the same years subtracted so the climate
    signal is not mistaken for a station jump."""
    p = panel[panel.index.get_level_values("year") <= origin]
    w = p[metric].unstack("year")
    seg = p["segment"].unstack("year")
    anom = w - w.mean(axis=1).values[:, None]
    national = anom.mean(axis=0)  # simple common signal
    resid = anom - national
    adj = pd.DataFrame(0.0, index=w.index, columns=w.columns)
    for city in w.index:
        s = seg.loc[city].dropna()
        if s.empty:
            continue
        cur = s.iloc[-1]
        changes = s.index[s.diff().fillna(0) != 0]
        r = resid.loc[city]
        for yr in sorted(changes, reverse=True):
            before = r[(r.index < yr) & (r.index >= yr - 10)].dropna()
            after = r[(r.index >= yr) & (r.index < yr + 10)].dropna()
            if len(before) < min_years or len(after) < min_years:
                continue
            jump = after.mean() - before.mean()
            adj.loc[city, adj.columns < yr] += jump
        _ = cur
    return adj


# --------------------------------------------------------------------------- forecasters

def f_mean(panel, meta, origin, years, k=30, metrics=METRICS):
    """Mean of the last k valid years (k=30 is the challenge's baseline)."""
    return _long({m: _broadcast(_window(wide(panel, m, origin), origin, k).mean(axis=1), years) for m in metrics})


def f_last(panel, meta, origin, years, metrics=METRICS):
    out = {}
    for m in metrics:
        w = wide(panel, m, origin)
        out[m] = _broadcast(w.ffill(axis=1).iloc[:, -1], years)
    return _long(out)


def f_city_trend(panel, meta, origin, years, k=30, metrics=METRICS):
    """Per-city OLS line over the last k years, extrapolated (the 'Linear Trend' baseline)."""
    out = {}
    for m in metrics:
        w = _window(wide(panel, m, origin), origin, k)
        x = np.array(w.columns, float)
        preds = {}
        for city, row in w.iterrows():
            ok = row.notna().values
            if ok.sum() < 5:
                preds[city] = [row.mean()] * len(years)
                continue
            b, a = np.polyfit(x[ok], row.values[ok], 1)
            preds[city] = [a + b * y for y in years]
        out[m] = pd.DataFrame(preds, index=list(years)).T
    return _long(out)


def regional_slope(w: pd.DataFrame, meta: pd.DataFrame, origin: int, k: int, radius_km: float) -> pd.Series:
    """Trend (units/yr) per city estimated from the pooled anomalies of all cities within
    `radius_km`. Pooling makes the slope far less noisy than a single-station fit."""
    win = _window(w, origin, k)
    anom = win.sub(win.mean(axis=1), axis=0)
    x = np.array(win.columns, float)
    lat = np.radians(meta.loc[win.index, "Lat"].values)
    lon = np.radians(meta.loc[win.index, "Lon"].values)
    d = 6371 * np.arccos(np.clip(
        np.sin(lat)[:, None] * np.sin(lat)[None, :]
        + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.cos(lon[:, None] - lon[None, :]), -1, 1))
    slopes = {}
    A = anom.values
    for i, city in enumerate(win.index):
        nb = d[i] <= radius_km
        series = np.nanmean(A[nb], axis=0)
        ok = ~np.isnan(series)
        slopes[city] = np.polyfit(x[ok], series[ok], 1)[0] if ok.sum() >= 5 else 0.0
    return pd.Series(slopes)


def f_level_trend(panel, meta, origin, years, *, level_k=None, trend_k=None, radius_km=None,
                  damp=None, adjust_stations=None, metrics=METRICS, params=None):
    """Level = mean of the last `level_k` years (station-adjusted if asked), plus a regionally
    pooled trend fitted on the last `trend_k` years, applied from the level window's centre
    to each target year and multiplied by `damp` (0 = no trend, 1 = full trend).

    `params` may give per-metric dicts overriding the keyword defaults."""
    defaults = dict(level_k=15, trend_k=40, radius_km=800, damp=1.0, adjust_stations=True)
    out = {}
    for m in metrics:
        cfg = dict(defaults)
        cfg.update({k: v for k, v in dict(level_k=level_k, trend_k=trend_k, radius_km=radius_km,
                                          damp=damp, adjust_stations=adjust_stations).items() if v is not None})
        if params and m in params:
            cfg.update(params[m])
        if cfg.get("trend_start"):  # fixed start year instead of a fixed window length
            cfg["trend_k"] = origin - cfg["trend_start"] + 1
        w = wide(panel, m, origin)
        if cfg["adjust_stations"]:
            w = w + segment_offsets(panel, m, origin)
        lvl_win = _window(w, origin, cfg["level_k"])
        level = lvl_win.mean(axis=1)
        centre = lvl_win.apply(lambda r: np.mean([c for c, v in r.items() if pd.notna(v)]) if r.notna().any() else origin, axis=1)
        slope = regional_slope(w, meta, origin, cfg["trend_k"], cfg["radius_km"]) if cfg["damp"] else 0 * level
        out[m] = pd.DataFrame({y: level + cfg["damp"] * slope * (y - centre) for y in years})
        if m in ("cdd", "total_precip_in"):
            out[m] = out[m].clip(lower=0)
    return _long(out)


def f_cdd_from_temp(panel, meta, origin, years, *, temp_params=None, level_k=15, beta_k=30):
    """CDD driven by the temperature forecast: each city's CDD moves with its annual mean
    temperature at a sensitivity beta (°F-days per °F) fitted on its last `beta_k` years.
    pred = mean CDD (last level_k yrs) + beta * (temp forecast - mean temp (last level_k yrs))."""
    temp = f_level_trend(panel, meta, origin, years, metrics=["annual_avg_temp"],
                         params={"annual_avg_temp": temp_params} if temp_params else None)
    tw = wide(panel, "annual_avg_temp", origin) + segment_offsets(panel, "annual_avg_temp", origin)
    cw = wide(panel, "cdd", origin) + segment_offsets(panel, "cdd", origin)
    t_win, c_win = _window(tw, origin, beta_k), _window(cw, origin, beta_k)
    beta = {}
    for city in t_win.index:
        ok = t_win.loc[city].notna() & c_win.loc[city].notna()
        beta[city] = np.polyfit(t_win.loc[city][ok], c_win.loc[city][ok], 1)[0] if ok.sum() >= 10 else 0.0
    beta = pd.Series(beta)
    t_lvl, c_lvl = _window(tw, origin, level_k).mean(axis=1), _window(cw, origin, level_k).mean(axis=1)
    tp = temp.pivot(index="city_id", columns="year", values="predicted_value")
    cdd = tp.sub(t_lvl, axis=0).mul(beta, axis=0).add(c_lvl, axis=0).clip(lower=0)
    return _long({"cdd": cdd})


_CURVES = {}


def load_cdd_curves(data_dir: Path = DATA) -> pd.DataFrame:
    """(city_id, year) x shift table from prepare_data.py: CDD of that year with every day s °F warmer."""
    if data_dir not in _CURVES:
        c = pd.read_parquet(data_dir / "cdd_shift_curves.parquet")
        if c["Year"].max() > CUTOFF:
            raise RuntimeError(f"Data rule: found weather dated after {CUTOFF}; refusing to continue.")
        _CURVES[data_dir] = c.rename(columns={"ID": "city_id", "Year": "year"}).set_index(["city_id", "year"])
    return _CURVES[data_dir]


def f_cdd_delta(panel, meta, origin, years, *, temp_params=None, k=15):
    """Delta-method CDD: replay the daily weather of the city's last `k` valid CDD years,
    with every day warmed by (temperature forecast - mean temperature of those years), and
    average the resulting CDD. Captures that a city with many days just under 65 °F gains
    more CDD per degree than one that is rarely warm."""
    curves = load_cdd_curves()
    shifts = np.array([float(c[1:]) for c in curves.columns])
    temp = f_level_trend(panel, meta, origin, years, metrics=["annual_avg_temp"],
                         params={"annual_avg_temp": temp_params} if temp_params else None)
    tp = temp.pivot(index="city_id", columns="year", values="predicted_value")
    cw, tw = wide(panel, "cdd", origin), wide(panel, "annual_avg_temp", origin)
    out = {}
    for city in tp.index:
        yrs = [y for y in cw.columns[::-1] if pd.notna(cw.loc[city, y]) and pd.notna(tw.loc[city, y])
               and (city, y) in curves.index][:k]
        if not yrs:
            out[city] = pd.Series(np.nan, index=tp.columns)
            continue
        C = curves.loc[[(city, y) for y in yrs]].to_numpy()
        base_t = tw.loc[city, yrs].mean()
        out[city] = pd.Series({y: np.mean([np.interp(tp.loc[city, y] - base_t, shifts, row) for row in C])
                               for y in tp.columns})
    return _long({"cdd": pd.DataFrame(out).T})


def bias_corrected(forecaster, metrics, *, lookback=10, shrink=1.0, horizon=7):
    """Wrap a forecaster with a correction learned from its own past errors.

    At origin O, rerun `forecaster` from each earlier origin o = O-lookback .. O-1, compare
    with the years already observed by O (years o+1 .. min(o+horizon, O)), and subtract
    `shrink` x the mean signed error (all cities pooled) from the new forecast. Only data
    up to O is touched, so the wrapper keeps the no-look-ahead property."""
    cache = {}

    def run(panel, meta, o, years):
        key = (id(panel), o)
        if key not in cache:
            cache[key] = forecaster(panel, meta, o, list(range(o + 1, o + 1 + horizon)))
        return cache[key]

    def f(panel, meta, origin, years):
        pred = forecaster(panel, meta, origin, years)
        truth = panel[metrics].stack().rename("truth").reset_index()
        truth.columns = ["city_id", "year", "metric", "truth"]
        errs = []
        for o in range(origin - lookback, origin):
            past = run(panel, meta, o, None)
            past = past[past.metric.isin(metrics) & (past.year <= origin)]
            errs.append(past.merge(truth, on=["city_id", "year", "metric"]))
        e = pd.concat(errs).dropna()
        bias = (e.predicted_value - e.truth).groupby(e.metric).mean()
        pred = pred.copy()
        adj = pred.metric.map(bias).fillna(0) * shrink
        pred["predicted_value"] = pred.predicted_value - adj
        lo = pred.metric.isin(["cdd", "total_precip_in"])
        pred.loc[lo, "predicted_value"] = pred.loc[lo, "predicted_value"].clip(lower=0)
        return pred

    return f


def blend(*preds: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight average of forecasts (rows matched on city_id, year, metric)."""
    return (pd.concat(preds).groupby(["city_id", "year", "metric"], as_index=False)["predicted_value"].mean())


# --------------------------------------------------------------------------- scoring

def score(pred: pd.DataFrame, panel: pd.DataFrame, baseline: pd.DataFrame, origin: int) -> dict:
    """Challenge score: S_m = 1 - MAE_model / MAE_baseline per metric, scored only on valid
    truth rows of city-metrics with >= 10 valid years in the 30 years before origin."""
    truth = panel[METRICS].stack().rename("truth").reset_index().rename(columns={"level_2": "metric"})
    hist = panel[(panel.index.get_level_values("year") > origin - 30) & (panel.index.get_level_values("year") <= origin)]
    counts = hist[METRICS].notna().groupby("city_id").sum().stack().rename("n").reset_index().rename(columns={"level_1": "metric"})
    df = (pred.merge(truth, on=["city_id", "year", "metric"])
              .merge(baseline.rename(columns={"predicted_value": "base"}), on=["city_id", "year", "metric"])
              .merge(counts, on=["city_id", "metric"]))
    df = df[df["n"] >= 10].dropna(subset=["truth", "predicted_value", "base"])
    res = {}
    for m in METRICS:
        d = df[df.metric == m]
        mae, mae_b = (d.predicted_value - d.truth).abs().mean(), (d.base - d.truth).abs().mean()
        res[m] = 1 - mae / mae_b
        res[m + "_mae"] = mae
    res["score"] = 100 * max(0.0, np.mean([res[m] for m in METRICS]))
    return res


def backtest(forecaster, panel, meta, origins, horizon=7, **kw) -> pd.DataFrame:
    rows = []
    for o in origins:
        years = list(range(o + 1, o + 1 + horizon))
        base = f_mean(panel, meta, o, years, k=30)
        pred = forecaster(panel, meta, o, years, **kw)
        rows.append({"origin": o, **score(pred, panel, base, o)})
    return pd.DataFrame(rows)
