"""Rolling-origin backtests on data up to 2018 only.

For each origin year o, every model sees years <= o and forecasts o+1..o+7, scored with the
challenge formula against the 30-year average. Origins 1997-2004 are used to choose settings
("tune"); 2005-2011 are held out to check that the choice generalises ("check").

    python code/backtest.py compare   # every model we tried -> code/experiments.csv (~10 min)
    python code/backtest.py grid      # settings grid of the level + pooled-trend model (~5 min)
"""

import sys

import pandas as pd

import climate_lib as cl

TUNE = list(range(1997, 2005))
CHECK = list(range(2005, 2012))


def summarise(name, f, panel, meta):
    rows = []
    for split, origins in [("tune", TUNE), ("check", CHECK)]:
        r = cl.backtest(f, panel, meta, origins).drop(columns="origin").mean()
        rows.append({"model": name, "split": split, **r.to_dict()})
        print(f"{name:34s} {split:5s} " + " ".join(f"{m}={r[m]:+.3f}" for m in cl.METRICS), flush=True)
    return rows


def compare(panel, meta):
    import learned
    from forecast import PARAMS, model as final

    B = {m: learned.FeatureBuilder(panel, meta, m) for m in cl.METRICS}

    def lt(**kw):
        return lambda p, mt, o, y: cl.f_level_trend(p, mt, o, y, **kw)

    def ml(make, span=None):
        return lambda p, mt, o, y: learned.f_learned(p, mt, o, y, builders=B, make_model=make, train_span=span)

    T = PARAMS["annual_avg_temp"]

    def cdd_temp(p, mt, o, y):
        return cl.f_cdd_from_temp(p, mt, o, y, temp_params=T)

    def temp(**kw):
        return lambda p, mt, o, y: cl.f_level_trend(p, mt, o, y, metrics=["annual_avg_temp"],
                                                    params={"annual_avg_temp": {**T, **kw}})

    def tmax_tmin(p, mt, o, y):
        a = cl.f_level_trend(p, mt, o, y, metrics=["tmax", "tmin"], params={"tmax": T, "tmin": T})
        a = a.groupby(["city_id", "year"], as_index=False)["predicted_value"].mean()
        return a.assign(metric="annual_avg_temp")

    def gbm_cdd(p, mt, o, y):
        return learned.f_learned(p, mt, o, y, metrics=["cdd"], builders=B, make_model=learned.make_gbm, train_span=30)

    models = [
        ("baseline: last year", lambda p, mt, o, y: cl.f_last(p, mt, o, y)),
        ("baseline: city linear trend 30y", lambda p, mt, o, y: cl.f_city_trend(p, mt, o, y, k=30)),
        ("mean of last 10 years", lambda p, mt, o, y: cl.f_mean(p, mt, o, y, k=10)),
        ("mean of last 15 years", lambda p, mt, o, y: cl.f_mean(p, mt, o, y, k=15)),
        ("level15 + pooled trend (one setting)", lt(level_k=15, trend_k=50, radius_km=1000)),
        ("level+trend, per-metric PARAMS", lt(params=PARAMS)),
        ("ridge, all origins since 1940", ml(learned.make_ridge)),
        ("ridge, last 30 origins", ml(lambda: learned.make_ridge(100), 30)),
        ("GBM, all origins since 1940", ml(learned.make_gbm)),
        ("GBM, last 30 origins", ml(learned.make_gbm, 30)),
        ("temp: trend fitted since 1975", temp(trend_start=1975)),
        ("temp: trend x1.5 (amplified)", temp(trend_start=1975, damp=1.5)),
        ("temp: Tmax and Tmin forecast separately", tmax_tmin),
        ("CDD from temperature forecast", cdd_temp),
        ("CDD: from temperature + GBM (previous final)", lambda *a: cl.blend(cdd_temp(*a), gbm_cdd(*a))),
        ("CDD: delta method, 15 years", lambda p, mt, o, y: cl.f_cdd_delta(p, mt, o, y, temp_params=T, k=15)),
        ("CDD: delta method, 30 years", lambda p, mt, o, y: cl.f_cdd_delta(p, mt, o, y, temp_params=T, k=30)),
        ("bias correction from own past errors (temp, CDD; 15 yrs, x0.5)",
         cl.bias_corrected(final, ["annual_avg_temp", "cdd"], lookback=15, shrink=0.5)),
        ("FINAL (forecast.py)", final),
    ]
    rows = []
    for name, f in models:
        rows += summarise(name, f, panel, meta)
    out = pd.DataFrame(rows)
    out.round(4).to_csv(cl.ROOT / "code" / "experiments.csv", index=False)
    return out


def grid(panel, meta):
    rows = []
    for adj in (False, True):
        for lk in (10, 15, 30):
            for tk in (30, 50):
                for r in (500, 1000, 5000):
                    for d in (0.5, 1.0):
                        name = f"adj={adj:d} L={lk} T={tk} R={r} d={d}"
                        f = (lambda p, mt, o, y, kw=dict(level_k=lk, trend_k=tk, radius_km=r, damp=d,
                                                         adjust_stations=adj): cl.f_level_trend(p, mt, o, y, **kw))
                        rows += summarise(name, f, panel, meta)
    pd.DataFrame(rows).round(4).to_csv(cl.ROOT / "code" / "grid_results.csv", index=False)


def bias(panel, meta):
    """Mean signed error (forecast - observed) of the final model per origin and metric."""
    from forecast import model

    truth = panel[cl.METRICS].stack().rename("truth").reset_index()
    truth.columns = ["city_id", "year", "metric", "truth"]
    rows = []
    for o in range(1985, 2012):
        p = model(panel, meta, o, list(range(o + 1, o + 8))).merge(truth, on=["city_id", "year", "metric"]).dropna()
        rows.append({"origin": o, **(p.predicted_value - p.truth).groupby(p.metric).mean().to_dict()})
    out = pd.DataFrame(rows)
    out.round(3).to_csv(cl.ROOT / "code" / "bias_by_origin.csv", index=False)
    print(out.round(3).to_string(index=False))
    print(f"temperature too cold at {(out.annual_avg_temp < 0).sum()} of {len(out)} origins; "
          f"mean {out.annual_avg_temp.mean():+.2f} °F")


if __name__ == "__main__":
    panel, meta = cl.load_panel()
    {"compare": compare, "grid": grid, "bias": bias}[sys.argv[1] if len(sys.argv) > 1 else "compare"](panel, meta)
