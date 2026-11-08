"""Learned forecaster: a model trained across cities and past forecast origins.

For a forecast origin o and horizon h (1..7), the target is how far year o+h lands from the
city's 30-year mean, in units of the city's 30-year standard deviation:

    z = (value[o+h] - mean30[o]) / sd30[o]

Features, all computed from years <= o:
    own_k   recent anomaly of the city: mean of last k years minus mean30, / sd30 (k = 1,3,5,10,15,20)
    reg_k   the same, averaged over cities within RADIUS_KM (regional signal, less noisy)
    nat_k   the same, averaged over all 210 cities
    slope*  pooled regional and own trends over 30 and 50 years, times the lead time
            from the 30-year window centre (h + 14.5), / sd30
    h       horizon

Training rows for a test origin O come only from origins o with o + 7 <= O, so every target
the model learns from is itself in the past at O. The prediction is mean30 + sd30 * z_hat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import climate_lib as cl

RADIUS_KM = 1000
WINDOWS = (1, 3, 5, 10, 15, 20)
HORIZON = 7
FIRST_TRAIN_ORIGIN = 1940


def neighbour_weights(meta: pd.DataFrame, cities, radius_km=RADIUS_KM) -> np.ndarray:
    lat = np.radians(meta.loc[cities, "Lat"].values)
    lon = np.radians(meta.loc[cities, "Lon"].values)
    d = 6371 * np.arccos(np.clip(
        np.sin(lat)[:, None] * np.sin(lat)[None, :]
        + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.cos(lon[:, None] - lon[None, :]), -1, 1))
    return (d <= radius_km).astype(float)


def _nanmean_w(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Weighted mean over cities (rows of X) ignoring NaN; W is cities x cities."""
    ok = ~np.isnan(X)
    return (W @ np.where(ok, X, 0)) / np.maximum(W @ ok, 1e-9)


def _slope(X: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS slope per row of X (cities x years) ignoring NaN."""
    ok = ~np.isnan(X)
    n = ok.sum(1)
    xm = (ok * x).sum(1) / np.maximum(n, 1)
    ym = np.nansum(X, 1) / np.maximum(n, 1)
    dx = np.where(ok, x[None, :] - xm[:, None], 0)
    s = (dx * np.where(ok, X - ym[:, None], 0)).sum(1) / np.maximum((dx ** 2).sum(1), 1e-9)
    return np.where(n >= 10, s, 0.0)


class FeatureBuilder:
    """Builds features for every (city, origin, horizon) of one metric. Station offsets are
    recomputed at each origin with data up to that origin only."""

    def __init__(self, panel, meta, metric, adjust_stations=True):
        self.panel, self.meta, self.metric, self.adjust = panel, meta, metric, adjust_stations
        self.full = cl.wide(panel, metric, cl.CUTOFF)
        self.cities = self.full.index
        self.W = neighbour_weights(meta, self.cities)
        self.Wall = np.ones_like(self.W)
        self._cache = {}

    def origin_block(self, o: int) -> pd.DataFrame:
        if o in self._cache:
            return self._cache[o]
        w = self.full.loc[:, [y for y in self.full.columns if y <= o]]
        if self.adjust:
            w = w + cl.segment_offsets(self.panel, self.metric, o)
        last = w.loc[:, [y for y in w.columns if y > o - 50]]
        years = np.array(last.columns, float)
        X = last.values
        w30 = X[:, years > o - 30]
        mean30 = np.nanmean(w30, 1)
        sd30 = np.nanstd(w30, 1)
        n30 = (~np.isnan(w30)).sum(1)
        sd = np.where(sd30 > 1e-6, sd30, np.nan)
        A = (X - mean30[:, None]) / sd[:, None]  # standardised anomalies
        A_reg, A_nat = _nanmean_w(self.W, A), _nanmean_w(self.Wall, A)
        f = {"mean30": mean30, "sd30": sd30, "n30": n30}
        for k in WINDOWS:
            sel = years > o - k
            f[f"own_{k}"] = np.nanmean(A[:, sel], 1)
            f[f"reg_{k}"] = np.nanmean(A_reg[:, sel], 1)
            f[f"nat_{k}"] = np.nanmean(A_nat[:, sel], 1)
        for k in (30, 50):
            sel = years > o - k
            f[f"own_slope{k}"] = _slope(A[:, sel], years[sel])
            f[f"reg_slope{k}"] = _slope(A_reg[:, sel], years[sel])
        block = pd.DataFrame(f, index=self.cities)
        for c in [c for c in block.columns if c.startswith(("own_", "reg_", "nat_"))]:
            block[c] = block[c].fillna(0.0)
        self._cache[o] = block
        return block

    def rows(self, origins, with_target=True) -> pd.DataFrame:
        out = []
        for o in origins:
            b = self.origin_block(o)
            for h in range(1, HORIZON + 1):
                r = b.copy()
                r["h"] = h
                lead = h + 14.5
                for c in [c for c in b.columns if "slope" in c]:
                    r[c + "_x_lead"] = b[c] * lead
                r["origin"], r["year"] = o, o + h
                if with_target:
                    if o + h > cl.CUTOFF:
                        continue
                    r["truth"] = self.full[o + h]
                    r["z"] = (r["truth"] - r["mean30"]) / r["sd30"]
                out.append(r.reset_index().rename(columns={"index": "city_id", "ID": "city_id"}))
        return pd.concat(out, ignore_index=True)


FEATURES = ([f"{s}_{k}" for s in ("own", "reg", "nat") for k in WINDOWS]
            + [f"{s}_slope{k}_x_lead" for s in ("own", "reg") for k in (30, 50)])


def make_ridge(alpha=10.0):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=alpha)


def make_gbm():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.03,
                                         max_leaf_nodes=15, min_samples_leaf=200, random_state=0)


def f_learned(panel, meta, origin, years, *, make_model=make_ridge, metrics=cl.METRICS,
              builders=None, clip_z=4.0, return_models=False, train_span=None, features=None):
    """Fit one model per metric on origins FIRST_TRAIN_ORIGIN..origin-7 (or only the last
    `train_span` of them), predict origin+h."""
    features = features or FEATURES
    preds, models = [], {}
    for m in metrics:
        fb = builders[m] if builders else FeatureBuilder(panel, meta, m)
        start = FIRST_TRAIN_ORIGIN if train_span is None else origin - HORIZON + 1 - train_span
        train = fb.rows(range(start, origin - HORIZON + 1))
        train = train[(train.n30 >= 10) & train.z.notna() & np.isfinite(train.z)]
        train = train[train.z.abs() <= clip_z]
        model = make_model().fit(train[features], train.z)
        models[m] = model
        test = fb.rows([origin], with_target=False)
        test = test[test.year.isin(years)]
        z = model.predict(test[features])
        val = test.mean30 + test.sd30.fillna(0) * z
        if m in ("cdd", "total_precip_in"):
            val = val.clip(lower=0)
        preds.append(pd.DataFrame({"city_id": test.city_id, "year": test.year, "metric": m,
                                   "predicted_value": val.values}))
    out = pd.concat(preds, ignore_index=True)
    return (out, models) if return_models else out
