"""Download the curated inputs into data/ (run once; the files are also committed).

Sources: CMU-OSPO/protostar-climate-train-to-2018 (annual_indices, coverage_and_stations,
city_info, daily). Stops if any row is dated after 2018 (data rule).

From `daily` (47 MB, not copied) we keep only a compact table, data/cdd_shift_curves.parquet:
for each city and year since 1940, the CDD that year *would* have had if every day had been
s °F warmer, for s = -3, -2.75, ..., +6. This is what the "delta method" CDD forecast needs.
"""

import numpy as np
import pandas as pd

from climate_lib import CUTOFF, DATA

REPO = "hf://datasets/CMU-OSPO/protostar-climate-train-to-2018"
SHIFTS = np.round(np.arange(-3, 6.01, 0.25), 2)
FIRST_CURVE_YEAR = 1940


def cdd_shift_curves(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.dropna(subset=["tmax", "tmin"]).copy()
    d["year"] = pd.to_datetime(d["Date"]).dt.year
    if d["year"].max() > CUTOFF:
        raise SystemExit(f"Data rule: daily has rows after {CUTOFF}; stopping.")
    d = d[d["year"] >= FIRST_CURVE_YEAR]
    tmean = ((d["tmax"] + d["tmin"]) / 2).to_numpy()
    keys = [d["ID"].to_numpy(), d["year"].to_numpy()]
    cols = {f"s{s:+.2f}": pd.Series(np.maximum(0, tmean + s - 65)).groupby(keys).sum() for s in SHIFTS}
    out = pd.DataFrame(cols)
    out.index.names = ["ID", "Year"]
    return out.reset_index()


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    for subset in ["annual_indices", "coverage_and_stations", "city_info"]:
        df = pd.read_parquet(f"{REPO}/{subset}")
        if "Year" in df and df["Year"].max() > CUTOFF:
            raise SystemExit(f"Data rule: {subset} has rows after {CUTOFF}; stopping.")
        df.to_parquet(DATA / f"{subset}.parquet", index=False)
        print(f"{subset}: {len(df)} rows -> data/{subset}.parquet")

    curves = cdd_shift_curves(pd.read_parquet(f"{REPO}/daily", columns=["ID", "Date", "tmax", "tmin"]))
    curves.round(1).to_parquet(DATA / "cdd_shift_curves.parquet", index=False)
    print(f"cdd_shift_curves: {len(curves)} rows -> data/cdd_shift_curves.parquet")
