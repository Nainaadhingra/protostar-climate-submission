# Data card: `data/`

The curated inputs `code/forecast.py` reads. Together with `code/`, they reproduce `submission.csv`. They come from one source, the challenge training dataset, and no other data is used. The `daily` subset itself (47 MB) is not copied: `code/prepare_data.py` downloads it from the Hub and condenses it into `cdd_shift_curves.parquet`.

## Files

| File | What it holds | Source | License | Years covered | How we prepared it | Used for |
|---|---|---|---|---|---|---|
| `annual_indices.parquet` | One row per city and year: yearly averages, totals, extremes, `cdd`, missing-day counts (28,052 rows) | [Challenge dataset](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018), `annual_indices` | CC0-1.0 | 1851–2018 | Copied unchanged by `code/prepare_data.py` | Forecast input |
| `coverage_and_stations.parquet` | One row per city and year: days with both temperatures (`Days Both`), station segment, station changes (28,052 rows) | Same dataset, `coverage_and_stations` | CC0-1.0 | 1851–2018 | Copied unchanged | Forecast input (validity of CDD years, station-change correction) |
| `cdd_shift_curves.parquet` | One row per city and year since 1940 (16,180 rows): the CDD that year would have had if every day were s °F warmer, columns `s-3.00` … `s+6.00` in 0.25 °F steps | Same dataset, `daily` (Tmax, Tmin) | CC0-1.0 | 1940–2018 | `code/prepare_data.py`: daily mean = (Tmax + Tmin)/2 on days with both values; sum of max(0, mean + s − 65) per year. Column `s+0.00` equals the official `cdd` exactly | Forecast input (delta-method CDD) |
| `city_info.parquet` | Stations combined into each city's record and their periods (461 rows) | Same dataset, `city_info` | CC0-1.0 | 1851–2018 | Copied unchanged | Reference only (not read by the forecast) |

All the preparation happens in code, so it can be read and changed. `climate_lib.load_panel()`:

- turns ±inf in the extreme columns (years with no data) into missing values;
- keeps a metric's value for a year only if it passes the challenge's rule: at most 10 missing days of Tmax and Tmin (temperature metrics), of precipitation (total precipitation), and, for CDD, also at most 10 days without both temperatures (`Days Both`);
- stops with an error if any row is dated after 2018.

At each forecast origin, `segment_offsets()` puts years recorded at earlier stations on the level of the current one. The step for each station change is the mean of up to 10 years after the change minus up to 10 years before, after removing the national mean anomaly. It uses only years up to that origin.

## How the files fit together

- **Join key:** `ID` (= `city_id` in `submission.csv`, the city's current GHCN station) and `Year`.
- **Units:** °F, inches, °F-days, as in the source. No conversions.
- **Known limitations:** stations are mostly at airports, so they do not capture neighborhood heat. Most years are stored in whole °F. 32 station changes fall in 1989–2018, and a few cities have long gaps (Sioux City has no data for 1896–1995). Under our reading of the validity rule, 10 city-metrics have fewer than 10 valid years in 1989–2018: Belvidere and Huntsville (most metrics), Escondido (CDD), and Grand Island and Elko (precipitation). The challenge counts 9, so our CDD rule probably differs on one borderline case. These rows are still forecast: the model falls back to the valid years it has.

## Data rule check

- [x] Every file marked "Forecast input" is weather data up to 2018-12-31, or data that does not respond to the weather.
- [x] No file holds 2019–2025 data. `outputs/cdd_change_by_city.csv` is built from our own forecasts, not from observations.

## Citation

Lai, Y., & Dzombak, D. (2019). Compiled historical daily temperature and precipitation data for selected 210 U.S. cities. Carnegie Mellon University. https://doi.org/10.1184/R1/7890488.v6. Lai, Y., & Dzombak, D. (2019). Historical changes of annual temperature and precipitation indices at selected 210 U.S. cities. https://doi.org/10.1184/R1/7961012.v6
