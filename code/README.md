# Code

Run everything from the repository root. Python 3.14; `pip install -r code/requirements.txt`.

| File | What it does | Runtime |
|---|---|---|
| `prepare_data.py` | Downloads the challenge tables into `data/` as Parquet and condenses `daily` into `cdd_shift_curves.parquet`; stops if any row is after 2018. Optional: `data/` is already committed. | ~1 min |
| `climate_lib.py` | Reusable library: loads the data with the challenge's validity rules (≤ 10 missing days), corrects station changes, the forecasters (window mean, linear trend, level + regionally pooled trend, CDD-from-temperature, delta-method CDD, blending), and the challenge score and rolling-origin backtest. | – |
| `learned.py` | Learned forecaster (tested, not in the final model): features from each city's and its region's recent anomalies and trends, one model per metric trained across cities on past forecast origins (ridge or gradient boosting). | – |
| `forecast.py` | **The final model.** Backtests it (writes `backtest_results.csv`) and writes `submission.csv`. | ~1 min |
| `check_leakage.py` | Look-ahead test: reruns the model from past origins with all later data scrambled and requires identical forecasts. `leak_test(f)` works for any forecaster. | ~1 min |
| `backtest.py` | `compare`: every model we tried, on the same backtest → `experiments.csv`. `grid`: settings grid of the level + trend model → `grid_results.csv`. `bias`: the final model's mean signed error per origin → `bias_by_origin.csv`. | ~15 min / ~5 min / ~1 min |
| `export.py` | Forecasts as `outputs/forecasts.geojson` (GIS, web maps) and `outputs/forecasts_wide.csv`. | seconds |
| `policy_analysis.py` | Forecast change in cooling demand per city → `outputs/cdd_change_by_city.csv` (numbers in the Policy Idea). | seconds |
| `check_submission.py` | From the template: the same checks the challenge Space runs. | seconds |
| `AGENTS.md` | From the template: instructions for AI agents (data rule), plus our team notes. | – |

## Reusing it for another forecast

Every forecaster has the signature `f(panel, meta, origin, years) -> DataFrame(city_id, year, metric, predicted_value)` and reads only years ≤ `origin`. So:

```python
import climate_lib as cl
from forecast import model

panel, meta = cl.load_panel()                      # or any panel with the same columns
pred = model(panel, meta, origin=2018, years=range(2019, 2026))
print(cl.backtest(model, panel, meta, origins=range(2000, 2012)))   # score any forecaster

from check_leakage import leak_test
assert leak_test(model)                                            # certify: no look-ahead
```

To add a metric, add its column to `SOURCE_COLUMN` and its validity rule in `load_panel`. To plug in a new model, write a function with the signature above and add it to `compare()` in `backtest.py`. With the full KiltHub record (to 2023), `origin=2023` gives forecasts beyond 2025 with no code change.
