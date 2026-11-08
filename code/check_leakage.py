"""Look-ahead test: the final model must not use any year after its forecast origin.

For past origins (e.g. 2004), every value dated after the origin, in all inputs, is replaced
with random noise. The forecast from that origin must be exactly the same as with the real
data. If any number changes, the model peeked at the future and the test fails.

    python code/check_leakage.py

`leak_test(forecaster)` works for any forecaster with the climate_lib signature.
"""

import sys

import numpy as np
import pandas as pd

import climate_lib as cl

ORIGINS = [1998, 2004, 2011]


def scramble_after(df: pd.DataFrame, origin: int, cols, rng) -> pd.DataFrame:
    df = df.copy()
    future = df.index.get_level_values("year") > origin
    for c in cols:
        df.loc[future, c] = rng.normal(df[c].mean(), 3 * df[c].std(), future.sum())
    return df


def leak_test(forecaster, origins=ORIGINS, verbose=True) -> bool:
    """True if `forecaster` gives identical forecasts with all post-origin data scrambled."""
    rng = np.random.default_rng(0)
    panel, meta = cl.load_panel()
    curves = cl.load_cdd_curves()
    clean = True
    for o in origins:
        years = list(range(o + 1, o + 8))
        real = forecaster(panel, meta, o, years)
        cl._CURVES[cl.DATA] = scramble_after(curves, o, curves.columns, rng)
        try:
            fake = forecaster(scramble_after(panel, o, cl.METRICS + ["tmax", "tmin"], rng), meta, o, years)
        finally:
            cl._CURVES[cl.DATA] = curves
        m = real.merge(fake, on=["city_id", "year", "metric"], suffixes=("_real", "_noise"))
        diff = (m.predicted_value_real - m.predicted_value_noise).abs().max()
        ok = len(m) == len(real) and np.isclose(diff, 0, atol=1e-9)
        clean &= ok
        if verbose:
            print(f"origin {o}: {len(m)} forecasts, max change with future scrambled = {diff:.2e} "
                  f"-> {'OK' if ok else 'LEAK'}")
    return clean


if __name__ == "__main__":
    from forecast import model

    if not leak_test(model):
        sys.exit("Look-ahead detected.")
    # the test itself must be able to fail: a model that peeks 3 years ahead
    assert not leak_test(lambda p, m, o, y: cl.f_mean(p, m, o + 3, y, k=15), [2004], verbose=False)
    print("OK: no look-ahead (and a deliberately leaky model is caught).")
