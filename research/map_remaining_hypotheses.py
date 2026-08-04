import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASET = Path("data/market_context_expected_r_enriched.csv")
FEATURES = [
    "time",
    "strength_usdjpy_pressure_1h", "strength_usdjpy_pressure_4h",
    "strength_usdjpy_pressure_1d", "strength_usdjpy_pressure_1w",
    "strength_usdjpy_pressure_1m", "strength_all_agreement_1h",
    "strength_all_agreement_4h", "strength_all_agreement_1d",
    "strength_usd_dispersion_1h", "strength_jpy_dispersion_1h",
]


def sign(series):
    return np.sign(series).astype("int8")


def main():
    rows = pd.read_csv(DATASET, usecols=FEATURES, parse_dates=["time"], low_memory=False)
    rows["time"] = pd.to_datetime(rows["time"], utc=True, errors="coerce")
    h = rows.dropna(subset=FEATURES).loc[lambda x: x["time"].dt.minute == 0].copy()
    dispersion = h["strength_usd_dispersion_1h"] + h["strength_jpy_dispersion_1h"]
    q25 = dispersion.shift(1).rolling(120).quantile(.25)
    q50 = dispersion.shift(1).rolling(120).median()
    q75 = dispersion.shift(1).rolling(120).quantile(.75)
    p1, p4, pd1 = (sign(h[x]) for x in (
        "strength_usdjpy_pressure_1h", "strength_usdjpy_pressure_4h", "strength_usdjpy_pressure_1d"
    ))
    pw, pm = sign(h["strength_usdjpy_pressure_1w"]), sign(h["strength_usdjpy_pressure_1m"])
    a1 = h["strength_all_agreement_1h"]
    strong = (a1 >= .75) | (a1 <= .25)
    neutral = a1.between(.40, .60)
    mismatch_1d = (p1 != 0) & (pd1 != 0) & (p1 != pd1)
    aligned_1d = (p1 != 0) & (p1 == pd1)
    short_long_conflict = (p1 == p4) & (pw == pm) & (p1 != 0) & (pw != 0) & (p1 != pw)

    masks = {
        "cross_asset_dispersion_expansion": (dispersion > q75) & (dispersion.shift(1) <= q75.shift(1)),
        "cross_asset_dispersion_convergence": (dispersion < q50) & (dispersion.shift(1) >= q75.shift(1)),
        "multi_horizon_pressure_agreement": (p1 == p4) & (p4 == pd1) & (p1 != 0) & ~((p1.shift(1) == p4.shift(1)) & (p4.shift(1) == pd1.shift(1))),
        "multi_horizon_pressure_disagreement": mismatch_1d & ~mismatch_1d.shift(1, fill_value=False),
        "agreement_to_disagreement": neutral & strong.shift(1, fill_value=False),
        "disagreement_to_agreement": strong & neutral.shift(1, fill_value=False),
        "weekly_monthly_hierarchy_conflict": short_long_conflict & ~short_long_conflict.shift(1, fill_value=False),
        "low_dispersion_direction_emergence": (dispersion < q25) & (dispersion.shift(1) >= q25.shift(1)) & strong,
        "h1_d1_disagreement_resolution": aligned_1d & mismatch_1d.shift(1, fill_value=False),
    }
    result = {
        "hourly_rows": len(h),
        "period_start": str(h["time"].min()),
        "period_end": str(h["time"].max()),
        "candidate_counts": {name: int(mask.fillna(False).sum()) for name, mask in masks.items()},
        "outcome_columns_read": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
