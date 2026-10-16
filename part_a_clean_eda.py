"""Part A: clean the raw export, build the modelling table, write checks and delay evidence."""
import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES

os.makedirs("outputs", exist_ok=True)
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

dedup = raw.drop_duplicates()
before = {
    "missing_distance_km_before": int(dedup["distance_km"].isna().sum()),
    "missing_parcel_weight_before": int((dedup["parcel_weight_kg"].isna()).sum()),
    "sentinel_weight_rows": int((dedup["parcel_weight_kg"] > 100).sum()),
    "missing_declared_value_before": int(dedup["declared_value"].isna().sum()),
    "missing_payment_mode_before": int(dedup["payment_mode"].isna().sum()),
    "service_type_spellings_before": int(dedup["service_type"].nunique()),
}

clean, medians = clean_shipments(raw)
feat = build_features(clean, hubs)
feat["is_delayed"] = (feat["delivery_hours"] > feat["promised_hours"]).astype(int)
feat.to_csv("data/modelling_table.csv", index=False)

q1, q3 = feat["delivery_hours"].quantile(0.25), feat["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
checks = {
    "raw_rows": len(raw),
    "duplicate_rows_removed": len(raw) - len(dedup),
    "rows_after_dedup": len(dedup),
    **before,
    "service_type_spellings_after": int(clean["service_type"].nunique()),
    "rows_after_hub_merge": len(feat),
    "median_distance_km": medians["distance_km"],
    "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"],
    "nulls_in_features_and_target": int(feat[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()),
    "delayed_count": int(feat["is_delayed"].sum()),
    "delay_rate": feat["is_delayed"].mean(),
    "hours_q1": q1, "hours_q3": q3,
    "hours_lower_bound": lo, "hours_upper_bound": hi,
    "hours_outlier_count": int(((feat["delivery_hours"] < lo) | (feat["delivery_hours"] > hi)).sum()),
    "declared_value_mean_inr": feat["declared_value_inr"].mean(),
    "declared_value_median_inr": feat["declared_value_inr"].median(),
    "log_declared_value_mean": feat["log_declared_value"].mean(),
    "log_declared_value_median": feat["log_declared_value"].median(),
}
pd.DataFrame({"metric": list(checks), "value": [round(float(v), 4) for v in checks.values()]}).to_csv(
    "outputs/part_a_checks.csv", index=False)

feat["distance_band"] = pd.cut(feat["distance_km"], bins=[0, 200, 400, 600, 2000],
                               labels=["0-200", "201-400", "401-600", "601+"], right=True)
rows = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = feat.groupby(dim, observed=True)
    for name, sub in g:
        rows.append({"dimension": dim, "group": str(name), "shipments": len(sub),
                     "delayed": int(sub["is_delayed"].sum()),
                     "delay_rate": round(sub["is_delayed"].mean(), 4),
                     "mean_hours": round(sub["delivery_hours"].mean(), 4)})
pd.DataFrame(rows).to_csv("outputs/part_a_delay_by_group.csv", index=False)

corr = feat.select_dtypes(include=[np.number]).corr().round(4)
corr.to_csv("outputs/part_a_correlations.csv")
print("Part A done:", len(feat), "rows")
