"""Part B: delivery-hours regression."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from src.shipment_prep import FEATURES

df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df["delivery_hours"]
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)


def metrics(name, split, yt, yp):
    mse = mean_squared_error(yt, yp)
    return {"model": name, "split": split, "MAE": mean_absolute_error(yt, yp), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(yt, yp)}


def rnd(frame):
    return frame.round(4)


# scaling without leakage
scaler = StandardScaler().fit(X_tr)
tr_s, te_s = scaler.transform(X_tr), scaler.transform(X_te)
rnd(pd.DataFrame({"metric": ["train_rows", "val_rows", "test_rows", "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
                  "value": [len(X_tr), len(X_val), len(X_te), np.abs(tr_s.mean(axis=0)).max(), np.abs(te_s.mean(axis=0)).max()]})
    ).to_csv("outputs/part_b_checks.csv", index=False)

# baseline
rows = [metrics("mean_baseline", "val", y_val, np.full(len(y_val), y_tr.mean()))]

# tree depth diagnostic
depth_rows = []
for d in [1, 2, 3, 4, 6, 8, 12, None]:
    t = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_tr, y_tr)
    depth_rows.append({"max_depth": "unlimited" if d is None else d,
                       "train_rmse": np.sqrt(mean_squared_error(y_tr, t.predict(X_tr))),
                       "val_rmse": np.sqrt(mean_squared_error(y_val, t.predict(X_val)))})
depth = pd.DataFrame(depth_rows)
best_i = int(depth["val_rmse"].round(12).values.argmin())  # first minimum -> smaller depth on ties
best_depth = [1, 2, 3, 4, 6, 8, 12, None][best_i]
rnd(depth).to_csv("outputs/part_b_tree_depth.csv", index=False)

# candidates
models = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    rows.append(metrics(name, "val", y_val, m.predict(X_val)))

# learning rate experiment
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    g = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_tr, y_tr)
    p = g.predict(X_val)
    lr_rows.append({"learning_rate": lr, "val_rmse": np.sqrt(mean_squared_error(y_val, p)), "val_r2": r2_score(y_val, p)})
rnd(pd.DataFrame(lr_rows)).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# rank, cross-validate the top two, final test
val_df = pd.DataFrame(rows)
ranked = val_df[val_df["model"] != "mean_baseline"].sort_values("RMSE", kind="stable")["model"].tolist()
top2 = ranked[:2]
kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_all, y_all = pd.concat([X_tr, X_val]), pd.concat([y_tr, y_val])
cv_rows = []
for name in top2:
    s = cross_val_score(models[name], X_all, y_all, cv=kf)
    cv_rows.append({"model": name, "mean_r2": s.mean(), "std_r2": s.std()})
rnd(pd.DataFrame(cv_rows)).to_csv("outputs/part_b_cv.csv", index=False)

final = ranked[0]
rows.append(metrics(final, "test", y_te, models[final].predict(X_te)))
rnd(pd.DataFrame(rows)).to_csv("outputs/part_b_metrics.csv", index=False)
print("best depth", best_depth, "ranked", ranked, "final", final)
