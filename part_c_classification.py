"""Part C: delay classifier, feature selection, tuning and alert threshold."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from src.shipment_prep import FEATURES, NOISE_COLUMNS

df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df["is_delayed"]
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print("sizes", len(X_tr), len(X_val), len(X_te), "delayed", y_tr.sum(), y_val.sum(), y_te.sum())


def cm_row(yt, yp):
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "accuracy": acc,
            "precision": prec, "recall": rec, "f1": f1}


# 2. blind baseline + classifiers
rows = [{"model": "blind_all_on_time", **cm_row(y_val, np.zeros(len(y_val), dtype=int))}]
lr = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]).fit(X_tr, y_tr)
rows.append({"model": "logistic_regression", **cm_row(y_val, lr.predict(X_val))})
best_d, best_f1 = None, -1
for d in [2, 3, 4, 5, 6, 8, 10]:
    t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_tr, y_tr)
    f = cm_row(y_val, t.predict(X_val))["f1"]
    if f > best_f1:  # strict > keeps the smaller d on ties
        best_d, best_f1 = d, f
t = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42).fit(X_tr, y_tr)
rows.append({"model": f"decision_tree_depth_{best_d}", **cm_row(y_val, t.predict(X_val))})
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
rows.append({"model": "random_forest_default", **cm_row(y_val, rf.predict(X_val))})
pd.DataFrame(rows).round(4).to_csv("outputs/part_c_models.csv", index=False)

# 3. feature scores
sel_all = SelectKBest(f_classif, k="all").fit(X_tr, y_tr)
fs = pd.DataFrame({"feature": FEATURES, "f_score": sel_all.scores_, "rf_importance": rf.feature_importances_})
fs["f_rank"] = fs["f_score"].rank(ascending=False, method="first").astype(int)
fs["is_noise"] = fs["feature"].isin(NOISE_COLUMNS)
fs = fs[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].sort_values("f_rank")
fs.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)

k_rows = []
for k in [5, 7, 10, 12, 17]:
    p = Pipeline([("sel", SelectKBest(f_classif, k=k)), ("rf", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_tr, y_tr)
    k_rows.append({"k": k, "val_f1": cm_row(y_val, p.predict(X_val))["f1"]})
pd.DataFrame(k_rows).round(4).to_csv("outputs/part_c_selectk.csv", index=False)

# 4. tuning with k = 7 (selector fitted on training rows only)
sel7 = SelectKBest(f_classif, k=7).fit(X_tr, y_tr)
Xtr7, Xval7 = sel7.transform(X_tr), sel7.transform(X_val)
grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5).fit(Xtr7, y_tr)
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, n_iter=9, cv=5, random_state=42).fit(Xtr7, y_tr)
default7 = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr7, y_tr)
tune = [
    {"search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 135, "best_params": str(gs.best_params_),
     "best_cv_accuracy": gs.best_score_, "val_f1": cm_row(y_val, gs.best_estimator_.predict(Xval7))["f1"]},
    {"search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 45, "best_params": str(rs.best_params_),
     "best_cv_accuracy": rs.best_score_, "val_f1": cm_row(y_val, rs.best_estimator_.predict(Xval7))["f1"]},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
     "best_params": "{'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 1}",
     "best_cv_accuracy": np.nan, "val_f1": cm_row(y_val, default7.predict(Xval7))["f1"]},
]
tune_df = pd.DataFrame(tune)
tune_df.round(4).to_csv("outputs/part_c_tuning.csv", index=False)

# 5. choose the final configuration (ties -> default, grid, randomized)
order = ["default_forest_same_7_features", "grid_search", "randomized_search"]
vals = tune_df.set_index("search")["val_f1"].round(4)
top = vals.max()
chosen = next(s for s in order if vals[s] == top)
params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
if chosen == "grid_search":
    params = dict(gs.best_params_)
elif chosen == "randomized_search":
    params = dict(rs.best_params_)
final = Pipeline([("selector", SelectKBest(f_classif, k=7)),
                  ("model", RandomForestClassifier(random_state=42, **params))]).fit(X_tr, y_tr)

# 6. threshold
proba_val = final.predict_proba(X_val)[:, 1]
thr_rows = []
for t_ in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    thr_rows.append({"threshold": t_, **cm_row(y_val, (proba_val >= t_).astype(int))})
thr = pd.DataFrame(thr_rows)
thr.round(4).to_csv("outputs/part_c_thresholds.csv", index=False)
ok = thr[thr["recall"] >= 0.80]
threshold = float(ok["threshold"].max()) if len(ok) else 0.1

# 7. one test evaluation
proba_te = final.predict_proba(X_te)[:, 1]
test_row = {"threshold": threshold, **cm_row(y_te, (proba_te >= threshold).astype(int))}
pd.DataFrame([test_row]).round(4).to_csv("outputs/part_c_final_test.csv", index=False)
pd.DataFrame([{"selector_k": 7, "n_estimators": params["n_estimators"],
               "max_depth": "unlimited" if params["max_depth"] is None else params["max_depth"],
               "min_samples_leaf": params["min_samples_leaf"], "threshold": threshold,
               "chosen_from": chosen}]).to_csv("outputs/part_c_final_config.csv", index=False)
print("chosen", chosen, params, "threshold", threshold)
