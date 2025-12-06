import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.tree import DecisionTreeRegressor, export_text
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ModuleNotFoundError:
    raise SystemExit(
        "ERROR: scikit-learn is not installed.\n"
        "Install it with:\n"
        "  pip install scikit-learn\n"
    )

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

EPS = 1e-6
DATA_PATH = Path("public_cases.json")
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(exist_ok=True)

def load_data(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            df = pd.read_json(path)
        except ValueError:
            df = pd.read_json(path, lines=True)
    else:
        rng = np.random.default_rng(RANDOM_STATE)
        df = pd.DataFrame({
            "trip_duration_days": rng.integers(1, 10, size=1000),
            "miles_traveled": rng.integers(10, 2000, size=1000),
            "total_receipts_amount": np.round(rng.uniform(50, 4000, size=1000), 2),
            "reimbursement": np.round(rng.uniform(60, 4200, size=1000), 2),
        })
        print("public_cases.json not found. Using synthetic data (1000 rows).")
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
   
    df = df.copy()

    if set(df.columns) == {"input", "expected_output"}:
        in_df = pd.json_normalize(df["input"])
        out_col = df["expected_output"]
        if len(out_col) > 0 and isinstance(out_col.iloc[0], (dict, list)):
            out_df = pd.json_normalize(out_col)
        else:
            out_df = pd.DataFrame({"reimbursement": out_col})
        df = pd.concat([in_df, out_df], axis=1)

    df.columns = [str(c).strip().lower() for c in df.columns]
    rename_map = {
        "trip_duration_days": "days",
        "miles_traveled": "distance",
        "total_receipts_amount": "receipt_total",
        "days": "days",
        "distance": "distance",
        "receipt_total": "receipt_total",
        "reimbursement": "reimbursement",
    }
    df = df.rename(columns=rename_map)

    required = ["days", "distance", "receipt_total", "reimbursement"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("Available columns now:", list(df.columns))
        raise KeyError(f"Missing required columns after flatten/rename: {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cost_per_mile"] = df["receipt_total"] / (df["distance"] + EPS)
    df["cost_per_day"]  = df["receipt_total"] / (df["days"] + EPS)
    df["miles_per_day"] = df["distance"] / (df["days"] + EPS)
    daily_cost = df["receipt_total"] / (df["days"] + EPS)
    df["short_trip_flag"]      = (df["distance"] < 100).astype(int)
    df["long_trip_flag"]       = (df["days"] >= 5).astype(int)
    df["high_daily_cost_flag"] = (daily_cost > 150).astype(int)
    return df

def split_train_test(df: pd.DataFrame):
    test_size = 250 / len(df) if len(df) >= 1000 else 0.25
    X = df.drop(columns=["reimbursement"])
    y = df["reimbursement"]
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)

def train_tree(X_train, y_train) -> GridSearchCV:
    tree = DecisionTreeRegressor(random_state=RANDOM_STATE)
    param_grid = {
        "max_depth": [3, 4, 5, 6, 8, None],
        "min_samples_leaf": [1, 3, 5, 10, 20],
        "min_samples_split": [2, 4, 10],
    }
    gscv = GridSearchCV(
        estimator=tree,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=0,
    )
    gscv.fit(X_train, y_train)
    return gscv

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    from sklearn.metrics import mean_squared_error
    import numpy as np
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    threshold = 0.05
    within_threshold = np.abs((y_test - pred) / y_test) <= threshold
    accuracy_within_threshold = np.mean(within_threshold) * 100
    return mae, rmse, r2, accuracy_within_threshold


def print_and_save_rules(best_tree: DecisionTreeRegressor, feature_names, out_path: Path):
    rules = export_text(best_tree, feature_names=list(feature_names), decimals=3)
    print("\n===== IF-THEN RULES (Decision Tree) =====\n")
    print(rules)
    out_path.write_text(rules, encoding="utf-8")
    print(f"\nRules saved to: {out_path.resolve()}")

def main():
    df_raw = load_data(DATA_PATH)
    df = basic_clean(df_raw)
    df = make_features(df)

    feature_cols = [
        "days","distance","receipt_total",
        "cost_per_mile","cost_per_day","miles_per_day",
        "short_trip_flag","long_trip_flag","high_daily_cost_flag",
    ]
    missing = [c for c in feature_cols + ["reimbursement"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {df.columns.tolist()}")

    df_model = df[feature_cols + ["reimbursement"]].copy()

    X_train, X_test, y_train, y_test = split_train_test(df_model)

    gscv = train_tree(X_train[feature_cols], y_train)
    best_tree = gscv.best_estimator_

    mae, rmse, r2, acc_in_thresh = evaluate(best_tree, X_test[feature_cols], y_test)
    print("===== EVALUATION (Decision Tree Rules) =====")
    print(f"Best Params: {gscv.best_params_}")
    print(f"MAE : {mae:,.3f}")
    print(f"RMSE: {rmse:,.3f}")
    print(f"R2  : {r2:,.3f}")
    print(f"Accuracy within 5%  : {acc_in_thresh:,.3f}%")


    importances = getattr(best_tree, "feature_importances_", None)
    if importances is not None:
        fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
        print("\nTop Feature Importances:")
        print(fi.head(10))
        fi.to_csv(OUT_DIR / "feature_importances.csv", header=["importance"])

    print_and_save_rules(best_tree, feature_cols, OUT_DIR / "tree_rules.txt")

if __name__ == "__main__":
    main()
