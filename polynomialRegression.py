import numpy as np
import json
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

def loadData(name):
    with open(name, 'r') as file:
        data = json.load(file)
    
    records = []
    for c in data:
        record = c['input']  
        record['reimbursement_amount'] = c['expected_output']
        records.append(record)

    df = pd.DataFrame(records)

    # Adding derived features
    EPS = 1e-6  #small constant to avoid division-by-zero
    df["cost_per_mile"] = df["total_receipts_amount"] / (df["miles_traveled"] + EPS)
    df["cost_per_day"]  = df["total_receipts_amount"] / (df["trip_duration_days"] + EPS)
    df["miles_per_day"] = df["miles_traveled"] / (df["trip_duration_days"] + EPS)
    df["short_trip_flag"] = (df["miles_traveled"] < 100).astype(int)
    df["long_trip_flag"] = (df["trip_duration_days"] >= 5).astype(int)

    daily_cost = df["total_receipts_amount"] / (df["trip_duration_days"] + EPS)
    df["high_daily_cost_flag"] = (daily_cost > 150).astype(int)

    inputFeatures = df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 
                        'cost_per_mile','cost_per_day', 'miles_per_day', 'short_trip_flag', 
                        'long_trip_flag', 'high_daily_cost_flag']]
    targetFeature = df['reimbursement_amount']

    return inputFeatures, targetFeature


def polynomialRegression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = Pipeline([
        ('poly_features', PolynomialFeatures(include_bias=False)),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    param_grid = {
        'poly_features__degree': [2, 3, 4]
    }

    gs_cv = GridSearchCV(estimator=model, param_grid = param_grid, verbose = 0, cv = 5, n_jobs=-1, scoring='neg_mean_absolute_error')
    gs_cv.fit(X_train, y_train)

    print(f"Best Parameter Values: {gs_cv.best_params_['poly_features__degree']}")

    y_pred = gs_cv.predict(X_test)
    y_pred = np.round(y_pred, 2)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    threshold = 0.05
    within_threshold = np.abs((y_test - y_pred) / y_test) <= threshold
    accuracy_within_threshold = np.mean(within_threshold) * 100

    print(f"\nModel Evaluation:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Accuracy within 5%: {accuracy_within_threshold:.2f}%")

    return gs_cv


def main():
    file = 'public_cases.json'
    descriptive, target = loadData(file)

    polynomialRegression(descriptive, target)

if __name__ == "__main__":
    main()