import numpy as np
import json
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

# Load and preprocess data to return descriptive and target features
def loadData(name):
    with open(name, 'r') as file:
        data = json.load(file)

    records = []
    for c in data:
        record = c['input']  
        record['reimbursement_amount'] = c['expected_output']
        records.append(record)

    df = pd.DataFrame(records)

    # Add derived features
    EPS = 1e-6  #Small constant to avoid division-by-zero
    df["cost_per_mile"] = df["total_receipts_amount"] / (df["miles_traveled"] + EPS)
    df["cost_per_day"]  = df["total_receipts_amount"] / (df["trip_duration_days"] + EPS)
    df["miles_per_day"] = df["miles_traveled"] / (df["trip_duration_days"] + EPS)
    df["short_trip_flag"] = (df["miles_traveled"] < 100).astype(int)
    df["long_trip_flag"] = (df["trip_duration_days"] >= 5).astype(int)

    daily_cost = df["total_receipts_amount"] / (df["trip_duration_days"] + EPS)
    df["high_daily_cost_flag"] = (daily_cost > 150).astype(int)

    # Define input features and target feature
    inputFeatures = df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 
                        'cost_per_mile','cost_per_day', 'miles_per_day', 'short_trip_flag', 
                        'long_trip_flag', 'high_daily_cost_flag']]
    
    targetFeature = df['reimbursement_amount']

    return inputFeatures, targetFeature

# Train and evaluate Gradient Boosting Regressor
def gradientBoostingRegression(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = GradientBoostingRegressor(
        random_state=42
    )

    # Perform grid search to find the best hyperparameters
    param_grid = {
        'n_estimators': [100, 200, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.85, 1.0]
    }
    gs_cv = GridSearchCV(estimator=model, param_grid = param_grid, verbose = 0, cv = 5, n_jobs=-1, scoring='neg_mean_absolute_error')
    gs_cv.fit(X_train, y_train)
    print(f"Best Parameter Values: {gs_cv.best_params_}")

    # Evaluate the model and display metrics
    y_pred = gs_cv.predict(X_test)
    y_pred = np.round(y_pred, 2)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    threshold = 0.05
    within_threshold = np.abs((y_test - y_pred) / y_test) <= threshold
    accuracy_within_threshold = np.mean(within_threshold) * 100

    print(f"\nModel Evaluation (Gradient Boosting):")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Accuracy within 5%: {accuracy_within_threshold:.2f}%")

    # Get feature importances and save them to a CSV file
    feature_importances = gs_cv.best_estimator_.feature_importances_
    feature_names = X.columns
    feat_imp_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
    feat_imp_df = feat_imp_df.sort_values('importance', ascending=False)
    feat_imp_df.to_csv('..\\artifacts\\gb_feat_imp.csv')

    return gs_cv


def main():
    file = '..\\public_cases.json'
    descriptive, target = loadData(file)

    gradientBoostingRegression(descriptive, target)

if __name__ == "__main__":
    main()