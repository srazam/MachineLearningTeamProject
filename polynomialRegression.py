import numpy as np
import json
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
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

    inputFeatures = df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']]
    targetFeature = df['reimbursement_amount']

    return inputFeatures, targetFeature


def polynomialRegression(X, y, degree = 2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ('poly_features', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred, 2)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Evaluation:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")

    return model


def main():
    file = 'public_cases.json'
    descriptive, target = loadData(file)

    model = polynomialRegression(descriptive, target, degree=2)

    sample_input = pd.DataFrame([{
        'trip_duration_days': 5,
        'miles_traveled': 900,
        'total_receipts_amount': 480.0
    }])

    prediction = model.predict(sample_input)
    print(f"\nPredicted reimbursement: ${prediction[0]:.2f}")

if __name__ == "__main__":
    main()