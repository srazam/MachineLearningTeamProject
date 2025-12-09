import joblib
import pandas as pd
import time

# Check for input errors
def checkErrors(days, miles, receipts):
    # Ensure that inputs are of correct type
    try:
        days = int(days)
        miles = int(miles)
        receipts = float(receipts)
    except ValueError:
        return "Error: Days and miles must be a whole number while receipts must be a float number. Try again."
    
    # Ensure that inputs are positive
    if int(days) <= 0 or int(miles) <= 0 or float(receipts) <= 0:
        return "Error: Negative numbers or zero entered. Try again."
    
    return "Input accepted."

def main():
    # Get user input until it's valid
    while True:
        user_days = input("Enter number of days (as a whole number): ")
        user_miles = input("Enter number of miles (as a whole number): ")
        user_receipts = input("Enter total dollar amount collected from receipts: ")

        print(checkErrors(user_days, user_miles, user_receipts), "\n")

        if checkErrors(user_days, user_miles, user_receipts) == "Input accepted.":
            break

    print("Calculating the predicted reimbursement...\n")

    start_time = time.time()

    # Convert inputs to appropriate types
    user_days = int(user_days)
    user_miles = int(user_miles)
    user_receipts = float(user_receipts)

    # Calculate derived features
    EPS = 1e-6 
    cost_per_mile = user_receipts / (user_miles + EPS)
    cost_per_day  = user_receipts / (user_days + EPS)
    miles_per_day = user_miles / (user_days + EPS)
    if user_days >= 5:
        long_trip_flag = 1
    else:
        long_trip_flag = 0

    # Load model and make the prediction
    imported_model = joblib.load('..\\artifacts\\finalModelgb.pkl')
    df = pd.DataFrame([{
        "miles_traveled": user_days, 
        "trip_duration_days": user_miles, 
        "total_receipts_amount": user_receipts,
        "cost_per_mile": cost_per_mile, 
        "cost_per_day": cost_per_day, 
        "miles_per_day": miles_per_day, 
        "long_trip_flag": long_trip_flag
        }])
    prediction = imported_model.predict(df)
    end_time = time.time()
    print(f"\nPredicted reimbursement: ${prediction[0]:.2f}")
    print(f"Execution time: {end_time - start_time:.2f} s")

if __name__ == "__main__":
    main()