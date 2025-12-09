# --- Setup & Load Data ---
import numpy as np 
import pandas as pd 
from pathlib import Path 
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler

EPS = 1e-6  #small constant to avoid division-by-zero
DATA_PATH = Path('..\\public_cases.json') #expected location of the dataset file

if DATA_PATH.exists(): #if the dataset file exists in the working directory...
    try:
        df = pd.read_json(DATA_PATH) #try reading it as a standard json array
    except ValueError:
        df = pd.read_json(DATA_PATH, lines=True)  #fallback: read as json Lines format if needed
else:
    #if the real file is missing, generate a tiny synthetic dataset so the script still runs
    rng = np.random.default_rng(42) # deterministic random generator for reproducibility
    df = pd.DataFrame({
        'trip_duration_days': rng.integers(1, 10, size=20), #1-9 days
        'miles_traveled': rng.integers(10, 1000, size=20), #10-999 miles
        'total_receipts_amount': np.round(rng.uniform(50, 2000, size=20), 2),#50-$2000
        'reimbursement': np.round(rng.uniform(60, 2200, size=20), 2)#target variable 
    })
    print('public_cases.json not found. using synthetic demo data.')

#normalize column names to the ones used in downstream code
df = df.rename(columns={
    'trip_duration_days': 'days', #rename to 'days'
    'miles_traveled': 'distance',  #rename to 'distance'
    'total_receipts_amount': 'receipt_total'#rename to 'receipt_total'
})

#derived features 
df['cost_per_mile'] = df['receipt_total'] / (df['distance'] + EPS)  #receipts per mile (avoid /0 with EPS)
df['cost_per_day']  = df['receipt_total'] / (df['days'] + EPS) #receipts per day
df['miles_per_day'] = df['distance'] / (df['days'] + EPS) #miles per day

#interaction&polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)  #set up to create degree-2 features (x, x^2, x*y, ...)
poly_arr = poly.fit_transform(df[['days','distance','receipt_total']])  #transform base numeric columns
poly_cols = poly.get_feature_names_out(['days','distance','receipt_total'])  #names for the new columns
df_poly = pd.DataFrame(poly_arr, columns=poly_cols, index=df.index) #dataFrame of polynomial features

#domain specific transformations
daily_cost = df['receipt_total'] / (df['days'] + EPS) #compute average daily cost as a helper
df['short_trip_flag'] = (df['distance'] < 100).astype(int) #1 if short trip (<100 miles), else 0
df['long_trip_flag']  = (df['days'] >= 5).astype(int)   #1 if long trip (>=5 days), else 0
df['high_daily_cost_flag'] = (daily_cost > 150).astype(int) #1 if daily cost exceeds threshold, else 0

# feature scaling & normalization
num_cols = ['days','distance','receipt_total','cost_per_mile','cost_per_day','miles_per_day']


std = StandardScaler()  #z-score scaling: mean=0, std=1
mm  = MinMaxScaler()# MinMax scaling: maps values into [0, 1]

std_df = pd.DataFrame(  #fit & transform numeric columns with standard scaler
    std.fit_transform(df[num_cols]),
    columns=[c+'_std' for c in num_cols],
    index=df.index
)

mm_df  = pd.DataFrame(  #fit & transform numeric columns with min-max scaler
    mm.fit_transform(df[num_cols]),
    columns=[c+'_mm'  for c in num_cols],
    index=df.index
)

#print samples to verify
print('Derived Features Sample:')
print(df[['days','distance','receipt_total','cost_per_mile','cost_per_day','miles_per_day']].head())

print('\n Domain Flags Sample:')
print(df[['short_trip_flag','long_trip_flag','high_daily_cost_flag']].head())

print('\n Standard Scaled Sample:')
print(std_df.head())

print('\n MinMax Scaled Sample:')
print(mm_df.head())
