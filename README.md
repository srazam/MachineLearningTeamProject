# Machine Learning Team Project - Documentation
This project is based on: https://github.com/8090-inc/top-coder-challenge/

**Scenario**: Your team has been hired as ML consultants by ACME Corporation. Their legacy reimbursement system has been running for 60 years, no one knows how it works, but it is still used daily. A new system has been built, but the ACME Corporation is confused by the differences in results. Your mission is to use machine learning to understand the original business logic and create a model that can explain and predict the legacy system’s behavior.

### How to Run
- For '.py' files: change directory into the 'MachineLearningTeamProject' folder and run 'python scriptName.py' in the terminal.
- For '.ipynb' files: run individual cells in order using the ctrl+Enter or press the play button on the top left of every cell. You can also run all cells at the same time by running the play all button at the top.

### Exploratory Data Analysis
The following files contains our work for the exploratory data analysis phase
- dataExp.ipynb: Data exploration report including a statistical summary of all variables, distribution analysis/visualization, correlation analysis, outlier detection/analysis, missing data assessment
- featureEngineering.py: Feature engineering strategies including derived features, interaction terms, polynomial features, domain-specific transformations, feature scaling, and normalization approaches
- TechnicalReport.md: Business logic hypothesis technical report incluidng analysis of PRD/interview transcripts, proposed business rules/logic patterns, feature importance hypotheses, potential non-linear relationships identification

### Model Evaluation Framework
For each model, Grid Search was used (where cross validaiton parameter = 5) and used MAE, RMSE, and accuracy within thresholds as evaluation metrics. 
- DecisionRule.py - implemented rule-based learning, specifically Decision rule extraction
- polynomialRegression.py - implemented a linear regression variant, poly nomial regression
- rf.ipynb - implemented tree-based methods, specifically Random Forest 

### System Regression
- mainModel.ipynb - shows final model training/testing, evaluation, artifacts, and pipeline. Rerun all cells in the file 
- productionReady.py - shows production-ready code

### Model Architecture
1. Data Processing: We retrieved data from 'public_cases.json' file (from https://github.com/8090-inc/top-coder-challenge/). We separate the input values/descriptive features into one data frame (miles_traveled, trip_duration_days, total_receipts_amount) and the expected_output into another. During feature engineering, we found which additional features to add. We added 3 derived features (cost per mile, cost per day, and miles per day) as well as a domain specific transformation (high_daily_cost_flag). Normalization was not used for this dataset for it was found to be more effective without it. Data was then split; 75% was used for training and 25% was used for testing.
2. Algorithm: We used a Random Forest algorithm to train and test our data on. Random Forest works by combining the outputs of several decision trees. The process starts with bootstrap sampling where random rows of data are selected with replacement (can appear in a sample more than once). Each bootstrap sample is used to train a tree. With each of these samples, feature sampling is then used where only a random subset of features is used to build each tree. It was found for this problem and out of the parameter values we tested, 200 trees were the best to use. The trees are then built by finding the best decision rules given its sample. Given the paramters we tested, we found that each tree shouldn't be deeper than 10 nodes, 5 samples are required to split an internal node, and that 2 samples can be in a node to be considered a leaf node. After the trees are trained, the entire random forest model makes a prediciton by averaging all of the outputs from the individual trees.
3. Integration: The user would enter 3 paramter values; one for miles traveled, one for number of days, and another for the total receipt amount. The derived features are then calculated using these values and all features are then used as input paramters for the model. The model then outputs one prediction value; the total amount in dollars for how much a person would be reimbursed based on the number days, miles, and receipt amount.

### Feature Engineering Rationale
We decided to not use polynomial features since we found that polynomial regression (a linear regression variant) was not the best model to use for this problem. The derived features we tested were:
- cost_per_mile = receipt_total / distance  
- cost_per_day  = receipt_total / days
- miles_per_day = distance / days
The domain specific transformations we used were:
daily_cost = receipt_total / days
- short_trip_flag = Check if the distance is less than 100
- long_trip_flag  = Check if the number of days is greater than or equal to 5   
- high_daily_cost_flag = Check if the daily cost (the receipt total / days) is greater than 150 
Since we selected random forest for the model, we looked at the feature importances when it was initially trained. It was found that total_receipts_amount, miles_traveled, and trip_duration_days was the most important (from greatest to least) and the least important features (almost zero) was high_daily_cost_flag and short_trip_flag. Therefore, the following features were used in the final model:
- miles_traveled
- trip_duration_days
- total_receipts_amount
- cost_per_mile
- cost_per_day
- miles_per_day
- long_trip_flag

### Deployment Instructions
The model is saved in the finalModel.pkl file. If the model is not present, run all cells in the 'mainModel.ipynb' file. Use the model by running the 'productionReady.py' file. This script takes 3 paramters where you enter the number of miles, days, and total receipt amount. If you don't enter a whole number for miles or day neither a dollar amount for total receipt amount, the script will prompt the user to keep entering values until they're correct.