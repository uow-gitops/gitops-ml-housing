# app/main.py

from __future__ import print_function
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime as dt
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib as mpl
warnings.filterwarnings("ignore")
import psycopg2
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os

def main_training():
    # Import dataset
    df = pd.read_csv('app/Resources/melbourne_housing.csv')
    print("Initial dataframe info:")
    print(df.head())
    print(df.info())
    
    # Drop unnecessary columns
    df.drop([
        "SellerG", "Method", "BuildingArea", "YearBuilt",
        "Lattitude", "Longtitude", "Bedroom2", "Address",
        "CouncilArea", "Suburb"
    ], axis=1, inplace=True)
    print("After dropping unnecessary columns, shape =", df.shape)

    # Rename columns
    melbourne_df = df.rename(columns={
        "Landsize": "Land Size",
        "Regionname": "Region",
        "Propertycount": "Property Count"
    })
    
    # Check and drop missing data
    print("Missing values before dropna:")
    print(melbourne_df.isna().sum())
    melbourne_df.dropna(inplace=True)
    print("Missing values after dropna:")
    print(melbourne_df.isna().sum())
    
    # Drop duplicate rows if any
    melbourne_df.drop_duplicates(inplace=True)
    
    # Change format of the 'Date' column; using dayfirst=True for dates like "13/08/2016"
    melbourne_df['Date'] = pd.to_datetime(melbourne_df['Date'], dayfirst=True)
    melbourne_df['year'] = melbourne_df['Date'].dt.year
    melbourne_df.drop(['Date'], axis=1, inplace=True)
    
    # Based on the heatmap (plotting code is commented out), drop extra columns
    melbourne_df.drop(['Postcode', 'year', 'Land Size', 'Property Count'], axis=1, inplace=True)
    
    # Data description
    print("Data description:")
    print(melbourne_df.describe())
    
    # Log transform Price
    melbourne_df["LogPrice"] = np.log(melbourne_df["Price"])
    
    # Copy for encoding
    housing_df = melbourne_df.copy()
    
    ### Encoding categorical features
    encode = LabelEncoder().fit(housing_df['type'])
    carpet = {x: i for i, x in enumerate(encode.classes_)}
    print("Type encoding:", carpet)
    
    encoder = LabelEncoder().fit(housing_df['region'])
    carp = {x: i for i, x in enumerate(encoder.classes_)}
    print("Region encoding:", carp)
    
    housing_df['type'] = encode.transform(housing_df['type'])
    housing_df['region'] = encoder.transform(housing_df['region'])
    
    ## Prepare features and target for training
    X = housing_df.drop(["logprice", "price"], axis=1)
    print("Training feature order:", X.columns.tolist())
    y = housing_df['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    ## Fit Linear Regression
    modelR = LinearRegression().fit(X_train_scaled, y_train)
    print(f"Linear Regression - Training Score: {modelR.score(X_train_scaled, y_train)}")
    print(f"Linear Regression - Testing Score:  {modelR.score(X_test_scaled, y_test)}")
    
    # Fit Random Forest with criterion='mse'
    model_rf = RandomForestRegressor(
        n_estimators=100,
        criterion='mse',
        random_state=42,
        max_depth=2
    ).fit(X_train, y_train)
    print(f"Random Forest - Training Score: {model_rf.score(X_train, y_train)}")
    print(f"Random Forest - Testing Score:  {model_rf.score(X_test_scaled, y_test)}")
    
    # Fit Decision Tree with criterion='mse'
    model_tree = DecisionTreeRegressor(
        criterion='mse',
        random_state=42
    ).fit(X_train, y_train)
    print(f"Decision Tree - Training Score: {model_tree.score(X_train, y_train)}")
    print(f"Decision Tree - Testing Score:  {model_tree.score(X_test, y_test)}")
    
    ## Hyperparameter search
    param_dists = {
        'criterion': ['mse', 'friedman_mse'],
        'max_depth': [3, 4, 7, None],
        'min_samples_split': np.arange(0.1, 1.1, 0.1),
        'min_samples_leaf': list(range(1, 21)),
        'max_features': ['auto', 'sqrt', 'log2', None]
    }
    model_cv = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=param_dists,
        n_iter=200,
        scoring='neg_mean_squared_error',
        cv=5,
        random_state=42
    ).fit(X_train_scaled, y_train)
    print(f"Randomized Search - Training Score: {model_cv.score(X_train_scaled, y_train)}")
    print(f"Randomized Search - Testing Score:  {model_cv.score(X_test_scaled, y_test)}")
    
    # SVR
    from sklearn.svm import SVR
    regressor = SVR(kernel="rbf").fit(X_train_scaled, y_train)
    print(f"SVR - Training Score: {regressor.score(X_train_scaled, y_train)}")
    print(f"SVR - Testing Score:  {regressor.score(X_test_scaled, y_test)}")
    
    # Lasso
    model_lasso = Lasso(alpha=1.0, max_iter=1000).fit(X_train_scaled, y_train)
    print(f"Lasso - Training Score: {model_lasso.score(X_train_scaled, y_train)}")
    print(f"Lasso - Testing Score:  {model_lasso.score(X_test_scaled, y_test)}")
    
    # Ridge
    model_Ridge = Ridge(alpha=100).fit(X_train_scaled, y_train)
    print(f"Ridge - Training Score: {model_Ridge.score(X_train_scaled, y_train)}")
    print(f"Ridge - Testing Score:  {model_Ridge.score(X_test_scaled, y_test)}")
    
    # Show some predictions
    y_pred = modelR.predict(X_test)
    print(pd.DataFrame({"Prediction": y_pred, "Actual": y_test}))
    y_pred = model_lasso.predict(X_test)
    print(pd.DataFrame({"Prediction": y_pred, "Actual": y_test}))
    y_pred = model_tree.predict(X_test)
    print(pd.DataFrame({"Prediction": y_pred, "Actual": y_test}))
    
    # ——— SAVE THE TREE MODEL TO app/model/model.pkl ———
    model_dir  = 'model'               # relative to cwd=app/
    model_file = os.path.join(model_dir, 'model.pkl')
    
    # 1) ensure directory
    os.makedirs(model_dir, exist_ok=True)
    print(f"Ensured directory exists: {model_dir}")
    
    # 2) remove stale
    if os.path.exists(model_file):
        os.remove(model_file)
        print(f"Removed old file: {model_file}")
    
    # 3) write new
    with open(model_file, 'wb') as f:
        pickle.dump(model_tree, f)
    print(f"Created new model file: {model_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "retrain":
        print("Retraining model...")
        main_training()
        print("Retraining complete. Exiting.")
        sys.exit(0)
    else:
        print("No retrain command provided. Exiting.")
        sys.exit(0)
