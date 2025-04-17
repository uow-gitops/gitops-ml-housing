# app/main.py

from __future__ import print_function
import warnings
import os
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
import psycopg2
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

warnings.filterwarnings("ignore")

# Base directory for this script (the "app" folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_PATH  = os.path.join(BASE_DIR, 'Resources', 'melbourne_housing.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'model')
MODEL_FILE = os.path.join(MODEL_DIR, 'model.pkl')

def main_training():
    # Import dataset
    df = pd.read_csv(DATA_PATH)
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
    
    # Drop duplicates
    melbourne_df.drop_duplicates(inplace=True)
    
    # Parse dates
    melbourne_df['Date'] = pd.to_datetime(melbourne_df['Date'], dayfirst=True)
    melbourne_df['year'] = melbourne_df['Date'].dt.year
    melbourne_df.drop(['Date'], axis=1, inplace=True)
    
    # ----------------- Plotting code commented out -----------------
    # plt.figure(figsize=(15,8))
    # numeric_corr = melbourne_df.select_dtypes(include=[np.number]).corr()
    # sns.heatmap(numeric_corr, annot=True, cmap='coolwarm')
    # plt.savefig('heatmap.png')
    # -----------------------------------------------------------------
    
    # Drop columns based on heatmap insights
    melbourne_df.drop([
        'Postcode', 'year', 'Land Size', 'Property Count'
    ], axis=1, inplace=True)
    
    # Describe data
    print("Data description:")
    print(melbourne_df.describe())
    
    # ----------------- Plotting code commented out -----------------
    # sns.distplot(melbourne_df["Price"], fit=norm)
    # fig = plt.figure()
    # prob = stats.probplot(melbourne_df["Price"], plot=plt)
    # -----------------------------------------------------------------
    
    # Log-transform price
    melbourne_df["LogPrice"] = np.log(melbourne_df["Price"])
    
    # ----------------- Plotting code commented out -----------------
    # dist_price = sns.distplot(melbourne_df["LogPrice"], fit=norm)
    # fig = plt.figure()
    # prob_log = stats.probplot(melbourne_df["LogPrice"], plot=plt)
    # plt.show()
    # -----------------------------------------------------------------
    
    # Helper to find outliers
    def finding_outliers(data, var):
        iqr   = data[var].quantile(0.75) - data[var].quantile(0.25)
        lower = data[var].quantile(0.25) - 1.5 * iqr
        upper = data[var].quantile(0.75) + 1.5 * iqr
        return data[(data[var] < lower) | (data[var] > upper)]
    
    # ----------------- Plotting code commented out -----------------
    # plt.figure(figsize=(8,8))
    # sns.boxplot(y="Price", data=melbourne_df)
    # finding_outliers(melbourne_df, "Price")
    # iqr_price = melbourne_df["Price"].quantile(0.75) - melbourne_df["Price"].quantile(0.25)
    # melbourne_df.loc[
    #     (finding_outliers(melbourne_df, "Price").index, "Price")
    # ] = melbourne_df["Price"].quantile(0.75) + 1.5 * iqr_price
    # plt.figure(figsize=(8,8))
    # sns.boxplot(y="Price", data=melbourne_df)
    # -----------------------------------------------------------------
    
    print(melbourne_df.groupby('Region')['Price'].mean())
    
    ###################### Data preparation
    
    # Optionally save cleaned CSV
    melbourne_df.to_csv(os.path.join(BASE_DIR, 'Resources', 'melbourne.csv'), index=False)
    melbourne_df.columns = [c.lower() for c in melbourne_df.columns]
    
    # Attempt DB import
    try:
        engine = create_engine('postgresql://postgres:Blome00228@localhost:5433/Housing')
        conn = engine.connect()
        melbourne_df.to_sql("melbourne", conn, if_exists='replace', index=False)
        housing_df = pd.read_sql('select * from "melbourne"', conn)
        conn.close()
        print("DB import successful.")
    except Exception as e:
        print("DB import failed, using local df. Error:", e)
        housing_df = melbourne_df.copy()
    
    # Encode categorical features
    encoder_type   = LabelEncoder().fit(housing_df['type'])
    encoder_region = LabelEncoder().fit(housing_df['region'])
    
    housing_df['type']   = encoder_type.transform(housing_df['type'])
    housing_df['region'] = encoder_region.transform(housing_df['region'])
    
    print("Type encoding:", dict(zip(encoder_type.classes_, encoder_type.transform(encoder_type.classes_))))
    print("Region encoding:", dict(zip(encoder_region.classes_, encoder_region.transform(encoder_region.classes_))))
    
    # Prepare features + target
    X = housing_df.drop(["logprice", "price"], axis=1)
    y = housing_df['price']
    
    # Train/test split + scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    # Linear Regression
    modelR = LinearRegression().fit(X_train_scaled, y_train)
    print("Linear Regression:", modelR.score(X_test_scaled, y_test))
    
    # Random Forest (using 'mse' under scikit‑learn 0.24.2)
    model_rf = RandomForestRegressor(
        n_estimators=100,
        criterion='mse',
        random_state=42,
        max_depth=2
    ).fit(X_train, y_train)
    print("Random Forest:", model_rf.score(X_test_scaled, y_test))
    
    # Decision Tree (using 'mse')
    model_tree = DecisionTreeRegressor(
        criterion='mse',
        random_state=42
    ).fit(X_train, y_train)
    print("Decision Tree:", model_tree.score(X_test, y_test))
    
    # Randomized Search over RF hyperparameters
    param_dists = {
        'criterion': ['mse', 'friedman_mse'],
        'max_depth': [3, 4, 7, None],
        'min_samples_split': np.arange(0.1, 1.1, 0.1),
        'min_samples_leaf': list(range(1,21)),
        'max_features': ['auto','sqrt','log2',None]
    }
    model_cv = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=param_dists,
        n_iter=200,
        scoring='neg_mean_squared_error',
        cv=5,
        random_state=42
    ).fit(X_train_scaled, y_train)
    print("RandSearch RF:", model_cv.score(X_test_scaled, y_test))
    
    # SVR
    from sklearn.svm import SVR
    svr = SVR(kernel="rbf").fit(X_train_scaled, y_train)
    print("SVR:", svr.score(X_test_scaled, y_test))
    
    # Lasso
    lasso = Lasso(alpha=1.0, max_iter=1000).fit(X_train_scaled, y_train)
    print("Lasso:", lasso.score(X_test_scaled, y_test))
    
    # Ridge
    ridge = Ridge(alpha=100).fit(X_train_scaled, y_train)
    print("Ridge:", ridge.score(X_test_scaled, y_test))
    
    # Save the trained Decision Tree model
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created directory: {MODEL_DIR}")
    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)
        print(f"Removed old file: {MODEL_FILE}")
    
    pickle.dump(model_tree, open(MODEL_FILE, 'wb'))
    print(f"Saved new model to: {MODEL_FILE}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "retrain":
        print("Retraining model...")
        main_training()
        print("Retraining complete.")
    else:
        print("Usage: python main.py retrain")
