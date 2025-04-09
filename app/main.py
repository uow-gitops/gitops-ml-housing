# import dependencies
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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib as mpl
warnings.filterwarnings("ignore")
import psycopg2
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
import pickle

def main_training():
    # import dataset
    df = pd.read_csv('app/Resources/melbourne_housing.csv')
    print("Initial dataframe info:")
    print(df.head())
    print(df.info())
    
    # Drop unnecessary columns
    df.drop(["SellerG", "Method", "BuildingArea", "YearBuilt", "Lattitude", "Longtitude", "Bedroom2", "Address", "CouncilArea", "Suburb"], axis=1, inplace=True)
    print("After dropping unnecessary columns, shape =", df.shape)

    # Rename columns
    melbourne_df = df.rename(columns={"Landsize": "Land Size",
                                      "Regionname": "Region",
                                      "Propertycount": "Property Count"})
    
    # Check and drop missing data
    print("Missing values before dropna:")
    print(melbourne_df.isna().sum())
    melbourne_df.dropna(inplace=True)
    print("Missing values after dropna:")
    print(melbourne_df.isna().sum())
    
    # Drop duplicate rows if any
    melbourne_df.drop_duplicates(inplace=True)
    
    # Change format of the 'Date' column using dayfirst=True for dates like "13/08/2016"
    melbourne_df['Date'] = pd.to_datetime(melbourne_df['Date'], dayfirst=True)
    melbourne_df['year'] = melbourne_df['Date'].dt.year
    melbourne_df.drop(['Date'], axis=1, inplace=True)
    
    # Generate and save heatmap of correlations
    plt.figure(figsize=(15,8))
    # Compute correlations only on numeric columns to avoid conversion errors.
    numeric_corr = melbourne_df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(numeric_corr, annot=True, cmap='coolwarm')
    plt.savefig('heatmap.png')
    
    # Based on the heatmap, drop additional columns
    melbourne_df.drop(['Postcode', 'year', 'Land Size', 'Property Count'], axis=1, inplace=True)
    
    # Describe data and plot distributions
    print("Data description:")
    print(melbourne_df.describe())
    sns.distplot(melbourne_df["Price"], fit=norm)
    fig = plt.figure()
    prob = stats.probplot(melbourne_df["Price"], plot=plt)
    
    # Log transform Price as its distribution is log-normal
    melbourne_df["LogPrice"] = np.log(melbourne_df["Price"])
    dist_price = sns.distplot(melbourne_df["LogPrice"], fit=norm)
    fig = plt.figure()
    prob_log = stats.probplot(melbourne_df["LogPrice"], plot=plt)
    plt.show()
    
    # Function to find outliers
    def finding_outliers(data, variable_name):
        iqr = data[variable_name].quantile(0.75) - data[variable_name].quantile(0.25)
        lower = data[variable_name].quantile(0.25) - 1.5 * iqr
        upper = data[variable_name].quantile(0.75) + 1.5 * iqr
        return data[(data[variable_name] < lower) | (data[variable_name] > upper)]
    
    # Plot and adjust for outliers in Price
    plt.figure(figsize=(8,8))
    sns.boxplot(y="Price", data=melbourne_df)
    finding_outliers(melbourne_df, "Price").sort_values("Price")
    iqr_price = melbourne_df["Price"].quantile(0.75) - melbourne_df["Price"].quantile(0.25)
    melbourne_df.loc[(finding_outliers(melbourne_df, "Price").index, "Price")] = melbourne_df["Price"].quantile(0.75) + 1.5 * iqr_price
    plt.figure(figsize=(8,8))
    sns.boxplot(y="Price", data=melbourne_df)
    
    # Plot and adjust for outliers in Rooms
    plt.figure(figsize=(8,8))
    sns.boxplot(y="Rooms", data=melbourne_df)
    finding_outliers(melbourne_df, "Rooms").sort_values("Rooms")
    iqr_rooms = melbourne_df["Rooms"].quantile(0.75) - melbourne_df["Rooms"].quantile(0.25)
    melbourne_df.loc[(finding_outliers(melbourne_df, "Rooms").index, "Rooms")] = melbourne_df["Rooms"].quantile(0.75) + 1.5 * iqr_rooms
    plt.figure(figsize=(8,8))
    sns.boxplot(y="Rooms", data=melbourne_df)
    
    # Plot and adjust for outliers in Bathroom
    plt.figure(figsize=(8,8))
    sns.boxplot(y="Bathroom", data=melbourne_df)
    finding_outliers(melbourne_df, "Bathroom").sort_values("Bathroom")
    iqr_bath = melbourne_df["Bathroom"].quantile(0.75) - melbourne_df["Bathroom"].quantile(0.25)
    melbourne_df.loc[(finding_outliers(melbourne_df, "Bathroom").index, "Bathroom")] = melbourne_df["Bathroom"].quantile(0.75) + 1.5 * iqr_bath
    plt.figure(figsize=(8,8))
    sns.boxplot(y="Bathroom", data=melbourne_df)
    plt.show()
    
    # Additional plots for analysis...
    plt.figure(figsize=(15,8))
    sns.countplot(x="Bathroom", data=melbourne_df)
    plt.figure(figsize=(15,8))
    sns.countplot(x="Rooms", data=melbourne_df)
    plt.figure(figsize=(15,8))
    sns.countplot(x="Type", data=melbourne_df)
    plt.figure(figsize=(15,8))
    sns.countplot(x="Car", data=melbourne_df)
    plt.figure(figsize=(15,8))
    ax = sns.countplot(x="Region", data=melbourne_df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35)
    plt.figure(figsize=(15,8))
    sns.barplot(x="Region", y="Price", data=melbourne_df)
    plt.xticks(rotation=45)
    print(melbourne_df.groupby('Region')['Price'].mean())
    
    ###################### Data preparation
    
    # Save cleaned data as CSV
    melbourne_df.to_csv('melbourne.csv', index=False)
    melbourne_df.columns = [c.lower() for c in melbourne_df.columns]
    
    # Connect to Postgres and import data into SQL
    try:
        engine = create_engine('postgresql://postgres:Blome00228@localhost:5433/Housing')
        conn = engine.connect()
        melbourne_df.to_sql("melbourne", conn, if_exists='replace', index=False)
        housing_df = pd.read_sql('select * from "melbourne"', conn)
        conn.close()
        print("Database import successful. Retrieved table:")
        print(housing_df)
    except Exception as e:
        print("Database connection failed, skipping DB import. Error:", e)
    
    ### Encoding categorical features
    encode = LabelEncoder().fit(housing_df['type']) if 'type' in housing_df.columns else LabelEncoder().fit(melbourne_df['type'])
    carpet = {x: i for i, x in enumerate(encode.classes_)}
    print("Type encoding:", carpet)
    
    encoder = LabelEncoder().fit(housing_df['region']) if 'region' in housing_df.columns else LabelEncoder().fit(melbourne_df['region'])
    carp = {x: i for i, x in enumerate(encoder.classes_)}
    print("Region encoding:", carp)
    
    # If a database table wasn't imported, use melbourne_df for encoding
    df_to_encode = housing_df if 'type' in housing_df.columns else melbourne_df
    df_to_encode['type'] = LabelEncoder().fit_transform(df_to_encode['type'])
    df_to_encode['region'] = LabelEncoder().fit_transform(df_to_encode['region'])
    
    ## Prepare features and target for training
    X = df_to_encode.drop(["logprice", "price"], axis=1)
    y = df_to_encode['price']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ## Fit Linear Regression model
    modelR = LinearRegression().fit(X_train_scaled, y_train)
    training_score = modelR.score(X_train_scaled, y_train)
    testing_score = modelR.score(X_test_scaled, y_test)
    print(f"Linear Regression - Training Score: {training_score}")
    print(f"Linear Regression - Testing Score: {testing_score}")
    
    # Fit Random Forest model
    model_rf = RandomForestRegressor(n_estimators=100, criterion='mse', random_state=42, max_depth=2).fit(X_train, y_train)
    training_score = model_rf.score(X_train, y_train)
    testing_score = model_rf.score(X_test_scaled, y_test)
    print(f"Random Forest - Training Score: {training_score}")
    print(f"Random Forest - Testing Score: {testing_score}")
    
    # Fit Decision Tree model
    model_tree = DecisionTreeRegressor(criterion='squared_error',
                                       splitter='best', max_depth=None, 
                                       min_samples_split=2, min_samples_leaf=1, 
                                       min_weight_fraction_leaf=0.0, max_features=None, 
                                       random_state=42, max_leaf_nodes=None, 
                                       min_impurity_decrease=0.0, ccp_alpha=0.0).fit(X_train, y_train)
    training_score = model_tree.score(X_train, y_train)
    testing_score = model_tree.score(X_test, y_test)
    print(f"Decision Tree - Training Score: {training_score}")
    print(f"Decision Tree - Testing Score: {testing_score}")
    
    ## Fit Randomized Search model
    param_dists = {'criterion': ['mean_squared_error', 'friedman_mse'],
                   'max_depth': [3, 4, 7, None],
                   'min_samples_split': np.arange(0.1, 1.1, 0.1),
                   'min_samples_leaf': list(range(1, 21)),
                   'max_features': ['auto', 'sqrt', 'log2', None]}
    
    model_cv = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=42),
                                  param_distributions=param_dists,
                                  n_iter=200,
                                  scoring='neg_mean_squared_error',
                                  cv=5,
                                  random_state=42).fit(X_train_scaled, y_train)
    training_score = model_cv.score(X_train_scaled, y_train)
    testing_score = model_cv.score(X_test_scaled, y_test)
    print(f"Randomized Search - Training Score: {training_score}")
    print(f"Randomized Search - Testing Score: {testing_score}")
    
    # Fit SVR model
    from sklearn.svm import SVR
    regressor = SVR(kernel="rbf").fit(X_train_scaled, y_train)
    training_score = regressor.score(X_train_scaled, y_train)
    testing_score = regressor.score(X_test_scaled, y_test)
    print(f"SVR - Training Score: {training_score}")
    print(f"SVR - Testing Score: {testing_score}")
    
    # Fit Lasso model
    model_lasso = Lasso(alpha=1.0, max_iter=1000).fit(X_train_scaled, y_train)
    training_score = model_lasso.score(X_train_scaled, y_train)
    testing_score = model_lasso.score(X_test_scaled, y_test)
    print(f"Lasso - Training Score: {training_score}")
    print(f"Lasso - Testing Score: {testing_score}")
    
    # Fit Ridge model
    model_Ridge = Ridge(alpha=100).fit(X_train, y_train)
    training_score = model_Ridge.score(X_train_scaled, y_train)
    testing_score = model_Ridge.score(X_test_scaled, y_test)
    print(f"Ridge - Training Score: {training_score}")
    print(f"Ridge - Testing Score: {testing_score}")
    
    # Make predictions and display sample comparisons
    y_pred = modelR.predict(X_test)
    print(pd.DataFrame({"Prediction": y_pred, "Actual": y_test}))
    
    y_pred = model_lasso.predict(X_test)
    print(pd.DataFrame({"Prediction": y_pred, "Actual": y_test}))
    
    y_pred = model_tree.predict(X_test)
    print(pd.DataFrame({"Prediction": y_pred, "Actual": y_test}))
    
    # Save the Decision Tree model
    pickle.dump(model_tree, open('model.pkl', 'wb'))

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
