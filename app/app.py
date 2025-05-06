#import libraries
import numpy as np
import flask
import pickle
import pandas as pd
from pandas import DataFrame
from flask import Flask, request, jsonify, render_template, url_for
from prometheus_client import Counter, Histogram, generate_latest, Gauge  # New
from sklearn.preprocessing import LabelEncoder, StandardScaler  # New


#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model/model.pkl', 'rb'))


# Metrics definitions
REQUEST_COUNT = Counter('request_count', 'Total number of prediction requests')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Time taken for prediction')
MODEL_ACCURACY = Gauge('model_accuracy', 'Latest model test accuracy')
LATEST_PREDICTION_ACCURACY = Gauge('latest_prediction_accuracy', 'Accuracy of last individual prediction')  # New

# Load cleaned dataset once at startup
try:
    df_clean = pd.read_csv('app/Resources/melbourne.csv')

    # 🔥 Drop any index column (like 'Unnamed: 0')
    df_clean = df_clean.loc[:, ~df_clean.columns.str.contains('^Unnamed')]

    # Lowercase all column names
    df_clean.columns = [c.lower() for c in df_clean.columns]

    # Encode type and region
    df_clean['type'] = LabelEncoder().fit_transform(df_clean['type'])
    df_clean['region'] = LabelEncoder().fit_transform(df_clean['region'])

    print("Cleaned dataset loaded and prepared successfully.")
except Exception as e:
    print(f"Error loading cleaned dataset: {e}")
    df_clean = pd.DataFrame()  # fallback


# Set overall model accuracy at startup
try:
    X_all = df_clean.drop(["logprice", "price"], axis=1)
    y_all = df_clean['price']
    scaler = StandardScaler().fit(X_all)
    X_all_scaled = scaler.transform(X_all)
    overall_accuracy = model.score(X_all_scaled, y_all)
    print(f"Loaded model accuracy on clean data: {overall_accuracy:.4f}")
    MODEL_ACCURACY.set(overall_accuracy)
except Exception as e:
    print(f"Could not calculate model accuracy: {e}")
    MODEL_ACCURACY.set(0.0)


def check_prediction_accuracy(input_features_array, prediction):

    try:
        input_features = input_features_array[0]
        input_df = pd.DataFrame([input_features], columns=['rooms', 'type', 'distance', 'bathroom', 'car', 'region'])

        # Cast inputs to int where needed
        rooms_val = int(input_df['rooms'][0])
        type_val = int(input_df['type'][0])
        distance_val = input_df['distance'][0]
        bathroom_val = int(input_df['bathroom'][0])
        car_val = int(input_df['car'][0])
        region_val = int(input_df['region'][0])

        # Do the matching
        matches = df_clean[
            (df_clean['rooms'] == rooms_val) &
            (df_clean['type'] == type_val) &
            (np.isclose(df_clean['distance'], distance_val, atol=0.5)) &
            (df_clean['bathroom'] == bathroom_val) &
            (df_clean['car'] == car_val) &
            (df_clean['region'] == region_val)
        ]

        print(f"Number of matches found: {len(matches)}")

        if not matches.empty:
            actual_price = matches.iloc[0]['price']
            error = abs((actual_price - prediction) / actual_price)
            accuracy = 1 - error
            print(f"Prediction: {prediction:.2f}, Actual: {actual_price:.2f}, Accuracy: {accuracy:.4f}")
            LATEST_PREDICTION_ACCURACY.set(accuracy)
        else:
            print("No exact match found in cleaned dataset for input features.")
            LATEST_PREDICTION_ACCURACY.set(0.0)
    except Exception as e:
        print(f"Error in check_prediction_accuracy: {e}")
        LATEST_PREDICTION_ACCURACY.set(0.0)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/', methods=['GET', 'POST'])
@PREDICTION_LATENCY.time()
def main():
    REQUEST_COUNT.inc()

    int_features = [float(x) for x in request.form.values()]

    form_input = {
    'type': int_features[0],
    'region': int_features[1],
    'rooms': int_features[2],
    'bathroom': int_features[3],
    'distance': int_features[4],
    'car': int_features[5]
    }

    # Reorder inputs to match model's expected order
    model_input_order = [
        form_input['rooms'],
        form_input['type'],
        form_input['distance'],
        form_input['bathroom'],
        form_input['car'],
        form_input['region']
    ]

    input_df = pd.DataFrame([model_input_order], columns=['rooms', 'type', 'distance', 'bathroom', 'car', 'region'])
    prediction = model.predict(input_df)[0]  # New
    
    check_prediction_accuracy([np.array(model_input_order)], prediction)
    
    return render_template('index.html', prediction_text ='The estimate price is :{}'.format(prediction))

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': 'text/plain; charset=utf-8'}

if __name__ == '__main__':  
   app.run(debug = True)

   #app.run(debug=True, host='0.0.0.0')  # ✅ REQUIRED for GKE
