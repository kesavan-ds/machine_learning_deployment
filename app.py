# Install Flask and Flask-Ngrok (if running in Colab and want to expose API)
# !pip install Flask flask-ngrok pandas scikit-learn

from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
# Uncomment the line below if you are running in Google Colab and want to expose your API
# run_with_ngrok(app)

# Load the best model and scaler
try:
    model = joblib.load('best_model_svm.pkl') # Assuming SVM was the best
    scaler = joblib.load('scaler.pkl')
    print("Model and Scaler loaded successfully!")
except FileNotFoundError:
    print("Error: Model or Scaler files not found. Make sure 'best_model_svm.pkl' and 'scaler.pkl' are in the same directory.")
    model = None
    scaler = None

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or Scaler not loaded. Please check server logs.'}), 500

    try:
        data = request.get_json(force=True)
        
        # Convert input data to a Pandas DataFrame for scaling
        # Ensure the order of features matches the training data
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        input_df = pd.DataFrame([data], columns=feature_names)
        
        # Scale the input features
        scaled_input = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[:, 1][0]
        
        # Return prediction as JSON
        return jsonify({
            'prediction': int(prediction),
            'probability_of_diabetes': float(prediction_proba)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# To run the app:
if __name__ == '__main__':
  #For local development:
  app.run(debug=True)
 # For Colab (with ngrok): uncomment run_with_ngrok(app) at the top
  app.run()
