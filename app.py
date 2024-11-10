from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib  # To load scaler if saved separately

app = Flask(__name__)
import os
print("Current working directory:", os.getcwd())

# Load the pre-trained CNN model
model = load_model('./heart_disease_cnn_model_final.h5')

# Load the saved scaler
scaler = joblib.load('scaler.save')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form fields
    age = float(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    fbs = int(request.form['fbs'])
    restecg = int(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = int(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = int(request.form['slope'])
    ca = int(request.form['ca'])
    thal = int(request.form['thal'])

    # Create an array with these inputs
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    # Scale input data to match model expectations
    input_data_scaled = scaler.transform(input_data)
    input_data_scaled = input_data_scaled.reshape((1, input_data_scaled.shape[1], 1))

    # Predict using the model
    prediction = model.predict(input_data_scaled)
    result = "Heart Disease Detected" if prediction[0] > 0.5 else "No Heart Disease Detected"
    
    # Render the result page with prediction
    return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
