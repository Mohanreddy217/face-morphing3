from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import cv2
import os
import csv
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
CSV_FILE = 'submissions.csv'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and scaler
model = joblib.load('face_morph_model.pkl')
scaler = joblib.load('scaler.pkl')

# Home Page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        message = request.form.get('message', '')
        file = request.files.get('image')

        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Preprocess image
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            img_flatten = img.flatten().reshape(1, -1)
            img_scaled = scaler.transform(img_flatten)

            # Predict
            prediction = model.predict(img_scaled)[0]
            result = 'Real' if prediction == 0 else 'Morphed'

            # Save to CSV
            with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([message, filename, result])

            return render_template('home.html', success=True)

    return render_template('home.html', success=False)

# About Page
@app.route('/about')
def about():
    return render_template('about.html')

# Upload Page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('image')
        message = request.form.get('message', '')

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Preprocess image
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            img_flatten = img.flatten().reshape(1, -1)
            img_scaled = scaler.transform(img_flatten)

            # Predict
            prediction = model.predict(img_scaled)[0]
            result = 'Real' if prediction == 0 else 'Morphed'

            # Save to CSV
            with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([message, filename, result])

            return render_template('upload.html', result=result)

    return render_template('upload.html')

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        img_flatten = img.flatten().reshape(1, -1)
        img_scaled = scaler.transform(img_flatten)

        prediction = model.predict(img_scaled)[0]
        result = 'Real' if prediction == 0 else 'Morphed'

        return jsonify({'result': result})

    except Exception as e:
        print("Prediction Error:", e)
        return jsonify({'error': 'Prediction failed'}), 500


# Invalid method handler for predict (for GET or others)
@app.route('/predict', methods=['GET'])
def predict_get_not_allowed():
    return jsonify({'error': 'GET method not allowed on this endpoint'}), 405

# Live Capture Page
@app.route('/live')
def live():
    return render_template('live.html')

if __name__ == '__main__':
    app.run(debug=True)
