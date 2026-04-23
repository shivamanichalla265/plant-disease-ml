from flask import Flask, render_template, request
import numpy as np
import cv2
import joblib
from PIL import Image

app = Flask(__name__)

model = joblib.load("model/plant_model.pkl")
classes = joblib.load("model/classes.pkl")

def extract_features(image):
    image = cv2.resize(image, (128, 128))

    mean = np.mean(image, axis=(0,1))
    std = np.std(image, axis=(0,1))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_mean = np.mean(edges)

    return np.hstack([mean, std, edge_mean]).reshape(1, -1)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None

    if request.method == 'POST':
        file = request.files['file']

        img = Image.open(file).convert('RGB')
        img = np.array(img)

        features = extract_features(img)

        probs = model.predict_proba(features)[0]
        pred = np.argmax(probs)

        prediction = classes[pred]
        confidence = round(probs[pred] * 100, 2)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence)

app.run(debug=True)