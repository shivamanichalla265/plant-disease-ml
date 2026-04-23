# 🌿 Plant Disease Detection System (Machine Learning)

## 📌 Overview
This project is a Machine Learning-based web application that detects plant leaf diseases from images. Users can upload an image of a plant leaf, and the system predicts whether the plant is healthy or affected by a disease.

The model uses image processing and classical Machine Learning (Random Forest) for classification, and Flask is used to deploy it as a web application.

---

## 🚀 Features
- Upload plant leaf images
- Predict disease type instantly
- Displays prediction confidence
- Simple and user-friendly web interface
- Lightweight ML model (fast predictions)

---

## 🧠 Tech Stack
- Python
- Flask (Web Framework)
- OpenCV (Image Processing)
- Scikit-learn (Machine Learning)
- NumPy, Pandas
- HTML, CSS (Frontend)

---

## 📂 Project Structure

plant-disease-ml/
│── dataset/
│── model/
│ ├── plant_model.pkl
│ ├── classes.pkl
│── templates/
│ ├── index.html
│── static/
│── train.py
│── app.py
│── download.py
│── README.md


---

## ⚙️ How It Works
1. Upload a leaf image through the web interface
2. Image is processed (resized + feature extraction)
3. Machine Learning model predicts disease type
4. Result is displayed with confidence score

---

## 🧪 Model Details
- Algorithm: Random Forest Classifier
- Features:
  - Color statistics (mean, std)
  - Edge detection (Canny)
- Dataset: PlantVillage dataset (subset used)

---

## ▶️ How to Run Locally

### 1. Install dependencies

pip install flask scikit-learn opencv-python numpy pillow joblib


### 2. Train the model

python train.py


### 3. Run the web app

python app.py


### 4. Open in browser

http://127.0.0.1:5000


---

## 📸 Example Output
- Input: Leaf image
- Output: Disease prediction (e.g., Early Blight)
- Confidence score (e.g., 87%)

---

## 📌 Future Improvements
- Upgrade to Deep Learning (CNN)
- Improve accuracy with larger dataset
- Add mobile-friendly UI
- Deploy online (Render / Vercel)

---

## 👨‍💻 Author
Shiva Mani  
B.Tech CSE (AI/ML Enthusiast)

---

## ⭐ If you like this project
Give a star ⭐ on the repository
