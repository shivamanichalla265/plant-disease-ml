import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

DATASET_PATH = "dataset"

def extract_features(image):
    image = cv2.resize(image, (128, 128))

    mean = np.mean(image, axis=(0,1))
    std = np.std(image, axis=(0,1))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_mean = np.mean(edges)

    return np.hstack([mean, std, edge_mean])

X, y = [], []
classes = sorted(os.listdir(DATASET_PATH))

print("Classes:", classes)

for label, folder in enumerate(classes):
    path = os.path.join(DATASET_PATH, folder)

    for img_name in os.listdir(path):
        img = cv2.imread(os.path.join(path, img_name))

        if img is not None:
            X.append(extract_features(img))
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

joblib.dump(model, "model/plant_model.pkl")
joblib.dump(classes, "model/classes.pkl")