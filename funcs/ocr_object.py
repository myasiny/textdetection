#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


def train_mlp(train_path, model_path):
    # X and y sets
    X, y = [], []

    for folder in train_path:
        for filename in os.listdir(folder):
            # Check file extension
            if filename.split(".")[1].lower() not in ("png", "jpg", "jpeg"):
                continue

            # Load image
            img_path = folder + filename
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
            else:
                raise Exception("Image not found: {}".format(img_path))

            # Normalize image
            mask = np.zeros(img.shape[:2], dtype="uint8")
            w, h = int(img.shape[1] / 2), int(img.shape[0] / 2)
            cv2.circle(mask, (w, h), int(min(w, h) / 2), 255, -1)
            hist = cv2.calcHist([img], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            flat = cv2.normalize(hist, hist).flatten()

            X.append(flat)
            y.append(folder.split("/")[-2])

    model = MLPClassifier(solver="lbfgs")

    # Train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model & calculate accuracy
    model.fit(X_train, y_train)
    print("*-*-*\nTrain-MLP Model Score: {}%".format(model.score(X_test, y_test) * 100))

    pickle.dump(model, open(model_path + "model-2.pkl", "wb"))


def money_to_text(img_path, model_path):
    # Load image
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
    else:
        raise Exception("Image not found: {}".format(img_path))

    # Resize image
    d = 1024 / img.shape[1]
    dim = (1024, int(img.shape[0] * d))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    org = img.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    histe = clahe.apply(gray)

    # Apply Gaussian & detect circles
    blur = cv2.GaussianBlur(histe, (7, 7), 0)
    circ = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100)

    # Load model
    if os.path.exists(model_path):
        model = pickle.load(open(model_path, "rb"))
    else:
        raise Exception("Model not found: {}".format(model_path))

    amount = 0
    materials = []
    coordinates = []
    if circ is not None:
        # Loop over circle coordinates and diameters
        circ = np.round(circ[0, :]).astype("int")
        for (x, y, d) in circ:
            coordinates.append((x, y))

            roi = img[y - d:y + d, x - d:x + d]

            # Normalize image
            mask = np.zeros(roi.shape[:2], dtype="uint8")
            w, h = int(roi.shape[1] / 2), int(roi.shape[0] / 2)
            cv2.circle(mask, (w, h), d, 255, -1)
            hist = cv2.calcHist([roi], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            flat = cv2.normalize(hist, hist).flatten()

            # Predict label
            material = model.predict([flat])
            materials.append(material)

            # Draw boundary & put label
            cv2.circle(org, (x, y), d, (0, 255, 0), 2)
            cv2.putText(org, material[0], (x - 40, y), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2, cv2.LINE_AA)

            # Sum amount
            if material == "1kr":
                amount += 0.01
            elif material == "5kr":
                amount += 0.05
            elif material == "10kr":
                amount += 0.10
            elif material == "25kr":
                amount += 0.25
            elif material == "50kr":
                amount += 0.50
            elif material == "1tl":
                amount += 1.00

    # Combine original and processed images
    d = 768 / org.shape[1]
    dim = (768, int(org.shape[0] * d))
    image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    output = cv2.resize(org, dim, interpolation=cv2.INTER_AREA)

    cv2.putText(output, "Total amount: {}".format(amount), (5, output.shape[0] - 24), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2, cv2.LINE_AA)

    filename = img_path.split(".")[0] + "-result.png"
    cv2.imwrite(filename, np.hstack([image, output]))
    print("*-*-*\nMoney-To-Text Result for {}:\n\n{}".format(img_path, filename))


if __name__ == "__main__":
    print("DETECTS MONEY ON A GIVEN IMAGE AND CALCULATES TOTAL AMOUNT")
