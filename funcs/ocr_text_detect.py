#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression


def detection_to_text(img_path, model_path, min_confidence=0.5, width=320, height=320):
    # Load image & grab dimensions
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        org = img.copy()
    else:
        raise Exception("Image not found: {}".format(img_path))

    # Determine ratio in change for image size
    h, w = img.shape[:2]
    w_ratio = w / float(width)
    h_ratio = h / float(height)

    # Resize image & grab new dimensions
    img = cv2.resize(img, (width, height))
    h, w = img.shape[:2]

    # Load pre-trained EAST text detector
    if os.path.exists(model_path):
        net = cv2.dnn.readNet(model_path)
    else:
        raise Exception("Model not found: {}".format(model_path))

    # Convert image to blob & obtain layer sets
    blob = cv2.dnn.blobFromImage(img, 1.0, (w, h), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)

    layers = [  # From model architecture
        "feature_fusion/Conv_7/Sigmoid",  # Probability of a region containing text or not
        "feature_fusion/concat_3"  # Feature map to derive bounding box coordinates
    ]
    scores, geometry = net.forward(layers)

    rects = []  # Bounding box coordinates for text regions
    confidences = []  # Probability associated with each bounding box

    # Grab rows and columns from scores to get potential bounding box coordinates
    num_rows, num_cols = scores.shape[2:4]
    for y in range(0, num_rows):
        x_0 = geometry[0, 0, y]
        x_1 = geometry[0, 1, y]
        x_2 = geometry[0, 2, y]
        x_3 = geometry[0, 3, y]
        data_angle = geometry[0, 4, y]
        data_score = scores[0, 0, y]

        for x in range(0, num_cols):
            # Ignore if probability is not sufficient
            if data_score[x] < min_confidence:
                continue

            # Compute offset factor to have valid coordinates as feature maps are 4x smaller than image
            x_offset, y_offset = (x * 4.0, y * 4.0)

            # Compute sin and cosine
            angle = data_angle[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # Use geometry volume to derive width and height of bounding box
            h = x_0[x] + x_2[x]
            w = x_1[x] + x_3[x]

            # Compute starting and ending coordinates for bounding box
            x_end = int(x_offset + (cos * x_1[x]) + (sin * x_2[x]))
            y_end = int(y_offset - (sin * x_1[x]) + (cos * x_2[x]))
            x_start = int(x_end - w)
            y_start = int(y_end - h)

            rects.append((x_start, y_start, x_end, y_end))
            confidences.append(data_score[x])

    # Apply Non-maxima Suppression to avoid overlapping bounding boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # Loop over bounding boxes
    for (x_starting, y_starting, x_ending, y_ending) in boxes:
        # Scale bounding box coordinates based on the ratios
        x_starting = int(x_starting * w_ratio)
        y_starting = int(y_starting * h_ratio)
        x_ending = int(x_ending * w_ratio)
        y_ending = int(y_ending * h_ratio)

        # Draw bounding box on image
        cv2.rectangle(org, (x_starting, y_starting), (x_ending, y_ending), (0, 255, 0), 2)

        # Region of interest
        roi = org[y_starting:y_ending, x_starting:x_ending]

    # Save result as image file
    filename = img_path.split(".")[0] + "-result.png"
    cv2.imwrite(filename, org)
    print("*-*-*\nDetection-To-Text Result for {}:\n\n{}".format(img_path, filename))


if __name__ == "__main__":
    print("DETECTS BOUNDARY BOXES ON A GIVEN IMAGE TO EXTRACT TEXT")
