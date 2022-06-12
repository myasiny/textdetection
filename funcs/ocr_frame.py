#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import imutils
import numpy as np
from imutils import contours


def credit_card_to_text(img_path, frame_path, process="thresh"):
    # Credit card types
    first_no = {"3": "American Express", "4": "Visa", "5": "MasterCard", "6": "Discover"}

    # Load frame & convert to grayscale
    if os.path.exists(frame_path):
        img_1 = cv2.imread(frame_path)
        frame = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    else:
        raise Exception("Frame not found: {}".format(frame_path))

    # Apply pre-processing to frame
    if process == "thresh":
        frame = cv2.threshold(frame, 10, 255, cv2.THRESH_BINARY_INV)[1]  # Threshold
    else:
        raise Exception("Pre-processing method not known: {}".format(process))

    # Find outlines, get tuple value based on OpenCV version & sort from left to right
    frame_conts = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_conts = imutils.grab_contours(frame_conts)
    frame_conts = contours.sort_contours(frame_conts, method="left-to-right")[0]

    # Loop through outlines, compute bounding box & extract digit
    digits = {}
    for (digit, contour) in enumerate(frame_conts):
        x, y, w, h = cv2.boundingRect(contour)  # Coordinates, width & height
        roi = frame[y:y+h, x:x+w]  # Region of interest
        roi = cv2.resize(roi, (57, 88))
        digits[digit] = roi

    # Initialize matrices to apply operations (ex. Top-hat Transform)
    kernel_rectng = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    kernel_square = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Load image & convert to grayscale
    if os.path.exists(img_path):
        img_2 = cv2.imread(img_path)
        img_2 = imutils.resize(img_2, width=300)
        gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    else:
        raise Exception("Image not found: {}".format(img_path))

    # Apply Top-Hat Transform to extract elements
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_rectng)

    # Apply Sobel Operator to detect edges
    sobel = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)

    # Apply Min-max Normalization to scale into 0-255
    sobel = np.absolute(sobel)
    min_val, max_val = (np.min(sobel), np.max(sobel))
    sobel = 255 * ((sobel - min_val) / (max_val - min_val))
    sobel = sobel.astype("uint8")

    # Apply Otsu to close gaps between digit areas & separate pixels into foreground and background
    sobel = cv2.morphologyEx(sobel, cv2.MORPH_CLOSE, kernel_rectng)
    thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Apply closing again to binary image & determine candidate outlines
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_square)

    # Find outlines
    img_conts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_conts = imutils.grab_contours(img_conts)

    # Loop through outlines, compute bounding box & extract digit locations
    locs = []
    for (i, c) in enumerate(img_conts):
        x, y, w, h = cv2.boundingRect(c)  # Coordinates, width & height
        ratio = w / float(h)  # Aspect ratio
        if 2.5 < ratio < 4.0:
            if (40 < w < 55) and (10 < h < 20):
                locs.append((x, y, w, h))

    # Sort from left to right
    locs = sorted(locs, key=lambda x:x[0])

    # Loop through each 4-digit groups
    output = []
    for (i, (g_x, g_y, g_w, g_h)) in enumerate(locs):
        group_output = []

        # Extract group region of interest of 4 digits
        group_roi = gray[g_y-5:g_y+g_h+5, g_x-5:g_x+g_w+5]
        group_roi = cv2.threshold(group_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Detect outlines of each digit
        digit_conts = cv2.findContours(group_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_conts = imutils.grab_contours(digit_conts)
        digit_conts = contours.sort_contours(digit_conts, method="left-to-right")[0]

        # Loop through digit outlines
        for c in digit_conts:
            # Compute bounding box of individual digit
            d_x, d_y, d_w, d_h = cv2.boundingRect(c)
            d_roi = group_roi[d_y:d_y+d_h, d_x:d_x+d_w]
            d_roi = cv2.resize(d_roi, (57, 88))

            scores = []
            for (digit, digit_roi) in digits.items():
                # Calculate score based on template matching
                match = cv2.matchTemplate(d_roi, digit_roi, cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(match)
                scores.append(score)

            group_output.append(str(np.argmax(scores)))

        output.extend(group_output)

    print("*-*-*\n")
    try:
        print("Credit-Card-To-Text Result for {}:\n\n{} #{}".format(img_path, first_no[output[0]], "".join(output)))
    except KeyError:
        print("Credit-Card-To-Text Result for {}:\n\n{} #{}".format(img_path, "Unknown", "".join(output)))
    except IndexError:
        raise Exception("Credit card number not detected: {}".format(img_path))


if __name__ == "__main__":
    print("EXTRACTS CARD NUMBER FROM A GIVEN IMAGE OF CREDIT CARD BY USING A TEMPLATE")
