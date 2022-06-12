#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import pytesseract
from PIL import Image
from pdf2image import convert_from_path


def pdf_to_jpg(pdf_name, pdf_path=""):
    files = []
    page_no = 1
    for page in convert_from_path(pdf_path + pdf_name):
        filename = pdf_name.split(".")[0] + str(page_no) + ".jpg"
        page.save(pdf_path + filename, "JPEG")
        files.append(filename)
        page_no += 1
    return files


def img_to_text(img_path, process="thresh"):
    # Load image & convert to grayscale
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise Exception("Image not found: {}".format(img_path))

    # Apply pre-processing
    if process == "blur":
        img_gray = cv2.medianBlur(img_gray, 3)  # Blur
    elif process == "thresh":
        img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # Threshold
    else:
        raise Exception("Pre-processing method not known: {}".format(process))

    # Save temporarily, load as PIL image & delete temporary save
    filename = "temporary.png"
    cv2.imwrite(filename, img_gray)
    result = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)

    print("*-*-*\nImage-To-Text Result for {}:\n\n{}".format(img_path, result))


if __name__ == "__main__":
    print("EXTRACTS ALL TEXT FROM A GIVEN IMAGE OR PDF FILE")
