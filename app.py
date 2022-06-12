#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from funcs import ocr
from funcs import ocr_frame
from funcs import ocr_text_detect
from funcs import ocr_object

__author__ = ["Muhammed Yasin Yildirim"]
__credits__ = ["Adrian Rosebrock"]


if __name__ == "__main__":
    ocr.img_to_text(img_path="data/test-1.png")
    ocr.img_to_text(img_path="data/test-2.png", process="blur")

    pages = ocr.pdf_to_jpg(pdf_name="test-5.pdf", pdf_path="data/")
    ocr.img_to_text(img_path="data/{}".format(pages[0]), process="blur")

    ocr_frame.credit_card_to_text(img_path="data/test-3.png", frame_path="data/template-1.png")
    ocr_frame.credit_card_to_text(img_path="data/test-4.png", frame_path="data/template-1.png")

    ocr_text_detect.detection_to_text(img_path="data/test-6.png", model_path="data/model-1.pb")
    ocr_text_detect.detection_to_text(img_path="data/test-7.png", model_path="data/model-1.pb")

    ocr_object.train_mlp(train_path=["data/money/1kr/",
                                     "data/money/5kr/",
                                     "data/money/10kr/",
                                     "data/money/25kr/",
                                     "data/money/50kr/",
                                     "data/money/1tl/"], model_path="data/")
    ocr_object.money_to_text(img_path="data/test-8.png", model_path="data/model-2.pkl")
