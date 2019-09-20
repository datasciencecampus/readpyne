# stdlib
import os
import cv2

# third party
import numpy as np
import pandas as pd

# project
import readpyne as rp

test_img = cv2.imread("tests/test_resources/test_receipt.jpeg")


def test_extract_lines():
    # test extraction from folder of images
    from_folder = rp.extract("tests/test_resources/test_receipt.jpeg")
    assert isinstance(from_folder, tuple)
    assert isinstance(from_folder[0], type(np.array([])))
    assert len(from_folder[0].shape) == 3

    img = cv2.imread("tests/test_resources/test_receipt.jpeg")
    from_cv2 = rp.extract(img)
    assert isinstance(from_cv2, tuple)
    assert isinstance(from_cv2[0], type(np.array([])))
    assert len(from_cv2[0].shape) == 3


def test_ocr_text_extract():
    imgs = rp.extract("tests/test_resources/test_receipt.jpeg")
    result = rp.ocr.ocr_textM(imgs)

    assert len(imgs) == len(result)
    assert isinstance(result[0][0], str)
    assert isinstance(result[0][1], type(np.zeros(1)))

    assert all([isinstance(e, str) for e in rp.text._text(result)])


def test_ocr_item_pipe():
    imgs = rp.extract("tests/test_resources/test_receipt.jpeg")
    items = rp.item_pipe(imgs)

    # check type
    assert isinstance(items, type(pd.DataFrame([1, 2, 3])))

    assert len(imgs) == len(items)
    assert len(items.columns) == 2


def test_extract():
    # check if it works through both types
    post = rp.extract(test_img)
    post = rp.extract("tests/test_resources/test_receipt.jpeg")

    # check type
    assert isinstance(post, tuple)
