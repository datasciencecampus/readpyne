"""
readpyne.ocr
==============

The OCR functionality of the package

"""
# stdlib
import re

from multiprocessing import cpu_count
from multiprocessing.dummy import Pool

# third party
import cv2
import pytesseract
import numpy as np
import toolz as fp
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

# project


def ocr_img_preprocess(img):
    """
    This is the default preprocessor that goes before the ocr
    Current preprocessing (note: might forget to update this)

    1. Convert to grayscale
    2. Threshold (img, 0, 255, Binary Thresh Invert | Thresh Otsu)
    3. Median blur with size 3
    4. Dilate, kernel size (1,1), iterations = 1
    5. Bitwise not (restore normal colors)

    Parameters
    ----------
    img : np.array
        A numpy array representing an image (3 channels)

    """
    post = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    post = cv2.threshold(post, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    post = cv2.medianBlur(post, 3)
    post = cv2.dilate(post, np.ones((1, 1), np.uint8), iterations=1)
    post = cv2.bitwise_not(post)
    return post


def ocr_text(
    img,
    pyt_config="--psm 7 --oem 1",
    ret_preprocessed=True,
    preprocessor=ocr_img_preprocess,
):
    """
    Take in an image and after some preprocessing shove it into tesseract.
    Push through tesseract.

    Parameters
    ----------
    img : np.array
        A numpy array representing an image (3 channels)

    pyt_config : str
        Config to pass to pytesseract.

    ret_preprocessed : bool
        If True, this will return the image after preprocessing

    preprocessor : function
        function with a type signature of ``Image -> Image`` that will
        preprocess the image before ``OCR``

        by default the ``ocr_img_preprocess`` is used

    Returns
    -------
    tuple
        A tuple containing the extracted text and the image (original or preprocessed)
    """
    post = preprocessor(img)
    text = pytesseract.image_to_string(post, lang="eng", config=pyt_config)

    if ret_preprocessed:
        return text, post
    else:
        return text, img


def ocr_textM(imgs, jobs=cpu_count(), **kwargs):
    """
    A mapping function across a list of images for ``ocr_text``.
    Takes in a list of images and applies ocr_text.

    Note
    ----
    Does not return a ``map`` object. Uses a list comprehension so
    you get a ``list``.
    

    Parameters
    ----------
    imgs : list
        list of numpy arrays representing images.

    jobs : int
        integer telling how many processes to spin up to do the extraction.
        if ``0`` none will be spun up and it will be done without any concurrency
        by default set to number of cores (docs might say otherwise)

    **kwargs
        Any other keyword arguments you want to pass to ``ocr_text`` such as
        custom configs for tesseract and perhaps diverging preprocessing in the
        future.

    Returns
    -------
    list
        A list of tuples ``[(str, np.array)]``
    """
    print("[INFO] Extracting text from subsets")
    if jobs:
        print("[INFO] Running with multiple threads")
        with Pool(jobs) as e:
            return list(e.map(ocr_text, imgs))
    else:
        return [ocr_text(im, **kwargs) for im in tqdm(imgs)]
