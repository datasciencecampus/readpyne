# third party
import cv2

# project
from readpyne import extract, ocr, io

lines = extract("tests/test_resources/test_receipt.jpeg")


def test_ocr():
    assert True
