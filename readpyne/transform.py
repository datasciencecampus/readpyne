"""
transform
========

Functions used for transforming images

"""

# stdlib
import os
import glob

# third party
import cv2
import numpy as np

# project
from readpyne.io import load_validate, get_data


def order_points(pts):
    """

    This function finds the corner points of a contour and adds them to a list

    Parameters
    ------------
    pts : numpy.ndarray
        array of coordinates that represent the lines of a rectangle

    Returns
    ------------
    list
        list containing the four corner points of the given rectangle

    """

    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    """

    This function calculates the maximum height and width of a given rectangle
    and applies a perspective transform over an image.

    Parameters
    ------------
    image : numpy.ndarray
        input image to apply the transform to

    pts : numpy.ndarray
        array of coordinates that represent the lines of a contour

    Returns
    ------------
    numpy.ndarray
        warped version of the input image

    """
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(round(widthA), round(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(round(heightA), round(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def grab_contours(edged, n_contours):
    """

    This function finds the contours of an image,
    checks which version of OpenCV is being used and changes the value of cnts respectively.
    Then it keeps the largest contour that has 4 corners and discards the rest.

    Parameters
    ------------
    edged : numpy.ndarray
        image that has been loaded with cv2 and edited to show edges.

    n_contours : int
        number of largest contours you are trying to extract.

    Returns
   ------------
    list
        list of the contours of receipts found by grab_contours

    """

    def _approx_contour(c):
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        return approx

    # find the contours in the edged image
    cnts = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # raise error if tuple isn't of length 2
    else:
        raise InvalidLengthError(
            "Contour tuples be of length 2. Please make sure you are using OpenCV v4"
        )

    # remove duplicate contours by filtering out negative numbers
    # returns all values in cnts that have a True value in areas
    cnts = list(filter(lambda c: cv2.contourArea(c, True) > 0, cnts))
    # sort the contours by size and take the top n_contours contours
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:n_contours]
    # loop over the contours
    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    return [a for a in map(_approx_contour, cnts) if len(a) == 4]


def crop_image(image, quality=2, multiple_receipts=False):
    """

    This function is responsible for cropping and transforming images to prepare them for use.

    Parameters
    ------------
    image : numpy.ndarray
        image extracted with io.load_validate

    quality : int
        custom quality setting. Lower = less detail but straighter lines.
        too high can confuse the edge detection. Defaults to 2.
        1 = low quality
        2 = medium quality
        3 = high quality

    multiple_receipts : bool
        specify whether you're extracting one or multiple receipts from an image.
        defaults to False.

    Returns
    ------------
    list
        An list containing numpy.ndarrays of cropped images.

    """

    def _process(img):
        # convert the image to grayscale, blur it, and find edges
        # in the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 50, 250, L2gradient=True)

        return edged

    # apply the four point transform to obtain a top-down
    # view of the original image
    def _four_point_pipe(image, contour):
        warped = four_point_transform(image, contour.reshape(4, 2) * ratio)
        (h, w) = warped.shape[0:2]
        bw = round(w / 50)
        bh = round(h / 50)

        return warped[bh : h - bh, bw : w - bw]

    if multiple_receipts:
        n_contours = 5
    else:
        n_contours = 1

    # define quality
    quality_dict = {1: 500, 2: 700, 3: 1000}
    quality = quality_dict[quality]
    # compute the ratio of the old height
    # to the new height and resize it
    ratio = image.shape[0] / quality
    image_new = cv2.resize(image, (round(image.shape[1] / ratio), quality))

    edged = _process(image_new)
    # apply the grab_contours function
    screenCnt = grab_contours(edged, n_contours)

    cropped = [_four_point_pipe(image, contour) for contour in screenCnt]
    # update current status
    print(f"[INFO] {len(cropped)} receipts cropped from image.")

    return cropped


def crop_images_from_folder(folder_path):
    """

    This function is responsible for cropping and transforming multiple images
    from a folder to prepare them for use.

    Parameters
    ------------
    folder_path : str
        string containing the path to a folder containing image files

    Returns
    ------------
    list
        list containing numpy.ndarrays of the cropped images

    """

    def _image_check(image):
        """
        Automatically check if the width of the image is greater than the height
        and assume that if that is the case, there are multiple receipts. 
        Returns a tuple with quality and boolean for multiple receipts
        """
        (h, w) = image.shape[:2]
        return (1, True) if w / h > 1.1 else (2, False)

    # load all images from folder_path
    images = get_data(folder_path, multiple_receipts=True)
    # loop through the images and check the dimensions
    # apply the crop_image function to get the cropped images
    cropped = [crop_image(img, *_image_check(img)) for img in images]
    return [img for crop in cropped for img in crop]
