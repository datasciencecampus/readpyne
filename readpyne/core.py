"""
readpyne.core
=============

the core functionality for readpyne.

"""
# stdlib
import warnings

from pkg_resources import resource_filename

# third party
import cv2
import toolz as fp
import numpy as np
import pandas as pd
import pytesseract as pt
import matplotlib.pyplot as plt

from tqdm import tqdm
from imutils.object_detection import non_max_suppression

# project
from readpyne.io import show
from readpyne.exceptions import *
from readpyne.decorators import timeit, unfold_args

# processing for the image before historgrams are created
def process(img):
    """
    Function that performs preprocessing before histograms are created.
    
    Parameters
    ----------
    img : numpy.array
        A numpy array of shape (n,m,3) containing an image.
        Usually this will be a subset of a larger receipt.

    Returns
    -------
    numpy.array
        A numpy array representation of the input image after 
        all the processing was applied.
    
    """
    thresh = lambda arr: cv2.threshold(arr, 230, 255, cv2.THRESH_BINARY)[1]
    normalized = lambda arr: (arr - 255) / 255
    do = fp.compose(normalized, thresh)
    return do(img)


# expand the bounding boxes to the width of the image
def expand(arr, shape, pad):
    """
    Function to expand an array of image coordinates to the full width of 
    the image.

    Parameters
    ----------
    arr : numpy.array
        A two dimensional array of coordinates which are in the form of 
        ``startX, startY, endX, endY``

    shape : tuple
        A tuple containing the shape of the original receipt image.
        Easily accessible through the use of the ``.shape`` method on an
        image.

    Returns
    -------
    numpy.array
        A numpy array of the same shape as the original array but with
        the ``x`` values in each row expanded to the width of the image.

    """

    def _expand(row):
        h = row[3] - row[1]
        p1, p2 = int(h * pad[0]), int(h * pad[1])
        return np.array([0, row[1] - p1, shape[1], row[3] + p2])

    return np.apply_along_axis(_expand, 1, arr)


# make bins
def binit(l_arrays, n_features=100):
    """
    Takes in a list of 3 1d arrays of length ``n`` and firstly it bins it
    into a set number of features dictated by ``n_features``. This produces 
    an array of length ``n_features``. Then it stacks the three arrays 
    into a 1d array of length ``3 * n_features``

    Note
    ____

    This function does not check if the len of the input list of arrays is 3.

    Parameters
    ----------
    l_arrays : list
        A list of arrays of length ``n``. Usually this will be a vertically
        collapsed image that has been passed through ``cv2.split`` to split its 
        channels.

    n_features : int
        An integer telling the function how many features to produce per array.
        This dictates the shape of end array as the resulting array will have a 
        length of ``3 * n_features``
    
    Returns
    -------
    numpy.array
        A numpy array of length ``3 * n_features``
    """

    def _bin(arr):
        arr = pd.Series(arr)
        bins = pd.qcut(arr.index, n_features, labels=list(range(n_features)))
        return arr.groupby(bins).mean().values

    return np.hstack([_bin(array) for array in l_arrays])


def hist(img):
    """
    Histogram creation function. This function takes an input image and then 
    collapses it vertically by using ``np.mean``. It does this for each channel
    of the image as it uses ``cv2.split`` to get each channel.


    Parameters
    ----------
    img : numpy.array
        This is a numpy array representation of an input image. Expected shape
        for this array is ``(n,m,3)`` where ``n`` is the height and ``m`` is the 
        width of the image.

    Returns
    -------
    list
        A list of numpy arrays. Each array will be of length ``m`` where 
        ``m`` is the width of the input image.

    """
    return [channel.mean(axis=0) for channel in cv2.split(process(img))]


# a function to create histogram of an image and put into bins
hbin = fp.compose(binit, hist)


def resize(img):
    """
    This function resizes an input image to the correct dimensions for the 
    ``EAST`` model that is used for text detection. 

    The dimensions for ``EAST`` have to be divisable by 32. This function pads 
    the bottom and the right side of the image by a as many pixels as it needs
    for this to happen. 

    Note
    ----
    
    The image is padded with white as the color and this is then propogated to 
    the rest of the pipeline under normal circumstances.

    Parameters
    ----------
    img : numpy.array
        This is a numpy array representation of an input image. Expected shape
        for this array is ``(n,m,3)`` where ``n`` is the height and ``m`` is the 
        width of the image.

    Returns
    -------
    img : numpy.array
        The white padded image with dimensions now divisable by 32.

    """

    def dim32(n):
        d, m = divmod(n, 32)
        return 32 - m if m else 0

    def get_dims(img):
        return list(map(dim32, img.shape[:2]))

    bottom, right = get_dims(img)
    return cv2.copyMakeBorder(
        img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )


def decode(scores, geo, min_east_confidence):
    """

    This takes the geometries and confidence scores and produces bounding box values.
    The inputs to this function come from the EAST model with 2 layers. 

    Note
    ----
    THIS FUNCTION IS UNTESTED

    This function borrows heavily from: 
    https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
    
    Parameters
    ----------
    scores : numpy.array
        numpy array of size ``(number of found boxes, )`` indicating the assigned
        confidence scores for each box.

    geo : numpy.array
        numpy array coming out from EAST
        that describes where the boxes are.

    min_east_confidence : float
        confidence level at which to cut off the boxes

    Returns
    -------
    numpy.array
        A numpy array of shape ``(n, 5)`` containing the confidences and box locations
        for the boxes that are of a certain confidence.

    """
    # TODO: stop looping, start vectoring
    # TODO: filter before you do anything
    rects, confidences = [], []

    numRows, numCols = scores.shape[2:4]
    rects, confidences = [], []
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        d0 = geo[0, 0, y]
        d1 = geo[0, 1, y]
        d2 = geo[0, 2, y]
        d3 = geo[0, 3, y]
        ag = geo[0, 4, y]
        for x in range(0, numCols):

            if scoresData[x] < min_east_confidence:
                continue

            offsetX, offsetY = x * 4.0, y * 4.0

            angle = ag[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = d0[x] + d2[x]
            w = d1[x] + d3[x]

            endx = int(offsetX + (cos * d1[x]) + (sin * d2[x]))
            endy = int(offsetY - (sin * d1[x]) + (cos * d2[x]))

            startx = int(endx - w)
            starty = int(endy - h)

            rects.append((startx, starty, endx, endy))
            confidences.append(scoresData[x])

    if not (bool(len(confidences)) and bool(len(rects))):
        raise EASTDecoderError("no boxes detected at given confidence")

    return np.concatenate([np.array(confidences).reshape(-1, 1), np.array(rects)], 1)


def forward(blob, layers=["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]):
    """
    
    Take in a ``cv2 blob`` and pass it forward through an ``EAST`` model.

    Note
    ----
    The layers in the model by default are ``"feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"``

    Parameters
    ----------
    blob : cv2.blob
        image that has been ``blobified`` by cv2.
        (see the blobify function documentation)

    layers : list[str]
        list of string indicating which layers of the NN to pass the image through

    Returns
    -------
    numpy.array
        Scores array for each box found.
    
    numpy.array
        Bounding box locations from EAST

    """
    net = cv2.dnn.readNet(resource_filename("readpyne", "models/EAST.pb"))
    net.setInput(blob)
    return net.forward(layers)


def blobify(img):
    """
    
    Take in an image and pass it through ``cv2.dnn.blobFromImage``

    Parameters
    ----------
    blob : cv2.blob
        An image that has been ``blobified`` by cv2.
        (see the cv2.dnn.blobFromImage function documentation)
    
    Returns
    -------
    np.array

    """
    H, W = img.shape[:2]
    return cv2.dnn.blobFromImage(img, 1.0, (W, H), crop=False)


def get_subsets(img, boxes):
    """
    
    Take an image and box locations. Then cut out these boxes from the given image.
    
    Parameters
    ----------
    img : numpy.array
        numpy array representation of an image.

    boxes : iterable
        iterable (most likely a numpy.array) containing box coordinates.
        
    Returns
    -------
    list
        list of subsets.
    """

    def _cut(shape):
        return img[shape[1] : shape[3], shape[0] : shape[2], ...]

    return [_cut(box) for box in boxes]


def boxes(img, context):
    """

    Take in an image, resize it, predict boxes. Then perform 
    expansion of the boxes to the width of the receipt and then 
    perform ``non_max_supression``. 
 
    Parameters
    ----------
    img : numpy.array
        A numpy array representation of an image.

    context : dict
        parameter dictionary which contains default settings
        for various functions
        # TODO: Write better summary of how to use this

    Returns
    -------
    np.array
        The original image

    list
        Predicted subsets for the image.
    """
    img = resize(img)
    east_decode = unfold_args(fp.partial(decode, **context["boxes"]))
    arr = fp.compose(east_decode, forward, blobify)(img)

    rects, conf = arr[:, 1:], arr[:, 0]
    boxes = non_max_suppression(
        expand(rects, img.shape, **context["expand"]), probs=conf
    )

    # preserve order by sorting on startx
    sorted_boxes = pd.DataFrame(boxes).sort_values(1).values

    return img, get_subsets(img, sorted_boxes)


boxesM = fp.partial(map, boxes)


def features(img, subsets):
    """
    Take an image and its subsets created from ``boxes`` and 
    produce histogram based features for each subset. 

    Parameters
    ----------
    img : numpy.array
        numpy array representation of an image.

    subsets : list
        list of numpy arrays of the subsets.

    Returns
    -------
    list
        subsets originally passed into the function

    list
        list of 1d numpy arrays
    """

    def _valid(sub):
        predicate = all(sub.shape)
        return predicate

    # filter out any subsets that have a 0 dimension
    subsetsFiltered = list(filter(_valid, subsets))

    # user feedback
    print(f"[INFO] {len(subsets)} subsets found")
    diff = len(subsets) - len(subsetsFiltered)
    if diff:
        print(f"[WARN] Due to 0 dimensions {diff} subset(s) were removed.")
    print("[INFO] Creating features from subsets")

    return subsetsFiltered, [hbin(sub) for sub in tqdm(subsetsFiltered)]


featuresM = fp.partial(map, unfold_args(features))


def stack(features):
    """
    Stack features. Basically take in a list containing
    tuples of subsets and histogram based features and 
    stack them all up. 

    Parameters
    ----------
    features : list
        List of type ``[([subsets], [features]), ...]``

    Returns
    -------
    tuple
        tuple of type ``([all_subsets],[all_features])``
    """

    def _stack(entry1, entry2):
        return (entry1[0] + entry2[0], entry1[1] + entry2[1])

    return fp.reduce(_stack, features)
