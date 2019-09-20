"""
readpyne.io
===========

All input output functions

"""
# stdlib
import os
import pickle
import warnings

from multiprocessing import cpu_count
from multiprocessing.dummy import Pool


# third party
import cv2
import numpy as np
import toolz as fp
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

# project
import readpyne as rp
from readpyne.text import _text
from readpyne.exceptions import *
from readpyne.decorators import unfold_args, timeit


def load_validate(filename, multiple_receipts=False):
    """
    A thin wrapper on top of ``cv2.imread`` that makes sure the
    image is loaded and if its sideways.

    Parameters
    ----------
    filename : str
        path to the file

    multiple_receipts : bool
        specify whether the image is a single receipt
        or multiple receipts on one image.
        defaults to False.
 
    Returns
    -------
    np.array
        numpy array of the image

    """
    img = cv2.imread(filename)

    # validate it loaded
    if isinstance(img, type(None)):
        warnings.warn(f"file '{filename}' failed to load as an image", Warning)
        return np.array([])

    if not multiple_receipts:
        # validate that width is less than 10% larger than height (sideways)
        H, W = img.shape[:2]
        if W / H > 1.1:
            raise ImageLoadError(f"Sorry, file '{filename}' looks like its sideways")

        return img
    else:
        return img


def get_data(folder="data/training", jobs=cpu_count(), **kwargs):
    """
    Load all images in a folder.

    Parameters
    ----------
    folder : str 
        string path to where the images are.

    jobs : int
        number of threads to start for the loading
        default: this will be set to the number of cores
        although the documenation might state otherwise.
        will stop using threading if set to 0

    **kwargs
        anything passed into the keyword arguments will be
        passed onto the ``load_validate`` function

    Returns
    -------
    list
        list of images
    """

    def clean_empty(arrays):
        """Remove empty arrays"""
        return [array for array in arrays if array.shape != (0,)]

    print("[INFO] Loading images from folder")
    image_paths = [f"{folder}/{filename}" for filename in os.listdir(folder)]

    if jobs:
        print("[INFO] Running with multiple threads")
        with Pool(jobs) as pool:
            images = list(
                pool.map(
                    fp.partial(load_validate, **kwargs),
                    image_paths,
                )
            )
    else:
        images = [load_validate(path, **kwargs) for path in tqdm(image_paths)]

    # remove all the empty images
    images = clean_empty(images)

    # crash out if no images are loaded
    if not len(images):
        raise ImageLoadError("Sorry, no images were loaded from the folder")

    return images


def cutout_save(path, img, subsets):
    """
    Take an image and its subsets and export it to the given path.

    Parameters
    ----------
    path : str
        A path to be used to save all subsets.
    
    img : numpy array 
        A numpy array representing the image.

    subsets : list
        A list of numpy array of each subset.

    Returns
    -------
    None

    """
    # given an image and boxes, cut them out, create histograms and save
    # TODO: Throws a segmentation error? WTH? probably matplotlib or np
    print(f"[INFO] Saving: {path}")

    cv2.imwrite(f"{path}/img.jpg", img)

    for i, sub in enumerate(subsets):

        cv2.imwrite(f"{path}/{i}.jpg", sub)

        plt.plot(rp.core.hbin(sub), color="r")
        plt.savefig(f"{path}/{i}_histogram.jpg")
        plt.close()


def save_images(image_list, path="outputs/training", offset=0):
    """
    A list of images exports to a given folder.

    Parameters
    ----------
    image_list : list 
        A list of images represented as numpy.arrays
    
    path : string
        A path to the folder.

    offset : int
        this integer dictates by how many to shift the names of the
        resulting images. for example if ``offset`` is 2, the images
        will start being numbered from 2 onwards.

    Returns
    -------
    None
    """
    [
        cv2.imwrite(f"{path}/{i+offset}.jpg", img)
        for i, img in tqdm(enumerate(image_list))
    ]


def save_stack(subs, features, folder):
    """
    Get subsets and features and export them.

    Parameters
    ----------
    subs : list[np.array]
        list of numpy arrays representing subsets of the image.
    
    features : list[np.array]
        list of features.

    Returns
    -------
    subs
        Same as input
    features
        Same as input

    """
    print("[INFO] Exporting images:")
    save_images(subs, path=folder, offset=2)
    df = pd.DataFrame(features)
    df["labels"] = np.zeros(len(df))
    df.to_csv(f"{folder}/training.csv", index=False)
    return subs, features


@timeit
def interactive_labelling(subs, features, output_folder=None):
    """

    Function with an event loop to label the subsets of training images.
    
    Note
    ----
    This function is untested.

    Parameters
    ----------
    subs : list[np.array]
        list of image representing found lines in a receipt

    features : list[np.array]
        list representing the features for line classifier

    output_folder : str
        path to the output folder (without the final /) where the
        labeled data is to be saved.

    Returns
    -------
    pd.DataFrame
        if successful will return pandas dataframe with features and final
        column as `labels`

    """

    def _get_response(img):
        show(img)
        response = input("[INPUT] Was the image shown a correct line (1/0/u):")

        while response not in [1, 0, "u"]:
            try:
                response = int(response)
                if response not in [1, 0]:
                    print("[ERROR] Accepted numerical values are 1 or 0.")
                    response = input(
                        "[INPUT] Was the image shown a correct line (1/0):"
                    )
            except ValueError:
                print("[ERROR] Could not convert to integer. Use 1 or 0 only.")
                response = input("[INPUT] Was the image shown a correct line (1/0):")

        return response

    features = pd.DataFrame(features)
    features["labels"] = np.zeros(len(features))

    print(
        "[INFO] You will be shown images and asked to label them with `1` or `0`.",
        "[INFO] If you make an error, you can return to the previous image by entering 'u'.",
        "[INFO] You can quickly close the image window by pressing `q`.",
        sep="\n",
    )

    i = 0
    num_imgs = len(subs)
    while i < num_imgs:
        print(
            f"[INFO] You have completed {i} out of {num_imgs} images - {round(100*i/num_imgs,1)}%"
        )
        x = _get_response(subs[i])
        if x == "u" and i != 0:
            i -= 1
        elif x == "u" and i == 0:
            print("[WARN] This is the first image, cannot go back to previous image.")
        else:
            features.loc[i, "labels"] = x
            i += 1

    if output_folder:
        print(f"[INFO] Saving outputs to {output_folder}")
        features.to_csv(f"{output_folder}/training.csv", index=False)
    return features


def show(img):
    """
    Use the matplotlib pyplot function to show the image.

    Note
    ----
    This function is untested.

    Parameters
    ----------
    img : numpy array 
        A numpy array representing the image.

    Returns
    -------
    None

    """
    plt.imshow(img, cmap="gray")
    plt.show()


def export_raw_ocr(ocr_results, filename="ocr_results.txt"):
    """
    Export the result of the ocr line extraction provided by ``ocr.ocr_textM``.

    Parameters
    ----------
    ocr_results : list
        list of tuples containing a ``str`` of text and a ``numpy.array`` for the
        image.

    filename : str
        a string path to the while which will be used to save the ocr text.
        (default: ``"ocr_results.txt"``)

    Returns
    -------
    None

    """

    with open(filename, "w+") as handle:
        handle.write("\n".join(_text(ocr_results)))


def show_ocr(ocr_results):
    """
    Show ``OCR`` results from the ``ocr_textM`` function.

    Parameters
    ----------
    ocr_results : list
        list of tuples containing a ``str`` of text and a ``numpy.array`` for the
        image.

    Returns
    -------
    None
    """

    def _plot(ax, text, image):
        x.imshow(image, cmap="gray")
        x.set_title(text)

    fig, ax = plt.subplots(len(ocr_results), sharex=True)
    print("[INFO] Creating OCR plot")
    if len(ocr_results) == 1:
        x, (text, image) = ax, ocr_results[0]
        _plot(x, text, image)
    else:
        for x, (text, image) in tqdm(zip(ax, ocr_results)):
            _plot(x, text, image)

    plt.show()


def load_model(path):
    """
    Load a model from a string path.

    Parameters
    ----------
    path : str
        A path to the model

    Returns
    -------
    sklearn model

    """
    with open(path, "rb") as f:
        return pickle.load(f)
