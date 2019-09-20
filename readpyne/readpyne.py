"""
readpyne
========

The key functions and tools that allow the end user to use the library with no fuss

"""
# stdlib
import os
import warnings
from operator import itemgetter
from pkg_resources import resource_filename

# third party
import cv2
import numpy as np
import toolz as fp
import pandas as pd

# project
from readpyne import core, io
from readpyne.ocr import ocr_textM
from readpyne.model import default_context
from readpyne.decorators import unfold_args, mute
from readpyne.exceptions import *  # this is safe as it only contains exceptions
from readpyne.text import _text, default_cleaner, extract_re


@mute
def shopscan(img, shops="(shop1|shop2|shop3)", header_frac=0.1, tail=False):
    """
    Takes in an image of a receipt, then based on a certain set of shop names
    and a header percentage will try to detect if the shopname is in the header.

    Useful for detecting shop type and such.

    Note
    ----
    Setting ``header_frac`` to either 1 or 0 will just run the whole receipt

    Parameters
    ----------
    img : numpy.ndarray
        image of the receipt

    shops : str
        regex string similary to ``(shop1|shop2|shop3)``. In other words,
        the shop has to be captured in a group

    header_frac : float
        a floating point number representing how much of the height to search

    tail : bool
        if ``True`` the function will look at a given fraction from the bottom
        instead of from the top

    Returns
    -------
    str
       the shop name that it found. if none are found, it returns ''
    """

    if header_frac:
        header_height = int(img.shape[0] * header_frac)
    else:
        header_height = img.shape[0]

    if not tail:
        header = img[:header_height, ...]
    else:
        header = img[-header_height:, ...]

    # pad header to make sure any shop names at the top are not cut off
    header = cv2.copyMakeBorder(
        header, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    header_lines = extract(header, override_prediction=True)
    text_lines = _text(ocr_textM(header_lines))
    shops = [extract_re(s.lower(), expr=shops, strict=True)[0] for s in text_lines]
    # filter empty strings and get the first one of the non empty list
    clean = [shop for shop in shops if shop]

    if clean:
        return clean[0]
    else:
        warnings.warn("[WARN] No shop found")
        return ""


def extract(
    image,
    classifier=None,
    context=default_context,
    output_folder=None,
    return_negatives=False,
    override_prediction=False,
):
    """
    extract(image, classifier, context=default_context, output_folder=None, return_negatives=False, override_prediction=False)
    A function utilised the core of the package to extract required lines and by
    default classifies the required and non-required lines.

    Note
    ----
    Needs refactoring.

    Parameters
    ----------
    image : np.array or str
        image as loaded by ``cv2.imread`` or string path to the image on disk

    classifier : sklearn model or str
        sklearn model for classification or a string to a pickled model
        loads the last trained model.
        Current default model is loaded if nothing else is provided.

    context : dict
        parameter dictionary which contains default settings
        for various functions
        # TODO: Write better summary of how to use this

    output_folder : str
        if provided will save the predicted lines

    override_prediction : bool
        if ``True`` then it overwrites any filtering done by the model and turns this
        into a regular pipeline of getting just the subsets

    expand : dict
        experimental feature. This will eventually accept a dictionary of parameters
        which will be trickled down into the core making testing easier. at the moment
        we can only change the vertical padding of the system

    Returns
    -------
    list | tuple
        a list of cutout lines in numpy array form if ``return_negatives`` is disabled,
        else a tuple containing both positive predictions and negatives (1s and 0s)

    """
    # logic for classifier assessment
    if classifier:
        if isinstance(classifier, str):
            classifier = io.load_model(classifier)
        else:
            classifier = classifier
    else:
        classifier = io.load_model(resource_filename("readpyne", "models/classifier.pkl"))

    pipe = fp.compose(
        unfold_args(core.features), fp.partial(core.boxes, context=context)
    )

    if isinstance(image, str):
        pipe = fp.compose(pipe, io.load_validate)

    subsets, features = pipe(image)

    # return the subsets raw without doing any other work
    if override_prediction:
        print(
            "[WARN] You have chosen not to use the classifier and hence full list of lines is returned"
        )
        return subsets

    # Use the model to predict
    prediction = classifier.predict(features)

    # get the zero and non-zero indices
    bindices_zero = prediction == 0
    zeros = np.arange(len(prediction))[bindices_zero]
    nonzeros = np.arange(len(prediction))[~bindices_zero]

    # Try to get the subsets that classify as non-zero
    try:
        positives = itemgetter(*nonzeros)(subsets.copy())
    except:
        raise NoPositivesFound("Could not get positive (1's) from subsets")

    # Make sure in the case of only 1 line found, we still return a list and
    # not an array.
    if not isinstance(positives, tuple) and isinstance(positives, type(np.zeros(1))):
        positives = (positives,)

    print(f"[INFO] {len(positives)} item lines found by the classifier")

    # output positives if this is provided
    if output_folder:
        io.save_images(positives, path=output_folder)

    # if required return negatives as a tuple
    if return_negatives:
        try:
            negatives = itemgetter(*zeros)(subsets.copy())

        except:
            raise Exception("Could not get 0's from subsets")

        if not isinstance(negatives, tuple) and isinstance(
            negatives, type(np.zeros(1))
        ):
            negatives = (negatives,)

        print(f"[INFO] {len(negatives)} non-item lines found by the classifier")
        # override positives to contain the final results
        positives = (positives, negatives)

    return positives


def extract_from_folder(folder, **kwargs):
    """
    A folder version of the ``extract`` function.

    Parameters
    ----------
    folder : str
        path the the folder from which to extract the images from

    **kwrags
        any other key word arguments to pass to the extract function.
        see ``readpyne.model.extract`` function documentation for more
        on which arguments are accepted

    Returns
    -------
    list
        list containing the results of applying the ``extract`` function on
        each image
    """
    images = io.get_data(folder)
    return [extract(im, **kwargs) for im in images]


def item_pipe(
    imgs, cleaner=default_cleaner, extractor=extract_re, expr="(.+?)(-?\d+.\d+$)"
):
    """
    This function abstracts a lot of work done by by ``ocr_textM`` and
    other function into one longer function that just takes in images of
    lines and exports a pandas dataframe.

    The opinionated parts of this pipeline are the regular expression
    (``expr``) that splits the string into item and price and the
    ``default_cleaner`` (see docs) function that is passed to the
    ``cleaner`` parameter by default. Both of which you can swap out.


    Note
    ----
    Currently the OCR picks up minus signs as ``~``. This is dealt with
    by substituting all occurances of ``~`` with ``-`` before using
    regular expressions to parse things. This is necessary to pick up
    negative values of price. This is performed by the ``default_cleaner``
    function.

    Parameters
    ----------
    imgs : list
        list of numpy arrays representing images.

    cleaner : function
        a function with the type signature ``String -> String`` that performs
        whatever cleaning is needed prior to splitting it by the regular expr.

    extractor : function
        a function with a signature ``String -> (String, String)`` that extracts
        the item and the price from a given string. by default such a function is
        provided (see docs for ``extract_re``)

    expr : str
        str of the regexp to pass into the ``expr`` parameter into the extractor

    Returns
    -------
    pandas.DataFrame
        pandas dataframe containing the extracted items and prices
    """
    clean = fp.partial(map, cleaner)
    get = fp.partial(map, fp.partial(extractor, expr=expr))

    pipe = fp.compose(pd.DataFrame, get, clean, _text, ocr_textM)

    df = pipe(imgs)
    df.columns = ["item", "price"]

    return df


def extras_pipe(
    imgs,
    expr={
        "date": "(\d{1,2}\/\d{1,2}\/\d{1,2} \d{1,2}:\d{1,2})",
        "shop": "(shop1|shop2)",
    },
    unique=True,
):
    """
    Pipeline function to extract the extra bits we need.

    In essence provide it with all the non-item lines and a set of regular expressions.
    It will then go through each expression for each image and see what it can extract.

    Note
    ----
    Under active development, might not be nice to use in this current API format, in
    which case this will be changed. This should be done by the NLP side to be fair, since
    its just regular expressions

    Parameters
    ----------
    imgs : list
        list of numpy arrays representing images.

    expr : dict[str:str]
        dictionary with a name for an extra item and its corresponding regular expression

    Returns
    -------
    dict
       dictionary with signature ``str:[str]``. each item in the dictionary is going to be
       a list of the same length as imgs with either found outputs from the regular expression
       or just an empty string. each key will be the same key as the ones within ``expr``

    """

    def _pick(xs):
        if unique:
            return list(filter(bool, set(xs)))
        else:
            return xs

    print("[INFO] Extracting extra information")
    texts = fp.compose(_text, ocr_textM)(imgs)
    clean = fp.compose(str.lower, str.strip)
    return {
        k: _pick([extract_re(clean(text), expr=r, strict=True)[0] for text in texts])
        for k, r in expr.items()
    }
