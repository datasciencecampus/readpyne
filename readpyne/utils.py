"""
readpyne.utils
==============

A small selection of utility functions

"""
# third party
import numpy as np
from textacy.similarity import levenshtein


def build_distance_matrix(iter1, iter2, function=levenshtein):
    """
    Build a distance matrix given two lists of text and a function used
    to compare the entries

    If iter1 is of length ``m`` and iter2 is of length ``n`` then the
    resulting matrix is ``m x n``

    Parameters
    ----------

    iter1 : list
        iterable of any kind but developed with lists

    iter2 : list
        iterable of any kind but developed with lists

    function : callable
        a callable (function) that takes in 2 positional arguments and
        returns a value. mapped across a cartesian product of iterables
        to return a matrix

    Returns
    -------

    numpy.ndarray

    """
    return np.array([[function(t2, t1) for t2 in iter2] for t1 in iter1])


def fuzzy_compare_iter(
    iterable1, iterable2, function=levenshtein, threshold=0.8, fill_val="N/A"
):
    """
    Used to compare two lists (or iterables). Their comparison will be based on the values
    outputed for each pair by the ``function`` given.

    The idea is to first create a distance matrix using the function as the comparator,
    then retrieve any pairs that pass a given value threshold. If a value from the first 
    iterable doesn't have anything in the second iterable that passed the threshold
    the fill value will be returned


    The method:
    x = [str] of length n
    y = [str] of length m

    (1) compute an n x m matrix of levenshtein distances ``D`` between each string
        in ``x`` and each string in ``y``.

    (2) for each row from ``i`` to ``n`` in the resulting matrix D, map across each
        row and find the maximum value

    (3) if the maximum value is not greater than ``threshold``, value of a new variable
        ``z`` will be set ``fill_val``, if the value is larger than ``threshold`` then
        the column index is returned.

    (4) finally any column indices generated in step (3) are turned into the corresponding
        values from ``y`` and the ``fill_val`` indicating missing rows will be propagated

    Example
    -------
    a = [ 1, 2, 3 ]
    b = [ 0, 0, 100 ]
    distance = lambda x, y: y - x

    fuzzy_compare_iter(a, b, function=distance, threshold=0, fill_val="MISSING")

    Out:
        ["MISSING", "MISSING", 97]

    Parameters
    ----------

    iterable1 : list | iterable
        something to use as a basis. the length of the return will be based on the
        length of this iterable

    iterable2 : list | iterable
        essential similar to iterable1, but will be used more as lookup

    function : callable
        callable function that takes 2 postional arguments which will provide a score
        as this is developed for text, it is set to the levenshtein distance

    threshold : numeric
        value to compare the distance matrix by

    fill_val
        anything passed into this function will be the placeholder that will be passed
        if a value for iterable1 has nothing to close enough in iterable2

    Returns
    -------
    list
    """
    # TODO: Can this break if np.argmax returns two locations of maximum value?
    arr = build_distance_matrix(iterable1, iterable2, function=function)
    idx = np.apply_along_axis(
        lambda row: -1 if np.max(row) < threshold else np.argmax(row), 1, arr
    )
    return [iterable2[index] if index != -1 else fill_val for index in idx]


def quality_metric(lines, gold):
    """
    Compare a list of text lines to the ``gold standard`` (ground truth)
    for a given receipt. This metric takes into account both the character
    level OCR quality and the recall of lines.

    The key method here is that of the ``fuzzy_compare_iter`` function with
    a levenshtein (edit) distance and 2 lists of strings.

    By using ``fuzzy_compare_iter`` and then filtering values with a placeholder
    we can then check how many rows were recalled 'close enough' (using the default)
    threshold value in ``fuzzy_compare_iter``. TODO: Add parametrisation of this fn.

    Once we have the lines we recalled, we then use levenstein distance to compare them
    and use that as our precision metric.

    Finally the number of lines recalled 'close enough' divided by the number of lines
    in the gold standard is our ``recall`` approximation and the precision of is measured
    by how well the characters in each of the recalled lines match using levenshtein
    distance across the whole string.

    To get the final metric we use a harmonic mean of precision and recall.

    Parameters
    ----------
    lines : list[str]
        list of text strings

    gold : list[str]
        list of text strings representing ground truth

    Returns
    -------
    float
        an approximation of a makeshift F1-score

    """
    filt = lambda tup: tup[1] != "N/A"
    edit = lambda tup: levenshtein(tup[0], tup[1])

    matched = list(
        filter(
            filt,
            zip(
                lines,
                fuzzy_compare_iter(lines, gold, function=levenshtein, fill_val="N/A"),
            ),
        )
    )

    precision = np.mean(list(map(edit, matched)))
    recall = len(matched) / len(gold)
    if precision and recall:
        return 2 / ((1 / precision) + (1 / recall))

    else:
        raise ZeroDivisionError(
            "recall or precision is 0, can't compute the final score"
        )
