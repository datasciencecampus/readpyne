# stdlib
import re


def _text(ocr_results):
    """
    Helper function to extract text out of the ocr results

    Parameters
    ----------
    ocr_results : list
        list of tuples containing a ``str`` of text and a ``numpy.array`` for the
        image.

    Returns
    -------
    list
        list of strings
    """
    return [text for text, _ in ocr_results]


def default_cleaner(text):
    """
    Function that applies default cleaning before regex is applied in the
    item_pipe.

    Current cleaning:
        - replace ``~`` with ``-``

    Parameters
    ----------
    text : str
        string to be cleaned

    Returns
    -------
    str
       string after cleaning
    """
    text = text.replace("~", "-")
    return text


def extract_re(s, expr="(.+?)(-?\d+.\d+$)", strict=False):
    """
    This function extracts the item names and the prices.
    It uses a regular expression which seperates out the item and the
    price into two groups leaving you with a tuple of strings each
    representing required item name and price.

    Note
    ----
    At the moment if no price is found all the initial text is dumped
    into the item string and the price label is assigned ``Not Found``
    as the text.

    Parameters
    ----------
    s : str
        string to process

    expr : str
        the regular expression string to use

    Returns
    -------
    tuple
        a tuple of signature ``(str, str)`` containing the item and price
        if strict is true, if no price is found, the output will be ``(str,)``
    """
    search = re.search(expr, s)
    if search:
        return search.groups()
    elif not strict:
        return (s, "Not Found")
    else:
        return ("",)
