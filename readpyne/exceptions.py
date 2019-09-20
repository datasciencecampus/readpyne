"""
readpyne.exceptions
===========

A collection of domain specific exceptions.
These are here to clarify what went wrong with the
program.

"""


class NoPositivesFound(Exception):
    """
    An exception that indicates that no positively classified lines were found
    by the classifier in the image.
    """

    pass


class ImageLoadError(Exception):
    """
    Used when image or images fail to load.
    """

    pass


class EASTDecoderError(Exception):
    """
    Used when east decoder function fails
    """

    pass


class InvalidLengthError(Exception):
    """
    Used when a tuple or list of unsupported length is passed into a function.
    """
