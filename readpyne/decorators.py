"""
readpyne.decorators
===================

A few helpful decorators for the rest of the code.

"""
# stdlib
import sys
import time
import tempfile
from functools import wraps

# timeit decorator
def timeit(fn):
    """
    A simple time decorator. Its not as cool as timeit.
    Only runs once.

    Parameters
    ----------
    fn : function 
        A function to be timed

    Returns
    -------
    function
    """

    @wraps(fn)
    def timed(*args, **kw):
        ts = time.time()
        result = fn(*args, **kw)
        te = time.time()
        print(f"[TIME] {fn.__name__.upper()} took {te-ts:.6f} seconds")
        return result

    return timed


# pattern match
def unfold_args(fn):
    """
    Take in a function that takes in positional arguments
    and makes it so you can pass in an unfoldable iterable as
    the singular argument which then gets unfolded into the
    positional arguments.

    Parameters
    ----------
    fn : function 

    Returns
    -------
    function
    """

    @wraps(fn)
    def _wrapped(to_expand):
        return fn(*to_expand)

    return _wrapped


# redirect standard output into a tmpfile
def mute(fn):
    @wraps(fn)
    def _wrapped(*args, **kwargs):
        original = sys.stdout
        with tempfile.TemporaryFile(mode="w+") as tmpfile:
            sys.stdout = tmpfile
            result = fn(*args, **kwargs)
            sys.stdout = original
        return result

    return _wrapped
