# stdlib
import os
import tempfile

# third party
import cv2
import numpy as np
import pandas as pd

from sklearn.utils import validation

# testing
import pytest
from hypothesis import settings, given, strategies as st
from hypothesis.extra import numpy as hnp

# project
from readpyne import io
from readpyne.exceptions import *


def test_get_data():
    # test normal loading
    post = io.get_data("tests/test_resources")

    # test if list is returned
    assert isinstance(post, list)

    # test if entry in list is of type arrays
    assert isinstance(post[0], type(np.zeros(1)))

    # assert that if no image is loaded we get error
    with pytest.raises(ImageLoadError):
        with tempfile.TemporaryDirectory() as tmpdirname:
            io.get_data(tmpdirname)


@given(
    hnp.arrays(np.float, shape=(100, 100)),
    st.lists(hnp.arrays(np.float, shape=(100, 100))),
)
@settings(deadline=None)
def test_cutout_save(img, subsets):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # save
        io.cutout_save(tmpdirname, img, subsets)

        # check number of files in directory is sensible
        assert len(os.listdir(tmpdirname)) == 2 * len(subsets) + 1


@given(st.lists(hnp.arrays(np.float, shape=(100, 100))))
def test_save_images(image_list):
    with tempfile.TemporaryDirectory() as tmpdirname:

        # save
        io.save_images(image_list, path=tmpdirname)

        # check number of files saved makes sense
        assert len(os.listdir(tmpdirname)) == len(image_list)


@given(
    st.lists(hnp.arrays(np.float, shape=(100, 100))),
    st.lists(hnp.arrays(np.float, shape=(300,)), min_size=1),
)
def test_save_stack(subs, features):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # save
        io.save_stack(subs, features, tmpdirname)

        # assert length of outputs is sensible
        assert len(os.listdir(tmpdirname)) == len(subs) + 1

        # assert csv was saved
        df = pd.read_csv(f"{tmpdirname}/training.csv")

        assert df.shape == (len(features), 301)

        # assert it contains a labels column in the end
        assert "labels" in df.columns
        assert df.iloc[:, -1].name == "labels"


@given(
    st.lists(
        st.tuples(
            st.from_regex("\w+", fullmatch=True), hnp.arrays(np.float, shape=(50, 100))
        )
    )
)
def test_export_raw_ocr(ocr_results):
    with tempfile.TemporaryDirectory() as tmpdirname:
        path = os.path.join(tmpdirname, "tmpfile.txt")
        io.export_raw_ocr(ocr_results, filename=path)

        with open(path, "r") as handle:
            lines = handle.read().splitlines()
            assert lines == io._text(ocr_results)


def test_default_load_model():
    model = io.load_model("readpyne/models/classifier.pkl")

    # check fit method exists and predict exists
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert callable(getattr(model, "fit"))
    assert callable(getattr(model, "predict"))
