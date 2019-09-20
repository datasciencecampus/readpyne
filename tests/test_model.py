# stdlib
import os
import tempfile as tf

from pkg_resources import resource_filename

# third party
import cv2
import numpy as np
import pandas as pd

# testing
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp

# project
from readpyne import io, model

dims = (100, 50, 3)
n, m, c = dims
img_strategy = hnp.arrays(np.uint8, (n, m, c), elements=st.integers(1, 255))
default_model = io.load_model(resource_filename("readpyne", "models/classifier.pkl"))


def test_make_training_data():
    with tf.TemporaryDirectory() as tmpdirname:
        # run with input folder
        res = model.make_training_data(
            "tests/test_resources", tmpdirname, interactive=False
        )
        # check res == true
        assert res == True

        contents = os.listdir(tmpdirname)

        # make sure the training.csv is created in there and it has
        # the right features associated with it
        assert "training.csv" in contents

        df = pd.read_csv(f"{tmpdirname}/training.csv")

        assert df.shape[1] == 301
        assert "labels" in df.columns


@given(
    hnp.arrays(
        np.float, shape=(100, 300), elements=st.floats(min_value=0.0, max_value=1.0)
    ),
    hnp.arrays(np.uint8, shape=(100,), elements=st.integers(min_value=1, max_value=1)),
)
def test_status(x, y):
    # just check it runs
    # Can throw warnings due to no predicted samples
    model.status("testing", x, y.reshape(-1, 1), default_model)


@given(
    st.lists(
        st.tuples(
            st.floats(min_value=1, max_value=20),
            st.floats(min_value=1, max_value=20),
            st.integers(min_value=0, max_value=1),
        ),
        min_size=100,
    )
)
@settings(deadline=None)
def test_train_model(df):
    df = pd.DataFrame(list(map(list, df)))
    with tf.TemporaryDirectory() as tmpdirname:

        # check if bits run
        post = model.train_model(df, grid_cv=2, grid_params={})
        post_no_frac = model.train_model(df, grid_cv=2, frac_test=0)

        prediction = post[0].predict(np.array([[1, 20], [12, 4], [12, 5]]))

        # check we get prediction for all 3 rows
        assert len(prediction) == 3


@given(
    st.lists(
        st.tuples(
            st.floats(min_value=1, max_value=20),
            st.floats(min_value=1, max_value=20),
            st.integers(min_value=0, max_value=1),
        ),
        min_size=130,
        max_size=150,
    )
)
@settings(deadline=None)
def test_plot_scaling(df):
    df = pd.DataFrame(list(map(list, df)))
    post = model.plot_scaling(df, plot=False)

    # check that the output is an array and it all runs
    assert isinstance(post, type(np.zeros(1)))
