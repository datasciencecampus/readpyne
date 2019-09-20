# third party
import cv2
import numpy as np

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

# project
from readpyne import core, model

dims = (100, 50, 3)
n, m, c = dims
test_img = cv2.imread("tests/test_resources/test_receipt.jpeg")
img_strategy = hnp.arrays(np.uint8, (n, m, c), elements=st.integers(1, 255))


@given(img_strategy)
def test_process(img):
    post = core.process(img)

    # check type
    assert isinstance(post, type(np.zeros(1)))
    # make sure the dimensions don't get squashed
    assert img.shape[2] == 3
    assert img.shape == post.shape


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=100),
            st.integers(min_value=0, max_value=100),
            st.integers(min_value=120, max_value=255),
            st.integers(min_value=120, max_value=255),
        ),
        min_size=2,
    )
)
def test_expand(arr):
    # make the array by hand
    arr = np.array([list(a) for a in arr])

    shape = (1000, 200, 3)
    exp = core.expand(arr, shape, pad=(0.2, 0.3))

    # check type
    assert isinstance(arr, type(np.zeros(1)))
    # check shape being preserved
    assert arr.shape == exp.shape
    # check all the coords in x0 are 0
    assert (exp[:, 0] == 0).all()
    # check that all values in the x end col are the same
    assert (exp[:, 2] == shape[1]).all()
    # check the other dimensions are not the same
    assert (exp[:, 1] != arr[:, 1]).all()
    assert (exp[:, 3] != arr[:, 3]).all()


@given(img_strategy)
def test_hist(img):
    post = core.hist(img)

    # check type
    assert isinstance(post, list)

    # check len of return
    assert len(post) == img.shape[2]

    # check dimensions of return
    assert len(post[0].shape) == 1


@given(st.lists(hnp.arrays(int, 1000), min_size=3, max_size=3))
def test_binit(l_arrays):
    n_features = 100
    post = core.binit(l_arrays, n_features)

    # make sure we get back and array
    assert isinstance(post, np.ndarray)
    # check length of the output array
    assert len(post) == 3 * n_features


@given(img_strategy)
def test_resize(img):
    post = core.resize(img)

    a, b, c = img.shape
    x, y, z = post.shape

    # make sure its divisable by 32
    assert x % 32 == 0
    assert y % 32 == 0
    # make sure we're getting the same nuber of channels
    assert c == z


@given(img_strategy)
@settings(deadline=None)
def test_forward(img):
    blob = core.blobify(core.resize(img))
    scores, geo = core.forward(blob)

    # check type
    assert isinstance(scores, type(np.zeros(1)))
    assert isinstance(geo, type(np.zeros(1)))

    # enforce shape
    assert len(scores.shape) == 4
    assert len(geo.shape) == 4

    assert scores.shape[1] == 1
    assert geo.shape[1] == 5


@given(img_strategy)
def test_blobify(img):
    blob = core.blobify(img)

    old_shape = img.shape
    new_shape = (1, old_shape[2], old_shape[0], old_shape[1])

    assert isinstance(blob, type(np.zeros(1)))
    assert new_shape == blob.shape


@given(
    hnp.arrays(np.uint8, shape=(100, 50, 3)),
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=0),
            st.integers(min_value=0, max_value=50),
            st.integers(min_value=50, max_value=50),
            st.integers(min_value=55, max_value=100),
        ),
        min_size=1,
    ),
)
def test_get_subsets(img, boxes):
    boxes = np.array(list(map(list, boxes)))
    post = core.get_subsets(img, boxes)

    # check type
    assert isinstance(post, list)

    # check the number of channels is respected
    assert post[0].shape[2] == img.shape[2]


def test_boxes():
    context = model.default_context
    post, subsets = core.boxes(test_img, context)

    # check type
    assert isinstance(post, type(np.zeros(1)))
    assert isinstance(subsets, list)

    # check image is the same
    assert np.array_equal(core.resize(test_img), post)

    # check that the subsets have been expanded
    assert all([sub.shape[1] == post.shape[1] for sub in subsets])


@given(
    hnp.arrays(np.uint8, shape=(100, 50, 3)),
    st.lists(
        hnp.arrays(
            np.uint8,
            shape=(20, 50, 3),
            elements=st.integers(min_value=0, max_value=255),
        ),
        min_size=1,
    ),
)
@settings(deadline=None)
def test_features(img, subsets):
    subs, arrs = core.features(img, subsets)

    # check type
    assert isinstance(subs, list)
    assert isinstance(arrs, list)


@given(
    st.lists(st.tuples(st.lists(st.integers()), st.lists(st.integers())), min_size=1)
)
def test_stack(features):
    post = core.stack(features)

    # check type
    assert isinstance(post, tuple)

    # check if the output entries in tuple are lists
    assert all([isinstance(e, list) for e in post])
