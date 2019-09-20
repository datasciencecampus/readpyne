"""
readpyne.model
==============

Model training, prediction and data making functions

"""
# stdlib
import pickle

# third party
import numpy as np
import toolz as fp
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# project
from readpyne import core, io

default_context = {"expand": {"pad": (0.2, 0.3)}, "boxes": {"min_east_confidence": 0.4}}


def make_training_data(
    input_folder, output_folder=None, interactive=True, context=default_context
):
    """
    Make training data from a folder of images.

    Parameters
    ----------
    input_folder : str
        Folder where the images are stored

    output_folder : str
        Folder where will the data will be saved. If not provided then
        data won't be saved and will be just returned.

    interactive : bool (default: True)
        If ``True`` will enable a interactive labelling mode where images
        shown one by one and the user is asked to label them 1 or 0 one by
        one. If ``output_folder`` is provided this will be respected.

    Returns
    -------
    bool or pd.DataFrame
        if the data was interactively labelled, the labelled dataframe will
        be returned, else a boolean will be returned indicating successfull
        completion and saving of the training data.

    """
    images = io.get_data(folder=input_folder)
    print("[Info] Extracting subsets for training data")
    subs, features = fp.compose(
        core.stack,
        core.featuresM,
        fp.partial(map, fp.partial(core.boxes, context=context)),
    )(images)

    if interactive:

        labelled = io.interactive_labelling(subs, features, output_folder)
        return labelled

    else:

        if output_folder:
            print(f"[INFO] Saving outputs to {output_folder}")
            io.save_stack(subs, features, output_folder)

        return True


def status(name, x, y, model):
    """
    This function is responsible for reporting the quality of the model.

    Parameters
    ----------
    name : str
        A string that will be the title of report

    x : numpy.array
        A numpy array with training features.

    y : numpy.array
        A numpy array with labels
    
    model : sklearn model
        A model to be scored

    Returns
    -------
    None

    """
    print(f"\n-- {name}")
    print(f"confusion: {confusion_matrix(y, model.predict(x)).ravel()}")
    print(f"    order: tn, fp, fn, tp")
    print(f"      acc: {model.score(x, y)}")
    print(f"       f1: {f1_score(y, model.predict(x))}")


def train_model(
    df,
    report=False,
    save_path=None,
    frac_test=0.25,
    sk_model=KNeighborsClassifier,
    model_params={"n_neighbors": 2},
    grid_params={"n_neighbors": [2, 3, 4, 5]},
    grid_cv=5,
    scaling_plot=False,
):
    """
    Given a set of data an labels. Train a sklearn model.

    Parameters
    ----------
    df : pd.DataFrame (or str)
        the dataframe containing the training features and labels
        anything until the last column as features (``df.iloc[:,:-1]``)
        and the last column is the labels (``df.iloc[:,-1]``)

        if a string is provided, it will attempt to load the dataset from
        the ``.csv`` file

    report : bool
        A boolean that tells you if you need the reporting procedure to run.

    save_path : str
        A path to save the model to

    frac_test : float
        A float indicating the amount of data to keep for testing.

    sk_model : sklearn model object
        sklearn model that will be trained. An instance of it will be created and then 
        trained.
    
    model_params : dict
        A dict of parameters to be passed to the sklearn model.

    grid_params : dict or None
        parameter dictionary for the GridSearchCV class in sklearn (see its documentation)
        for more information. If provided this enables gridsearch to be done for best
        performing hyperparameters

    grid_cv : int
        specifies how many times crossvalidation is to be done in GridSearchCV
        (ignored if grid_params is set to None)

    scaling_plot : bool
        if ``True`` will plot the scaling of the given model with increase in data
        for this to work, ``report`` has to be set to True
    
    Returns
    -------
    sklearn model
        Trained sklearn model

    tuple
        A tuple containing the untouched test data ``(X_test, y_test)``

    """

    if isinstance(df, str):
        df = pd.read_csv(df)

    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=frac_test)

    model = sk_model(**model_params)

    # check if gridsearch parameters are provided and make a gridsearch classifier
    if grid_params:
        print(f"[INFO] Performing gridsearch on these parameters: {grid_params}")
        model = GridSearchCV(model, grid_params, cv=grid_cv)
        model.fit(X_train, y_train)
        model = model.best_estimator_

    else:
        model.fit(X_train, y_train)

    if report:
        status("train", X_train, y_train, model)
        status("test", X_test, y_test, model)
        if scaling_plot:
            plot_scaling(pd.concat([X, y], axis=1), plot=True, model_type=sk_model)

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(model, f)

    return model, (X_test, y_test)


def plot_scaling(df, plot, model_type=KNeighborsClassifier, splits=100):
    """
    Plot the scaling of data given a sklearn classifier and a training dataset.

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe containing the training feature columns and labels. The last column
        should be the label column.

    plot : bool
        if `True` this will plot the data before returning

    model_type : sklearn classifier
        this has to be a classifier following the sklearn API (uses ``.fit`` and ``.predict``)

    splits : int
        number of splits to do of your data

    Returns
    -------
    np.array
        numpy array containing the ``size of data``, ``accuracy`` score and ``f1`` score.
        This is the data that is used to plot the chart internally.
    """

    def _show(arr):
        plt.plot(arr[:, 0], arr[:, 1], label="accuracy")
        plt.plot(arr[:, 0], arr[:, 2], label="f1")
        plt.legend()
        plt.show()
        return arr

    def _split(df):
        size = int(len(df) / splits)
        res = [df]
        for i in range(splits - 1):
            df = df.iloc[size:]
            res.append(df)

        return res

    def _get_score(df):

        features, labels = df.iloc[:, :-1], df.iloc[:, -1]

        (X_train, X_test, y_train, y_test) = train_test_split(
            features, labels, test_size=0.25, random_state=1993
        )

        model = KNeighborsClassifier(n_neighbors=2)
        model.fit(X_train, y_train)

        return (
            len(X_train),
            model.score(X_test, y_test),
            f1_score(model.predict(X_test), y_test),
        )

    # get data splits
    splits = _split(df)

    # get scores for each dataset size
    print("[INFO] Getting scores for each of the data splits")
    scores = np.array([_get_score(split) for split in tqdm(splits)])

    if plot:
        print("[INFO] Plotting the scores")
        return _show(scores)
    else:
        return scores
