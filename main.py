# NOTE THIS WONT RUN AND IS JUST ILLUSTRATIVE
# Eventually this will be rewritten in sphinx to be part of docs in a nice format
# Fuller documentation is available in https://datasciencecampus.github.io/readpyne/


# third party
import pandas as pd

# project
import readpyne as rp

from readpyne import io
from readpyne import ocr
from readpyne import utils
from readpyne import model as m
from readpyne import transform as tr

# --------------------------------------------------------------------------------
# It is important to know what shop the receipt is from before moving on to any
# extraction. In order to check what shop it belongs to use the following function
shop = rp.shopscan(img_as_ndarray)

# This will work for a few shops only by default.

# --------------------------------------------------------------------------------
# Example of getting training data which can be used to train a better line
# classifier. Note: make sure that the folder only contains images
data = m.make_training_data(
    # Folder with receipt images that will be turned into training data
    input_folder="path/to/training_data_folder",
    # Where to save the cutout lines and the csv that needs to be labeled
    output_folder="path/to/training_output_folder",
    # This specifies if you want the new interactive behaviour to be enabled
    # True by default!
    interactive=True,
    # Context
    # This is a key parameter. The usage of this parameter is detailed in the other
    # instructions below
    context={"dictionary": "with", "specific": "configuration", "parameters": "!"},
)

# If interactive = True
# In the case you used interactive mode, your function call will return the training
# data as a dataframe. If you provided an output folder alongside interactive mode
# it will also save the training data in that folder.

# If Interactive == False
# You should then proceed to adjust the `label` column in the `training.csv` file
# and mark the lines that you want to extract as 1s. How do you correspond the csv
# to the subsets I hear you ask? Well each subset will be labeled from 2 to n.jpg
# Each line in the csv (except obviously the headings hence the labeling from 2)
# corresponds to a given subset of the image.

# NOTE ON CONTEXT:
# This is a functionality which allows tailoring the pipeline for a given shop.
# Most shops will have different padding hence this will be important to not only
# training data making but also to the process of extraction.

# I would recommend finding dictionaries of parameters that work well for each shop
# type and utilise the ``shopscan`` function to tailor the data generation and
# extraction parts of the package.

# To see the default context for this function do the following
from readpyne.model import default_context

print(default_context)

# --------------------------------------------------------------------------------
# Example of taking prelabelled data and training a model (see previous step on
# how to get data)
training_data = pd.read_csv("location/to/labeled/training_data.csv")

model, (X_test, y_test) = m.train_model(
    # Training features
    df=training_data,
    # Should you get a little bit of feedback on the f1-score and accuracy after
    # training
    report=True,
    # Where to save the trained sklearn model
    save_path="path/to/save/classifier/model.pkl"
    # sk_model & model_params can be passed see documentation
)

# --------------------------------------------------------------------------------
# Example of extracting lines from an image
# If no classifier path is provided a default classifier will be used
# which is only trained on one store type and has loose ground truth
# labels (see docs on how to provide a classifier from previous step)

# NOTE: As with the ``make_training_data`` function above, you can and should
# use the context dictionary for extraction. It would ideally be the same as the one
# used to train the model for a given shop. So the padding and other parameters
# are respected.

# From a image path:
imgs = rp.extract(input_image_as_str)
# From Image
imgs = rp.extract(input_img_loaded_with_io_load_validate)
# From a path to a folder
imgs = rp.extract_from_folder(folder_path)

# I've added aditional functionality!
# If you run the following command, you will get a tuple out (items, non-items).
imgs, non_items = rp.extract(input_see_above, return_negatives=True)

# Afterwards you can proceed to run the rest of the pipeline as normal on the 'imgs'
# However please see further down how to use the `non_items` to get date and shop name

# Please note! That if you specify `override_prediction = True` in the extract
# function, the line classification will be avoided and one can see just what the
# text line detection is finding in terms of lines.

# --------------------------------------------------------------------------------
# If you wanted to then use the lines found by the above code and to put them into
# a pandas dataframe, then you can simply use the code below. If you wanted more
# control, look at the code further down in the next section it will show how to get
# the text, extract it and save it.
df_with_items = rp.item_pipe(imgs)

# If you ran line 70 which created an extra variable `non_items` you can now run
# an extra bit of the pipeline (currently tuned only to only formats specific to a
# single shop). This pipeline will extract 2 things: date and shop name if it finds
# them.
date_shopname = rp.extras_pipe(non_items)

# --------------------------------------------------------------------------------
# EXTRAS
# --------------------------------------------------------------------------------
# Get text for each subset. So imgs here is the list of lines extracted by the # code so far.
text_and_images = ocr.ocr_textM(imgs)

# Quality of the whole pipeline
# If gold standard measurements are available, meaning you have a receipt
# that has been typed up and you have the required lines from it in a text
# file and/or list, you can use the quality metric provided to compare the
# recall of the filtering and the quality of the OCR on the recalled lines
# using the following code
utils.quality_metric(
    ["lines", "extracted", "by", "ocr"], ["correct", "lines", "expected"]
)

# Show the predicted text labels in a matplolib window. At the moment its not
# perfect as the labels overlap the images. But you can resize the window until it
# works.
# NOTE: exporting the labels somehow will be added in future versions
io.show_ocr(text_and_images)

# You can also export the text that it finds into a txt file.
io.export_ocr(text_and_images, filename="my_ocr_findings.txt")

# Below is the functionality to crop receipts from larger images, given that the
# contrast between the receipts and the background is sufficient.
tr.crop_image(img_as_ndarray)

# The quality settings determines the size of the image before detecting the edges.
# This can be either 1, 2 or 3. A higher is more precise but can cause problems with edge
# detection depending on the quality of the input image.
tr.crop_image(img_as_ndarray, quality=2)

# You can crop one or multiple receipts from an image depending on whether the parameter
# multiple_receipts = True or False.
# NOTE: It is advisable to use quality=1 when cropping multiple receipts from one image.
tr.crop_image(img_as_ndarray, quality=1, multiple_receipts=True)

# Below is a function to crop receipts from multiple images, both regular images and images
# with more than one receipts on them. It automatically predicts whether or not the image
# should have the multiple_receipts paramater set to True.
tr.crop_images_from_folder(folder_path)
