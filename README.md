[![Build Status](https://travis-ci.com/datasciencecampus/readpyne.svg?token=er1Tq53jBBtypkGES25q&branch=master)](https://travis-ci.com/datasciencecampus/readpyne)

# readpyne

<img width="200" height="200" src="http://cdn.onlinewebfonts.com/svg/img_568517.png">

This repo contains a toolkit that allows one to feed in a number of shop receipts (or similar data) and after some training allows extraction of given lines from the receipts. For example if you were interested in extracting just the items bought from receipts this would be suitable for that purpose. 

The toolkit contains everything you need to:
- (a) create training data 
- (b) quickly label entries
- (c) train a model to classify lines 
- (d) ocr lines that vere extracted and filtered using the model 
- (e) pipe these steps together into a coherent pipeline once a model is trained

## Note on the state of the repo:
This repo is decomissioned, but functional. It has been developed and then superseeded with other tools. However it does contain value as a product and is shared for that purpose. The value here is the whole framework of doing this process and not specifically the efficiency of the OCR or Text detection. Forks are welcome to swap these parts of the library with more effective algorithms. 

Consequentially, the tests will not run as the `test_resources` folder has been removed as it contained resources that were not public. Furthermore the model that comes packed with the repository was trained on a specific set of images and will not do well at line extraction for arbitrary text. Please retrain a model for your tasks. 

# Installation

## Prerequisites
- python 3.7+
- tesseract 4+ (see [link](https://github.com/tesseract-ocr/tesseract))
- virtual environment (completely optional but recommended, see [link](https://pipenv.readthedocs.io/en/latest/))

## To install
In your terminal use: 
```sh
pip install git+ssh://git@github.com/datasciencecampus/readpyne.git@master
```
# Usage & documentation
For example of usage as well as documentation of the API please refer to:
https://datasciencecampus.github.io/readpyne/ 
