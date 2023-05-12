# On Device Species Classifier

This repository contains the code to:
- download image and metadata for moth species from GBIF (Global Biodiversity Information Facility)
- train a deep learning model using the downloaded data to classify species of moths.
- compress the model for deployment on device

The code is organized in the following way:

1. data_download: contains the code to download the training data from GBIF which is saved in the gbif_data directory.
1. model_training: contains the code to train the models
1. model_compression: contains the code to compress the trained model
