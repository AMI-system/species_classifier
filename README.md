# Species Classifier
This repository contains the code to download image and metadata for moth species from GBIF ([Global Biodiversity Information Facility](https://www.gbif.org/)) and train a deep learning model using the downloaded data to classify species of moths.

The code is organized in the following way:
1. data_download: contains the code to download the training data from GBIF which is saved in the `gbif_data` directory.
2. model_training: contains the code to train the models


There are two models types: 
 1. using pytorch based on [work by the Rolnick Lab](https://github.com/RolnickLab/gbif-species-trainer), and
 2. using Tensorflow


