# from evaluation.micro_accuracy_batch import MicroAccuracyBatch
# from evaluation.micro_accuracy_batch import add_batch_microacc, final_microacc
# from evaluation.macro_accuracy_batch import MacroAccuracyBatch
# from evaluation.macro_accuracy_batch import (
#     add_batch_macroacc,
#     final_macroacc,
#     taxon_accuracy,
# )
# from evaluation.confusion_matrix_data import confusion_matrix_data
# from evaluation.confusion_data_conversion import ConfusionDataConvert
# from torchvision import transforms
# import torch
import os
import numpy as np
import tensorflow as tf
import csv
import json
# from PIL import Image
# import PIL
# import onnx
# from typing import Literal
# from typing_extensions import Literal
# import matplotlib.pyplot as plt
# import timm
import wandb

wandb.init(
    project="gbif",
    entity="kg-test", 
    tags="tflite"
)

wandb.init(settings=wandb.Settings(start_method="fork"))

counter = 0
for image_batch, label_batch in test_dataloader:
    print('new batch', counter)
    image_batch, label_batch = image_batch.to(device), label_batch.to(device)

    for i in range(len(image_batch)):
        image = image_batch[i]
        label = label_batch[i]

        # For the tensorflow lite model
        interpreter.set_tensor(input_details[0]['index'], image.unsqueeze(0))
        interpreter.invoke()
        outputs_tf = interpreter.get_tensor(output_details[0]['index'])
        prediction_tf = np.squeeze(outputs_tf)
        prediction_tf = prediction_tf.argsort()[::-1]

        in_top_10_tf = 1 if int(label) in prediction_tf[0:10] else 0
        in_top_3_tf = 1 if int(label) in prediction_tf[0:3] else 0
        in_top_1_tf = 1 if int(label) == prediction_tf[0] else 0
        
        species_to_family_gt(taxon_hierar, ['Leucania loreyi'])[0]

        true_species = species_list[int(label)]
        pytorch_species = species_list[int(prediction_py1[0])]
        tflite_species = species_list[prediction_tf[0]]

        line = {'counter':counter, 'True_label':true_species, 'True_family':species_to_family_gt(taxon_hierar, [true_species])[0], 
                'True_genus':species_to_family_gt(taxon_hierar, [true_species])[0], 'True_label_index':str(int(label)),
                'TFLite_prediction':tflite_species,'TFLite_family':species_to_family_gt(taxon_hierar, [tflite_species])[0], 
                'TFLite_family':species_to_family_gt(taxon_hierar, [tflite_species])[0], 'TFLite_prediction_index':str(prediction_tf[0]), 
                'TFLite_top10':str(int(in_top_10_tf)), 'TFLite_top3':str(int(in_top_3_tf)), 'TFLite_top1':str(int(in_top_1_tf))
               }

        wandb.log(line)
        counter = counter + 1  
        
wandb.finish()