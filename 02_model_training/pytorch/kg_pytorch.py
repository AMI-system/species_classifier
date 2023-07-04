from torchvision import transforms
import torch
from timm.layers import *
import os
import numpy as np
import tensorflow as tf
import csv
import json
from PIL import Image
import PIL
import onnx
from typing import Literal
from typing_extensions import Literal


from onnx_tf.backend import prepare

from data2 import dataloader

# Load in the pytorch model
model_py = torch.load("/bask/projects/v/vjgo8416-amber/projects/on_device_classifier/outputs/turing-macro_v01_efficientnetv2-b3_2023-06-27-10-45.pt", map_location='cpu')


# Load in the test data
test_dataloader = dataloader.build_webdataset_pipeline(
        sharedurl="../../../../data/gbif_macro_data/datasets/macro/test/test-500-{000000..000013}.tar",
        input_size=1000,
        batch_size=64,
        is_training=False,
        num_workers=4,
        preprocess_mode="tf",
    )
print("images loaded")

image_dummy, label_dummy = next(iter(test_dataloader))
image_batch = image_dummy.to("cpu", non_blocking=True)
image = image_batch[0]
image = image.unsqueeze(0)

# convert the pytorch model to TF Lite
reconvert_model = True

if reconvert_model:
    
    model = model_py.eval()
    #sample_data_np = np.transpose(image, (2, 0, 1))[np.newaxis, :, :, :]
    #sample_data_torch = torch.from_numpy(image)

    # Convert to onnx
    torch.onnx.export(
                model=model,
                args=image,
                f="/bask/projects/v/vjgo8416-amber/projects/on_device_classifier/outputs/onnx_file.onnx",
                #verbose=False,
                export_params=True,
                do_constant_folding=False,
                input_names=['input'],
                opset_version=12,
                output_names=['output']
    )


    # Convert to tf
    print("\n\nonnx to tf\n\n")
    onnx_model = onnx.load("/bask/projects/v/vjgo8416-amber/projects/on_device_classifier/outputs/onnx_file.onnx")
    onnx.checker.check_model(onnx_model)
    tf_rep = prepare(onnx_model, device='CPU')
    tf_rep.export_graph("/bask/projects/v/vjgo8416-amber/projects/on_device_classifier/outputs/tf_file")

    # Convert to tfLite
    converter = tf.lite.TFLiteConverter.from_saved_model("/bask/projects/v/vjgo8416-amber/projects/on_device_classifier/outputs/tf_file")
    tflite_model = converter.convert()
    with open("/bask/projects/v/vjgo8416-amber/projects/on_device_classifier/outputs/compressed_model.tflite", 'wb') as f:
        f.write(tflite_model)

    model = tf.lite.Interpreter(model_path="/bask/projects/v/vjgo8416-amber/projects/on_device_classifier/outputs/compressed_model.tflite")
    model.allocate_tensors()

    print(model)

print("\n\nThe model is converted!!\n\n")

def padding(image):
    """returns the padding transformation required based on image shape"""

    height, width = np.shape(image)[0], np.shape(image)[1]

    if height < width:
        pad_transform = transforms.Pad(padding=[0, 0, 0, width - height])
    elif height > width:
        pad_transform = transforms.Pad(padding=[0, 0, height - width, 0])
    else:
        return None

    return pad_transform


# Label info for the species names
f = open("/bask/projects/v/vjgo8416-amber/projects/on_device_classifier/02_model_training/pytorch/data2/01_uk_macro_data_numeric_labels.json")
label_info = json.load(f)
species_list = label_info["species_list"]
print(len(species_list), " species in total")





device="cpu"

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/bask/projects/v/vjgo8416-amber/projects/on_device_classifier/outputs/compressed_model.tflite")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()
print("tflite model loaded")

headers = ['True_label', 'Pytorch_prediction', 'TFLite_prediction']

f = open('myfile.csv', 'w', newline="")

# create the csv writer
writer = csv.writer(f, delimiter=';')
writer.writerow(headers)

for image_batch, label_batch in test_dataloader:


    image_batch, label_batch = image_batch.to(
        device, non_blocking=True
    ), label_batch.to(device, non_blocking=True)
    
    for i in range(len(image_batch)):
        image = image_batch[i]


        # For pytorch model
        outputs_py = model_py(image.unsqueeze(0))
        prediction_py = int(torch.max(outputs_py.data, 1)[1].numpy())
        
        interpreter.set_tensor(input_details[0]['index'], image.unsqueeze(0))
        interpreter.invoke()
        outputs_tf = interpreter.get_tensor(output_details[0]['index'])
        
        prediction_tf = np.squeeze(outputs_tf)
        prediction_tf = int(prediction_tf.argsort()[-1:][::-1])
        
        true_label = int(label_batch[i].numpy())
        print("true: ", true_label, species_list[true_label], ", "
            ", py: ", prediction_py, species_list[prediction_py], ", "
            ", tf: ", prediction_tf, species_list[prediction_tf])
        line = [str(int(true_label)), str(prediction_py), str(prediction_tf)]
        writer.writerow(line)

f.close()