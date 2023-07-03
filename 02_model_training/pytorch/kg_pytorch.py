from torchvision import transforms
import torch
from timm.layers import *
import os
import numpy as np
import tensorflow as tf
import csv
import json

from data2 import dataloader

model_py = torch.load("/bask/projects/v/vjgo8416-amber/projects/on_device_classifier/outputs/turing-macro_v01_efficientnetv2-b3_2023-06-27-10-45.pt", map_location='cpu')

from PIL import Image
import PIL

reconvert_model = False

if reconvert_model:
    # Load in the pytorch model
    model = torch.load("/bask/projects/v/vjgo8416-amber/projects/on_device_classifier/outputs/turing-macro_v01_efficientnetv2-b3_2023-06-27-10-45.pt", map_location='cpu')
    model = model.eval()

    # Convert to onnx
    torch.onnx.export(
                model=model,
                args=sample_data_torch,
                f="/bask/projects/v/vjgo8416-amber/projects/on_device_classifier/outputs/onnx_file.onnx",
                #verbose=False,
                #export_params=True,
                #do_constant_folding=False,
                input_names=['input'],
                opset_version=12,
                output_names=['output']
    )


    # Convert to tf
    print("onnx to tf")
    import onnx
    from onnx_tf.backend import prepare
    onnx_model = onnx.load("/bask/projects/v/vjgo8416-amber/projects/on_device_classifier/outputs/onnx_file.onnx")
    onnx.checker.check_model(onnx_model)
    tf_rep = prepare(onnx_model, device='CPU')
    tf_rep.export_graph("/bask/projects/v/vjgo8416-amber/projects/on_device_classifier/outputs/tf_file")

    # Convert to tfLite
    converter = tf.lite.TFLiteConverter.from_saved_model("/bask/projects/v/vjgo8416-amber/projects/on_device_classifier/outputs/tf_file")
    tflite_model = converter.convert()
    with open("/bask/projects/v/vjgo8416-amber/projects/on_device_classifier/outputs/compressed_model.tflite", 'wb') as f:
        f.write(tflite_model)
        
    import tensorflow as tf 

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


# image_path = "/bask/homes/f/fspo1218/amber/data/gbif_macro_data/gbif_macro/Abraxas/Geometridae/Abraxas grossulariata/3906809928.jpg"

# # create a dictionary
# label_list = {
#   "genus_list": "Abraxas",
#   "family_list": "Geometridae",
#   "species_list": "Abraxas grossulariata"
# }

# img_resize=1000

# not_found_img = 0
# corrupt_img = 0

# if not os.path.isfile(image_path):
#     print(f"File {image_path} not found")
#     not_found_img += 1


# # check issue with image opening; completely corrupt
# try:
#     image = Image.open(image_path)
#     image = image.convert("RGB")
# except PIL.UnidentifiedImageError:
#     print(f"Unidentified Image Error on file {image_path}")
#     corrupt_img += 1
# except OSError:
#     print(f"OSError Error on file {image_path}")
#     corrupt_img += 1

# print(image)


# print("\n\nThe image is loaded!!\n\n")

# padding_transform = padding(image)
# if padding_transform:
#     image = padding_transform(image)
    
# print(image)
    
# print("\n\nThe image is padded!!\n\n")

# transformer = transforms.Compose([transforms.Resize((img_resize, img_resize))])

    


# # check for partial image corruption
# try:
#     image = transformer(image)
# except ValueError:
#     print(f"Partial corruption of file {image_path}")
#     corrupt_img += 1

# print(image)

# print("\n\nThe image is transformed!!\n\n")

# species_list = label_list["species_list"]
# label = label_list["species_list"]

# print(label)

# print("\n\nDone my image stuff!!\n\n")


# img = image

# img = (np.float32(img)) 
# print(img.shape)

# # img = np.expand_dims(img, axis=0)
# # imga = np.array(np.random.random_sample(img), dtype=np.uint8)

# print(img.shape)

#input = trans(img)

# input = input.view(1, 3, 1000,1000)


label_info = "/bask/projects/v/vjgo8416-amber/projects/on_device_classifier/02_model_training/pytorch/data2/01_uk_macro_data_numeric_labels.json"


# read the json
with open(label_info, 'r') as myfile:
    data=myfile.read()

species = data[2]
print(species)




test_dataloader = dataloader.build_webdataset_pipeline(
        sharedurl="../../../../data/gbif_macro_data/datasets/macro/test/test-500-{000000..000013}.tar",
        input_size=1000,
        batch_size=64,
        is_training=False,
        num_workers=4,
        preprocess_mode="tf",
    )

device="cpu"

# Load the TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path="/bask/projects/v/vjgo8416-amber/projects/on_device_classifier/outputs/compressed_model.tflite")

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# interpreter.allocate_tensors()

headers = ['True_label', 'Pytorch_prediction', 'TFLite_prediction']

# f = open('myfile.csv', 'w', newline="")

# # create the csv writer
# writer = csv.writer(f, delimiter=';')
# writer.writerow(headers)

for image_batch, label_batch in test_dataloader:


    image_batch, label_batch = image_batch.to(
        device, non_blocking=True
    ), label_batch.to(device, non_blocking=True)
    
    #image = image_batch[0]

    # For pytorch model
    outputs_py = model_py(image_batch) #.unsqueeze(0))
    print("i")
    
    # interpreter.set_tensor(input_details[0]['index'], image_batch)#.unsqueeze(0))
    # interpreter.invoke()

    # # The function `get_tensor()` returns a copy of the tensor data.
    # # Use `tensor()` in order to get a pointer to the tensor.
    # outputs_tf = interpreter.get_tensor(output_details[0]['index'])


    
    # prediction_py = int(torch.max(outputs_py.data, 1)[1].numpy())
    # prediction_tf = np.squeeze(outputs_tf)
    # prediction_tf = prediction_tf.argsort()[-1:][::-1]

    # print("true: ", label_batch[0])
    
    # line = [str(label_batch[0]), str(prediction_py), str(prediction_tf)]
    # writer.writerow(line)

# f.close()