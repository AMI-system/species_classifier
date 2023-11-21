"""
Author: Aditya Jain
Date  : August 3, 2022
About : Main function for building a model
"""
from models.resnet50 import Resnet50
from models.efficientnet import EfficientNet

from models.resnet50 import Resnet502
from models.efficientnet import EfficientNet2


def build_model(config):
    model_name = config["model"]["type"]

    if model_name == "resnet50":
        return Resnet50(config)
    elif model_name == "efficientnetv2-b3":
        return EfficientNet(config).get_model()
    else:
        raise RuntimeError(f"Model {config.model_name} not implemented")
        
def build_model2(num_classes: int, model_type: str):
    if model_type == "resnet50":
        return Resnet502(num_classes)
    elif model_type == "efficientnetv2-b3":
        return EfficientNet2(num_classes).get_model()
    else:
        raise RuntimeError(f"Model {model_type} not implemented.")