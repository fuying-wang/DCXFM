import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .models.gloria_model import GLoRIA
from .models import IMAGE_MODELS
from .models import text_model



def build_gloria_model(cfg):
    gloria_model = GLoRIA(cfg)
    return gloria_model

def build_img_model(cfg):
    image_model = IMAGE_MODELS[cfg.phase.lower()]
    return image_model(cfg)


def build_text_model(cfg):
    return text_model.BertEncoder(cfg)