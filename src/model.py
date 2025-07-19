# src/model.py
import torch
from torchvision import models
from .config import MODEL_PATH

def get_model2(num_classes):
    model = models.vgg16(weights='IMAGENET1K_V1')
    model.classifier[6] = torch.nn.Linear(4096, num_classes)
    return model

def get_model(num_classes):
    # ResNet18으로 변경
    model = models.resnet50(weights='IMAGENET1K_V1')  # 최신 torchvision
    # fc 레이어만 교체
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def save_model(model):
    torch.save(model.state_dict(), MODEL_PATH)

def load_model(num_classes, device):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model
