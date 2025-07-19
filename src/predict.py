# src/predict.py
import torch
from torchvision import transforms
from PIL import Image
import json
from .model import load_model
from .config import CLASS_NAMES_PATH, IMG_SIZE

def load_class_names():
    with open(CLASS_NAMES_PATH, encoding='utf-8') as f:
        class_names = json.load(f)
    return class_names

def predict_image(img_path, topk=3):
    class_names = load_class_names()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(len(class_names), device)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_idxs = torch.topk(probs, topk)
    results = [(class_names[top_idxs[0][i]], top_probs[0][i].item()) for i in range(topk)]
    return results
