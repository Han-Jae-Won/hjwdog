# dog_finder.py
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class DogFinder:
    def __init__(self, model_path, class_names, img_size=224):
        self.model = self._load_model(model_path, len(class_names))
        self.class_names = class_names
        self.img_size = img_size

    def _load_model(self, model_path, num_classes):
        from torchvision import models
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model

    def _preprocess(self, img):
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return transform(img).unsqueeze(0)

    def predict_breed(self, pil_img):
        x = self._preprocess(pil_img)
        with torch.no_grad():
            out = self.model(x)
            idx = torch.argmax(out, 1).item()
        return self.class_names[str(idx)]

    def find_breed_frames(self, video_path, target_breed, interval=30):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % interval == 0:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                breed = self.predict_breed(pil_img)
                if breed == target_breed:
                    frames.append((frame_num, pil_img))
            frame_num += 1
        cap.release()
        return frames
