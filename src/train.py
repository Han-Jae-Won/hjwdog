# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import os
from torchvision import models
from src.dataset import get_data_loaders
from src.config import DATA_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, CLASS_NAMES_PATH, MODEL_PATH

def main():
    # 데이터 로딩 (품종당 30장 제한)
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        DATA_DIR, IMG_SIZE, BATCH_SIZE, limit_per_class=60
    )
    num_classes = len(class_names)
    print('num_classes==>', num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class CustomResNetClassifier(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            base = models.resnet50(weights='IMAGENET1K_V1')
            self.feature = nn.Sequential(*list(base.children())[:-1])
            in_features = base.fc.in_features
            self.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        def forward(self, x):
            x = self.feature(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    # ResNet50 모델 생성s
    # 기존 코드 수정
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    #model = CustomResNetClassifier(num_classes)
    #model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)

        # 검증
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch+1}: loss {avg_loss:.4f} / val acc {val_acc:.4f}")

    # 모델 저장
    torch.save(model.state_dict(), MODEL_PATH)

    # 클래스 이름 저장 (예측, Streamlit 등에서 활용)
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(list(class_names), f, ensure_ascii=False)
    print(f"모델 및 클래스 정보 저장 완료: {MODEL_PATH}, {CLASS_NAMES_PATH}")

if __name__ == "__main__":
    main()
