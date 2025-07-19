# src/config.py
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'Images')
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 6
LEARNING_RATE = 1e-4
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'dog_breed_model.pth')
CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), '..', 'class_names.json')
