"""
Base Training Script for CropGuard
Shared preprocessing + utils for MobileNetV2 & EfficientNetB0
Dataset: PlantVillage/Tomato_* (4 classes)
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import pandas as pd

# Config - Matches Chapter 3.3-3.6
DATA_DIR = 'PlantVillage'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
NUM_CLASSES = 4
CLASS_NAMES = ['Tomato_healthy', 'Tomato_Bacterial_spot', 'Tomato_Septoria_leaf_spot', 'Tomato_Tomato_mosaic_virus']

def get_class_indices():
    """Print class mapping for app.py CLASSES dict"""
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    print("CLASS INDICES (update app.py CLASSES keys):")
    for i, cls in sorted(idx_to_class.items()):
        print(f"{i}: {{ \"name\": \"{cls.replace('_', ' — ')}\" }}")
    return class_to_idx, idx_to_class

def create_data_generators():
    """3.3 Data preprocessing + augmentation"""
    # Training augmentation (3.3 iii)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.15  # 70/15/15 split
    )
    
    # Validation: only rescale (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)
    
    # Load data
    train_gen = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        classes=CLASS_NAMES
    )
    
    val_gen = val_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        classes=CLASS_NAMES
    )
    
    print(f"Train batches: {train_gen.samples}, Val: {val_gen.samples}")
    return train_gen, val_gen

def plot_history(history, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title(f'{model_name} Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title(f'{model_name} Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_training_curves.png')
    plt.show()

if __name__ == '__main__':
    get_class_indices()

