"""
Evaluation Script (Chapter 3.7)
python evaluate.py --model mobilenet_best.h5
"""
import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from train_base import CLASS_NAMES

def evaluate_single_model(model_path, data_dir='PlantVillage'):
    model = load_model(model_path)
    
    # Test generator (deterministic, no shuffle)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False,  # Important for predictions
        classes=CLASS_NAMES
    )
    
    # Predictions
    print("🔮 Generating predictions...")
    preds = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    y_proba = np.max(preds, axis=1)
    
    # Metrics
    model_name = model_path.split('/')[-1].replace('.h5', '')
    print(f"\n{'='*60}")
    print(f"📊 {model_name.upper()} EVALUATION RESULTS")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    
    f1_macro = f1_score(y_true, y_pred, average='macro')
    acc = (y_true == y_pred).mean()
    print(f"\n🎯 SUMMARY:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {f1_macro:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, 
                yticklabels=CLASS_NAMES,
                square=True)
    plt.title(f'{model_name.upper()} Confusion Matrix\nAcc: {acc:.1%} | F1: {f1_macro:.3f}', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {'accuracy': acc, 'f1_macro': f1_macro}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to .h5 model')
    args = parser.parse_args()
    
    results = evaluate_single_model(args.model)
    print(f"\n✅ Saved: {args.model.split('/')[-1].replace('.h5', '')}_confusion_matrix.png")

if __name__ == '__main__':
    main()


