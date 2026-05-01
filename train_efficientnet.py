"""
EfficientNetB0 Training Script (Chapter 3.4 ii, 3.5)
python train_efficientnet.py
"""
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from train_base import create_data_generators, plot_history, get_class_indices, CLASS_NAMES, NUM_CLASSES

print("🧠 Training EfficientNetB0 for CropGuard...")

# Data
train_gen, val_gen = create_data_generators()

# Model (3.4 ii - compound scaling)
base = EfficientNetB0(
    input_shape=(*train_gen.image_shape[:-1], 3),
    include_top=False,
    weights='imagenet',
    pooling=None
)

# Freeze base
base.trainable = False

# Head
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base.input, outputs=predictions)

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    ModelCheckpoint('efficientnet_best.h5', monitor='val_accuracy', save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=6, min_lr=1e-7)
]

# Initial training
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=100,
    callbacks=callbacks
)

# Fine-tuning
base.trainable = True
for layer in base.layers[:120]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=callbacks
)

model.save('efficientnet_final.h5')
print("✅ EfficientNetB0 saved: efficientnet_best.h5 + efficientnet_final.h5")
plot_history(history, 'EfficientNetB0')


