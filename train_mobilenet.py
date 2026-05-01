"""
MobileNetV2 Training Script (Chapter 3.4 i, 3.5)
python train_mobilenet.py
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from train_base import create_data_generators, plot_history, get_class_indices, CLASS_NAMES, NUM_CLASSES

print("🚀 Training MobileNetV2 for CropGuard...")

# Data
train_gen, val_gen = create_data_generators()

# Model (3.4 i - depthwise separable convolutions)
base = MobileNetV2(
    input_shape=(*train_gen.image_shape[:-1], 3),
    include_top=False,
    weights='imagenet',
    pooling=None
)

# Freeze base layers initially (transfer learning 3.5 i)
base.trainable = False

# Custom head
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)  # 3.6 regularization
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base.input, outputs=predictions)

# Compile (3.5 iv)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks (3.5 v-vi)
callbacks = [
    ModelCheckpoint('mobilenet_best.h5', monitor='val_accuracy', save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=6, min_lr=1e-7)
]

# Train (3.5)
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=100,
    callbacks=callbacks
)

# Unfreeze for fine-tuning (3.5 ii)
base.trainable = True
for layer in base.layers[:100]:  # Fine-tune top layers
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-4),  # Lower LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=callbacks
)

# Final save
model.save('mobilenet_final.h5')
print("✅ MobileNetV2 saved: mobilenet_best.h5 + mobilenet_final.h5")
plot_history(history, 'MobileNetV2')


