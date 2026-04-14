import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json

# ── Settings ───────────────────────────────────────────────
TRAIN_PATH     = "data/processed/train"
VAL_PATH       = "data/processed/val"
TEST_PATH      = "data/processed/test"
MODEL_SAVE_PATH = "models/crop_disease_model.h5"
HISTORY_PATH   = "models/training_history.json"

IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 10
NUM_CLASSES = 38

# ── Create models folder ────────────────────────────────────
os.makedirs("models", exist_ok=True)

# ── Data Generators ─────────────────────────────────────────
print("Loading data...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = val_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print(f"Train images: {train_generator.samples}")
print(f"Val images:   {val_generator.samples}")
print(f"Test images:  {test_generator.samples}")
print(f"Classes:      {len(train_generator.class_indices)}")

# ── Save class names ────────────────────────────────────────
class_indices = train_generator.class_indices
class_names   = {v: k for k, v in class_indices.items()}
with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)
print("Class names saved!")

# ── Build Model ─────────────────────────────────────────────
print("\nBuilding model...")

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model layers first
base_model.trainable = False

# Add our custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model built successfully!")
print(f"Total layers: {len(model.layers)}")

# ── Callbacks ───────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        verbose=1
    )
]

# ── Train Model ─────────────────────────────────────────────
print("\nStarting training...")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# ── Save Training History ───────────────────────────────────
history_dict = {
    'accuracy':     history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'],
    'loss':         history.history['loss'],
    'val_loss':     history.history['val_loss']
}
with open(HISTORY_PATH, "w") as f:
    json.dump(history_dict, f)
print("Training history saved!")

# ── Evaluate on Test Set ────────────────────────────────────
print("\nEvaluating on test set...")
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss:     {test_loss:.4f}")

# ── Plot Training Graphs ────────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('models/training_graphs.png')
print("Training graphs saved to models/training_graphs.png")

print("\nDay 3 Complete! Model trained successfully!")