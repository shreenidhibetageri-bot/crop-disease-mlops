import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json

TRAIN_PATH      = "data/processed/train"
VAL_PATH        = "data/processed/val"
TEST_PATH       = "data/processed/test"
MODEL_SAVE_PATH = "models/crop_disease_model.h5"
HISTORY_PATH    = "models/training_history.json"

IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 32
NUM_CLASSES = 38

os.makedirs("models", exist_ok=True)

print("Loading data...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
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

print(f"Train : {train_generator.samples} images")
print(f"Val   : {val_generator.samples} images")
print(f"Test  : {test_generator.samples} images")

class_indices = train_generator.class_indices
class_names = {v: k for k, v in class_indices.items()}
with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)
print(f"Classes saved: {len(class_names)}")

print("\nBuilding model...")

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Total layers: {len(model.layers)}")

print("\nPhase 1: Training top layers...")

callbacks_p1 = [
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        verbose=1,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        verbose=1,
        min_lr=1e-7
    )
]

history1 = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=callbacks_p1,
    verbose=1
)

p1_best = max(history1.history['val_accuracy']) * 100
print(f"\nPhase 1 Best Accuracy: {p1_best:.2f}%")

print("\nPhase 2: Fine tuning base layers...")

base_model.trainable = True

for layer in base_model.layers[:100]:
    layer.trainable = False
for layer in base_model.layers[100:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_p2 = [
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        verbose=1,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        verbose=1,
        min_lr=1e-8
    )
]

history2 = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks_p2,
    verbose=1
)

p2_best = max(history2.history['val_accuracy']) * 100
print(f"\nPhase 2 Best Accuracy: {p2_best:.2f}%")

combined = {
    'accuracy':     history1.history['accuracy'] + history2.history['accuracy'],
    'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
    'loss':         history1.history['loss'] + history2.history['loss'],
    'val_loss':     history1.history['val_loss'] + history2.history['val_loss']
}
with open(HISTORY_PATH, "w") as f:
    json.dump(combined, f)
print("Training history saved!")

print("\nEvaluating on test set...")
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"\nFinal Test Accuracy : {test_accuracy * 100:.2f}%")
print(f"Final Test Loss     : {test_loss:.4f}")

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(combined['accuracy'],     label='Train', color='#1d9e75', linewidth=2)
plt.plot(combined['val_accuracy'], label='Val',   color='#0f6e56', linewidth=2, linestyle='--')
plt.axvline(x=len(history1.history['accuracy']), color='red', linestyle='--', label='Fine-tune start')
plt.title('Model Accuracy', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(combined['loss'],     label='Train', color='#e24b4a', linewidth=2)
plt.plot(combined['val_loss'], label='Val',   color='#c0392b', linewidth=2, linestyle='--')
plt.axvline(x=len(history1.history['loss']), color='red', linestyle='--', label='Fine-tune start')
plt.title('Model Loss', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('models/training_graphs.png', dpi=150)
print("Training graphs saved!")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print(f"Phase 1 Best : {p1_best:.2f}%")
print(f"Phase 2 Best : {p2_best:.2f}%")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print("="*50)
