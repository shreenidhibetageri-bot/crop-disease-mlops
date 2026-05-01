import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Settings ────────────────────────────────────────────────
MODEL_PATH      = "models/crop_disease_model.h5"
CLASS_NAMES_PATH = "models/class_names.json"
TEST_PATH       = "data/processed/test"
SAVE_PATH       = "models/performance"
os.makedirs(SAVE_PATH, exist_ok=True)

print("Loading model and data...")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# Load test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

print(f"Test images: {test_generator.samples}")
print(f"Classes: {len(class_names)}")

# ── Get Predictions ──────────────────────────────────────────
print("\nGenerating predictions...")
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# ── Calculate Metrics ────────────────────────────────────────
accuracy  = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes,
                           average='weighted', zero_division=0)
recall    = recall_score(true_classes, predicted_classes,
                        average='weighted', zero_division=0)
f1        = f1_score(true_classes, predicted_classes,
                    average='weighted', zero_division=0)

print("\n" + "="*50)
print("PERFORMANCE METRICS REPORT")
print("="*50)
print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")
print("="*50)

# Save metrics
metrics = {
    "accuracy":  round(accuracy*100, 2),
    "precision": round(precision*100, 2),
    "recall":    round(recall*100, 2),
    "f1_score":  round(f1*100, 2)
}
with open(f"{SAVE_PATH}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved!")

# ── Plot 1: Metrics Bar Chart ────────────────────────────────
print("\nGenerating performance charts...")
fig, ax = plt.subplots(figsize=(10, 6))
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metric_values = [accuracy*100, precision*100, recall*100, f1*100]
colors = ['#1d9e75', '#0f6e56', '#5dcaa5', '#085041']
bars = ax.bar(metric_names, metric_values, color=colors,
              width=0.5, edgecolor='white')
for bar, val in zip(bars, metric_values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{val:.2f}%', ha='center', va='bottom',
            fontweight='bold', fontsize=12)
ax.set_ylim(0, 115)
ax.set_title('Model Performance Metrics', fontsize=16,
             fontweight='bold', pad=20)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_xlabel('Metric', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SAVE_PATH}/1_performance_metrics.png", dpi=150)
plt.close()
print("✅ Performance metrics chart saved!")

# ── Plot 2: Confusion Matrix (Top 10 classes) ────────────────
print("Generating confusion matrix...")
top_classes_idx = list(range(min(10, len(class_names))))
mask_true = np.isin(true_classes, top_classes_idx)
mask_pred = np.isin(predicted_classes, top_classes_idx)
mask = mask_true & mask_pred

if mask.sum() > 0:
    cm = confusion_matrix(
        true_classes[mask],
        predicted_classes[mask],
        labels=top_classes_idx
    )
    top_class_names = [class_names[str(i)].split("___")[-1]
                       .replace("_", " ")[:15]
                       for i in top_classes_idx]
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=top_class_names,
                yticklabels=top_class_names,
                linewidths=0.5)
    plt.title('Confusion Matrix (Top 10 Classes)',
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Actual Disease', fontsize=11)
    plt.xlabel('Predicted Disease', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{SAVE_PATH}/2_confusion_matrix.png", dpi=150)
    plt.close()
    print("✅ Confusion matrix saved!")

# ── Plot 3: Training History ─────────────────────────────────
print("Generating training history chart...")
history_path = "models/training_history.json"
if os.path.exists(history_path):
    with open(history_path, "r") as f:
        history = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['accuracy'], label='Train',
             color='#1d9e75', linewidth=2)
    ax1.plot(history['val_accuracy'], label='Validation',
             color='#0f6e56', linewidth=2, linestyle='--')
    ax1.set_title('Model Accuracy Over Epochs',
                  fontweight='bold', fontsize=13)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history['loss'], label='Train',
             color='#e24b4a', linewidth=2)
    ax2.plot(history['val_loss'], label='Validation',
             color='#c0392b', linewidth=2, linestyle='--')
    ax2.set_title('Model Loss Over Epochs',
                  fontweight='bold', fontsize=13)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{SAVE_PATH}/3_training_history.png", dpi=150)
    plt.close()
    print("✅ Training history chart saved!")

# ── Plot 4: Per Class Accuracy ───────────────────────────────
print("Generating per class accuracy chart...")
class_accuracies = {}
for class_idx in range(len(class_names)):
    mask = true_classes == class_idx
    if mask.sum() > 0:
        correct = (predicted_classes[mask] == class_idx).sum()
        class_accuracies[class_names[str(class_idx)]] = \
            (correct / mask.sum()) * 100

sorted_acc = sorted(class_accuracies.items(),
                    key=lambda x: x[1], reverse=True)
top15 = sorted_acc[:15]

names = [c[0].split("___")[-1].replace("_", " ")[:20]
         for c in top15]
accs  = [c[1] for c in top15]
colors_bar = ['#1d9e75' if a >= 80 else
              '#f39c12' if a >= 50 else
              '#e24b4a' for a in accs]

plt.figure(figsize=(14, 7))
bars = plt.barh(names, accs, color=colors_bar)
for bar, val in zip(bars, accs):
    plt.text(val + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=9)
plt.title('Top 15 Classes by Accuracy\n'
          '(Green=Good, Orange=Medium, Red=Poor)',
          fontweight='bold', fontsize=13)
plt.xlabel('Accuracy (%)')
plt.xlim(0, 115)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SAVE_PATH}/4_per_class_accuracy.png", dpi=150)
plt.close()
print("✅ Per class accuracy chart saved!")

# ── Final Summary ────────────────────────────────────────────
print("\n" + "="*50)
print("PERFORMANCE ANALYSIS COMPLETE!")
print("="*50)
print(f"Accuracy  : {metrics['accuracy']}%")
print(f"Precision : {metrics['precision']}%")
print(f"Recall    : {metrics['recall']}%")
print(f"F1 Score  : {metrics['f1_score']}%")
print(f"\nCharts saved in: {SAVE_PATH}/")
for f in os.listdir(SAVE_PATH):
    print(f"  ✅ {f}")