import os
import json
import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Settings ────────────────────────────────────────────────
MODEL_PATH       = "models/crop_disease_model.h5"
CLASS_NAMES_PATH = "models/class_names.json"
HISTORY_PATH     = "models/training_history.json"
TEST_PATH        = "data/processed/test"

# ── Set MLflow Tracking ─────────────────────────────────────
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("Crop-Disease-Detection")

print("Starting MLflow tracking...")

with mlflow.start_run(run_name="EfficientNetB0-Final"):

    # ── Log Parameters ───────────────────────────────────────
    print("Logging parameters...")
    mlflow.log_param("model",        "EfficientNetB0")
    mlflow.log_param("dataset",      "PlantVillage")
    mlflow.log_param("num_classes",  38)
    mlflow.log_param("image_size",   "224x224")
    mlflow.log_param("batch_size",   32)
    mlflow.log_param("optimizer",    "Adam")
    mlflow.log_param("learning_rate_phase1", 0.001)
    mlflow.log_param("learning_rate_phase2", 0.0001)
    mlflow.log_param("train_images", 43429)
    mlflow.log_param("val_images",   5428)
    mlflow.log_param("test_images",  5448)
    mlflow.log_param("total_images", 54305)
    mlflow.log_param("framework",    "TensorFlow 2.12")
    mlflow.log_param("deployment",   "Kubernetes + Docker")
    print("Parameters logged!")

    # ── Log Metrics from History ─────────────────────────────
    print("Logging training metrics...")
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            history = json.load(f)

        for i, (acc, val_acc, loss, val_loss) in enumerate(zip(
            history['accuracy'],
            history['val_accuracy'],
            history['loss'],
            history['val_loss']
        )):
            mlflow.log_metric("train_accuracy", acc,     step=i)
            mlflow.log_metric("val_accuracy",   val_acc, step=i)
            mlflow.log_metric("train_loss",     loss,    step=i)
            mlflow.log_metric("val_loss",       val_loss, step=i)

        # Log best metrics
        mlflow.log_metric("best_val_accuracy", max(history['val_accuracy']))
        mlflow.log_metric("best_val_loss",     min(history['val_loss']))
        print("Training metrics logged!")

    # ── Evaluate on Test Set ─────────────────────────────────
    print("Evaluating model on test set...")
    model = tf.keras.models.load_model(MODEL_PATH)

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        TEST_PATH,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

    # Log test metrics
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_loss",     test_loss)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Loss    : {test_loss:.4f}")

    # ── Log Performance Charts ───────────────────────────────
    print("Logging artifacts...")
    if os.path.exists("models/training_graphs.png"):
        mlflow.log_artifact("models/training_graphs.png")

    if os.path.exists("models/performance"):
        for f in os.listdir("models/performance"):
            mlflow.log_artifact(f"models/performance/{f}")

    # ── Log Model ────────────────────────────────────────────
    print("Logging model...")
    mlflow.keras.log_model(model, "crop-disease-model")

    # ── Log Tags ─────────────────────────────────────────────
    mlflow.set_tag("project",    "Crop Disease Detection")
    mlflow.set_tag("college",    "Atria Institute of Technology")
    mlflow.set_tag("department", "ISE")
    mlflow.set_tag("team",       "Shreenidhi, Suvarna, Shravan")
    mlflow.set_tag("status",     "Production")

    print("\n" + "="*50)
    print("MLFLOW TRACKING COMPLETE!")
    print("="*50)
    print(f"Test Accuracy : {test_accuracy*100:.2f}%")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Best Val Acc  : {max(history['val_accuracy'])*100:.2f}%")
    print("\nRun this to see MLflow dashboard:")
    print("mlflow ui")
    print("Then open: http://localhost:5000")
    print("="*50)