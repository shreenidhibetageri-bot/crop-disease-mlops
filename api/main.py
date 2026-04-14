import os
import json
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import tensorflow as tf
from PIL import Image
import io

# ── App Setup ───────────────────────────────────────────────
app = FastAPI(
    title="Crop Disease Detection API",
    description="API for detecting crop diseases using Deep Learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Load Model ───────────────────────────────────────────────
MODEL_PATH      = "models/crop_disease_model.h5"
CLASS_NAMES_PATH = "models/class_names.json"

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("Model loaded successfully!")
print(f"Total classes: {len(class_names)}")

# ── Helper Function ──────────────────────────────────────────
def preprocess_image(image_bytes):
    """Convert uploaded image to model input format"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ── Routes ───────────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "message": "Crop Disease Detection API is running!",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health":  "/health",
            "docs":    "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model":  "loaded",
        "classes": len(class_names)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a crop leaf image and get disease prediction
    """
    try:
        # Read uploaded image
        image_bytes = await file.read()

        # Preprocess image
        img_array = preprocess_image(image_bytes)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0])) * 100

        # Get class name
        predicted_class = class_names[str(predicted_index)]

        # Check if healthy or diseased
        if "healthy" in predicted_class.lower():
            status = "Healthy"
        else:
            status = "Diseased"

        # Format the result nicely
        plant_name  = predicted_class.split("___")[0].replace("_", " ")
        disease_name = predicted_class.split("___")[1].replace("_", " ") if "___" in predicted_class else "Unknown"

        return {
            "status":       "success",
            "plant":        plant_name,
            "disease":      disease_name,
            "health_status": status,
            "confidence":   f"{confidence:.2f}%",
            "raw_class":    predicted_class
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/classes")
def get_classes():
    """Get all disease classes the model can detect"""
    return {
        "total_classes": len(class_names),
        "classes": list(class_names.values())
    }