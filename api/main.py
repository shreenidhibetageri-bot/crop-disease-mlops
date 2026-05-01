import os
import json
import time
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import tensorflow as tf
from PIL import Image
import io
from src.dosage import get_dosage
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)

# ── Prometheus Metrics ───────────────────────────────────────
REQUEST_COUNT = Counter(
    'crop_disease_requests_total',
    'Total number of prediction requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'crop_disease_request_duration_seconds',
    'Request duration in seconds',
    ['endpoint']
)

PREDICTION_COUNT = Counter(
    'crop_disease_predictions_total',
    'Total predictions by disease class',
    ['disease', 'status']
)

MODEL_ACCURACY = Gauge(
    'crop_disease_model_accuracy',
    'Current model accuracy'
)

HEALTHY_COUNT = Counter(
    'crop_disease_healthy_total',
    'Total healthy plant predictions'
)

DISEASED_COUNT = Counter(
    'crop_disease_diseased_total',
    'Total diseased plant predictions'
)

# Set model accuracy
MODEL_ACCURACY.set(97.76)

# ── App Setup ───────────────────────────────────────────────
app = FastAPI(
    title="Crop Disease Detection API",
    description="API for detecting crop diseases with dosage recommendations",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Load Model ───────────────────────────────────────────────
MODEL_PATH       = "models/crop_disease_model.h5"
CLASS_NAMES_PATH = "models/class_names.json"

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("Model loaded successfully!")
print(f"Total classes: {len(class_names)}")

# ── Middleware for tracking all requests ─────────────────────
@app.middleware("http")
async def track_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(
        endpoint=request.url.path
    ).observe(duration)

    return response

# ── Helper Function ──────────────────────────────────────────
def preprocess_image(image_bytes):
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
        "version": "2.0.0",
        "accuracy": "97.76%",
        "endpoints": {
            "predict":  "/predict",
            "health":   "/health",
            "metrics":  "/metrics",
            "docs":     "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status":   "healthy",
        "model":    "loaded",
        "accuracy": "97.76%",
        "classes":  len(class_names)
    }

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        start_time = time.time()

        # Read and preprocess image
        image_bytes = await file.read()
        img_array = preprocess_image(image_bytes)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0])) * 100

        # Get class name
        predicted_class = class_names[str(predicted_index)]

        # Get dosage recommendation
        dosage_info = get_dosage(predicted_class)

        # Check healthy or diseased
        if "healthy" in predicted_class.lower():
            status = "Healthy"
            HEALTHY_COUNT.inc()
        else:
            status = "Diseased"
            DISEASED_COUNT.inc()

        # Track prediction
        PREDICTION_COUNT.labels(
            disease=predicted_class,
            status=status
        ).inc()

        # Format result
        plant_name   = predicted_class.split("___")[0].replace("_", " ")
        disease_name = predicted_class.split("___")[1].replace("_", " ") \
                       if "___" in predicted_class else "Unknown"

        duration = time.time() - start_time

        return {
            "status":        "success",
            "plant":         plant_name,
            "disease":       disease_name,
            "health_status": status,
            "confidence":    f"{confidence:.2f}%",
            "raw_class":     predicted_class,
            "response_time": f"{duration:.3f}s",
            "recommendation": {
                "medicine":   dosage_info["medicine"],
                "dosage":     dosage_info["dosage"],
                "frequency":  dosage_info["frequency"],
                "precaution": dosage_info["precaution"],
                "severity":   dosage_info["severity"]
            }
        }

    except Exception as e:
        REQUEST_COUNT.labels(
            method="POST",
            endpoint="/predict",
            status=500
        ).inc()
        return {
            "status":  "error",
            "message": str(e)
        }

@app.get("/classes")
def get_classes():
    return {
        "total_classes": len(class_names),
        "classes":       list(class_names.values())
    }

@app.get("/stats")
def get_stats():
    """Get prediction statistics"""
    return {
        "model_accuracy": "97.76%",
        "total_classes":  len(class_names),
        "dataset":        "PlantVillage",
        "total_images":   54305,
        "deployment":     "Kubernetes + Docker",
        "monitoring":     "Prometheus + Grafana"
    }