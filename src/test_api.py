import requests
import json
import os
import sys
from datetime import datetime

# ── Settings ────────────────────────────────────────────────
API_URL  = "http://localhost:8000"
PASS     = "✅ PASS"
FAIL     = "❌ FAIL"
results  = []

print("=" * 60)
print("   CROPAI FULL SYSTEM TEST")
print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ── Helper Functions ─────────────────────────────────────────
def test(name, condition, details=""):
    status = PASS if condition else FAIL
    results.append({"name": name, "status": status, "details": details})
    print(f"{status}  {name}")
    if details:
        print(f"       {details}")

def get(endpoint):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=10)
        return r.json(), r.status_code
    except Exception as e:
        return None, 0

# ── TEST 1: API Health ───────────────────────────────────────
print("\n📡 TEST 1: API Connectivity")
print("-" * 40)

data, code = get("/")
test("API is running",
     code == 200,
     f"Status: {code}")
test("API version is 2.0.0",
     data and data.get("version") == "2.0.0",
     f"Version: {data.get('version') if data else 'N/A'}")
test("API has predict endpoint",
     data and "predict" in str(data),
     "Endpoint /predict found")

# ── TEST 2: Health Check ─────────────────────────────────────
print("\n🏥 TEST 2: Health Check")
print("-" * 40)

data, code = get("/health")
test("Health endpoint works",
     code == 200,
     f"Status: {code}")
test("Model is loaded",
     data and data.get("model") == "loaded",
     f"Model: {data.get('model') if data else 'N/A'}")
test("38 classes loaded",
     data and data.get("classes") == 38,
     f"Classes: {data.get('classes') if data else 'N/A'}")
test("Accuracy is 97.76%",
     data and "97.76" in str(data.get("accuracy", "")),
     f"Accuracy: {data.get('accuracy') if data else 'N/A'}")

# ── TEST 3: Classes Endpoint ─────────────────────────────────
print("\n🌿 TEST 3: Disease Classes")
print("-" * 40)

data, code = get("/classes")
test("Classes endpoint works",
     code == 200,
     f"Status: {code}")
test("Returns 38 classes",
     data and data.get("total_classes") == 38,
     f"Total: {data.get('total_classes') if data else 'N/A'}")
test("Tomato class exists",
     data and any("Tomato" in c for c in data.get("classes", [])),
     "Tomato diseases found")
test("Potato class exists",
     data and any("Potato" in c for c in data.get("classes", [])),
     "Potato diseases found")

# ── TEST 4: Stats Endpoint ───────────────────────────────────
print("\n📊 TEST 4: System Stats")
print("-" * 40)

data, code = get("/stats")
test("Stats endpoint works",
     code == 200,
     f"Status: {code}")
test("Accuracy in stats",
     data and "97.76" in str(data.get("model_accuracy", "")),
     f"Accuracy: {data.get('model_accuracy') if data else 'N/A'}")
test("Kubernetes in deployment",
     data and "Kubernetes" in str(data.get("deployment", "")),
     f"Deployment: {data.get('deployment') if data else 'N/A'}")

# ── TEST 5: Metrics Endpoint ─────────────────────────────────
print("\n📈 TEST 5: Prometheus Metrics")
print("-" * 40)

try:
    r = requests.get(f"{API_URL}/metrics", timeout=10)
    metrics_text = r.text
    test("Metrics endpoint works",
         r.status_code == 200,
         f"Status: {r.status_code}")
    test("Model accuracy metric exists",
         "crop_disease_model_accuracy" in metrics_text,
         "Metric found in Prometheus output")
    test("Request counter exists",
         "crop_disease_requests_total" in metrics_text,
         "Request counter found")
    test("Accuracy value is 97.76",
         "97.76" in metrics_text,
         "Accuracy value correct")
except Exception as e:
    test("Metrics endpoint works", False, str(e))

# ── TEST 6: MongoDB History ──────────────────────────────────
print("\n🗄️  TEST 6: MongoDB Integration")
print("-" * 40)

data, code = get("/history")
test("History endpoint works",
     code == 200,
     f"Status: {code}")
test("History returns data",
     data is not None,
     f"Total predictions: {data.get('total') if data else 0}")

data, code = get("/prediction-stats")
test("Prediction stats works",
     code == 200,
     f"Status: {code}")

# ── TEST 7: Image Prediction ─────────────────────────────────
print("\n🔍 TEST 7: Disease Prediction")
print("-" * 40)

# Find a test image
test_image_path = None
test_folder = "data/processed/test"

if os.path.exists(test_folder):
    for class_folder in os.listdir(test_folder):
        class_path = os.path.join(test_folder, class_folder)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            if images:
                test_image_path = os.path.join(class_path, images[0])
                test_class = class_folder
                break

if test_image_path and os.path.exists(test_image_path):
    try:
        with open(test_image_path, "rb") as f:
            files = {"file": ("test.jpg", f, "image/jpeg")}
            r = requests.post(
                f"{API_URL}/predict",
                files=files,
                timeout=30
            )
            pred_data = r.json()

        test("Prediction endpoint works",
             r.status_code == 200,
             f"Status: {r.status_code}")
        test("Prediction returns plant name",
             "plant" in pred_data,
             f"Plant: {pred_data.get('plant', 'N/A')}")
        test("Prediction returns disease",
             "disease" in pred_data,
             f"Disease: {pred_data.get('disease', 'N/A')}")
        test("Prediction returns confidence",
             "confidence" in pred_data,
             f"Confidence: {pred_data.get('confidence', 'N/A')}")
        test("Prediction returns dosage",
             "recommendation" in pred_data,
             f"Medicine: {pred_data.get('recommendation', {}).get('medicine', 'N/A')}")
        test("Health status returned",
             "health_status" in pred_data,
             f"Status: {pred_data.get('health_status', 'N/A')}")

        print(f"\n       Test image class: {test_class}")
        print(f"       Predicted plant : {pred_data.get('plant', 'N/A')}")
        print(f"       Predicted disease: {pred_data.get('disease', 'N/A')}")
        print(f"       Confidence      : {pred_data.get('confidence', 'N/A')}")
        print(f"       Medicine        : {pred_data.get('recommendation', {}).get('medicine', 'N/A')}")

    except Exception as e:
        test("Prediction endpoint works", False, str(e))
else:
    print("⚠️  No test image found - skipping prediction test")
    print("   Make sure data/processed/test folder exists")

# ── TEST 8: Project Files ────────────────────────────────────
print("\n📁 TEST 8: Project Structure")
print("-" * 40)

required_files = [
    "api/main.py",
    "src/train.py",
    "src/preprocess.py",
    "src/dosage.py",
    "src/database.py",
    "src/performance.py",
    "notebooks/eda.py",
    "docker/Dockerfile",
    "k8s/deployment.yaml",
    "k8s/service.yaml",
    "models/crop_disease_model.h5",
    "models/class_names.json",
    "frontend/index.html",
    "monitoring/dashboard.html",
    "mlflow/mlflow_tracking.py",
    ".github/workflows/ci-cd.yml",
    "requirements.txt",
    "README.md",
]

for file_path in required_files:
    exists = os.path.exists(file_path)
    test(f"File exists: {file_path}", exists, "")

# ── FINAL SUMMARY ────────────────────────────────────────────
print("\n" + "=" * 60)
print("   TEST SUMMARY")
print("=" * 60)

passed = sum(1 for r in results if r["status"] == PASS)
failed = sum(1 for r in results if r["status"] == FAIL)
total  = len(results)

print(f"Total Tests : {total}")
print(f"Passed      : {passed} ✅")
print(f"Failed      : {failed} ❌")
print(f"Pass Rate   : {(passed/total*100):.1f}%")
print("=" * 60)

if failed == 0:
    print("ALL TESTS PASSED! System is ready for Review 2!")
elif failed <= 3:
    print("MOSTLY PASSING! Minor issues to fix.")
else:
    print("SOME TESTS FAILED! Check the failures above.")

print("=" * 60)

# Save test report
report = {
    "timestamp":  datetime.now().isoformat(),
    "total":      total,
    "passed":     passed,
    "failed":     failed,
    "pass_rate":  f"{(passed/total*100):.1f}%",
    "results":    results
}

with open("models/test_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\nTest report saved to: models/test_report.json")