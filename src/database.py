import os
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# ── MongoDB Connection ───────────────────────────────────────
MONGO_URI = "mongodb+srv://shreenidhibetageri051_db_user:cropai123@tasktracker.m0yxo0c.mongodb.net/?appName=tasktracker"

# Database and collection names
DB_NAME         = "cropai_db"
PREDICTIONS_COL = "predictions"
USERS_COL       = "users"

# ── Connect to MongoDB ───────────────────────────────────────
def get_database():
    """Connect to MongoDB and return database"""
    try:
        client = MongoClient(MONGO_URI)
        # Test connection
        client.admin.command('ping')
        print("✅ Connected to MongoDB Atlas!")
        db = client[DB_NAME]
        return db
    except ConnectionFailure as e:
        print(f"❌ MongoDB connection failed: {e}")
        return None

# Initialize database
db = get_database()

# ── Prediction Functions ─────────────────────────────────────
def save_prediction(data: dict):
    """Save prediction result to MongoDB"""
    try:
        if db is None:
            return None

        collection = db[PREDICTIONS_COL]

        # Create prediction document
        prediction_doc = {
            "plant":         data.get("plant", "Unknown"),
            "disease":       data.get("disease", "Unknown"),
            "health_status": data.get("health_status", "Unknown"),
            "confidence":    data.get("confidence", "0%"),
            "raw_class":     data.get("raw_class", ""),
            "medicine":      data.get("recommendation", {}).get("medicine", ""),
            "dosage":        data.get("recommendation", {}).get("dosage", ""),
            "severity":      data.get("recommendation", {}).get("severity", ""),
            "timestamp":     datetime.now(),
            "date":          datetime.now().strftime("%Y-%m-%d"),
            "time":          datetime.now().strftime("%H:%M:%S")
        }

        result = collection.insert_one(prediction_doc)
        print(f"✅ Prediction saved to MongoDB: {result.inserted_id}")
        return str(result.inserted_id)

    except Exception as e:
        print(f"❌ Error saving prediction: {e}")
        return None

def get_all_predictions():
    """Get all predictions from MongoDB"""
    try:
        if db is None:
            return []

        collection = db[PREDICTIONS_COL]
        predictions = list(collection.find(
            {},
            {"_id": 0}
        ).sort("timestamp", -1).limit(50))
        return predictions

    except Exception as e:
        print(f"❌ Error getting predictions: {e}")
        return []

def get_prediction_stats():
    """Get prediction statistics"""
    try:
        if db is None:
            return {}

        collection = db[PREDICTIONS_COL]

        total      = collection.count_documents({})
        healthy    = collection.count_documents({"health_status": "Healthy"})
        diseased   = collection.count_documents({"health_status": "Diseased"})

        # Most common disease
        pipeline = [
            {"$match": {"health_status": "Diseased"}},
            {"$group": {"_id": "$disease", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 1}
        ]
        most_common = list(collection.aggregate(pipeline))
        top_disease = most_common[0]["_id"] if most_common else "None"

        return {
            "total_predictions": total,
            "healthy_count":     healthy,
            "diseased_count":    diseased,
            "top_disease":       top_disease,
            "healthy_percent":   f"{(healthy/total*100):.1f}%" if total > 0 else "0%",
            "diseased_percent":  f"{(diseased/total*100):.1f}%" if total > 0 else "0%"
        }

    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        return {}

def get_recent_predictions(limit=10):
    """Get recent predictions"""
    try:
        if db is None:
            return []

        collection = db[PREDICTIONS_COL]
        predictions = list(collection.find(
            {},
            {"_id": 0}
        ).sort("timestamp", -1).limit(limit))
        return predictions

    except Exception as e:
        print(f"❌ Error getting recent predictions: {e}")
        return []

# ── Test Connection ──────────────────────────────────────────
if __name__ == "__main__":
    print("Testing MongoDB connection...")
    db_test = get_database()

    if db_test is not None:
        print("✅ MongoDB is working!")
        print(f"Database: {DB_NAME}")

        # Test save
        test_data = {
            "plant":         "Tomato",
            "disease":       "Early Blight",
            "health_status": "Diseased",
            "confidence":    "95.23%",
            "raw_class":     "Tomato___Early_blight",
            "recommendation": {
                "medicine": "Mancozeb 75 WP",
                "dosage":   "2.5 grams per litre",
                "severity": "Medium"
            }
        }
        save_prediction(test_data)

        # Test stats
        stats = get_prediction_stats()
        print(f"Stats: {stats}")

        # Test recent
        recent = get_recent_predictions(5)
        print(f"Recent predictions: {len(recent)}")
        print("✅ All MongoDB functions working!")
    else:
        print("❌ MongoDB connection failed!")