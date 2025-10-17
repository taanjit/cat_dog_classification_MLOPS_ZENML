import mlflow
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import threading
import time

from mlflow.tracking import MlflowClient

app = FastAPI(
    title="Cat vs Dog Classifier API",
    description="FastAPI service with auto-updating Production model from MLflow",
    version="1.1.0"
)

MODEL_NAME = "cat_dog_classifier"
MODEL_STAGE = "Production"
POLL_INTERVAL = 60  # seconds to check for new production version

client = MlflowClient()
model = None
current_model_version = None
lock = threading.Lock()


def load_latest_production_model():
    """Loads the latest Production model from MLflow or local fallback."""
    global model, current_model_version

    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if not versions:
            print(f"⚠️ No Production version found for '{MODEL_NAME}'.")
            return

        latest_version = versions[0].version

        if latest_version != current_model_version:
            print(f"🔁 Detected new Production version: {latest_version}. Reloading model...")
            new_model = mlflow.keras.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
            with lock:
                model = new_model
                current_model_version = latest_version
            print(f"✅ Loaded model version {latest_version} from MLflow.")

    except Exception as e:
        print(f"⚠️ Could not load Production model from MLflow: {e}")
        # fallback
        fallback_path = "artifacts/cat_dog_model.keras"
        if os.path.exists(fallback_path):
            print(f"📦 Loading fallback model from {fallback_path}")
            with lock:
                model = load_model(fallback_path)
                current_model_version = "local"
        else:
            print("❌ No model available — please train and promote one first.")


def auto_refresh_model():
    """Background thread that checks MLflow for updated Production models."""
    while True:
        load_latest_production_model()
        time.sleep(POLL_INTERVAL)


@app.on_event("startup")
def startup_event():
    """Start the model auto-reloader thread."""
    print("🚀 Starting model auto-refresh thread...")
    thread = threading.Thread(target=auto_refresh_model, daemon=True)
    thread.start()


@app.get("/health")
def health_check():
    """Check if API and model are healthy."""
    with lock:
        status = "ready" if model else "no model loaded"
        version = current_model_version
    return {"status": status, "model_version": version}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict cat or dog using the latest loaded model."""
    with lock:
        active_model = model
        version = current_model_version

    if active_model is None:
        return {"error": "No model loaded. Train and promote one first."}

    # Save image temporarily
    img_path = f"temp_{file.filename}"
    with open(img_path, "wb") as f:
        f.write(await file.read())

    try:
        img = image.load_img(img_path, target_size=(180, 180))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = active_model.predict(img_array)[0][0]
        label = "Dog" if prediction > 0.5 else "Cat"

        return {
            "prediction": label,
            "confidence": float(prediction),
            "model_version": version
        }
    finally:
        os.remove(img_path)
