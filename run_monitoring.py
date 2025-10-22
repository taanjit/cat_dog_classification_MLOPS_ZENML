# run_monitoring_scheduler.py
import os
import time
import random
import pandas as pd
import schedule
from pipelines.monitoring_pipeline import monitoring_pipeline


def ensure_samples_exist():
    """Ensure reference and current sample CSVs exist for monitoring."""
    os.makedirs("data", exist_ok=True)

    # --- Reference Sample (Training Baseline) ---
    ref_path = "data/reference_sample.csv"
    if not os.path.exists(ref_path):
        print("⚙️ Generating reference_sample.csv from training data...")
        base_dir = "data/train"
        cat_dir = os.path.join(base_dir, "Cat")
        dog_dir = os.path.join(base_dir, "Dog")

        if not (os.path.exists(cat_dir) and os.path.exists(dog_dir)):
            raise FileNotFoundError("❌ Training data folders not found under data/train/")

        cats = [("Cat", f"Cat/{img}") for img in os.listdir(cat_dir) if img.lower().endswith((".jpg", ".png"))]
        dogs = [("Dog", f"Dog/{img}") for img in os.listdir(dog_dir) if img.lower().endswith((".jpg", ".png"))]

        sample_size = min(100, len(cats + dogs))
        df = pd.DataFrame(random.sample(cats + dogs, sample_size), columns=["label", "filepath"])
        df.to_csv(ref_path, index=False)
        print(f"✅ Created reference_sample.csv with {sample_size} rows")

    # --- Current Sample (Recent Production Data) ---
    cur_path = "data/current_sample.csv"
    if not os.path.exists(cur_path):
        print("⚙️ Generating current_sample.csv from validation data...")
        base_dir = "data/validation"
        cat_dir = os.path.join(base_dir, "Cat")
        dog_dir = os.path.join(base_dir, "Dog")

        if not (os.path.exists(cat_dir) and os.path.exists(dog_dir)):
            raise FileNotFoundError("❌ Validation data folders not found under data/validation/")

        cats = [("Cat", f"Cat/{img}") for img in os.listdir(cat_dir) if img.lower().endswith((".jpg", ".png"))]
        dogs = [("Dog", f"Dog/{img}") for img in os.listdir(dog_dir) if img.lower().endswith((".jpg", ".png"))]

        sample_size = min(100, len(cats + dogs))
        df = pd.DataFrame(random.sample(cats + dogs, sample_size), columns=["label", "filepath"])
        df.to_csv(cur_path, index=False)
        print(f"✅ Created current_sample.csv with {sample_size} rows")


def run_monitoring():
    """Run the monitoring pipeline once."""
    print("⏱️ Running scheduled monitoring pipeline...")
    ensure_samples_exist()

    reference_path = "data/reference_sample.csv"
    current_path = "data/current_sample.csv"

    # Dummy example predictions — replace with real predictions if available
    y_true = [0, 1, 0, 1, 1, 0, 0, 1]
    y_pred = [0, 1, 0, 1, 1, 0, 0, 1]

    try:
        monitoring_pipeline(
            reference_data_path=reference_path,
            current_data_path=current_path,
            y_true=y_true,
            y_pred=y_pred
        )
        print("✅ Monitoring pipeline run completed successfully.\n")
    except Exception as e:
        print(f"❌ Monitoring pipeline failed: {e}\n")


# --- Scheduler Setup ---
schedule.every(5).minutes.do(run_monitoring)

print("🚀 Monitoring scheduler started (every 5 minutes)...")
run_monitoring()  # Run once immediately on start

while True:
    schedule.run_pending()
    time.sleep(1)
