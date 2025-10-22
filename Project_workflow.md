# run_monitoring_scheduler.py
 - is a core part of your MLOps lifecycle.
 - It automates continuous monitoring of your model in production and acts as a “watchdog” for model health.

🧭 Purpose

This script automates model monitoring — it periodically checks whether your deployed model is still performing well or if your input data has drifted away from what it was trained on.

Essentially, it ensures your deployed ML model remains reliable and accurate over time.

⚙️ Why It’s Needed

Once your model is deployed:

- The input data in production changes (real-world patterns evolve).

- The model’s accuracy can degrade (known as concept drift or data drift).

- Manually checking for these changes is inefficient.

So this scheduler:

- Automatically samples new and old data.

- Runs monitoring logic every few minutes.

- Detects performance or data drift.

- (Optionally) retriggers retraining if drift is detected.

This is a key part of a MLOps feedback loop:

- Train → Deploy → Monitor → Retrain → Redeploy

1️⃣ Imports & Setup

Uses schedule to run tasks periodically (like a lightweight cron job).

Imports your monitoring_pipeline built with ZenML.

2️⃣ Ensuring Sample Data Exists

This function ensures that two CSVs exist:

    data/reference_sample.csv: a sample from training data (your baseline).

    data/current_sample.csv: a sample from current/production data (latest data).

If they don’t exist, it:

    Creates them automatically by taking random 100 samples from training and validation folders.

Each CSV contains columns:

    label | filepath


This ensures you always have reference and current datasets to compare for drift detection.

✅ Purpose: provide consistent baseline & comparison data for Evidently to analyze drift.

3️⃣ Running the Monitoring Pipeline

This function:

Loads the CSVs generated above.

Calls your ZenML pipeline (monitoring_pipeline) that runs:

    data_drift_detector: checks input drift using Evidently.

    performance_monitor: checks accuracy/precision/F1 metrics.

    alert_notifier: triggers alerts if performance drops or drift detected.

    Uses dummy y_true and y_pred arrays here (but can use real predictions if integrated with your model inference system).

✅ Purpose: automate periodic evaluation of the model’s health.

          ┌───────────────┐
          │ Model Training│
          └──────┬────────┘
                 │
        Deploy (MLflow/FastAPI)
                 │
        ┌────────▼────────┐
        │ Monitoring Loop │
        │  (Evidently)    │
        └──────┬──────────┘
               │
     Drift Detected? ─────── Yes ───> Retraining Triggered
               │
              No
               │
        Continue Monitoring




# run_pipeline.py
