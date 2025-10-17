# steps/data_drift_detector.py
from zenml import step
import os
import pandas as pd
from evidently.report import Report

# ✅ For evidently==0.4.33 (latest versions)
from evidently.metrics import DatasetDriftMetric


@step(enable_cache=False)
def data_drift_detector(reference_data_path: str, current_data_path: str) -> bool:
    """Detects data drift between reference and current datasets."""
    print("[data_drift_detector] 🚀 Running data drift detection...")

    # --- Load datasets ---
    reference_data = pd.read_csv(reference_data_path)
    current_data = pd.read_csv(current_data_path)

    # --- Run drift analysis ---
    report = Report(metrics=[DatasetDriftMetric()])
    report.run(reference_data=reference_data, current_data=current_data)

    # --- Save report ---
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/data_drift_report.html"
    report.save_html(report_path)
    print(f"[data_drift_detector] 📊 Drift report saved at: {report_path}")

    # --- Extract drift info ---
    try:
        report_dict = report.as_dict()
        drift_info = report_dict["metrics"][0]["result"]
        drift_detected = drift_info.get("dataset_drift", False)
        drift_share = drift_info.get("share_drifted_features", 0)
    except Exception as e:
        print(f"[data_drift_detector] ⚠️ Could not parse drift info: {e}")
        drift_detected = False
        drift_share = 0

    print(f"[data_drift_detector] 📈 Drift share: {drift_share:.2f}")
    print(f"[data_drift_detector] {'⚠️ Drift detected!' if drift_detected else '✅ No drift detected.'}")

    return drift_detected
