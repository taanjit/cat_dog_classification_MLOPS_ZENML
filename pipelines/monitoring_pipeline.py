# pipelines/monitoring_pipeline.py
from zenml import pipeline
from steps.performance_monitor import performance_monitor
from steps.data_drift_detector import data_drift_detector
from steps.alert_notifier import alert_notifier


@pipeline(enable_cache=False)
def monitoring_pipeline(reference_data_path: str, current_data_path: str, y_true: list, y_pred: list):
    """Pipeline for performance & data drift monitoring."""
    drift_detected = data_drift_detector(reference_data_path, current_data_path)
    performance_metrics = performance_monitor(y_true, y_pred)
    alert_notifier(drift_detected, performance_metrics)
