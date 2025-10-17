# steps/alert_notifier.py
from zenml import step


@step(enable_cache=False)
def alert_notifier(drift_detected: bool, performance_metrics: dict):
    """Notifies team if drift or performance degradation occurs."""
    print("[alert_notifier] 🔔 Checking alerts...")

    if drift_detected:
        print("⚠️ ALERT: Data drift detected! Consider retraining your model.")
    if performance_metrics["accuracy"] < 0.6:
        print("⚠️ ALERT: Model accuracy dropped below threshold!")
    if not drift_detected and performance_metrics["accuracy"] >= 0.6:
        print("✅ System stable — no alerts.")
