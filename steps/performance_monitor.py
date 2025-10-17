# steps/performance_monitor.py
from zenml import step
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@step(enable_cache=False)
def performance_monitor(y_true: list, y_pred: list) -> dict:
    """Computes model performance metrics."""
    print("[performance_monitor] 📈 Evaluating model performance...")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }

    for k, v in metrics.items():
        print(f"   {k}: {v:.3f}")

    return metrics
