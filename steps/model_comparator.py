# steps/model_comparator.py
from zenml import step
from mlflow.tracking import MlflowClient

@step(enable_cache=False)
def model_comparator(new_val_acc: float, model_name: str) -> bool:
    """
    Compare the newly trained model with the current Production model.
    Returns True if the new model is better and should be promoted.
    """
    print(f"[model_comparator] 🔍 Comparing new model (val_acc={new_val_acc:.4f}) "
          f"with Production model metrics...")

    client = MlflowClient()
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            print("[model_comparator] ℹ️ No Production model found — promoting new one by default.")
            return True

        prod_model = versions[0]
        prod_run = client.get_run(prod_model.run_id)
        prod_val_acc = float(prod_run.data.metrics.get("val_accuracy", 0.0))

        print(f"[model_comparator] 📊 Production val_acc={prod_val_acc:.4f}")
        if new_val_acc > prod_val_acc:
            print("[model_comparator] ✅ New model performs better — recommend promotion.")
            return True
        else:
            print("[model_comparator] 🚫 New model is worse — keeping current Production model.")
            return False
    except Exception as e:
        print(f"[model_comparator] ⚠️ Comparison failed: {e}. Promoting new model by default.")
        return True
