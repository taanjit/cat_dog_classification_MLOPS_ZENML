# steps/model_deployer.py
from zenml import step
import mlflow
from mlflow.tracking import MlflowClient
import os


@step(enable_cache=False)
def model_deployer(model_name: str) -> str:
    print(f"[model_deployer] 🚀 Fetching latest Production model for '{model_name}' from MLflow...")
    client = MlflowClient()

    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        print(f"[model_deployer] ⚠️ No Production version found for model '{model_name}'. Skipping deployment.")
        return "No production model available"

    prod_model = versions[0]
    model_uri = f"models:/{model_name}/{prod_model.version}"

    model = mlflow.keras.load_model(model_uri)
    os.makedirs("artifacts/deployed_model", exist_ok=True)
    model_path = f"artifacts/deployed_model/{model_name}_prod.keras"
    model.save(model_path)

    print(f"[model_deployer] ✅ Deployed latest Production model (v{prod_model.version}) to {model_path}")
    return model_path
