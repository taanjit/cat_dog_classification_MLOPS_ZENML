import subprocess
import os
from zenml import step
from typing import Tuple

@step
def data_loader(data_dir: str) -> Tuple[str, str]:
    """
    Step 1: Ensures DVC data is pulled and available locally.
    Returns train/validation directory paths.
    """

    # Pull the correct dataset version from DVC
    try:
        print("📦 Pulling latest data version from DVC...")
        subprocess.run(["dvc", "pull"], check=True)
        print("✅ DVC data pulled successfully.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ DVC pull failed: {e}")
        raise

    # Define train and validation directories
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError("Train or validation directory missing after DVC pull.")

    # Log current DVC status
    try:
        dvc_status = subprocess.check_output(["dvc", "status"]).decode().strip()
    except Exception as e:
        dvc_status = f"Could not fetch DVC status: {e}"

    print(f"🔖 DVC Data Status:\n{dvc_status}")

    return train_dir, val_dir
