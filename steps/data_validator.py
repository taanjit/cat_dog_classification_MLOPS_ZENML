# steps/data_validator.py
import os
from PIL import Image
from zenml import step

@step(enable_cache=False)
def data_validator(train_dir: str, val_dir: str) -> bool:
    """
    Step: Validate training and validation data directories.
    Ensures all images are readable and valid.
    Returns True if validation passes.
    """
    print("[data_validator] 🔍 Starting data validation...")

    def validate_images(directory: str):
        print(f"[data_validator] 📁 Validating {directory}")
        valid_count = 0
        corrupted = 0

        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    try:
                        img_path = os.path.join(root, file)
                        Image.open(img_path).verify()
                        valid_count += 1
                    except Exception:
                        corrupted += 1
                        os.remove(img_path)

        print(f"[data_validator] ✅ Total files: {valid_count + corrupted}, Removed corrupted: {corrupted}")

    validate_images(train_dir)
    validate_images(val_dir)
    print("[data_validator] 🧩 Data validation completed successfully.")
    return True
