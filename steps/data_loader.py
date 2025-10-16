from typing import Tuple
from zenml import step
import os

@step
def data_loader(data_dir: str) -> Tuple[str, str]:
    """
    Step 1: Return paths for training and validation directories.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError("Train or validation directory not found.")

    print(f"📁 Train dir: {train_dir}")
    print(f"📁 Validation dir: {val_dir}")
    return train_dir, val_dir
