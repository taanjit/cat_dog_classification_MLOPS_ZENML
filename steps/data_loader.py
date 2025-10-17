# steps/data_loader.py

from typing import Tuple
from zenml import step

@step
def data_loader(data_dir: str) -> Tuple[str, str]:
    """
    Step 1: Loads train and validation data paths.
    Returns both directories as outputs.
    """
    train_dir = f"{data_dir}/train"
    val_dir = f"{data_dir}/validation"
    print(f"📂 Loaded data from {data_dir}")
    return train_dir, val_dir



# UPDATE THE DVC
# dvc add data/train data/validation