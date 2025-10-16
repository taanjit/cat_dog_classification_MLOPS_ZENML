from typing import Tuple
from zenml import step

@step
def data_preprocessor(train_dir: str, val_dir: str) -> Tuple[str, str]:
    """
    Step 2: Data preprocessor — currently a placeholder.
    Just returns train/validation directories for use by the trainer.
    """
    print(f"✅ Preprocessing done (paths validated).")
    return train_dir, val_dir
