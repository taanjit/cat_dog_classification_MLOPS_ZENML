# pipelines/training_pipeline.py

from zenml import pipeline
from steps.data_loader import data_loader
from steps.data_preprocessor import data_preprocessor
from steps.model_trainer import model_trainer
from steps.model_comparator import model_comparator
from steps.model_deployer import model_deployer
from steps.data_validator import data_validator
from steps.data_version_checker import data_version_checker


@pipeline(enable_cache=False)
def training_pipeline(data_dir: str):
    """
    Main MLOps training pipeline for Cat vs Dog Classification.
    
    Pipeline flow:
    1️⃣ Check DVC for data updates.
    2️⃣ Load dataset paths.
    3️⃣ Validate data quality.
    4️⃣ Preprocess data.
    5️⃣ Train a CNN model and log to MLflow.
    6️⃣ Compare new model vs Production model.
    7️⃣ Deploy if the new model performs better.
    """

    # Step 0️⃣: DVC sync + check for dataset changes
    should_train = data_version_checker(data_dir)

    if not should_train:
        print("🟡 No data change detected — skipping training.")
        return

    # Step 1️⃣: Load dataset directories
    train_dir, val_dir = data_loader(data_dir)
    print(f"📂 Data loaded:\n   Train: {train_dir}\n   Validation: {val_dir}")

    # Step 2️⃣: Validate data quality
    data_validator(train_dir, val_dir)

    # Step 3️⃣: Preprocess data (if needed)
    train_dir, val_dir = data_preprocessor(train_dir, val_dir)

    # Step 4️⃣: Train model
    model_path, val_acc = model_trainer(train_dir, val_dir)
    print(f"🏋️ Model trained successfully. Val Accuracy: {val_acc}")

    # Step 5️⃣: Compare new model with existing Production model
    should_promote = model_comparator(
        new_val_acc=val_acc,
        model_name="cat_dog_classifier"
    )

    # Step 6️⃣: Promote and deploy best model
    if should_promote:
        print("🚀 Promoting new model to Production...")
        model_deployer(model_name="cat_dog_classifier")
    else:
        print("⏸️ Keeping existing Production model — new one not better.")
