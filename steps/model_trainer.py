# steps/model_trainer.py
import os
import subprocess
from typing import Tuple

import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from zenml import step

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


@step(enable_cache=False)
def model_trainer(train_dir: str, val_dir: str) -> Tuple[str, float]:
    """
    Step: Train and register a CNN model for cat–dog classification.
    Returns:
        local_model_path (str): path to saved model file
        val_acc (float): validation accuracy
    """

    print("[model_trainer] 🚀 Starting model training...")

    # --- 1️⃣ Prepare Data ---
    img_height, img_width = 180, 180
    batch_size = 32

    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="binary",
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="binary",
    )

    # --- 2️⃣ Build CNN Model ---
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(img_height, img_width, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # --- 3️⃣ MLflow Setup ---
    mlflow.set_experiment("cat_dog_classification")

    with mlflow.start_run(run_name="cnn_cat_dog") as run:
        # ✅ Log Git commit for reproducibility
        try:
            commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            mlflow.log_param("git_commit", commit_hash)
        except Exception as e:
            print(f"[model_trainer] ⚠️ Could not log git commit: {e}")

        # ✅ Log hyperparameters
        mlflow.log_params(
            {
                "img_height": img_height,
                "img_width": img_width,
                "batch_size": batch_size,
                "optimizer": "adam",
            }
        )

        # --- 4️⃣ Train the Model ---
        history = model.fit(train_gen, validation_data=val_gen, epochs=10)  # can increase epochs
        val_acc = history.history["val_accuracy"][-1]
        mlflow.log_metric("val_accuracy", val_acc)

        # --- 5️⃣ Save Model Locally ---
        os.makedirs("artifacts", exist_ok=True)
        local_model_path = "artifacts/cat_dog_model.keras"
        model.save(local_model_path)
        print(f"[model_trainer] 💾 Model saved locally at: {local_model_path}")

        # --- 6️⃣ Log and Register in MLflow ---
        mlflow.keras.log_model(
            model,
            "model",
            registered_model_name="cat_dog_classifier",
        )

        client = MlflowClient()
        model_name = "cat_dog_classifier"

        # --- 7️⃣ Promote Latest Model to Production ---
        try:
            latest_versions = client.get_latest_versions(model_name, stages=["None"])
            if latest_versions:
                latest_version = latest_versions[-1].version
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                print(f"[model_trainer] 🚀 Promoted model version {latest_version} to 'Production'")
            else:
                print("[model_trainer] ⚠️ No new model version found to promote.")
        except Exception as e:
            print(f"[model_trainer] ⚠️ Model registry update failed: {e}")

        # --- ✅ Return Model Path and Accuracy ---
        print(f"[model_trainer] ✅ Model trained successfully. Val Accuracy: {val_acc:.4f}")
        return local_model_path, val_acc
