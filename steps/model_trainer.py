# steps/model_trainer.py

from zenml import step
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow
import subprocess
import os


@step(experiment_tracker="mlflow_local")
def model_trainer(train_dir: str, val_dir: str) -> str:
    """
    Step 3: Train the CNN model and log experiments with MLflow.
    Includes Git commit hash + DVC version tracking for full reproducibility.
    """

    # --- ⚙️ Initialize Data Generators ---
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = datagen.flow_from_directory(
        train_dir, target_size=(150, 150), batch_size=32, class_mode="binary"
    )
    val_gen = datagen.flow_from_directory(
        val_dir, target_size=(150, 150), batch_size=32, class_mode="binary"
    )

    # --- 🧠 Define Model ---
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # --- 📊 Log Hyperparameters ---
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 5)
    mlflow.log_param("image_size", "(150,150)")

    # --- 🧾 Log Git Commit Hash ---
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )
        mlflow.log_param("git_commit", commit_hash)
        print(f"📘 Logged Git commit: {commit_hash}")
    except Exception as e:
        print(f"⚠️ Could not fetch Git commit hash: {e}")

    # --- 🗃️ Log DVC Data Version ---
    try:
        dvc_hash = subprocess.check_output(["dvc", "status", "-c"]).decode().strip()
        mlflow.log_param("dvc_status", dvc_hash)
        print(f"📘 Logged DVC data version: {dvc_hash}")
    except Exception as e:
        print(f"⚠️ Could not fetch DVC version: {e}")

    # --- 🚀 Train Model ---
    print("[model_trainer] 🚀 Starting model training...")
    history = model.fit(train_gen, validation_data=val_gen, epochs=5)

    # --- 📈 Log Metrics ---
    val_accuracy = history.history["val_accuracy"][-1]
    val_loss = history.history["val_loss"][-1]
    mlflow.log_metric("val_accuracy", val_accuracy)
    mlflow.log_metric("val_loss", val_loss)
    print(f"[model_trainer] 📊 Validation Accuracy: {val_accuracy * 100:.2f}%")

    # --- 💾 Save Model ---
    model_dir = "artifacts"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "cat_dog_model.keras")
    model.save(model_path)

    # --- 🔗 Log Model to MLflow ---
    mlflow.tensorflow.log_model(model, "model")
    print(f"[model_trainer] ✅ Model saved to: {model_path}")

    return model_path
