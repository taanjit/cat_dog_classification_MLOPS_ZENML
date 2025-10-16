from zenml import step
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

@step
def model_trainer(
    train_dir: str,
    val_dir: str,
    epochs: int = 5,
    batch_size: int = 32,
):
    """
    Train CNN model for cats vs dogs and save as a .keras file.
    """
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_gen = datagen.flow_from_directory(train_dir,
                                            target_size=(150, 150),
                                            batch_size=batch_size,
                                            class_mode="binary")
    val_gen = datagen.flow_from_directory(val_dir,
                                          target_size=(150, 150),
                                          batch_size=batch_size,
                                          class_mode="binary")

    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(150,150,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer=Adam(1e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    print("🚀 Starting model training...")
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    print("✅ Model training completed.")

    # evaluate
    _, accuracy = model.evaluate(val_gen)
    print(f"📊 Validation Accuracy: {accuracy*100:.2f}%")

    # ✅ save model to artifacts folder
    os.makedirs("artifacts", exist_ok=True)
    save_path = os.path.join("artifacts", "cat_dog_model.keras")
    model.save(save_path)
    print(f"💾 Model saved to: {save_path}")

    return save_path
