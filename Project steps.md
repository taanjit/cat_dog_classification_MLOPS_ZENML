🧭 Project Overview

We’ll use ZenML pipelines to orchestrate the workflow:

📁 cats-vs-dogs-zenml/
├── data/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── validation/
│       ├── cats/
│       └── dogs/
├── steps/
│   ├── data_loader.py
│   ├── data_preprocessor.py
│   ├── model_trainer.py
│   ├── model_evaluator.py
│   ├── predictor.py
├── pipelines/
│   ├── training_pipeline.py
│   └── inference_pipeline.py
├── app/
│   └── main.py   # FastAPI app for predictions
├── requirements.txt
└── run_pipeline.py

🧩 Step 1. Environment Setup
# Create environment
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

# Install dependencies
pip install zenml tensorflow pillow fastapi uvicorn
zenml init

🧱 Step 2. Data Loading Step — steps/data_loader.py
from zenml import step
from tensorflow.keras.preprocessing.image import ImageDataGenerator

@step
def data_loader(data_dir: str, target_size=(150, 150), batch_size=32):
    """
    Load training and validation data using ImageDataGenerator.
    """
    datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    val_gen = datagen.flow_from_directory(
        f"{data_dir}/validation",
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    return train_gen, val_gen

🧼 Step 3. Data Preprocessing Step — steps/data_preprocessor.py

(For now, this can be minimal, since rescaling is handled in ImageDataGenerator — but we include this step for version control and extensibility.)

from zenml import step

@step
def data_preprocessor(train_gen, val_gen):
    """
    Placeholder for preprocessing (could include augmentation, normalization, etc.)
    """
    # For now, just return as is
    return train_gen, val_gen

⚙️ Step 4. Model Training — steps/model_trainer.py
from zenml import step
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

@step
def model_trainer(train_gen, val_gen, epochs: int = 5):
    """
    Train a simple CNN model for Cats vs Dogs classification.
    """
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(train_gen, epochs=epochs, validation_data=val_gen)
    return model

📊 Step 5. Model Evaluation — steps/model_evaluator.py
from zenml import step

@step
def model_evaluator(model, val_gen):
    """
    Evaluate the model on validation data.
    """
    loss, accuracy = model.evaluate(val_gen)
    print(f"Validation Accuracy: {accuracy*100:.2f}%")
    return accuracy

🧠 Step 6. Prediction Step (for inference pipeline) — steps/predictor.py
from zenml import step
import numpy as np
from tensorflow.keras.preprocessing import image

@step
def predictor(model, img_path: str):
    """
    Make a single prediction using trained model.
    """
    img = image.load_img(img_path, target_size=(150,150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    return "dog" if pred[0][0] > 0.5 else "cat"

🔁 Step 7. Define Training Pipeline — pipelines/training_pipeline.py
from zenml import pipeline
from steps.data_loader import data_loader
from steps.data_preprocessor import data_preprocessor
from steps.model_trainer import model_trainer
from steps.model_evaluator import model_evaluator

@pipeline
def training_pipeline(data_dir: str):
    train_gen, val_gen = data_loader(data_dir)
    train_gen, val_gen = data_preprocessor(train_gen, val_gen)
    model = model_trainer(train_gen, val_gen)
    acc = model_evaluator(model, val_gen)

🧩 Step 8. Define Inference Pipeline — pipelines/inference_pipeline.py
from zenml import pipeline
from steps.predictor import predictor

@pipeline
def inference_pipeline(model, img_path: str):
    result = predictor(model, img_path)
    return result

⚡ Step 9. Run Training — run_pipeline.py
from pipelines.training_pipeline import training_pipeline

if __name__ == "__main__":
    pipeline = training_pipeline(data_dir="data")
    pipeline.run()


✅ This will:

Load data

Preprocess

Train model

Evaluate and version it automatically (ZenML handles versioning of artifacts)


💾 Step 10. Data and Model Version Control

ZenML automatically tracks:

Data versions (artifacts)

Model versions (via step outputs)

You can inspect them:

zenml artifact list
zenml model list
zenml pipeline list

🚀 Step 11. FastAPI App — app/main.py
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = FastAPI(title="Cats vs Dogs Classifier")

model = load_model("artifacts/latest_model")  # Or use ZenML artifact store path

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img_path = f"temp_{file.filename}"
    with open(img_path, "wb") as buffer:
        buffer.write(await file.read())

    img = image.load_img(img_path, target_size=(150,150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    result = "dog 🐶" if prediction[0][0] > 0.5 else "cat 🐱"
    return {"prediction": result}


Run the app:

uvicorn app.main:app --reload


Then open:
👉 http://127.0.0.1:8000/docs


1️⃣ Data Versioning
🎯 Goal

Track which data version each model was trained on — so you can reproduce experiments and detect drift later.

✅ ZenML-Compatible Techniques
Option A: DVC (Data Version Control) — Best for file-based datasets

DVC tracks datasets (even large ones) via .dvc metadata files in Git.

You can version-control the raw data in data/train/ and data/validation/.

Each pipeline run can log the data commit hash used.

Integration pattern:

pip install dvc
dvc init
dvc add data/train data/validation
git add data/*.dvc .gitignore
git commit -m "Versioned training data"