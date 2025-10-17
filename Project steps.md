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



# Branch data_version

🧩 1️⃣ Data Versioning
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


Then in your ZenML data_loader step:

import subprocess
from zenml import step

@step
def data_loader(data_dir: str):
    # Log the current DVC hash for traceability
    dvc_hash = subprocess.check_output(["dvc", "status", "-c"]).decode().strip()
    print(f"🔖 DVC data version: {dvc_hash}")
    return data_dir


Every pipeline run will automatically track the data hash (via ZenML’s metadata).

Option B: ZenML Artifact Store + Dataset Metadata

ZenML itself tracks all data as artifacts. You can explicitly log dataset hashes:

import hashlib, os
from zenml import step

@step
def data_loader(data_dir: str):
    hash_md5 = hashlib.md5()
    for root, _, files in os.walk(data_dir):
        for f in files:
            with open(os.path.join(root, f), "rb") as file:
                hash_md5.update(file.read())
    print(f"📦 Data version hash: {hash_md5.hexdigest()}")
    return data_dir


This integrates seamlessly with ZenML’s pipeline lineage view.

🧩 2️⃣ Feature Store
🎯 Goal

Have a single source of truth for engineered features — shared between training and inference.

✅ Best Tools
Option A: Feast (Open Source) — Most common

Feast lets you define feature sets (e.g., “image stats”, “color histograms”, etc.) and store them locally or in online databases (Redis, BigQuery, etc.).

Install & initialize:

pip install feast
feast init feature_repo
cd feature_repo


Example feature definition:

from feast import FeatureStore, Entity, FeatureView, Field
from feast.types import Float32

image = Entity(name="image_id")

feature_view = FeatureView(
    name="image_features",
    entities=[image],
    schema=[
        Field(name="brightness", dtype=Float32),
        Field(name="contrast", dtype=Float32),
    ],
    online=True,
    batch_source=...
)


You can fetch features in ZenML steps like:

from feast import FeatureStore

@step
def feature_loader(image_id: int):
    store = FeatureStore(repo_path="feature_repo")
    features = store.get_online_features(
        feature_refs=["image_features:brightness", "image_features:contrast"],
        entity_rows=[{"image_id": image_id}]
    ).to_dict()
    return features


Feast integrates directly with ZenML through its Feast integration:

zenml integration install feast


Then you can register a FeastFeatureStore as part of your ZenML stack.

Option B: Pandas / Parquet Feature Store (lightweight local setup)

If you’re staying local, maintain features as Parquet files (fast + versionable):

feature_store/
├── v1/
│   └── image_features.parquet
├── v2/
│   └── image_features.parquet


Your feature_store_step can simply load the latest version based on timestamp.

🧩 3️⃣ Data Validation
🎯 Goal

Automatically validate datasets before training to ensure:

No schema mismatches

No missing critical columns

No class imbalance drift

✅ Techniques
Option A: Great Expectations (Powerful & ZenML-integrated)

Install:

pip install great_expectations
zenml integration install great_expectations


Then create a validation step:

from zenml import step
import great_expectations as ge

@step
def data_validation_step(data_dir: str):
    df = ge.read_csv(f"{data_dir}/metadata.csv")
    df.expect_column_values_to_not_be_null("label")
    df.expect_column_values_to_be_in_set("label", ["cat", "dog"])
    df.expect_table_row_count_to_be_between(1000, 20000)
    validation_results = df.validate()
    if not validation_results["success"]:
        raise ValueError("❌ Data validation failed.")
    print("✅ Data validation passed.")
    return data_dir


Integrate before data_preprocessor in your pipeline:

from steps.data_validation import data_validation_step

@pipeline
def training_pipeline(data_dir: str):
    data_dir = data_loader(data_dir)
    data_dir = data_validation_step(data_dir)
    train_dir, val_dir = data_preprocessor(data_dir, data_dir)
    model = model_trainer(train_dir, val_dir)

Option B: TensorFlow Data Validation (TFDV) — For TF pipelines

If your data is TensorFlow-based (e.g., TFRecords), use:

pip install tensorflow-data-validation


and run schema checks like:

import tensorflow_data_validation as tfdv

@step
def data_validation_step(data_dir: str):
    stats = tfdv.generate_statistics_from_csv(f"{data_dir}/train.csv")
    schema = tfdv.infer_schema(stats)
    anomalies = tfdv.validate_statistics(stats, schema)
    if anomalies.anomaly_info:
        raise ValueError(f"Data anomalies detected: {anomalies}")

🧠 Putting it All Together — New ZenML Workflow
data_loader
   ↓
data_version_logger     (DVC or hash-based)
   ↓
data_validation_step    (Great Expectations)
   ↓
feature_store_loader    (Feast or local parquet)
   ↓
model_trainer
   ↓
model_evaluator

🧰 Recommended Stack
Category	Tool	Integration
Data Versioning	DVC or ZenML Metadata	Track dataset versions & hashes
Feature Store	Feast	Register as ZenML stack component
Data Validation	Great Expectations	ZenML data_validation_step
Orchestration	ZenML	Orchestrate and log pipeline runs
Deployment	FastAPI	Serve predictions

Would you like me to extend your current ZenML project (with actual working steps for data versioning + validation + feature store integration code)?
I can show the exact files to add (steps/data_validation.py, steps/data_version.py, etc.) with ready-to-run examples.


# ⚙️ 1. Experiment Tracking
✅ Goal:

Track hyperparameters, metrics, data version, and artifacts for each model training run.

💡 Techniques:
Tool	                    Type	        ZenML Integration	            Notes
MLflow	                    Open-source	    ✅ Native integration	    Most popular for experiment tracking, artifact logging
Weights & Biases (W&B)	    SaaS	        ✅ ZenML plugin	            Great dashboards, collaborative
Neptune.ai	                SaaS	        ✅ ZenML plugin	            Enterprise-grade, secure
ZenML Experiment Tracker	Built-in	    ✅ Default	                For small setups, easy to start
🔧 Implementation (MLflow example)

Step 1: Install MLflow integration

zenml integration install mlflow -y


Step 2: Register MLflow experiment tracker

zenml experiment-tracker register mlflow_tracker \
    --flavor=mlflow \
    --tracking_uri="file:./mlruns"


Step 3: Update your active stack

zenml stack update default -e mlflow_tracker


Step 4: Modify your model_trainer.py

from zenml import step
from zenml.client import Client
import mlflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

@step(experiment_tracker="mlflow_tracker")
def model_trainer(train_dir: str, val_dir: str):
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_gen = datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32, class_mode="binary")
    val_gen = datagen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=32, class_mode="binary")

    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("batch_size", 32)

    history = model.fit(train_gen, validation_data=val_gen, epochs=5)

    mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])
    mlflow.log_metric("val_loss", history.history["val_loss"][-1])

    model.save("artifacts/cat_dog_model.keras")
    mlflow.tensorflow.log_model(model, "model")

    return "artifacts/cat_dog_model.keras"


🖥️ Now every pipeline run will show up in the MLflow dashboard:

mlflow ui
# visit http://127.0.0.1:5000

🧩 2. Code Versioning
✅ Goal:

Associate each model with the exact code and environment that created it.

💡 Techniques:
Technique	Tool	How to Integrate
Git Commit Tagging	Git	Log commit hash to MLflow / ZenML metadata
ZenML Git Metadata	Built-in	ZenML automatically tracks the Git commit of your pipeline run
Git Hooks / CI/CD	GitHub Actions, GitLab	Auto-trigger pipelines on code pushes
🔧 Implementation

Add this snippet in your model_trainer step:

import subprocess

commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
mlflow.log_param("git_commit", commit_hash)


Now, your MLflow experiment will clearly show:

optimizer = adam
batch_size = 32
git_commit = a7f09b3


✅ This creates full traceability between data version (DVC) and code version (Git).

🧠 3. Training Pipeline Enhancements
✅ Goal:

Build modular, automated training using ZenML pipelines.

💡 Techniques:
Concept	Description
Pipelines	Orchestrate modular steps (data_loader → preprocess → train → evaluate)
Orchestrators	ZenML supports local, Airflow, Kubeflow, Argo, etc.
Hyperparameter tuning	Use ZenML’s integrations with Optuna, KerasTuner, or custom loops
Pipeline caching	ZenML automatically reuses previous step outputs to save time
Example ZenML Pipeline (enhanced)
from zenml import pipeline
from steps.data_loader import data_loader
from steps.data_preprocessor import data_preprocessor
from steps.model_trainer import model_trainer

@pipeline(enable_cache=True)
def training_pipeline(data_dir: str):
    train_dir, val_dir = data_loader(data_dir)
    train_dir, val_dir = data_preprocessor(train_dir, val_dir)
    model_path = model_trainer(train_dir, val_dir)
    return model_path

🧱 4. Putting It All Together — Your Workflow Now Looks Like:
📦 DVC for data versioning
   ↓
⚙️ ZenML pipeline orchestration
   ↓
🔬 MLflow for experiment tracking
   ↓
🧠 Git for code versioning
   ↓
📊 Model registry (optional)


Each run tracks:

Dataset version (DVC hash)

Code version (Git commit)

Hyperparameters and metrics (MLflow)

Model artifact (saved .keras + registered model)

Pipeline lineage (ZenML dashboard)

🚀 Optional Add-ons
Goal	Tool	Benefit
Feature Store	Feast, Hopsworks	Manage training/serving consistency
Data Validation	Great Expectations	Automatically validate schema & drift
Deployment	BentoML, FastAPI, MLflow Serving	Serve models as REST APIs

Would you like me to give you a code-level implementation showing how to wire together
ZenML + DVC + MLflow + Git tracking in one unified training_pipeline.py?
I can generate the complete working files (steps/, pipelines/, requirements.txt, etc.) for you.