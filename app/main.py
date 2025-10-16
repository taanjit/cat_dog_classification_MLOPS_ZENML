from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np, os

app = FastAPI(title="🐾 Cats vs Dogs Classifier")

MODEL_PATH = "artifacts/cat_dog_model.keras"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Train it first.")

model = load_model(MODEL_PATH)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    img = image.load_img(temp_path, target_size=(150,150))
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0][0]
    os.remove(temp_path)
    return {"prediction": "dog 🐶" if pred > 0.5 else "cat 🐱"}
