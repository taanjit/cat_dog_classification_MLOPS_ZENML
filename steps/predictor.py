from zenml import step
import numpy as np
from tensorflow.keras.preprocessing import image

@step
def predictor(model, img_path: str):
    """
    Step 5: Predict whether a given image is a cat or dog.
    """
    img = image.load_img(img_path, target_size=(150,150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    label = "dog 🐶" if pred[0][0] > 0.5 else "cat 🐱"

    print(f"Prediction for {img_path}: {label}")
    return label
