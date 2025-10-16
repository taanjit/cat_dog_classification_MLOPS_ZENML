from zenml import pipeline
from steps.predictor import predictor

@pipeline
def inference_pipeline(model, img_path: str):
    """
    Simple inference pipeline for making predictions using trained model.
    """
    result = predictor(model, img_path)
    return result
