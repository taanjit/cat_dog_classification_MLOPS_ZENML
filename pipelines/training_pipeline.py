from zenml import pipeline
from steps.data_loader import data_loader
from steps.data_preprocessor import data_preprocessor
from steps.model_trainer import model_trainer

@pipeline
def training_pipeline(data_dir: str):
    """
    Pipeline: Load data → preprocess (paths) → train model
    """
    train_dir, val_dir = data_loader(data_dir)
    train_dir, val_dir = data_preprocessor(train_dir, val_dir)
    model = model_trainer(train_dir, val_dir)
    return model
