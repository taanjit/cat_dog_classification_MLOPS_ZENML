from pipelines.training_pipeline import training_pipeline

if __name__ == "__main__":
    print("🏁 Starting training pipeline...")
    # ✅ Simply execute the pipeline
    training_pipeline(data_dir="data")



# Would you like me to extend this by showing how to automatically promote a new model to “Production” in MLflow when its accuracy exceeds the current one (i.e., a complete MLOps auto-promotion loop)?