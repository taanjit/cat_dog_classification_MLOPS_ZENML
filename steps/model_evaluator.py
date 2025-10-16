from zenml import step

@step
def model_evaluator(model, val_gen):
    """
    Step 4: Evaluate the trained model on validation data.
    """
    print("📊 Evaluating model...")
    loss, accuracy = model.evaluate(val_gen, verbose=1)
    print(f"✅ Validation Accuracy: {accuracy * 100:.2f}%")
    return accuracy
