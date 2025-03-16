import os
import pickle

# Define the model directory
MODEL_DIR = "models/trained_models/"

def load_model(model_name, version="v1"):
    """
    Loads a trained model from file.

    Parameters:
    - model_name (str): Name of the model (e.g., "random_forest", "xgboost")
    - version (str): Version number (e.g., "v1", "v2") or "latest" by default

    Returns:
    - Loaded model

    Example:
    >>> model = load_model("random_forest", "v1")
    """
    model_filename = f"{model_name}_{version}.pkl"  # Correct filename format
    model_path = os.path.join(MODEL_DIR, model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"✅ Model loaded: {model_path}")
    return model
