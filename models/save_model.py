import os
import pickle

# Define the model directory
MODEL_DIR = "models/trained_models/"
os.makedirs(MODEL_DIR, exist_ok=True)

def save_model(model, model_name, version="latest"):
    """
    Saves a trained model with versioning.

    Parameters:
    - model: Trained ML model
    - model_name (str): Name of the model (e.g., "random_forest", "xgboost")
    - version (str): Version number (e.g., "v1", "v2") or "latest" by default

    Saves the model as: models/trained_models/{model_name}_{version}.pkl
    """
    model_path = os.path.join(MODEL_DIR, f"{model_name}_{version}.pkl")
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"✅ Model saved: {model_path}")
    return model_path
