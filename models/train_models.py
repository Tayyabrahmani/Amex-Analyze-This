import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from models.model_factory import create_random_forest, create_xgboost
from models.save_model import save_model

# Paths
PROCESSED_DATA_PATH = "Input_Data/processed_data.csv"
MODEL_DIR = "models/trained_models/"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load version from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
MODEL_VERSION = config["model_version"]


def train_models():
    """Train models and save them."""

    # Load preprocessed data
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(columns=["default_ind"])
    y = df["default_ind"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create models
    rf_model = create_random_forest()
    xgb_model = create_xgboost()

    # Train models
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # Save models
    save_model(rf_model, "random_forest", MODEL_VERSION)
    save_model(xgb_model, "xgboost", MODEL_VERSION)
    print("Models trained and saved!")

if __name__ == "__main__":
    train_models()
