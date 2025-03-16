import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import yaml
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from models.model_factory import create_random_forest, create_xgboost
from mlflow_tracking.mlflow_setup import setup_mlflow

# Paths
PROCESSED_DATA_PATH = "Input_Data/processed_data.csv"
MODEL_DIR = "models/trained_models/"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load version from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
MODEL_VERSION = config["model_version"]

setup_mlflow()


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

    # 🔥 Register Models in MLflow (NO Local Saving)
    with mlflow.start_run():
        # Log and register Random Forest
        rf_model_uri = mlflow.sklearn.log_model(rf_model, "random_forest_model")
        mlflow.register_model(rf_model_uri.model_uri, "random_forest_model")

        # Log and register XGBoost
        xgb_model_uri = mlflow.sklearn.log_model(xgb_model, "xgboost_model")
        mlflow.register_model(xgb_model_uri.model_uri, "xgboost_model")

    print(f"Models trained and registered in MLflow as version: {MODEL_VERSION}")

if __name__ == "__main__":
    train_models()
