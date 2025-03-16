import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import yaml
import pandas as pd
import mlflow
from mlflow_tracking.mlflow_setup import setup_mlflow
from models.load_model import load_model
from models.scoring import scoring_function
from sklearn.model_selection import train_test_split

# Initialize MLflow and DagsHub once
setup_mlflow()

# Paths
PROCESSED_DATA_PATH = "Input_Data/processed_data.csv"
MODEL_DIR = "models/trained_models/"

# Load version from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
MODEL_VERSION = config["model_version"]


def evaluate_model(model_filename):
    """Evaluate a trained model using custom scoring and log results in MLFlow."""

    # Load data
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(columns=["default_ind"])
    y = df["default_ind"].map({1: "Default", 0: "No Default"})  # Convert to categorical labels

    # Split dataset
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load model
    model = load_model(model_filename)

    # Make predictions
    y_pred_numeric = model.predict(X_test)
    y_pred = pd.Series(y_pred_numeric).map({1: "Default", 0: "No Default"})

    # Apply custom scoring function
    scoring_results = scoring_function(y_test, y_pred)

    # Compute total cost & average points
    total_cost = scoring_results["Cost"].sum()
    total_points = scoring_results["Points"].sum()

    # Log evaluation metrics in MLFlow
    with mlflow.start_run():
        mlflow.log_param("model_name", model_filename)
        mlflow.log_metric("total_cost", total_cost)
        mlflow.log_metric("avg_points", total_points)

    # Print evaluation results
    print(f"Model: {model_filename}")
    print(f"Total Cost: ${total_cost}")
    print(f"Average Points: {total_points:.2f}")

if __name__ == "__main__":
    evaluate_model("random_forest")
