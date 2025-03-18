import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import yaml
import pandas as pd
import mlflow
from mlflow_tracking.mlflow_setup import setup_mlflow
from mlflow.tracking import MlflowClient
from models.scoring import scoring_function
from sklearn.model_selection import train_test_split

# Initialize MLflow and DagsHub once
setup_mlflow()

# Paths
PROCESSED_DATA_PATH = "Input_Data/processed_data.csv"

# Load version from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
MODEL_VERSION = config["model_version"]
MAX_BUDGET = config.get("max_process_amount", 50000)

def load_latest_model(model_name, stage="Staging"):
    """
    Load the latest version of a model from MLflow Model Registry.

    :param model_name: Name of the registered model.
    :param stage: Model stage ('Staging', 'Production', etc.).
    :return: Loaded MLflow model.
    """
    client = MlflowClient()
    
    # Fetch latest version in the specified stage
    latest_versions = client.search_model_versions(f"name='{model_name}'")
    if not latest_versions:
        raise ValueError(f"No model found for '{model_name}' in DagsHub MLflow!")

    # Get the latest model URI
    latest_model_info = sorted(latest_versions, key=lambda x: int(x.version), reverse=True)[0]
    model_uri = latest_model_info.source

    print(f"Loading model {model_name} from: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)

def compute_expected_points(prob_default):
    """Calculate expected points based on predicted probabilities."""
    p_d = prob_default  # Probability of 'Default' (1)
    p_nd = 1 - prob_default  # Probability of 'No Default' (0)

    # Expected points calculation
    expected_points = (p_d * 100) + (p_nd * 100) - (p_d * 50)
    return expected_points

def load_and_prepare_data():
    """Load and preprocess the dataset."""
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(columns=["default_ind"])
    y = df["default_ind"]

    # Split dataset
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_test, y_test

def process_applications(evaluation_df):
    """Process applications while maximizing points within budget."""
    remaining_budget = MAX_BUDGET
    total_points = 0
    total_processed = 0
    total_cost = 0

    for _, row in evaluation_df.iterrows():
        actual = row["Actual"]
        pred_prob = row["Predicted_Prob"]

        # Determine cost based on scoring table
        if pred_prob >= 0.5:  # Model predicts "Default"
            cost = 10
            points = 100 if actual == 1 else 0  # Gain 100 for correct, 0 for incorrect
        else:  # Model predicts "No Default"
            cost = 10 if actual == 1 else 5  # Cost $10 for incorrect, $5 for correct
            points = -50 if actual == 1 else 100  # Lose 50 for incorrect, gain 100 for correct

        # Check if budget allows processing
        if remaining_budget - cost < 0:
            break

        # Process application
        remaining_budget -= cost
        total_points += points
        total_cost += cost
        total_processed += 1

    return total_processed, total_cost, total_points, remaining_budget

def evaluate_model(model_name):
    """Evaluate a trained model and maximize total points while staying within budget."""
    # Load model
    model = load_latest_model(model_name, stage="Staging")

    # Load and preprocess data
    X_test, y_test = load_and_prepare_data()

    # Predict probabilities
    y_pred_probs = model.predict_proba(X_test)[:, 1]

    # Create evaluation DataFrame
    evaluation_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted_Prob": y_pred_probs,
    })

    # Compute expected points for each application
    evaluation_df["Expected_Points"] = evaluation_df["Predicted_Prob"].apply(compute_expected_points)

    # Sort applications by highest expected points
    evaluation_df = evaluation_df.sort_values(by="Expected_Points", ascending=False)

    # Process applications and get results
    total_processed, total_cost, total_points, remaining_budget = process_applications(evaluation_df)

    # Log evaluation metrics in MLFlow
    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("total_cost", total_cost)
        mlflow.log_metric("total_points", total_points)
        mlflow.log_metric("total_processed", total_processed)
        mlflow.log_metric("remaining_budget", remaining_budget)

    # Print evaluation results
    print(f"Model: {model_name}")
    print(f"Total Processed Applications: {total_processed}")
    print(f"Total Cost Spent: ${total_cost}")
    print(f"Total Points Scored: {total_points}")
    print(f"Remaining Budget: ${remaining_budget}")

if __name__ == "__main__":
    evaluate_model("random_forest_model")
    evaluate_model("xgboost_model")
