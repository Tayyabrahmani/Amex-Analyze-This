import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
from models.train_models import train_models
from models.evaluate_models import evaluate_model
from mlflow_tracking.mlflow_setup import setup_mlflow
# Initialize MLflow and DagsHub once
setup_mlflow()

def run_mlflow_experiment():
    """Run MLflow experiment separately after models are trained."""
    
    # Train models first
    train_models()

    # Evaluate models
    evaluate_model("random_forest")
    evaluate_model("xgboost")

    print("Experiments tracked successfully in MLFlow!")

if __name__ == "__main__":
    run_mlflow_experiment()
