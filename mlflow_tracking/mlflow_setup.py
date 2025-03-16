import mlflow
import dagshub

# DagsHub Credentials (Update with your username & repo name)
DAGSHUB_USERNAME = "tayyabrahmani"
DAGSHUB_REPO = "Amex-Analyze-This"

def setup_mlflow():
    """
    Initializes DagsHub and MLflow Tracking URI for all scripts.
    Call this function at the start of any script using MLflow.
    """
    tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("credit-card-default-prediction")
    
    # Initialize DagsHub Logging (Runs Only Once)
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)

    print(f"✅ MLflow & DagsHub initialized: {tracking_uri}")
