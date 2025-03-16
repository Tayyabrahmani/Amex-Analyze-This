import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

mlflow.set_tracking_uri("https://dagshub.com/tayyabrahmani/credit-card-prediction.mlflow")
mlflow.set_experiment("credit-card-default-prediction")

# Load dataset
data = pd.read_csv("Input_Data/Training_dataset_Original.csv")
X = data.drop(columns=["default_ind"])
y = data["default_ind"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Log parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("train_accuracy", model.score(X_train, y_train))
    mlflow.log_metric("test_accuracy", model.score(X_test, y_test))

    # Log Model
    mlflow.sklearn.log_model(model, "random_forest_model")
