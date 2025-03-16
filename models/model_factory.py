from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def create_random_forest(n_estimators=100):
    """Create and return a Random Forest model."""
    return RandomForestClassifier(n_estimators=n_estimators, random_state=42)

def create_xgboost():
    """Create and return an XGBoost model."""
    return xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
