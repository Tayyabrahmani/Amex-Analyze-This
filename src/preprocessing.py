import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Paths
RAW_DATA_PATH = "Input_Data/Training_dataset_Original.csv"
PROCESSED_DATA_PATH = "Input_Data/processed_data.csv"

def load_data(path):
    """Load raw dataset and replace 'na', 'N/A', 'missing' with NaN."""
    df = pd.read_csv(path, low_memory=False, dtype=str)  # Force reading everything as string
    df.replace(["na", "N/A", "missing"], np.nan, inplace=True)

    return df

def handle_missing_values(df):
    """Fill missing values with appropriate strategies."""
    # Convert numeric columns back to float (excluding default_ind)
    for col in df.columns:
        if col != "default_ind" and df[col].str.isnumeric().all():
            df[col] = pd.to_numeric(df[col])

    # Ensure 'default_ind' is treated as integer
    if "default_ind" in df.columns:
        df["default_ind"] = df["default_ind"].astype(int)

    # Impute numerical columns with median
    num_cols = df.select_dtypes(include=['number']).columns
    imputer = SimpleImputer(strategy='median')
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # Impute categorical columns with most frequent value
    cat_cols = df.select_dtypes(include=['object']).columns
    imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = imputer.fit_transform(df[cat_cols])

    return df

def encode_categorical_variables(df):
    """Encode categorical variables using Label Encoding."""
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        encoder = LabelEncoder()
        df[col] = df[col].astype(str)  # Convert all categorical columns to strings
        df[col] = encoder.fit_transform(df[col])
    return df

def scale_features(df):
    """Scale numerical features using Standard Scaler, excluding 'default_ind'."""
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['number']).columns

    # Exclude 'default_ind' from scaling
    num_cols = [col for col in num_cols if col != "default_ind"]

    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def save_processed_data(df, path):
    """Save the cleaned dataset."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def preprocess():
    """Main function to run preprocessing steps."""
    print("Loading data...")
    df = load_data(RAW_DATA_PATH)

    print("Handling missing values...")
    df = handle_missing_values(df)

    print("Encoding categorical variables...")
    df = encode_categorical_variables(df)

    print("Scaling features...")
    df = scale_features(df)

    print("Saving processed data...")
    save_processed_data(df, PROCESSED_DATA_PATH)

    print("Preprocessing completed!")

if __name__ == "__main__":
    preprocess()
