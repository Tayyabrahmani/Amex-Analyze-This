# Amex-Analyze-This: Credit Card Default Prediction Pipeline

This repository implements a **Credit Card Default Prediction Pipeline** using:
- **MLflow** (Experiment tracking & Model Registry)
- **DVC** (Data & pipeline versioning)
- **DagsHub** (Remote tracking for MLflow & DVC)

---

## 📂 Project Structure
```
📂 credit_card_prediction/
│── 📂 Input_Data/
│   ├── Training_dataset_Original.csv   # Raw dataset
│   ├── processed_data.csv              # Preprocessed dataset (DVC tracked)
│
│── 📂 src/
│   ├── preprocessing.py                 # Preprocessing (cleaning, encoding, scaling)
│
│── 📂 models/
│   ├── model_factory.py                 # Create ML models (RandomForest, XGBoost)
│   ├── train_models.py                  # Train & register models in MLflow
│   ├── evaluate_models.py               # Evaluate models from MLflow registry
│   ├── scoring.py                        # Custom scoring function (cost & points)
│
│── 📂 mlflow_tracking/
│   ├── mlflow_setup.py                  # Initialize MLflow & DagsHub
│   ├── mlflow_runner.py                  # Run MLFLow
│
│── 📂 utils/
│   ├── update_dvc.py                     # Auto-update `dvc.yaml`
│
│── .dvc/                                # DVC metadata directory
│── dvc.yaml                             # DVC pipeline definition
│── config.yaml                          # Stores model version dynamically
│── requirements.txt                     # Python dependencies
│── README.md                            # Project documentation
```

---

## 🚀 Setup Instructions

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Pip & Virtualenv
- MLflow & DVC

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/Amex-Analyze-This.git
   cd Amex-Analyze-This
   ```

2. Set up a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate   # For macOS/Linux
   venv\Scripts\activate    # For Windows
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Initialize MLflow and DVC:
   ```sh
   mlflow ui &   # Starts MLflow UI
   dvc init      # Initializes DVC
   ```

5. Run the Pipeline
    ```sh
    dvc repro
    ```
---

## 📌 Pipeline Workflow

### 1️⃣ Data Preprocessing
- **Handles missing values** (`na`, `N/A`, `missing` → `NaN`)
- **Encodes categorical variables**
- **Scales numerical features** (excluding target column)

Run preprocessing:
```sh
python src/preprocessing.py
```

---

### 2️⃣ Model Training & Registration
- **Trains Random Forest & XGBoost models**
- **Directly registers models in MLflow**

Run training:
```sh
python models/train_models.py
```

📌 **Check MLflow UI:**
```sh
mlflow ui
```
➡️ **http://127.0.0.1:5000** → **Models** → **Registered models**

---

### 3️⃣ Model Evaluation
- **Loads models from MLflow Model Registry**
- **Uses a custom scoring function** (cost-based evaluation)
- **Logs metrics (total cost, total points) in MLflow**

Run evaluation:
```sh
python models/evaluate_models.py
```

---

### 4️⃣ Track Everything in DVC
After training & evaluation, push tracked files:
```sh
dvc push
git add .
git commit -m "Updated pipeline & model version"
git push origin main
```

---

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.