import os
import sys
import joblib
import pandas as pd
from typing import Dict

from mlflow.tracking import MlflowClient

# Import mlflow only if needed, without failing startup
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not available, using local model fallback.", file=sys.stderr)

# Local model fallback path
MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model.pkl")

# MLflow model settings
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', '').strip()
MLFLOW_MODEL_NAME = os.getenv('MLFLOW_REGISTERED_NAME', os.getenv('MLFLOW_MODEL_NAME', 'ChurnModel'))
MLFLOW_MODEL_ALIAS = os.getenv('MLFLOW_ALIAS', os.getenv('MLFLOW_MODEL_ALIAS', 'champion')).strip()

model = None
model_source = None

# Attempt MLflow loading if configured
if MLFLOW_AVAILABLE and MLFLOW_TRACKING_URI:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        
        # Support both new Alias syntax (@champion) and legacy Stage syntax (/Production)
        model_version = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, MLFLOW_MODEL_ALIAS)
        
        # 2. Беремо прямий URI артефакту з конкретного Run (наприклад: runs:/<run_id>/model)
        model_uri = f"runs:/{model_version.run_id}/model"
            
        print(f'Loading model via {model_uri}')

        model = mlflow.sklearn.load_model(model_uri)
        model_source = f"MLflow registry ({model_uri})"
        print(f"✓ Model loaded successfully from {model_source}")
    except Exception as e:
        print(f"Warning: Could not load from MLflow ({e}). Falling back to local model...", file=sys.stderr)

# Fallback to local .pkl file
if model is None:
    try:
        model = joblib.load(MODEL_PATH)
        model_source = f"local file ({MODEL_PATH})"
        print(f"✓ Model loaded from {model_source}")
    except Exception as e:
        print(f"Warning: Could not load local model: {e}", file=sys.stderr)
        model = None
        model_source = None

if model is None:
    print("Warning: No model loaded - predictions will fail until a model becomes available.", file=sys.stderr)


def preprocess_features(features: Dict) -> pd.DataFrame:
    """Return a DataFrame formatted for the saved sklearn Pipeline.
    
    Performs minimal numeric coercion and supplies missing timestamp features.
    """
    df = pd.DataFrame([features])

    # Inject current date dynamically if required by feature pipeline schema
    if 'RecordDate' not in df.columns:
        df['RecordDate'] = pd.Timestamp.now().strftime('%Y-%m-%d')

    # Ensure numerical columns are converted cleanly
    for col in ['TotalCharges', 'MonthlyCharges', 'tenure']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill NA with neutral defaults
    df = df.fillna({
        'TotalCharges': 0.0,
        'MonthlyCharges': 0.0,
        'tenure': 0
    })

    return df


def predict_churn(features: Dict) -> Dict:
    """Accepts feature dictionary and returns prediction probabilities."""
    if model is None:
        return {"error": "Model not loaded"}

    try:
        X = preprocess_features(features)

        # Prefer predict_proba for probabilities, fall back to binary predict
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X)[0][1]
        elif hasattr(model, 'predict'):
            pred = model.predict(X)[0]
            prob = float(pred)
        else:
            return {"error": "Loaded model object does not support prediction methods."}

        pred = 1 if prob >= 0.5 else 0

        return {
            "churn_probability": round(float(prob), 4),
            "churn_prediction": int(pred),
            "model_source": model_source,
            "features_used": list(X.columns)
        }
    except Exception as e:
        return {"error": str(e)}