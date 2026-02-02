import pandas as pd
import joblib
from pathlib import Path


BASE_PATH = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = BASE_PATH / "data/raw"
PROCESSED_DATA_DIR = BASE_PATH / "data/processed"
MODEL_DIR = BASE_PATH / "models"



def load_train():
    """load train CSV"""
    train = pd.read_csv(RAW_DATA_DIR / "train.csv")
    return train


def load_test():
    """load test CSV"""
    test = pd.read_csv(RAW_DATA_DIR / "test.csv")
    return test


def load_store():
    """load store CSV"""
    store = pd.read_csv(RAW_DATA_DIR / "store.csv")
    return store


def load_processed_data(data_name=None):
    """
    Load the latest or specific processed pickle file from PROCESSED_DATA_DIR.
    
    Returns:
        object: The loaded object from the latest file, or None if no file is found.
    """
    data_files = list(PROCESSED_DATA_DIR.glob("*.pkl"))
    
    if not data_files:
        print("No processed pickle files found in the directory.")
        return None
    
    if data_name:
        data_path = PROCESSED_DATA_DIR / data_name
        
        if not data_path.exists():
            print(f"Processed data file '{data_name}' not found.")
            return None
    
    else:
        data_path = max(data_files, key=lambda f: f.stat().st_mtime)
    
    print(f"Loading latest file: {data_path.name}")
    
    data = joblib.load(data_path)
    print(f"Successfully loaded: {data_path.name}")
    return data


def load_model(model_name=None):
    """
    Load a latest or specific model file from MODEL_DIR.
    
    Returns:
        object: The loaded model, or None if not found.
    """
    model_files = list(MODEL_DIR.glob("*.pkl"))

    if not model_files:
        print("No model files found.")
        return None
    
    if model_name:
        model_path = MODEL_DIR / model_name
        if not model_path.exists():
            print(f"Model '{model_name}' not found.")
            return None

    else:
        model_path = max(model_files, key=lambda f: f.stat().st_mtime)

    print(f"Loading latest model: {model_path.name}")

    model = joblib.load(model_path)
    print("Model loaded successfully.")
    return model